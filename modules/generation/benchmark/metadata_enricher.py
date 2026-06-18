"""
Rich Metadata Extraction for OCL Constraints

Extracts operators, navigation depth, quantifier depth, and difficulty labels
to create research-grade benchmark metadata.

Now integrates the comprehensive complexity metrics from:
"A new set of metrics for measuring the complexity of OCL expressions"
(Jha, Monahan, Wu - STAF 2025)
"""

import re
from typing import List, Set, Dict, Any, Optional
from modules.core.models import OCLConstraint
from .coverage_tracker import nav_hops, quantifier_depth as coverage_quantifier_depth
from .complexity_calculator import (
    ComplexityWeights,
    ComplexityResult,
    compute_total_complexity,
    tc_to_difficulty_label,
    tc_to_score,
)

# Module-level complexity weights (configurable at runtime)
_complexity_weights: ComplexityWeights = ComplexityWeights()


def set_complexity_weights(weights: ComplexityWeights):
    """Set the complexity weights used for all enrichment calls."""
    global _complexity_weights
    _complexity_weights = weights


def get_complexity_weights() -> ComplexityWeights:
    """Get the current complexity weights."""
    return _complexity_weights


# OCL Operators (grouped by category)
COMPARISON_OPS = ['=', '<>', '<', '>', '<=', '>=']
LOGICAL_OPS = ['and', 'or', 'not', 'implies', 'xor']
COLLECTION_OPS = [
    'forAll', 'exists', 'select', 'reject', 'collect', 
    'size', 'isEmpty', 'notEmpty', 'includes', 'excludes',
    'includesAll', 'excludesAll', 'sum', 'count', 'any',
    'one', 'isUnique', 'sortedBy', 'closure', 'intersection',
    'union', 'including', 'excluding', 'symmetricDifference',
    'flatten', 'asSet', 'asBag', 'asSequence', 'asOrderedSet',
    'first', 'last', 'at', 'indexOf', 'append', 'prepend'
]
STRING_OPS = [
    'concat', 'substring', 'toUpper', 'toLower', 'size',
    'indexOf', 'equalsIgnoreCase', 'startsWith', 'endsWith',
    'characters', 'toInteger', 'toReal', 'matches'
]
ARITHMETIC_OPS = ['+', '-', '*', '/', 'abs', 'div', 'mod', 'max', 'min', 'round', 'floor']
TYPE_OPS = ['oclIsTypeOf', 'oclIsKindOf', 'oclAsType', 'allInstances']
OTHER_OPS = ['if', 'then', 'else', 'endif', 'let', 'in']

ALL_OPERATORS = (
    COMPARISON_OPS + LOGICAL_OPS + COLLECTION_OPS + 
    STRING_OPS + ARITHMETIC_OPS + TYPE_OPS + OTHER_OPS
)


def extract_operators(ocl: str) -> List[str]:
    """
    Extract all OCL operators used in a constraint.
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        List of operator names (deduplicated and sorted)
    """
    found_ops = set()
    ocl_lower = ocl.lower()
    
    # Extract word-based operators (forAll, exists, etc.)
    for op in ALL_OPERATORS:
        if isinstance(op, str) and op.isalpha():
            # Use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(op.lower()) + r'\b'
            if re.search(pattern, ocl_lower):
                found_ops.add(op)
    
    # Extract symbol operators (=, <>, +, -, etc.)
    for op in COMPARISON_OPS + ARITHMETIC_OPS:
        if not op.isalpha():
            if op in ocl:
                found_ops.add(op)
    
    return sorted(list(found_ops))


def count_navigation_depth(ocl: str) -> int:
    """
    Count maximum navigation depth (number of consecutive dot navigations).
    
    Example:
        self.library.books.author.name  -> depth = 4
        self.name -> depth = 1
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        Maximum navigation depth
    """
    # Extract the constraint body (remove context declaration)
    body = ocl
    if 'inv:' in ocl:
        body = ocl.split('inv:', 1)[1]
    
    # Find all navigation chains (sequences of identifiers separated by dots)
    # Pattern: word followed by one or more (.word)
    navigation_pattern = r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+'
    navigations = re.findall(navigation_pattern, body)
    
    if not navigations:
        return 0
    
    # Count dots in each navigation chain to get depth
    max_depth = 0
    for nav in navigations:
        # Count dots + 1 to get number of identifiers
        depth = nav.count('.') + 1
        max_depth = max(max_depth, depth)
    
    return max_depth


def count_quantifier_depth(ocl: str) -> int:
    """
    Count maximum nesting depth of quantifiers (forAll, exists, select, etc.).
    
    Example:
        self.books->forAll(b | b.authors->exists(a | a.age > 30))  -> depth = 2
    
    Args:
        ocl: OCL constraint text
        
    Returns:
        Maximum quantifier nesting depth
    """
    # Collection operations that introduce quantifier scope
    quantifiers = ['forAll', 'exists', 'select', 'reject', 'collect', 
                   'one', 'any', 'sortedBy', 'closure']
    
    max_depth = 0
    current_depth = 0
    
    # Tokenize OCL to track quantifier nesting
    tokens = re.findall(r'\b\w+\b|[(){}|]', ocl)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check if this is a quantifier
        if token in quantifiers:
            # Look ahead for opening parenthesis and pipe (quantifier pattern)
            if i + 2 < len(tokens):
                # Pattern: quantifier ( ... | ... )
                current_depth += 1
                max_depth = max(max_depth, current_depth)
        
        # Track scope ending with closing parenthesis
        elif token == ')':
            # Check if this closes a quantifier scope
            # Simple heuristic: decrease depth if we've entered quantifiers
            if current_depth > 0:
                current_depth -= 1
        
        i += 1
    
    return max_depth


def classify_difficulty(constraint: OCLConstraint,
                        metamodel=None,
                        all_constraints: Optional[List] = None) -> str:
    """
    Classify constraint difficulty using Total Complexity (TC) metrics.

    Uses the verification-driven complexity metrics from Jha et al. (STAF 2025)
    to compute TC and map it to a difficulty label.

    Difficulty levels (TC thresholds):
        - trivial: TC <= 3
        - easy:    TC <= 8
        - medium:  TC <= 16
        - hard:    TC <= 25
        - expert:  TC > 25

    Args:
        constraint: OCLConstraint with metadata
        metamodel: Optional Metamodel for cardinality-aware analysis
        all_constraints: Optional list of all constraints (for RUC)

    Returns:
        Difficulty label: "trivial", "easy", "medium", "hard", or "expert"
    """
    result = compute_total_complexity(
        constraint.ocl,
        metamodel=metamodel,
        context_class=constraint.context,
        all_constraints=all_constraints,
        weights=_complexity_weights,
    )
    return tc_to_difficulty_label(result.tc)


def enrich_constraint_metadata(
    constraint: OCLConstraint,
    metamodel=None,
    all_constraints: Optional[List] = None,
) -> OCLConstraint:
    """
    Enrich OCLConstraint with operators, navigation depth, quantifier depth,
    difficulty label, and full complexity metrics (TC-based).

    Args:
        constraint: OCLConstraint to enrich
        metamodel: Optional Metamodel for cardinality-aware NNR-C
        all_constraints: Optional list of all constraints (for RUC)

    Returns:
        Same constraint with updated metadata dict
    """
    # Legacy metadata (kept for backward compatibility)
    operators = extract_operators(constraint.ocl)
    nav_depth = count_navigation_depth(constraint.ocl)
    quant_depth = count_quantifier_depth(constraint.ocl)

    # Compute full complexity metrics
    complexity_result = compute_total_complexity(
        constraint.ocl,
        metamodel=metamodel,
        context_class=constraint.context,
        all_constraints=all_constraints,
        weights=_complexity_weights,
    )

    # Difficulty: prefer the steered complexity-profile label (e.g. easy/medium/
    # difficult) when present; otherwise derive it from Total Complexity.
    difficulty = constraint.metadata.get('profile') or tc_to_difficulty_label(complexity_result.tc)

    # Update metadata dict (preserve existing metadata)
    constraint.metadata.update({
        'operators_used': operators,
        'navigation_depth': nav_depth,
        'quantifier_depth': quant_depth,
        'difficulty': difficulty,
        'operator_count': len(operators),
        'complexity_metrics': complexity_result.to_dict(),
    })

    return constraint


def extract_families(constraint: OCLConstraint) -> List[str]:
    """
    Extract constraint families based on operators used.
    
    Families:
        - basic: Simple attribute/association constraints
        - collection: Collection operations
        - navigation: Multi-hop navigation
        - quantifier: Universal/existential quantifiers
        - arithmetic: Numeric operations
        - string: String operations
        - advanced: Advanced OCL features
    
    Args:
        constraint: OCLConstraint
        
    Returns:
        List of family names
    """
    ocl = constraint.ocl
    operators = constraint.metadata.get('operators_used', extract_operators(ocl))
    nav_depth = constraint.metadata.get('navigation_depth', count_navigation_depth(ocl))
    quant_depth = constraint.metadata.get('quantifier_depth', count_quantifier_depth(ocl))
    
    families = set()
    
    # Determine families
    if any(op in operators for op in COLLECTION_OPS):
        families.add('collection')
    
    if any(op in operators for op in ['forAll', 'exists']):
        families.add('quantifier')
    
    if nav_depth >= 2:
        families.add('navigation')
    
    if any(op in operators for op in ARITHMETIC_OPS):
        families.add('arithmetic')
    
    if any(op in operators for op in STRING_OPS):
        families.add('string')
    
    if any(op in operators for op in ['closure', 'iterate', 'oclIsTypeOf', 'allInstances', 'let']):
        families.add('advanced')
    
    # Default to basic if no specific family
    if not families:
        families.add('basic')
    
    return sorted(list(families))


def get_enrichment_summary(constraints: List[OCLConstraint]) -> Dict[str, Any]:
    """
    Get summary statistics of enriched constraints.

    Args:
        constraints: List of enriched OCLConstraints

    Returns:
        Dictionary with aggregated statistics including TC distribution
    """
    if not constraints:
        return {}

    difficulties = {}
    all_operators = set()
    nav_depths = []
    quant_depths = []
    families_count = {}
    tc_scores = []

    for c in constraints:
        # Difficulty distribution
        diff = c.metadata.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1

        # All operators used
        ops = c.metadata.get('operators_used', [])
        all_operators.update(ops)

        # Depth distributions
        nav_depths.append(c.metadata.get('navigation_depth', 0))
        quant_depths.append(c.metadata.get('quantifier_depth', 0))

        # Family distribution
        fams = extract_families(c)
        for fam in fams:
            families_count[fam] = families_count.get(fam, 0) + 1

        # TC scores
        cm = c.metadata.get('complexity_metrics', {})
        tc = cm.get('tc', 0.0)
        tc_scores.append(tc)

    summary = {
        'total_constraints': len(constraints),
        'difficulty_distribution': difficulties,
        'unique_operators': len(all_operators),
        'operators_used': sorted(list(all_operators)),
        'navigation_depth': {
            'max': max(nav_depths) if nav_depths else 0,
            'avg': sum(nav_depths) / len(nav_depths) if nav_depths else 0,
            'distribution': {i: nav_depths.count(i) for i in range(max(nav_depths) + 1)} if nav_depths else {}
        },
        'quantifier_depth': {
            'max': max(quant_depths) if quant_depths else 0,
            'avg': sum(quant_depths) / len(quant_depths) if quant_depths else 0,
            'distribution': {i: quant_depths.count(i) for i in range(max(quant_depths) + 1)} if quant_depths else {}
        },
        'family_distribution': families_count,
    }

    # TC distribution statistics
    if tc_scores:
        sorted_tc = sorted(tc_scores)
        n = len(sorted_tc)
        mean_tc = sum(sorted_tc) / n
        median_tc = sorted_tc[n // 2] if n % 2 == 1 else (sorted_tc[n // 2 - 1] + sorted_tc[n // 2]) / 2
        variance = sum((x - mean_tc) ** 2 for x in sorted_tc) / n
        stddev_tc = variance ** 0.5
        summary['total_complexity'] = {
            'min': round(sorted_tc[0], 3),
            'max': round(sorted_tc[-1], 3),
            'mean': round(mean_tc, 3),
            'median': round(median_tc, 3),
            'stddev': round(stddev_tc, 3),
        }

    return summary


def normalize_ocl(ocl: str) -> str:
    """Normalize OCL string for lightweight similarity comparisons."""
    s = re.sub(r"\s+", " ", ocl)
    s = s.replace("self.", "$")
    return s.strip().lower()


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over token sets."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def similarity(c1: OCLConstraint, c2: OCLConstraint) -> float:
    """Lightweight similarity between two constraints."""
    return jaccard(normalize_ocl(c1.ocl), normalize_ocl(c2.ocl))


def difficulty_score(ocl: str) -> int:
    """
    Coarse difficulty bucket based on Total Complexity (TC).

    Returns: 0 (easy), 1 (medium), 2 (hard)

    Backward compatible signature — now TC-driven instead of
    simple hops+depth heuristic.
    """
    result = compute_total_complexity(
        ocl,
        weights=_complexity_weights,
    )
    return tc_to_score(result.tc)
