"""
Complexity Calculator for OCL Expressions

Implements the comprehensive complexity metrics from:
"A new set of metrics for measuring the complexity of OCL expressions"
(Jha, Monahan, Wu - STAF 2025)

Three complexity dimensions:
    1. Structural Complexity (NNR-C, NNC+, DN-CA, WNC, TNC, VRC, WNO, WNM)
    2. Computational Complexity (OC, TCC, CIC)
    3. Dependency Complexity (RUC)

Total Complexity (TC) = w_s * Structural + w_c * Computational + w_d * Dependency
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


# =============================================================================
# Table 1: Verification-driven weights for OCL constructs
# =============================================================================

DEFAULT_OPERATOR_WEIGHTS: Dict[str, float] = {
    # Comparison operators (weight 1.0) - lightweight in verification
    '=': 1.0, '<>': 1.0, '>': 1.0, '<': 1.0, '>=': 1.0, '<=': 1.0,

    # Logical operators (weight 1.5) - require evaluating multiple conditions
    'and': 1.5, 'or': 1.5, 'not': 1.5,

    # Higher logical operators (weight 2.0) - introduce logical dependencies
    'implies': 2.0, 'xor': 2.0,

    # Collection traversal (weight 2.0) - involve iterating over collections
    'select': 2.0, 'collect': 2.0, 'reject': 2.0, 'flatten': 2.0,

    # Quantifiers & higher-order (weight 3.0) - nested iteration, especially
    # when applied to large collections
    'forAll': 3.0, 'exists': 3.0, 'closure': 3.0, 'iterate': 3.0,

    # Type conversion (weight 2.0) - introduces validation overhead
    'oclAsType': 2.0,

    # Set conversion (weight 1.5) - lightweight collection operation
    'asSet': 1.5,

    # --- Operations counted by WNO but NOT by OC -----------------------------
    # OC (compute_oc) sums only the verification-relevant subset
    # (_VERIFICATION_OPS). The arithmetic and string operations below are
    # counted by WNO (which sums every weighted operator), making WNO a strict
    # superset so the two metrics are no longer identical.
    # Arithmetic symbols ('-' is omitted to avoid clashing with the '->' arrow).
    '+': 1.0, '*': 1.0, '/': 1.0,
    'mod': 1.0, 'div': 1.0, 'abs': 1.0, 'max': 1.0, 'min': 1.0,
    'round': 1.0, 'floor': 1.0,
    # String operations.
    'concat': 1.0, 'substring': 1.0, 'toUpper': 1.0, 'toLower': 1.0,
    'indexOf': 1.0, 'toInteger': 1.0, 'toReal': 1.0,
}


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ComplexityWeights:
    """
    User-configurable weights for all complexity calculations.

    Attributes:
        operator_weights: Verification-driven weights per OCL construct (Table 1 defaults)
        structural_weight: Weight for structural dimension in TC (w_s)
        computational_weight: Weight for computational dimension in TC (w_c)
        dependency_weight: Weight for dependency dimension in TC (w_d)
        tnc_alpha: Weight for NNR-C in TNC formula
        tnc_beta: Weight for WNC in TNC formula
        tnc_gamma: Weight for DN-CA in TNC formula
        single_nav_weight: Weight for 1..1 navigations in NNR-C
        collection_nav_weight: Weight for 1..* navigations in NNR-C
        collection_op_depth_multiplier: Multiplier for navigation depth inside collection ops (DN-CA)
    """
    operator_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OPERATOR_WEIGHTS))
    # Dimension weights for TC
    structural_weight: float = 1.0
    computational_weight: float = 1.0
    dependency_weight: float = 1.0
    # TNC sub-weights: TNC = alpha * NNR-C + beta * WNC + gamma * DN-CA
    tnc_alpha: float = 0.4
    tnc_beta: float = 0.3
    tnc_gamma: float = 0.3
    # Navigation cardinality weights (NNR-C)
    single_nav_weight: float = 1.0
    collection_nav_weight: float = 2.0
    # DN-CA depth multiplier when inside collection operations
    collection_op_depth_multiplier: float = 1.5

    def get_operator_weight(self, op: str) -> float:
        """Get the weight for an operator, defaulting to 1.0 for unknown operators."""
        return self.operator_weights.get(op, 1.0)


@dataclass
class ComplexityResult:
    """
    Complete complexity measurement result for a single OCL constraint.

    All three dimensions (structural, computational, dependency) plus
    the combined Total Complexity (TC) score.
    """
    # --- Structural Complexity Metrics ---
    nnr_c: float = 0.0    # Cardinality-aware navigation reference count
    nnc_plus: float = 0.0  # Class navigation count with depth weighting
    dn_ca: float = 0.0    # Context-aware depth of navigation
    wnc: float = 0.0      # Weighted navigation chains
    tnc: float = 0.0      # Total navigation complexity (combined)
    vrc: int = 0           # Variable reference count
    wno: float = 0.0      # Weighted number of operations
    wnm: float = 0.0      # Weighted number of messages
    structural_total: float = 0.0

    # --- Computational Complexity Metrics ---
    oc: float = 0.0       # Operator complexity
    tcc: float = 0.0      # Type conversion complexity
    cic: float = 0.0      # Collection iteration complexity
    computational_total: float = 0.0

    # --- Dependency Complexity Metrics ---
    ruc: int = 0           # Reused constraint count
    dependency_total: float = 0.0

    # --- Total Complexity ---
    tc: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all metrics to a dictionary."""
        return {
            'structural': {
                'nnr_c': round(self.nnr_c, 3),
                'nnc_plus': round(self.nnc_plus, 3),
                'dn_ca': round(self.dn_ca, 3),
                'wnc': round(self.wnc, 3),
                'tnc': round(self.tnc, 3),
                'vrc': self.vrc,
                'wno': round(self.wno, 3),
                'wnm': round(self.wnm, 3),
                'total': round(self.structural_total, 3),
            },
            'computational': {
                'oc': round(self.oc, 3),
                'tcc': round(self.tcc, 3),
                'cic': round(self.cic, 3),
                'total': round(self.computational_total, 3),
            },
            'dependency': {
                'ruc': self.ruc,
                'total': round(self.dependency_total, 3),
            },
            'tc': round(self.tc, 3),
        }


# =============================================================================
# Helper: Extract constraint body (remove context declaration)
# =============================================================================

def _get_body(ocl: str) -> str:
    """Extract the constraint body after 'inv:'."""
    if 'inv:' in ocl:
        return ocl.split('inv:', 1)[1].strip()
    return ocl.strip()


# =============================================================================
# Structural Complexity Metrics
# =============================================================================

# Regex for navigation chains: self.a.b.c or var.x.y
_NAV_CHAIN_RE = re.compile(r'\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)')

# Collection operations that introduce iteration scope
_COLLECTION_ITER_OPS = {'forAll', 'exists', 'select', 'reject', 'collect',
                        'one', 'any', 'sortedBy', 'closure', 'iterate'}

# Regex for collection op invocations: ->opName(
_COLLECTION_OP_RE = re.compile(
    r'->\s*(' + '|'.join(_COLLECTION_ITER_OPS) + r')\s*\('
)

# Regex for iterator variable introduction: ->op(var |
_ITERATOR_VAR_RE = re.compile(
    r'->\s*(?:' + '|'.join(_COLLECTION_ITER_OPS) + r')\s*\(\s*([a-zA-Z_]\w*)\s*\|'
)

# Regex for let variable introduction: let varName
_LET_VAR_RE = re.compile(r'\blet\s+([a-zA-Z_]\w*)\b')

# Regex for message calls: .methodName() or ->methodName()
_MESSAGE_RE = re.compile(r'(?:\.|->)\s*([a-zA-Z_]\w*)\s*\(')


def compute_nnr_c(ocl: str, metamodel=None, context_class: str = '',
                  weights: ComplexityWeights = None) -> float:
    """
    NNR-C: Cardinality-Aware Navigation Reference Count.

    Quantifies navigated relationships, weighting 1..* navigations higher
    than 1..1 navigations because collections introduce iteration overhead.

    Falls back to equal weighting (1.0) when metamodel is unavailable.
    """
    if weights is None:
        weights = ComplexityWeights()

    body = _get_body(ocl)
    chains = _NAV_CHAIN_RE.findall(body)
    total = 0.0

    for chain in chains:
        parts = chain.split('.')
        # Walk the chain, resolving cardinality where possible
        current_class = context_class if parts[0] == 'self' else ''
        start = 1 if parts[0] == 'self' else 0

        for part in parts[start:]:
            if metamodel and current_class:
                # Try as association first
                assoc = next(
                    (a for a in metamodel.get_associations_for(current_class)
                     if a.ref_name == part), None
                )
                if assoc:
                    w = weights.collection_nav_weight if assoc.is_collection else weights.single_nav_weight
                    total += w
                    current_class = assoc.target_class
                    continue
                # Try as attribute
                attr = next(
                    (a for a in metamodel.get_attributes_for(current_class)
                     if a.name == part), None
                )
                if attr:
                    total += weights.single_nav_weight
                    current_class = ''  # attributes are endpoints
                    continue
            # Fallback: unknown navigation step
            total += weights.single_nav_weight

    return total


def compute_nnc_plus(ocl: str) -> float:
    """
    NNC+: Class Navigation Count with depth weighting.

    Deeper navigation paths get higher weights:
      self.customer     -> weight 1
      self.membership.card -> weight 2
    """
    body = _get_body(ocl)
    chains = _NAV_CHAIN_RE.findall(body)
    total = 0.0

    for chain in chains:
        parts = chain.split('.')
        start = 1 if parts[0] == 'self' else 0
        depth = len(parts) - start  # number of navigation steps
        # Weight = depth (deeper chains are more complex)
        total += depth

    return total


def _is_inside_collection_op(ocl: str, position: int) -> bool:
    """
    Check if a position in the OCL string is inside a collection operation body.
    Uses parenthesis depth tracking from the last collection op opening.
    """
    # Find all collection op starts before this position
    depth = 0
    in_collection_op = False
    i = 0
    text = ocl[:position]

    for match in _COLLECTION_OP_RE.finditer(text):
        in_collection_op = True

    # Simpler heuristic: count collection ops before position vs closing parens
    ops_before = len(_COLLECTION_OP_RE.findall(text))
    # Count unmatched opening parens from collection ops
    if ops_before > 0:
        # Track paren depth after last collection op
        last_op = None
        for m in _COLLECTION_OP_RE.finditer(text):
            last_op = m
        if last_op:
            after_op = text[last_op.end():]
            paren_depth = 1  # the opening paren of the collection op
            for ch in after_op:
                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                if paren_depth <= 0:
                    return False
            return paren_depth > 0
    return False


def compute_dn_ca(ocl: str, weights: ComplexityWeights = None) -> float:
    """
    DN-CA: Context-Aware Depth of Navigation.

    Measures maximum navigation depth, with a multiplier when the navigation
    occurs inside collection-based operations (forAll, select, etc.).
    """
    if weights is None:
        weights = ComplexityWeights()

    body = _get_body(ocl)
    max_depth = 0.0

    for match in _NAV_CHAIN_RE.finditer(body):
        chain = match.group()
        parts = chain.split('.')
        start = 1 if parts[0] == 'self' else 0
        depth = len(parts) - start

        # Apply collection op multiplier if inside a quantifier/iterator
        if _is_inside_collection_op(body, match.start()):
            depth *= weights.collection_op_depth_multiplier

        max_depth = max(max_depth, depth)

    return max_depth


def compute_wnc(ocl: str) -> float:
    """
    WNC: Weighted Navigation Chains.

    Each navigation chain's weight equals its length (depth).
    Chains inside collection operations are doubled.
    """
    body = _get_body(ocl)
    total = 0.0

    for match in _NAV_CHAIN_RE.finditer(body):
        chain = match.group()
        parts = chain.split('.')
        start = 1 if parts[0] == 'self' else 0
        chain_length = len(parts) - start

        # Double weight when inside collection ops
        multiplier = 2.0 if _is_inside_collection_op(body, match.start()) else 1.0
        total += chain_length * multiplier

    return total


def compute_tnc(nnr_c: float, wnc: float, dn_ca: float,
                weights: ComplexityWeights = None) -> float:
    """
    TNC: Total Navigation Complexity.

    TNC = alpha * NNR-C + beta * WNC + gamma * DN-CA

    Combines navigation metrics to avoid redundancy in counting.
    """
    if weights is None:
        weights = ComplexityWeights()
    return (weights.tnc_alpha * nnr_c +
            weights.tnc_beta * wnc +
            weights.tnc_gamma * dn_ca)


def compute_vrc(ocl: str) -> int:
    """
    VRC: Variable Reference Count.

    Counts distinct variables introduced by iterator patterns (forAll(b|...),
    select(x|...)) and let expressions (let v = ...).
    """
    body = _get_body(ocl)
    variables = set()

    # Iterator variables: ->forAll(b |, ->select(x |, etc.
    for match in _ITERATOR_VAR_RE.finditer(body):
        variables.add(match.group(1))

    # Let variables: let varName
    for match in _LET_VAR_RE.finditer(body):
        variables.add(match.group(1))

    return len(variables)


def compute_wno(ocl: str, weights: ComplexityWeights = None) -> float:
    """
    WNO: Weighted Number of Operations.

    Sum of Table 1 weights for every occurrence of every OCL operator/keyword.
    """
    if weights is None:
        weights = ComplexityWeights()

    body = _get_body(ocl)
    total = 0.0

    for op, w in weights.operator_weights.items():
        if op.isalpha():
            # Word-boundary matching for keyword operators
            count = len(re.findall(r'\b' + re.escape(op) + r'\b', body))
        else:
            # Symbol operators: escape for regex and count
            count = len(re.findall(re.escape(op), body))
        total += count * w

    return total


def compute_wnm(ocl: str) -> float:
    """
    WNM: Weighted Number of Messages.

    Counts method/message calls like .size(), .isEmpty(), ->forAll(), etc.
    Each message has weight 1.
    """
    body = _get_body(ocl)
    messages = _MESSAGE_RE.findall(body)
    return float(len(messages))


# =============================================================================
# Computational Complexity Metrics
# =============================================================================

# Operators relevant to verification cost (OC focuses on these)
_VERIFICATION_OPS = {
    # Comparison
    '=', '<>', '>', '<', '>=', '<=',
    # Logical
    'and', 'or', 'not', 'implies', 'xor',
    # Collection / Quantifiers
    'forAll', 'exists', 'select', 'reject', 'collect',
    'closure', 'iterate', 'flatten',
    # Type
    'oclAsType',
    # Set
    'asSet',
}

# Type conversion operations with their weights
_TYPE_CONVERSION_OPS = {
    'asSet': 1.5,
    'asBag': 1.5,
    'asSequence': 1.5,
    'asOrderedSet': 1.5,
    'flatten': 2.0,
    'oclAsType': 2.0,
}


def compute_oc(ocl: str, weights: ComplexityWeights = None) -> float:
    """
    OC: Operator Complexity.

    Sum of Table 1 weights for comparison, logical, and collection operators.
    Focuses on constructs that most significantly affect solver performance.
    """
    if weights is None:
        weights = ComplexityWeights()

    body = _get_body(ocl)
    total = 0.0

    for op in _VERIFICATION_OPS:
        w = weights.get_operator_weight(op)
        if op.isalpha():
            count = len(re.findall(r'\b' + re.escape(op) + r'\b', body))
        else:
            count = len(re.findall(re.escape(op), body))
        total += count * w

    return total


def compute_tcc(ocl: str, weights: ComplexityWeights = None) -> float:
    """
    TCC: Type Conversion Complexity.

    Weighted sum of type conversion operations (asSet, flatten, oclAsType, etc.).
    """
    if weights is None:
        weights = ComplexityWeights()

    body = _get_body(ocl)
    total = 0.0

    for op, default_w in _TYPE_CONVERSION_OPS.items():
        # Use user-overridden weight if available, else use default
        w = weights.operator_weights.get(op, default_w)
        count = len(re.findall(r'\b' + re.escape(op) + r'\b', body))
        total += count * w

    return total


def compute_cic(ocl: str) -> float:
    """
    CIC: Collection Iteration Complexity.

    Measures the cost of iterating over collections.
    Nested iterations get multiplied by nesting level.
    Uses a stack-based approach to track quantifier nesting.
    """
    body = _get_body(ocl)

    # Base weights for collection iteration ops
    iter_weights = {
        'forAll': 3.0, 'exists': 3.0, 'closure': 3.0, 'iterate': 3.0,
        'select': 2.0, 'collect': 2.0, 'reject': 2.0,
        'one': 2.0, 'any': 2.0, 'sortedBy': 2.0,
    }

    total = 0.0
    nesting_level = 0
    paren_stack = []  # Track which parens belong to collection ops

    # Tokenize: find collection ops and parens
    tokens = re.finditer(
        r'->(\w+)\s*\(|(\()|(\))',
        body
    )

    for match in tokens:
        op_name = match.group(1)
        open_paren = match.group(2)
        close_paren = match.group(3)

        if op_name and op_name in iter_weights:
            nesting_level += 1
            paren_stack.append(True)  # This paren belongs to a collection op
            # Weight = base_weight * nesting_level
            total += iter_weights[op_name] * nesting_level
        elif open_paren:
            paren_stack.append(False)  # Regular paren
        elif close_paren:
            if paren_stack:
                was_collection = paren_stack.pop()
                if was_collection:
                    nesting_level = max(0, nesting_level - 1)

    return total


# =============================================================================
# Dependency Complexity Metrics
# =============================================================================

def compute_ruc(ocl: str, context: str,
                all_constraints: Optional[List] = None) -> int:
    """
    RUC: Reused Constraint Count.

    Counts how many other constraints share the same context and have
    overlapping navigation paths, indicating dependency/coupling.
    """
    if not all_constraints:
        return 0

    body = _get_body(ocl)
    my_navs = set(_NAV_CHAIN_RE.findall(body))

    if not my_navs:
        return 0

    count = 0
    for other in all_constraints:
        if other.ocl == ocl:
            continue
        if other.context != context:
            continue
        other_body = _get_body(other.ocl)
        other_navs = set(_NAV_CHAIN_RE.findall(other_body))
        # Count overlap
        if my_navs & other_navs:
            count += 1

    return count


# =============================================================================
# Total Complexity
# =============================================================================

def compute_total_complexity(
    ocl: str,
    metamodel=None,
    context_class: str = '',
    all_constraints: Optional[List] = None,
    weights: ComplexityWeights = None,
) -> ComplexityResult:
    """
    Compute all complexity metrics for a single OCL constraint.

    TC = w_s * Structural + w_c * Computational + w_d * Dependency

    Args:
        ocl: OCL constraint text
        metamodel: Optional Metamodel for cardinality-aware analysis
        context_class: The context class name
        all_constraints: Optional list of all constraints (for RUC)
        weights: User-configurable complexity weights

    Returns:
        ComplexityResult with all individual metrics and TC score
    """
    if weights is None:
        weights = ComplexityWeights()

    result = ComplexityResult()

    # --- Structural ---
    result.nnr_c = compute_nnr_c(ocl, metamodel, context_class, weights)
    result.nnc_plus = compute_nnc_plus(ocl)
    result.dn_ca = compute_dn_ca(ocl, weights)
    result.wnc = compute_wnc(ocl)
    result.tnc = compute_tnc(result.nnr_c, result.wnc, result.dn_ca, weights)
    result.vrc = compute_vrc(ocl)
    result.wno = compute_wno(ocl, weights)
    result.wnm = compute_wnm(ocl)

    result.structural_total = result.tnc + result.vrc + result.wno + result.wnm

    # --- Computational ---
    result.oc = compute_oc(ocl, weights)
    result.tcc = compute_tcc(ocl, weights)
    result.cic = compute_cic(ocl)

    result.computational_total = result.oc + result.tcc + result.cic

    # --- Dependency ---
    result.ruc = compute_ruc(ocl, context_class, all_constraints)
    result.dependency_total = float(result.ruc)

    # --- Total Complexity ---
    result.tc = (
        weights.structural_weight * result.structural_total +
        weights.computational_weight * result.computational_total +
        weights.dependency_weight * result.dependency_total
    )

    return result


# =============================================================================
# Backward-compatible difficulty mapping
# =============================================================================

def tc_to_difficulty_label(tc: float) -> str:
    """
    Map Total Complexity score to a human-readable difficulty label.

    Thresholds based on empirical analysis of OCL expressions:
        TC <= 3    -> trivial
        TC <= 8    -> easy
        TC <= 16   -> medium
        TC <= 25   -> hard
        TC > 25    -> expert
    """
    if tc <= 3.0:
        return "trivial"
    if tc <= 8.0:
        return "easy"
    if tc <= 16.0:
        return "medium"
    if tc <= 25.0:
        return "hard"
    return "expert"


def tc_to_score(tc: float) -> int:
    """
    Map TC to a coarse 0/1/2 difficulty score.

    Backward compatible with the old difficulty_score() function:
        0 = easy (TC <= 8)
        1 = medium (TC <= 16)
        2 = hard (TC > 16)
    """
    if tc <= 8.0:
        return 0
    if tc <= 16.0:
        return 1
    return 2
