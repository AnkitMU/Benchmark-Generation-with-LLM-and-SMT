"""
Creates intentionally unsatisfiable (UNSAT) constraints by mutating
satisfiable (SAT) constraints. This is useful for:
- Testing solver completeness
- Creating balanced benchmarks with known SAT/UNSAT ratio
- Validating verification frameworks
"""

import random
import re
from typing import List, Optional, Tuple, Dict, Any
from modules.core.models import OCLConstraint, Metamodel


class OCLMutationHelper:
    """Utilities to preserve valid OCL syntax while mutating constraints."""

    _INV_REGEX = re.compile(
        r'(?s)^(?P<header>.*?)(?P<inv>\binv\b\s*(?:[A-Za-z_][\w-]*\s*)?:)(?P<body>.*)$'
    )

    @staticmethod
    def split_ocl(ocl_string: str) -> Tuple[str, str, str]:
        """
        Split an OCL constraint into (header, inv_token, body).

        Examples:
          "context C\\ninv: x > 0" -> ("context C", "inv:", "x > 0")
          "context C inv myInv: x > 0" -> ("context C", "inv myInv:", "x > 0")
        """
        match = OCLMutationHelper._INV_REGEX.match(ocl_string)
        if match:
            header = match.group('header').rstrip()
            inv_token = match.group('inv').strip()
            body = match.group('body').strip()
            return header, inv_token, body
        # Fallback: treat whole string as body
        return "", "inv:", ocl_string.strip()

    @staticmethod
    def _prefix(header: str) -> str:
        header = header.rstrip()
        return f"{header}\n" if header else ""

    @staticmethod
    def build_unsat_ocl(header: str, inv_token: str, original_body: str, contradiction: str) -> str:
        """Construct a syntactically safe UNSAT OCL string."""
        prefix = OCLMutationHelper._prefix(header)
        return f"{prefix}{inv_token} ({original_body}) and ({contradiction})"

    @staticmethod
    def build_negated_ocl(header: str, inv_token: str, original_body: str) -> str:
        """Construct a syntactically safe negated OCL string."""
        prefix = OCLMutationHelper._prefix(header)
        return f"{prefix}{inv_token} not({original_body})"


class UnsatMutationStrategy:
    """Base class for UNSAT mutation strategies"""
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        """Check if strategy can be applied to constraint"""
        raise NotImplementedError
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        """Apply mutation to create UNSAT variant"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__

    def _wrap(self, constraint: OCLConstraint, new_ocl: str, mutation: str,
              contradiction_ocl: str = None,
              contradiction_pattern_id: str = None) -> OCLConstraint:
        """Create a new UNSAT constraint with consistent metadata.

        Args:
            constraint: Original SAT constraint
            new_ocl: Full mutated OCL text (for output/display)
            mutation: Name of the mutation strategy used
            contradiction_ocl: Standalone contradiction as full OCL string.
                Used during Z3 verification: the original constraint and
                contradiction are sent as two separate constraints to the
                pattern-based encoder, which cannot parse the combined
                mutated OCL.
            contradiction_pattern_id: Pattern ID for the contradiction
                constraint (may differ from original, e.g. forAll→exists).
                If None, uses the original constraint's pattern_id.
        """
        meta = {**constraint.metadata, 'is_unsat': True, 'mutation': mutation,
                'original_pattern_id': constraint.pattern_id}
        if contradiction_ocl:
            meta['contradiction_ocl'] = contradiction_ocl
        if contradiction_pattern_id:
            meta['contradiction_pattern_id'] = contradiction_pattern_id
        return OCLConstraint(
            ocl=new_ocl,
            pattern_id=f"{constraint.pattern_id}_unsat",
            pattern_name=f"{constraint.pattern_name} (UNSAT)",
            context=constraint.context,
            parameters=constraint.parameters,
            metadata=meta
        )


class ContradictoryBoundsStrategy(UnsatMutationStrategy):
    """
    Create contradictory bounds: x > 10 and x < 5
    """

    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        ocl = constraint.ocl
        # Check for numeric comparison
        return bool(re.search(r'[<>=]', ocl) and re.search(r'\d+', ocl))

    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        # Find numeric comparisons inside the body
        pattern = r'([\w\.]+(?:->\w+\(\))?)\s*([<>]=?|=)\s*(\d+)'
        match = re.search(pattern, body)
        if not match:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        var, op, val_str = match.groups()
        val = int(val_str)

        if op == '>':
            contradiction = f'{var} < {val - 5}'
        elif op == '<':
            contradiction = f'{var} > {val + 5}'
        elif op == '>=':
            contradiction = f'{var} < {val}'
        elif op == '<=':
            contradiction = f'{var} > {val}'
        else:
            contradiction = f'{var} <> {val}'

        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        contra_ocl = f"context {constraint.context}\ninv: {contradiction}"
        return self._wrap(constraint, new_ocl, 'contradictory_bounds',
                          contradiction_ocl=contra_ocl)


class EmptyCollectionStrategy(UnsatMutationStrategy):
    """
    Require collection to be both empty and non-empty
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        body = OCLMutationHelper.split_ocl(constraint.ocl)[2]
        return 'notEmpty()' in body or 'size()' in body
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        if 'notEmpty()' in body:
            coll_pattern = r'([\w.]+)->notEmpty\(\)'
            match = re.search(coll_pattern, body)
            if not match:
                return SimpleNegationStrategy().apply(constraint, metamodel)
            collection = match.group(1)
            contradiction = f'{collection}->isEmpty()'
            new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
            contra_ocl = f"context {constraint.context}\ninv: {contradiction}"
            return self._wrap(constraint, new_ocl, 'empty_collection',
                              contradiction_ocl=contra_ocl)

        # Match size() > N or size() >= N (with N > 0)
        size_pattern = r'([\w.]+)->size\(\)\s*(>=?)\s*(\d+)'
        match = re.search(size_pattern, body)
        if match:
            collection = match.group(1)
            op = match.group(2)
            val = int(match.group(3))
            if op == '>=' and val == 0:
                return SimpleNegationStrategy().apply(constraint, metamodel)
            if op == '>':
                contradiction = f'{collection}->size() <= {val}'
            else:  # >=
                contradiction = f'{collection}->size() < {val}'
            new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
            contra_ocl = f"context {constraint.context}\ninv: {contradiction}"
            return self._wrap(constraint, new_ocl, 'empty_collection',
                              contradiction_ocl=contra_ocl)

        return SimpleNegationStrategy().apply(constraint, metamodel)


class TypeContradictionStrategy(UnsatMutationStrategy):
    """
    Create type contradictions: x.oclIsTypeOf(A) and x.oclIsTypeOf(B)
    """
    
    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        return 'oclIsTypeOf' in constraint.ocl
    
    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        pattern = r'(\w+)\.oclIsTypeOf\((\w+)\)'
        match = re.search(pattern, body)
        if not match:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        var, type1 = match.groups()
        classes = [c for c in metamodel.get_class_names() if c != type1]
        if not classes:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        type2 = random.choice(classes)
        contradiction = f'{var}.oclIsTypeOf({type2})'
        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        contra_ocl = f"context {constraint.context}\ninv: {contradiction}"
        return self._wrap(constraint, new_ocl, 'type_contradiction',
                          contradiction_ocl=contra_ocl)


class UniversalNegationStrategy(UnsatMutationStrategy):
    """
    Negate universal quantifier: forAll(...) and exists(not(...))
    """

    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        return '->forAll(' in constraint.ocl

    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)

        # Find the collection->forAll( prefix and then balance parentheses
        # to extract the correct condition, handling nested parens safely.
        fa_match = re.search(r'([\w.]+)->forAll\(\s*(\w+)\s*\|\s*', body)
        if not fa_match:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        collection = fa_match.group(1)
        var = fa_match.group(2)

        # Balance parentheses from the start of the condition to find its end
        cond_start = fa_match.end()
        depth = 1  # we are inside forAll(
        pos = cond_start
        while pos < len(body) and depth > 0:
            if body[pos] == '(':
                depth += 1
            elif body[pos] == ')':
                depth -= 1
            pos += 1

        if depth != 0:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        # pos-1 is the closing ')' of forAll; condition is everything before it
        condition = body[cond_start:pos - 1].strip()
        if not condition:
            return SimpleNegationStrategy().apply(constraint, metamodel)

        contradiction = f'{collection}->exists({var} | not({condition}))'
        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        contra_ocl = f"context {constraint.context}\ninv: {contradiction}"
        # Original is forAll → encoder uses forall_nested; contradiction is
        # exists → encoder needs exists_nested.
        return self._wrap(constraint, new_ocl, 'universal_negation',
                          contradiction_ocl=contra_ocl,
                          contradiction_pattern_id='exists_nested')


class SimpleNegationStrategy(UnsatMutationStrategy):
    """
    Conjoin constraint with its own negation: P and not(P).
    This is a tautological contradiction — always UNSAT — and serves as the
    guaranteed fallback when more interesting strategies fail Z3 verification.
    """

    def can_apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> bool:
        return True  # Always applicable

    def apply(self, constraint: OCLConstraint, metamodel: Metamodel) -> OCLConstraint:
        header, inv_token, body = OCLMutationHelper.split_ocl(constraint.ocl)
        contradiction = f'not({body})'
        new_ocl = OCLMutationHelper.build_unsat_ocl(header, inv_token, body, contradiction)
        return self._wrap(constraint, new_ocl, 'self_contradiction')


# All available strategies
ALL_STRATEGIES = [
    ContradictoryBoundsStrategy(),
    EmptyCollectionStrategy(),
    TypeContradictionStrategy(),
    UniversalNegationStrategy(),
    SimpleNegationStrategy()  # Fallback
]


def generate_unsat_variant(
    constraint: OCLConstraint,
    metamodel: Metamodel,
    strategies: Optional[List[UnsatMutationStrategy]] = None
) -> OCLConstraint:
    """
    Generate UNSAT variant of a constraint.
    
    Args:
        constraint: SAT constraint to mutate
        metamodel: Metamodel for context
        strategies: List of strategies to try (default: ALL_STRATEGIES)
        
    Returns:
        UNSAT variant of constraint
    """
    if strategies is None:
        strategies = ALL_STRATEGIES
    
    # Try strategies in order until one works
    for strategy in strategies:
        if strategy.can_apply(constraint, metamodel):
            return strategy.apply(constraint, metamodel)
    
    # Fallback: simple negation
    return SimpleNegationStrategy().apply(constraint, metamodel)


def generate_mixed_sat_unsat_set(
    constraints: List[OCLConstraint],
    metamodel: Metamodel,
    unsat_ratio: float = 0.3,
    strategies: Optional[List[UnsatMutationStrategy]] = None
) -> Tuple[List[OCLConstraint], Dict[int, str]]:
    """
    Generate mixed SAT/UNSAT constraint set from SAT-only set.
    
    Args:
        constraints: List of SAT constraints
        metamodel: Metamodel
        unsat_ratio: Target ratio of UNSAT constraints (0.0 - 1.0)
        strategies: Mutation strategies to use
        
    Returns:
        Tuple of (mixed_constraints, unsat_indices)
        where unsat_indices maps index -> mutation_strategy_name
    """
    if not (0.0 <= unsat_ratio <= 1.0):
        raise ValueError("unsat_ratio must be between 0.0 and 1.0")
    
    n = len(constraints)
    n_unsat = int(n * unsat_ratio)
    
    if n_unsat == 0:
        return constraints, {}
    
    # Randomly select indices to mutate
    indices_to_mutate = random.sample(range(n), min(n_unsat, n))
    
    result = []
    unsat_map = {}
    
    for i, constraint in enumerate(constraints):
        if i in indices_to_mutate:
            # Generate UNSAT variant
            unsat_variant = generate_unsat_variant(constraint, metamodel, strategies)
            result.append(unsat_variant)
            unsat_map[i] = unsat_variant.metadata.get('mutation', 'unknown')
        else:
            # Keep SAT constraint
            result.append(constraint)
    
    return result, unsat_map


def verify_unsat_generation(
    unsat_variant: OCLConstraint,
    verifier,
    original_constraint: OCLConstraint = None
) -> Tuple[bool, str]:
    """
    Verify that an UNSAT variant is actually unsatisfiable using Z3.

    The Z3 encoder is pattern-based and parses OCL text with pattern-specific
    regexes. A combined mutated OCL like ``(P) and (¬P)`` cannot be parsed
    correctly — the encoder only sees the first match.

    Strategy: send the **original** constraint and the **contradiction** as
    two separate constraints to the batch verifier.  If the pair is jointly
    UNSAT, the mutation is valid.

    Args:
        unsat_variant: Generated UNSAT variant to check
        verifier: An initialized FrameworkConstraintVerifier instance
        original_constraint: The original SAT constraint (used with
            contradiction_ocl for two-constraint verification)

    Returns:
        Tuple of (is_unsat, message)
    """
    if verifier is None or not getattr(verifier, 'framework_available', False):
        return True, "Verifier unavailable (assume correct)"

    try:
        contradiction_ocl = unsat_variant.metadata.get('contradiction_ocl')
        orig_pattern_id = unsat_variant.metadata.get('original_pattern_id')
        contra_pattern_id = unsat_variant.metadata.get('contradiction_pattern_id')

        # Two-constraint approach: send original + contradiction separately
        if contradiction_ocl and original_constraint is not None:
            # Use the contradiction-specific pattern ID if the mutation
            # produces a different pattern type (e.g. forAll→exists).
            # Fall back to the original constraint's pattern ID when the
            # contradiction is the same pattern type.
            contra_constraint = OCLConstraint(
                ocl=contradiction_ocl,
                pattern_id=contra_pattern_id or orig_pattern_id or original_constraint.pattern_id,
                pattern_name=f"{original_constraint.pattern_name}_contradiction",
                context=unsat_variant.context,
                parameters=original_constraint.parameters,
                metadata={}
            )
            results = verifier.verify_batch(
                [original_constraint, contra_constraint], silent=True
            )
            # Joint result
            joint = results[0].solver_result if results else 'unknown'
            if joint == 'unsat':
                return True, "Confirmed UNSAT by Z3 (two-constraint)"
            elif joint == 'sat':
                return False, "Z3 found SAT — contradiction is compatible with original"
            else:
                return False, f"Z3 returned {joint} — cannot confirm UNSAT"

        # Fallback: single-constraint verification (legacy / self_contradiction)
        result = verifier.verify(unsat_variant)

        if result.solver_result == 'unsat':
            return True, "Confirmed UNSAT by Z3"
        elif result.solver_result == 'sat':
            return False, "Z3 found SAT — mutation did not produce a genuine UNSAT"
        else:
            return False, f"Z3 returned {result.solver_result} — cannot confirm UNSAT"
    except Exception as e:
        return False, f"Verification failed: {e}"


def get_mutation_statistics(
    constraints: List[OCLConstraint]
) -> Dict[str, Any]:
    """
    Get statistics about UNSAT mutations in constraint set.
    
    Args:
        constraints: List of constraints (mixed SAT/UNSAT)
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total': len(constraints),
        'unsat_count': 0,
        'sat_count': 0,
        'mutations': {}
    }
    
    for c in constraints:
        if c.metadata.get('is_unsat', False):
            stats['unsat_count'] += 1
            mutation = c.metadata.get('mutation', 'unknown')
            stats['mutations'][mutation] = stats['mutations'].get(mutation, 0) + 1
        else:
            stats['sat_count'] += 1
    
    if stats['total'] > 0:
        stats['unsat_ratio'] = stats['unsat_count'] / stats['total']
    else:
        stats['unsat_ratio'] = 0.0
    
    return stats
