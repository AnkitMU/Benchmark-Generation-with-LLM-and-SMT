"""
Advanced benchmark engine with coverage tracking, diversity filtering, and adaptive generation.
"""
from __future__ import annotations
import random
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from modules.core.models import Metamodel, OCLConstraint
from modules.synthesis.pattern_engine.pattern_registry import PatternRegistry
from modules.generation.composer.ocl_generator import OCLGenerator

from .bench_config import BenchmarkProfile, FAMILY_KEYS, OPERATORS, TYPES, TC_DIFFICULTY_LEVELS
from .coverage_tracker import compute_coverage, count_operators, nav_hops, quantifier_depth
from .metadata_enricher import similarity, difficulty_score, get_complexity_weights
from .complexity_calculator import compute_total_complexity, tc_to_difficulty_label, ComplexityWeights, compute_ruc


# Tier ordering used by the construct-and-select mechanism (matches
# tc_to_difficulty_label thresholds: trivial<=3, easy<=8, medium<=16,
# hard<=25, expert>25).
TIER_ORDER = ["trivial", "easy", "medium", "hard", "expert"]

# Supported generation mechanisms (see BenchmarkEngineV2.generation_mode).
VALID_GENERATION_MODES = ("construct_select", "steered", "legacy")


@dataclass
class _PooledConstraint:
    """A constructed candidate with its MEASURED complexity.

    Used only by the construct-and-select path.  Structural/Computational
    are measured exactly by the complexity calculator (dependency deferred
    to the final suite); ``tier`` is derived from the measured TC.
    """
    constraint: 'OCLConstraint'
    structural: float
    computational: float
    tc: float
    tier: str


def classify_family(pattern_id: str, category: str) -> str:
    """Classify pattern into family."""
    pid = pattern_id.lower()
    cat = (category or "").lower()
    
    if cat == "string" or pid.startswith("string_") or "regex" in pid:
        return "string"
    # Conditional (if-then-else) expressions get their own family instead of
    # falling through to the "cardinality" catch-all (or, for the arithmetic-
    # categorised if_then_else_expression, into arithmetic). The numeric min/max
    # idioms stay arithmetic — their ids carry no 'conditional'/'if_then_else'.
    if "conditional" in pid or "if_then_else" in pid:
        return "conditional"
    if cat == "arithmetic" or any(k in pid for k in ["numeric", "range", "division", "abs_min_max"]):
        return "arithmetic"
    if any(k in pid for k in ["forall", "exists", "select_operation", "collect_operation", "one_operation", "any_operation"]):
        return "quantified"
    if cat == "navigation" or "navigation" in pid:
        return "navigation"
    if "unique" in pid or pid in ("uniqueness_constraint", "all_different"):
        return "uniqueness"
    if any(k in pid for k in ["size", "isempty", "notempty", "includes", "excludes", "cardinality"]):
        return "cardinality"
    if "enum" in pid or "forbiddenliteral" in pid or "subset" in pid:
        return "enum"
    if any(k in pid for k in ["ocliskindof", "oclastype", "ocl_is", "type_check"]):
        return "type_checks"
    if cat in ("collection", "set_operations"):
        return "cardinality"
    if cat == "ocl_library":
        return "type_checks"
    return "cardinality"


# Quantified collection patterns iterate a collection association and synthesise a
# body over its target class. The value maps each pattern to the attribute KIND its
# target must expose so the synthesised body is both valid OCL and encodable:
#   'numeric' — the live forAll/select handlers accept only `var.attr OP <number>`;
#   'usable'  — exists/any/one drop the predicate but still need a parseable body,
#               so any numeric/boolean/string attr works (`_attr_predicate`);
#   'any'     — collect(x|x.attr)->notEmpty() is Boolean for an attr of any type.
# The same map gates applicability AND restricts the chosen 'collection' to a viable
# one, so applicable == instantiable on every metamodel (not just car_rental).
_QUANT_COLLECTION_KIND = {
    'forall_nested': 'numeric',
    'select_operation': 'numeric',
    'exists_constraint': 'usable',
    'any_operation': 'usable',
    'one_operation': 'usable',
    'collect_operation': 'any',
}

# Operator / bound pools for diversified synthesised predicates. Without this the
# generator emitted a uniform `attr >= 0` everywhere, which made quantifier bodies a
# monotone cliche and produced byte-identical constraints across classes. Restricted
# to integer RHS and the four magnitude comparisons so the SMT forAll/select handlers
# (which parse only `var.attr OP <int>`) still encode the body. Lower bounds pair with
# >/>= and upper bounds with </<= so the predicate stays plausible (not `cost < 0`).
_PRED_NUM_OPS = (">=", ">", "<=", "<")
_PRED_LOWER_BOUNDS = (0, 1, 2, 5, 10)
_PRED_UPPER_BOUNDS = (10, 50, 100, 1000)
_PRED_BOOL = ("= true", "= false", "<> true", "<> false")


class CoverageState:
    """Live coverage tracking during generation with TC-based complexity awareness."""
    def __init__(self, metamodel: Metamodel, targets: 'BenchmarkProfile',
                 complexity_weights: 'ComplexityWeights' = None):
        self.metamodel = metamodel
        self.targets = targets.coverage
        self.complexity_config = targets.complexity
        self.complexity_weights = complexity_weights
        self.constraints: List[OCLConstraint] = []

        # Track usage
        self.classes_used: Set[str] = set()
        self.attributes_used: Set[Tuple[str, str]] = set()
        self.associations_used: Set[Tuple[str, str]] = set()
        self.operator_counts: Dict[str, int] = {op: 0 for op in OPERATORS}
        self.hop_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.depth_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self.type_counts: Dict[str, int] = {t: 0 for t in TYPES}

        # TC-based difficulty tracking (5 buckets)
        self.difficulty_counts: Dict[str, int] = {k: 0 for k in TC_DIFFICULTY_LEVELS}
        self.tc_scores: List[float] = []

        # Per-dimension score tracking (Structural & Computational)
        self.structural_scores: List[float] = []
        self.computational_scores: List[float] = []

    def add_constraint(self, c: OCLConstraint):
        """Add constraint and update coverage including TC and per-dimension metrics."""
        self.constraints.append(c)
        self.classes_used.add(c.context)

        ocl = c.ocl
        # Operators
        ops = count_operators(ocl)
        for k, v in ops.items():
            self.operator_counts[k] += v

        # Hops
        hops = nav_hops(ocl)
        bucket = 0 if hops == 0 else (1 if hops == 1 else 2)
        self.hop_counts[bucket] += 1

        # Depth
        depth = quantifier_depth(ocl)
        bucket = 0 if depth == 0 else (1 if depth == 1 else 2)
        self.depth_counts[bucket] += 1

        # TC-based difficulty (5 buckets) + per-dimension tracking
        tc_result = compute_total_complexity(ocl, metamodel=self.metamodel, context_class=c.context,
                                             weights=self.complexity_weights)
        tc = tc_result.tc
        self.tc_scores.append(tc)
        self.structural_scores.append(tc_result.structural_total)
        self.computational_scores.append(tc_result.computational_total)
        diff_label = tc_to_difficulty_label(tc)
        self.difficulty_counts[diff_label] = self.difficulty_counts.get(diff_label, 0) + 1

    def avg_tc(self) -> float:
        """Return average TC of generated constraints."""
        return sum(self.tc_scores) / len(self.tc_scores) if self.tc_scores else 0.0

    def avg_structural(self) -> float:
        """Return average structural complexity of generated constraints."""
        return sum(self.structural_scores) / len(self.structural_scores) if self.structural_scores else 0.0

    def avg_computational(self) -> float:
        """Return average computational complexity of generated constraints."""
        return sum(self.computational_scores) / len(self.computational_scores) if self.computational_scores else 0.0

    def score(self) -> float:
        """Compute overall coverage score (0-1) including TC difficulty mix."""
        total_classes = len(self.metamodel.classes)

        scores = []

        # Class coverage
        if total_classes > 0:
            target_classes = self.targets.class_context_pct / 100.0 * total_classes
            scores.append(min(1.0, len(self.classes_used) / max(1, target_classes)))

        # Operator coverage
        for op in OPERATORS:
            target = self.targets.operator_mins.get(op, 0)
            if target > 0:
                scores.append(min(1.0, self.operator_counts[op] / target))

        # Hop coverage
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.nav_hops.get(key, 0)
            if target > 0:
                scores.append(min(1.0, self.hop_counts[k] / target))

        # Depth coverage
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.quantifier_depth.get(key, 0)
            if target > 0:
                scores.append(min(1.0, self.depth_counts[k] / target))

        # TC difficulty mix (5 buckets from paper)
        total = sum(self.difficulty_counts.values()) or 1
        tc_mix = self.complexity_config.tc_difficulty_mix
        for key in TC_DIFFICULTY_LEVELS:
            target_pct = tc_mix.get(key, 0) / 100.0
            actual_pct = self.difficulty_counts.get(key, 0) / total
            scores.append(1.0 - abs(target_pct - actual_pct))

        # TC range penalty: penalize if average TC is outside target range
        if self.tc_scores:
            avg = self.avg_tc()
            min_tc = self.complexity_config.min_tc
            max_tc = self.complexity_config.max_tc
            if min_tc <= avg <= max_tc:
                scores.append(1.0)
            else:
                # Distance from target range, normalized
                dist = min(abs(avg - min_tc), abs(avg - max_tc))
                range_size = max(max_tc - min_tc, 1.0)
                scores.append(max(0.0, 1.0 - dist / range_size))

        # Per-dimension range penalties (Structural)
        if self.structural_scores and self.complexity_config.structural_enabled:
            s_min = self.complexity_config.structural_target_min
            s_max = self.complexity_config.structural_target_max
            if s_min is not None and s_max is not None:
                avg_s = self.avg_structural()
                if s_min <= avg_s <= s_max:
                    scores.append(1.0)
                else:
                    dist = min(abs(avg_s - s_min), abs(avg_s - s_max))
                    range_size = max(s_max - s_min, 1.0)
                    scores.append(max(0.0, 1.0 - dist / range_size))

        # Per-dimension range penalties (Computational)
        if self.computational_scores and self.complexity_config.computational_enabled:
            c_min = self.complexity_config.computational_target_min
            c_max = self.complexity_config.computational_target_max
            if c_min is not None and c_max is not None:
                avg_c = self.avg_computational()
                if c_min <= avg_c <= c_max:
                    scores.append(1.0)
                else:
                    dist = min(abs(avg_c - c_min), abs(avg_c - c_max))
                    range_size = max(c_max - c_min, 1.0)
                    scores.append(max(0.0, 1.0 - dist / range_size))

        return sum(scores) / max(1, len(scores)) if scores else 0.0

    def deficits(self) -> List[Tuple[str, int, int]]:
        """Return list of (target_name, achieved, needed) for unmet targets."""
        gaps = []

        # Operators
        for op in OPERATORS:
            target = self.targets.operator_mins.get(op, 0)
            if target > 0 and self.operator_counts[op] < target:
                gaps.append((f"op:{op}", self.operator_counts[op], target))

        # Hops
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.nav_hops.get(key, 0)
            if target > 0 and self.hop_counts[k] < target:
                gaps.append((f"hops:{key}", self.hop_counts[k], target))

        # Depth
        for k in [0, 1, 2]:
            key = str(k) if k < 2 else "2plus"
            target = self.targets.quantifier_depth.get(key, 0)
            if target > 0 and self.depth_counts[k] < target:
                gaps.append((f"depth:{key}", self.depth_counts[k], target))

        return gaps


class BenchmarkEngineV2:
    """Advanced benchmark generator with full coverage and diversity."""
    
    # C: CENTRALIZED EXCLUDE LIST - Single source of truth for all phases
    EXCLUDE_LIST = {
        # Tautologies (always true/false)
        'isEmpty_notEmpty',                 # Always true: isEmpty() or notEmpty()
        'self_not_null',                    # Always true in OCL

        # Not a constraint (expression without assertion)
        'set_difference',                   # Just returns a set
        'closure_operation',                # Just returns a closure
        'product_operation',                # Just returns a product
        'abs_min_max',                      # Expression without comparison

        # Degenerate + unverifiable null assertion
        'attribute_null_check',             # 'self.attr = null' forces an attribute
                                            # always-null (degenerate invariant); the
                                            # encoder cannot model '= null', so it is
                                            # unverifiable AND silently contradicts
                                            # 'attribute_not_null' on the same attr.

        # Semantically degenerate arithmetic
        'numeric_product_constraint',       # 'self.a * self.b > N' — a product of two
                                            # model attributes is almost never a real
                                            # invariant (mileageEnd*mileageStart,
                                            # seats*doors); meaningful only across
                                            # specific differing roles (count x price)
                                            # that don't occur as clean pairs here.

        # Malformed / tautological templates
        'string_operation',                 # 'self.attr.{op}({params})' where the op
                                            # options already carry '()' (toLower()),
                                            # so it renders 'attr.toLower()(value)' —
                                            # non-parseable, non-boolean, value leak.
        'collection_asSet',                 # 'asSet()->size() <= size()' is ALWAYS true
                                            # (asSet never increases cardinality).

        # Not implemented in encoder
        'oclAsType',                        # Causes type errors
        'oclAsType_cast',                   # Causes type errors

        # Type check issues
        'oclIsTypeOf_check',                # Often wrong types
        'oclIsKindOf_check',                # Often wrong types
        'allInstances_check',               # Encoder errors

        # Patterns that commonly generate tautologies
        'two_attributes_equal',             # Often picks same attribute twice

        # Collection operations with encoding issues
        'collection_sortedBy',              # sortedBy() causes type errors with booleans

        # Broken pattern templates
        'at_least_one_defined',             # Generates malformed OCL
        'null_check',                       # Uses oclIsUndefined() - low value
    }
    
    # B: SAT/UNSAT PATTERN TAGS - Patterns likely to generate SAT vs UNSAT
    SAT_LIKELY_PATTERNS = {
        # Lenient constraints - usually satisfiable
        'numeric_non_negative', 'numeric_positive', 'string_min_length',
        'collection_not_empty_simple', 'boolean_is_true', 'boolean_is_false',
        'attribute_not_null_simple', 'implies_simple',
    }
    
    UNSAT_LIKELY_PATTERNS = {
        # Strict/conflicting constraints - harder to satisfy
        'numeric_bounded',  # Can conflict with other bounds
        'xor_condition',  # Exclusive conditions
        'acyclicity',  # Graph constraints
        'all_different',  # Uniqueness across collection
    }
    
    # D: SMT-COMPATIBLE PATTERNS - Patterns that map cleanly to Z3
    SMT_VERIFIED_PATTERNS = {
        # Numeric
        'numeric_positive', 'numeric_non_negative', 'numeric_bounded',
        'numeric_comparison', 'range_constraint', 'numeric_sum_constraint',
        'numeric_difference_constraint', 'numeric_even', 'numeric_odd',
        # Boolean
        'boolean_guard', 'boolean_is_true', 'boolean_is_false',
        # Collection
        'size_constraint', 'collection_not_empty_simple', 'collection_min_size',
        'collection_max_size', 'collection_has_size',
        # String (basic)
        'string_min_length', 'string_max_length', 'string_exact_length',
        # Logic
        'implies_simple', 'all_attributes_defined', 'at_least_one_defined',
    }

    NUMERIC_TYPES = {
        'Integer', 'Real', 'Double', 'Float', 'EInt', 'EDouble', 'EFloat', 'ELong', 'EShort'
    }
    INTEGER_TYPES = {
        'Integer', 'int', 'EInt', 'Long', 'long', 'ELong', 'Short', 'short', 'EShort'
    }
    BOOLEAN_TYPES = {'Boolean', 'EBoolean'}
    STRING_TYPES = {'String', 'EString'}

    def _normalize_type(self, t: Optional[str]) -> str:
        if not t:
            return 'unknown'
        t_lower = t.lower()
        if t in self.NUMERIC_TYPES:
            return 'numeric'
        if t in self.BOOLEAN_TYPES or t_lower == 'boolean':
            return 'boolean'
        if t in self.STRING_TYPES or t_lower == 'string':
            return 'string'
        if 'date' in t_lower or 'time' in t_lower or 'timestamp' in t_lower:
            return 'date'
        return 'other'

    def _types_compatible(self, t1: Optional[str], t2: Optional[str]) -> bool:
        g1 = self._normalize_type(t1)
        g2 = self._normalize_type(t2)
        if g1 == 'unknown' or g2 == 'unknown':
            return True
        if g1 == g2:
            return True
        # Allow numeric mixing (Integer/Real/etc.)
        if g1 == 'numeric' and g2 == 'numeric':
            return True
        return False

    def _is_integer_type(self, t: Optional[str]) -> bool:
        if not t:
            return False
        if t in self.INTEGER_TYPES:
            return True
        t_lower = t.lower()
        if t_lower in {x.lower() for x in self.INTEGER_TYPES}:
            return True
        return any(x in t_lower for x in ['int', 'long', 'short'])
    
    def _attribute_names_compatible(self, first_attr: str, second_attr: str,
                                     context: str = None) -> bool:
        """
        Check if two attributes are semantically compatible for comparison.

        Priority: LLM matrix (Phi-4) → keyword heuristics (fallback).
        """
        # 1. Try LLM semantic matrix first (O(1) lookup, pre-computed)
        if context and hasattr(self, 'semantic_matrix') and self.semantic_matrix:
            result = self.semantic_matrix.is_comparable(context, first_attr, second_attr)
            if result is not None:
                return result
            # If pair not in matrix, fall through to heuristics

        # 2. Domain-family heuristic (richer than the ad-hoc keyword blocks below):
        #    two measurable attributes are only comparable if they share a domain
        #    family, so e.g. longitude (geo) vs staffCount (count) or totalAmount
        #    (money) vs mileageStart (distance) are rejected.
        try:
            from modules.semantic.llm_semantic_analyzer import (
                _classify_attr_domain, _domain_family, _NEVER_CROSS_FAMILIES,
            )
            fa = _domain_family(_classify_attr_domain(first_attr))
            fb = _domain_family(_classify_attr_domain(second_attr))
            if fa and fb and fa != fb:
                return False                              # two known, different families
            if (fa in _NEVER_CROSS_FAMILIES and fb != fa) or \
               (fb in _NEVER_CROSS_FAMILIES and fa != fb):
                return False                              # never-cross family vs anything else
        except Exception:
            pass

        # 3. Fallback: boolean-naming heuristic only. The money/distance/date keyword
        # blocks are superseded by the domain-family check above (which recognises
        # 'balance' as money and avoids substring traps like 'rate' matching 'at').
        first_lower = first_attr.lower()
        second_lower = second_attr.lower()

        # Boolean-like naming conventions: don't compare a flag with a quantity.
        # Use PREFIX matching (isActive, hasLicense) — a substring test wrongly flags
        # 'discount' ('dIScount'), 'this', 'list', etc.
        _bool_pref = ('is', 'has', 'can', 'should', 'are', 'was', 'will')
        is_first_bool = first_lower.startswith(_bool_pref)
        is_second_bool = second_lower.startswith(_bool_pref)
        if is_first_bool != is_second_bool:
            return False

        return True

    def _get_attribute_type(self, context: str, attr_name: str) -> Optional[str]:
        attr = next((a for a in self.metamodel.get_attributes_for(context) if a.name == attr_name), None)
        return attr.type if attr else None

    def _get_association(self, context: str, ref_name: str):
        return next((a for a in self.metamodel.get_associations_for(context) if a.ref_name == ref_name), None)

    def _has_usable_attr(self, cls: str) -> bool:
        """True if `cls` exposes a numeric/boolean/string attribute that
        `_attr_predicate` can turn into a self-contained boolean tail. Pure
        (no RNG) so it is safe to call from applicability checks."""
        return any(
            self._normalize_type(a.type) in ('numeric', 'boolean', 'string')
            for a in self.metamodel.get_attributes_for(cls)
        )

    def _can_navigate(self, context: str) -> bool:
        """Pure feasibility oracle for `_synth_navigation_path`: True iff some
        single-valued association (one or two hops) reaches a class with a usable
        attribute."""
        for a in self.metamodel.get_single_associations(context):
            if self._has_usable_attr(a.target_class):
                return True
            for b in self.metamodel.get_single_associations(a.target_class):
                if self._has_usable_attr(b.target_class):
                    return True
        return False

    def _has_numeric_attr(self, cls: str) -> bool:
        """True if `cls` exposes a numeric attribute (pure, no RNG)."""
        return any(
            self._normalize_type(a.type) == 'numeric'
            for a in self.metamodel.get_attributes_for(cls)
        )

    def _viable_collections(self, context: str, kind: str) -> List[str]:
        """Ref-names of the context's collection associations whose target class can
        support a synthesised body of the given `kind` ('numeric' | 'usable' | 'any').
        Restricting the chosen collection to this set keeps applicable == instantiable
        regardless of metamodel (some collections may reach attribute-poor classes)."""
        out = []
        for a in self.metamodel.get_collection_associations(context):
            if kind == 'numeric' and self._has_numeric_attr(a.target_class):
                out.append(a.ref_name)
            elif kind == 'usable' and self._has_usable_attr(a.target_class):
                out.append(a.ref_name)
            elif kind == 'any' and self.metamodel.get_attributes_for(a.target_class):
                out.append(a.ref_name)
        return out

    def _attr_predicate(self, cls: str, var: Optional[str] = None,
                        numeric_only: bool = False) -> Optional[str]:
        """Pick an attribute of `cls` and return a self-contained boolean predicate,
        e.g. 'price >= 0', 'active = true', "name <> ''". With `var` set the tail is
        iterator-prefixed ('x.price >= 0') for quantifier bodies; without it there is
        no prefix ('price >= 0') for navigation heads / template-supplied prefixes.

        The RHS never references the LHS, so the tail stays valid under any prefix.
        The operator and constant are varied (not a fixed `>= 0`) for structural and
        value diversity; `numeric_only` keeps the form to a numeric comparison the
        live forAll/select SMT handlers can parse (`var.attr OP <int>`).
        Returns None when `cls` exposes no suitable attribute."""
        attrs = self.metamodel.get_attributes_for(cls)
        prefix = f"{var}." if var else ""
        numeric = [a for a in attrs if self._normalize_type(a.type) == 'numeric']
        if numeric:
            op = random.choice(_PRED_NUM_OPS)
            bound = random.choice(_PRED_LOWER_BOUNDS if op in (">=", ">") else _PRED_UPPER_BOUNDS)
            return f"{prefix}{random.choice(numeric).name} {op} {bound}"
        if numeric_only:
            return None
        booleans = [a for a in attrs if self._normalize_type(a.type) == 'boolean']
        if booleans:
            return f"{prefix}{random.choice(booleans).name} {random.choice(_PRED_BOOL)}"
        strings = [a for a in attrs if self._normalize_type(a.type) == 'string']
        if strings:
            return f"{prefix}{random.choice(strings).name} <> ''"
        return None

    def _synth_navigation_path(self, context: str) -> Optional[str]:
        """Synthesise a boolean navigation expression from single-valued
        associations. Prefers a two-hop chain ('self.a.b.attr >= 0') for genuine
        navigation depth, falling back to one hop ('self.a.attr >= 0'). Returns
        None if no single-valued association reaches a class with a usable
        attribute."""
        singles = self.metamodel.get_single_associations(context)
        for a in singles:
            for b in self.metamodel.get_single_associations(a.target_class):
                pred = self._attr_predicate(b.target_class)
                if pred:
                    return f"self.{a.ref_name}.{b.ref_name}.{pred}"
        for a in singles:
            pred = self._attr_predicate(a.target_class)
            if pred:
                return f"self.{a.ref_name}.{pred}"
        return None

    def _synth_nested_predicate(self, context: str, params: Dict) -> Optional[str]:
        """Synthesise the body of a collection iteration over the target class of
        the chosen 'collection' association, e.g. 'price >= 0'. The pattern
        template prepends '{iterator}.', so the predicate carries no prefix."""
        assoc = self._get_association(context, params.get('collection'))
        if not assoc:
            return None
        return self._attr_predicate(assoc.target_class)

    # ── Cross-class semantic helpers ──────────────────────────────────────

    def _resolve_cross_attr_type(self, context: str, attr_name: str,
                                  params: Dict) -> Optional[str]:
        """
        Find the declared type of an attribute that belongs to an associated
        class rather than the context class itself.

        Resolution priority:
          1. 'collection' param hint  → narrows to one class, always unambiguous
          2. Single match across all direct associations → unambiguous
          3. Multiple matches, same type → safe to return
          4. Multiple matches, different types → fail closed (return None, log warning)
        """
        # Priority 1: use 'collection' param as a class hint
        collection_name = params.get('collection')
        if collection_name:
            assoc = self._get_association(context, collection_name)
            if assoc:
                for a in self.metamodel.get_attributes_for(assoc.target_class):
                    if a.name == attr_name:
                        return a.type

        # Priority 2: search all directly associated classes
        matches = []
        for assoc in self.metamodel.get_associations_for(context):
            for a in self.metamodel.get_attributes_for(assoc.target_class):
                if a.name == attr_name:
                    matches.append((assoc.target_class, a.type))

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0][1]

        # Multiple matches: safe if all share the same type
        unique_types = {t for _, t in matches}
        if len(unique_types) == 1:
            return unique_types.pop()

        # Conflicting types across classes — fail closed
        logger.warning(
            "[ambiguous resolution] Attribute '%s' found in multiple "
            "associated classes of '%s' with conflicting types: %s. "
            "Cannot resolve — cross-class type validation skipped.",
            attr_name, context, matches
        )
        return None

    def _resolve_navigation_path(self, context: str,
                                  path_parts: List[str]) -> Optional[str]:
        """
        Walk a list of association names from context and return the final
        class name.  Returns None if any step cannot be resolved.
        e.g. context='Customer', path_parts=['rental','vehicle'] -> 'Vehicle'
        """
        current = context
        for part in path_parts:
            assoc = self._get_association(current, part)
            if not assoc:
                return None
            current = assoc.target_class
        return current

    def _resolve_cross_attr_type_multihop(self, context: str,
                                           attr_name: str,
                                           params: Dict) -> Optional[str]:
        """
        Extended resolver that first tries single-hop lookup, then uses
        'collection' + 'nested_collection' params to attempt a two-hop
        resolution (Fix 3).
        """
        # Single-hop first
        result = self._resolve_cross_attr_type(context, attr_name, params)
        if result is not None:
            return result

        # Two-hop: collection → nested_collection → attr
        collection_name = params.get('collection')
        nested_name = params.get('nested_collection')
        if collection_name and nested_name:
            final_class = self._resolve_navigation_path(
                context, [collection_name, nested_name]
            )
            if final_class:
                for a in self.metamodel.get_attributes_for(final_class):
                    if a.name == attr_name:
                        return a.type

        return None

    def _cross_class_semantics_compatible(self, attr1: str, attr2: str) -> bool:
        """
        Stronger domain-family check for cross-class attribute pairs.
        Uses _DOMAIN_PATTERNS / _domain_family from llm_semantic_analyzer
        (the same classifier used in post-processing) rather than the simpler
        keyword heuristic in _attribute_names_compatible().

        Logic:
          - Never-cross domain on either side → reject (unless both same family)
          - Both in different *known* families → reject
          - Both in same approved family → accept
          - Either domain unknown → admit conservatively (True)
        """
        try:
            from modules.semantic.llm_semantic_analyzer import (
                _classify_attr_domain, _domain_family, _NEVER_CROSS_FAMILIES,
            )
            fa = _domain_family(_classify_attr_domain(attr1))
            fb = _domain_family(_classify_attr_domain(attr2))

            # Hard reject: never-cross family
            if fa in _NEVER_CROSS_FAMILIES or fb in _NEVER_CROSS_FAMILIES:
                return fa == fb

            # Both known, different families → reject
            if fa and fb and fa != fb:
                return False

            # Both known, same family → accept
            # One or both unknown → admit by default
            return True
        except ImportError:
            # Fallback to the engine's own keyword heuristic
            return self._attribute_names_compatible(attr1, attr2)

    def _association_names_compatible(self, assoc1: str, assoc2: str) -> bool:
        """
        Conservative guard for two-hop collection chains (Fix 2).
        Only rejects when one association name clearly falls into a
        never-cross domain (identifiers, labels, units, versions).
        Most association names are neutral nouns and will pass through.
        This is NOT a full semantic filter — it is a narrow sanity check.
        """
        try:
            from modules.semantic.llm_semantic_analyzer import (
                _classify_attr_domain, _domain_family, _NEVER_CROSS_FAMILIES,
            )
            fa = _domain_family(_classify_attr_domain(assoc1))
            fb = _domain_family(_classify_attr_domain(assoc2))
            if fa in _NEVER_CROSS_FAMILIES or fb in _NEVER_CROSS_FAMILIES:
                return fa == fb   # allow only if both are in the same never-cross family
        except ImportError:
            pass
        return True  # conservative default: admit
    
    def __init__(self, metamodel: Metamodel, enable_semantic_validation: bool = True,
                 verification_enabled: bool = False, semantic_matrix=None,
                 generation_mode: str = "construct_select"):
        logger.info("="*80)
        logger.info("Initializing BenchmarkEngineV2")
        logger.info(f"Metamodel: {len(metamodel.classes)} classes")
        logger.info(f"Semantic validation: {'ENABLED' if enable_semantic_validation else 'DISABLED'}")
        logger.info(f"LLM semantic matrix: {'LOADED' if semantic_matrix else 'NOT AVAILABLE (using heuristics)'}")
        logger.info(f"Generation mode: {generation_mode}")

        self.metamodel = metamodel
        self.verification_enabled = verification_enabled
        self.semantic_matrix = semantic_matrix  # LLM-based compatibility matrix (or None)
        # Generation mechanism:
        #   "construct_select" (default) — over-generate a candidate pool from the
        #        pattern library, MEASURE exact (Structural, Computational) per
        #        candidate, then select to fill each difficulty tier's quota
        #        exactly; feasibility gaps are reported, never padded.
        #   "legacy" — the older TC-steering path (kept for comparison/fallback).
        self.generation_mode = self._normalize_generation_mode(generation_mode)
        self.registry = PatternRegistry()
        self.generator = OCLGenerator(pattern_registry=self.registry, metamodel=metamodel)
        logger.debug(f"Pattern registry loaded: {len(self.registry.get_all_patterns())} patterns")
        
        # A: FAILURE TRACKING - Track pattern/context failures for adaptive weighting
        self.pattern_fail_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.pattern_success_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # Track string attributes already constrained to avoid null-check conflicts
        self._string_constrained_attrs: Dict[str, set] = defaultdict(set)
        
        # Index patterns by family
        logger.debug("Indexing patterns by family...")
        self.patterns_by_family: Dict[str, List] = {k: [] for k in FAMILY_KEYS}
        self.all_patterns = self.registry.get_all_patterns()
        for p in self.all_patterns:
            fam = classify_family(p.id, getattr(p.category, "value", str(p.category)))
            if fam not in self.patterns_by_family:
                self.patterns_by_family[fam] = []
            self.patterns_by_family[fam].append(p)
        
        logger.debug(f"Pattern families: {', '.join(f'{k}={len(v)}' for k, v in self.patterns_by_family.items() if v)}")
        
        # TIER 2: Initialize semantic validator (ENABLED BY DEFAULT)
        logger.debug("Initializing semantic validator...")
        self.semantic_validator = None
        self.pattern_template_validator = None
        if enable_semantic_validation:
            try:
                logger.debug("Loading semantic_rules module...")
                import sys
                from pathlib import Path
                config_path = Path(__file__).parent.parent.parent.parent / 'config'
                if str(config_path) not in sys.path:
                    sys.path.insert(0, str(config_path))
                from semantic_rules import SemanticValidator, is_valid_pattern_template
                self.semantic_validator = SemanticValidator()
                self.pattern_template_validator = is_valid_pattern_template
                
                # Validate all loaded patterns
                invalid_count = 0
                for pattern in self.all_patterns:
                    template = getattr(pattern, 'template', '')
                    if template:
                        is_valid, reason = self.pattern_template_validator(pattern.id, template)
                        if not is_valid:
                            invalid_count += 1
                            print(f"Pattern '{pattern.id}' has invalid template: {reason}")
                
                print(f"Semantic validation enabled (Tier 2)")
                if invalid_count > 0:
                    print(f"   Found {invalid_count} patterns with invalid templates (will be skipped during generation)")
            except ImportError as e:
                print(f"Semantic validator not available: {e}")
                print("   Falling back to basic validation (Tier 1 only)")
        
        # TIER 3: Initialize advanced semantic analyzers
        from modules.semantic.metamodel.structure_analyzer import StructureAnalyzer
        from modules.semantic.metamodel.pattern_suggester import PatternSuggester
        from modules.semantic.metamodel.dependency_graph import DependencyGraph
        from modules.semantic.metamodel.invariant_detector import InvariantDetector
        
        self.structure_analyzer = StructureAnalyzer(metamodel)
        self.pattern_suggester = PatternSuggester(metamodel)
        self.dependency_graph = DependencyGraph(metamodel)
        self.invariant_detector = InvariantDetector(metamodel)
        print("Advanced semantic analysis enabled (Tier 3): Structure, Patterns, Dependencies, Invariants")
    
    def generate(self, profile: BenchmarkProfile, progress_callback=None) -> List[OCLConstraint]:
        """Generate constraints with full coverage tracking and diversity."""
        logger.info("\n" + "="*80)
        logger.info("GENERATE() - Starting constraint generation")
        logger.info("="*80)
        
        # RNG seeding is handled by the suite controller using the profile seed
        # Reset per-run string constraint tracking
        self._string_constrained_attrs = defaultdict(set)
        # Populated by _generate_steered; written to report.json by the controller.
        self.steered_report = None

        # Dispatch: construct-and-select is the default mechanism.  It replaces
        # the TC-steering heuristics below with an over-generate -> measure ->
        # stratified-select pipeline (exact tier histogram by construction;
        # honest feasibility gaps).  The legacy path is preserved for fallback.
        if self.generation_mode == "construct_select":
            return self._generate_construct_select(profile, progress_callback)
        if self.generation_mode == "steered":
            return self._generate_steered(profile, progress_callback)

        coverage = CoverageState(self.metamodel, profile,
                                  complexity_weights=get_complexity_weights())
        self._coverage = coverage  # Store reference for complexity steering
        total = profile.quantities.invariants
        
        logger.info(f"Target: {total} invariants")
        logger.info(f"Classes: {len(self.metamodel.get_class_names())}")
        logger.info(f"Enabled patterns: {len(profile.library.enabled) if profile.library.enabled else len(self.all_patterns)}")
        logger.info(f"Exclude-list size: {len(self.EXCLUDE_LIST)} patterns")
        
        print(f"\n=== BENCHMARK GENERATION START ===")
        print(f"Target: {total} invariants")
        print(f"Classes: {len(self.metamodel.get_class_names())}")
        print(f"Enabled patterns: {len(profile.library.enabled) if profile.library.enabled else len(self.all_patterns)}")
        
        # Initialize empty constraint list and class quota
        constraints = []
        class_quota = {c: 0 for c in self.metamodel.get_class_names()}

        # Phase 1: Family-based generation
        print(f"\n--- Phase 1: Family-based generation ---")
        logger.info("\nPHASE 1: Family-based generation")
        plan = self._plan_families(profile)
        logger.debug(f"Family plan: {plan}")
        print(f"\nFamily plan: {plan}")
        
        for family, count in sorted(plan.items(), key=lambda x: -x[1]):
            logger.debug(f"\n--- Processing family '{family}' (need {count} constraints) ---")
            patterns = self._get_enabled_patterns(family, profile)
            if not patterns:
                logger.warning(f"Skipping family {family}: no enabled patterns")
                print(f"  Skipping family {family}: no enabled patterns")
                continue
            
            logger.info(f"Generating {count} constraints for family '{family}' ({len(patterns)} patterns available)")
            print(f"\n  Generating {count} constraints for family '{family}' ({len(patterns)} patterns available)")

            # Phase 1 uses a retry-until-quota loop (proposed algorithm):
            # keeps attempting until the family quota is filled or K_f attempts exhausted,
            # rather than a fixed for-loop that wastes slots on rejections.
            accepted_f = 0
            attempts_f = 0
            K_f = count * 3  # allow up to 3x attempts to fill the family quota

            while accepted_f < count and attempts_f < K_f and len(constraints) < total:
                attempts_f += 1

                # Select context with room (enhanced with complexity weighting)
                logger.debug(f"  Attempt {attempts_f}/{K_f}: Selecting context...")
                context = self._select_context(class_quota, profile, coverage)
                if not context:
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: No context available (quota exhausted)")
                    break

                logger.debug(f"  Attempt {attempts_f}/{K_f}: Selected context '{context}' (quota: {class_quota[context]})")

                # Sample pattern with weights (enhanced with semantic suggestions)
                logger.debug(f"  Attempt {attempts_f}/{K_f}: Sampling pattern from {len(patterns)} options...")
                pattern = self._weighted_sample(patterns, profile, context=context)
                logger.debug(f"  Attempt {attempts_f}/{K_f}: Selected pattern '{pattern.id}'")

                # Check if pattern is applicable to this context before trying
                if not self._is_pattern_applicable(pattern, context):
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Pattern '{pattern.id}' not applicable to '{context}' - skipping")
                    continue

                logger.debug(f"  Attempt {attempts_f}/{K_f}: Pattern '{pattern.id}' is applicable to '{context}'")

                # Generate
                try:
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Generating parameters...")
                    params = self._gen_params(pattern, context)
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Parameters: {params}")

                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Generating constraint...")
                    c = self.generator.generate(pattern.id, context, params)
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Generated: {c.ocl}")

                    # Track successful generation
                    self.pattern_success_counts[(pattern.id, context)] += 1
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Pattern succeeded")

                    # Check diversity (AST similarity)
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Checking diversity...")
                    if not self._is_diverse(c, constraints, profile):
                        logger.debug(f"  Attempt {attempts_f}/{K_f}: Failed diversity check - too similar to existing")
                        continue

                    logger.debug(f"  Attempt {attempts_f}/{K_f}: Diversity check passed")

                    # Context-aware semantic validation
                    if self.semantic_validator:
                        logger.debug(f"  Attempt {attempts_f}/{K_f}: Running semantic validation...")
                        is_valid, reason = self.semantic_validator.validate_parameters(
                            pattern.id, context, params
                        )
                        if not is_valid:
                            logger.debug(f"  Attempt {attempts_f}/{K_f}: Semantic validation failed: {reason}")
                            continue
                        logger.debug(f"  Attempt {attempts_f}/{K_f}: Semantic validation passed")

                    constraints.append(c)
                    self._track_string_constraints(pattern, params, context)
                    coverage.add_constraint(c)
                    class_quota[context] += 1
                    accepted_f += 1
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: CONSTRAINT ADDED - Accepted: {accepted_f}/{count}, Total: {len(constraints)}/{total}")
                    logger.debug(f"    Pattern: {pattern.id}")
                    logger.debug(f"    Context: {context}")
                    logger.debug(f"    Expression: {c.ocl}")
                    if progress_callback:
                        progress_callback(len(constraints), total, coverage.score())
                except Exception as e:
                    # Track pattern failure — continue, do not consume a quota slot
                    self.pattern_fail_counts[(pattern.id, context)] += 1
                    logger.debug(f"  Attempt {attempts_f}/{K_f}: GENERATION FAILED")
                    logger.debug(f"    Pattern: {pattern.id}")
                    logger.debug(f"    Context: {context}")
                    logger.debug(f"    Error: {type(e).__name__}: {e}")

            if accepted_f < count:
                logger.warning(f"  Family '{family}': generated {accepted_f}/{count} after {attempts_f} attempts (K_f={K_f})")
                print(f"  Family '{family}': generated {accepted_f}/{count} constraints in {attempts_f} attempts")
        
        # Phase 2: Coverage-driven backfill
        # Uses a stall counter (K_STALL) instead of breaking on first error —
        # transient failures increment the counter; a successful add resets it.
        # Hard breaks are kept only for structural impossibilities (no deficits,
        # no pattern for deficit type, no context available).
        print(f"\n--- Phase 2: Coverage-driven backfill (target: {total - len(constraints)} more constraints) ---")
        phase2_attempts = 0
        phase2_generated = 0
        K_STALL = 20   # max consecutive failures before giving up
        stall = 0

        while len(constraints) < total and stall < K_STALL:
            phase2_attempts += 1

            deficits = coverage.deficits()
            if not deficits:
                print(f"  Phase 2 stopped: No coverage deficits found (generated {phase2_generated} in {phase2_attempts} attempts)")
                break

            # Pick first deficit and find a pattern to address it
            target_name, _, _ = deficits[0]
            pattern = self._pattern_for_deficit(target_name, profile)
            if not pattern:
                print(f"  Phase 2 stopped: No pattern found for deficit '{target_name}'")
                break

            context = self._select_context(class_quota, profile, coverage)
            if not context:
                print(f"  Phase 2 stopped: No context available - all classes at quota")
                break

            try:
                params = self._gen_params(pattern, context)
                c = self.generator.generate(pattern.id, context, params)
                if self._is_diverse(c, constraints, profile):
                    constraints.append(c)
                    self._track_string_constraints(pattern, params, context)
                    coverage.add_constraint(c)
                    class_quota[context] += 1
                    phase2_generated += 1
                    stall = 0  # reset stall counter on successful add

                    if phase2_generated % 10 == 0:
                        print(f"  Phase 2: Generated {phase2_generated} constraints (total: {len(constraints)}/{total})")

                    if progress_callback:
                        progress_callback(len(constraints), total, coverage.score())
                else:
                    stall += 1  # diversity check failed
                    logger.debug(f"  Phase 2: diversity failed (stall {stall}/{K_STALL})")
            except Exception as e:
                # Track failure, increment stall — do not break on first error
                self.pattern_fail_counts[(pattern.id, context)] += 1
                stall += 1
                logger.debug(f"  Phase 2: generation error (stall {stall}/{K_STALL}) - {type(e).__name__}: {str(e)[:60]}")

        if stall >= K_STALL:
            print(f"  Phase 2 stopped: stall limit reached ({K_STALL} consecutive failures, {phase2_generated} generated)")
        if phase2_generated > 0:
            print(f"  Phase 2 complete: Generated {phase2_generated} additional constraints")
        
        # Phase 2.5: Simple backfill to reach target count (no coverage requirements)
        if len(constraints) < total:
            print(f"\n--- Phase 2.5: Simple backfill (need {total - len(constraints)} more) ---")
            phase25_generated = 0
            max_attempts = (total - len(constraints)) * 10  # Try up to 10x the needed amount
            
            for attempt in range(max_attempts):
                if len(constraints) >= total:
                    break
                
                # C: Use centralized exclude-list
                pattern = random.choice([p for p in self.all_patterns if p.id not in self.EXCLUDE_LIST])
                
                # Pick random context
                context = self._select_context(class_quota, profile, coverage)
                if not context:
                    # All classes at max, can't generate more
                    print(f"  Phase 2.5 stopped: All classes at quota (generated {phase25_generated} in {attempt+1} attempts)")
                    break
                
                try:
                    params = self._gen_params(pattern, context)
                    c = self.generator.generate(pattern.id, context, params)
                    
                    # Check diversity
                    if self._is_diverse(c, constraints, profile):
                        constraints.append(c)
                        self._track_string_constraints(pattern, params, context)
                        coverage.add_constraint(c)
                        class_quota[context] += 1
                        phase25_generated += 1
                        
                        if phase25_generated % 10 == 0:
                            print(f"  Phase 2.5: Generated {phase25_generated} constraints (total: {len(constraints)}/{total})")
                except Exception:
                    # Silently continue on errors
                    continue
            
            if phase25_generated > 0:
                print(f"  Phase 2.5 complete: Generated {phase25_generated} additional constraints")
        
        # Phase 3: Redundancy pruning (if enabled)
        if profile.redundancy.implication_mode != "off":
            before = len(constraints)
            constraints = self._prune_redundant(constraints, profile)
            print(f"\nPhase 3: Pruned {before - len(constraints)} redundant constraints")
        
        print(f"\n=== GENERATION COMPLETE ===")
        print(f"Total generated: {len(constraints)} / {total} requested")
        
        # A: Print failure statistics
        self._print_failure_stats()

        return constraints

    # ------------------------------------------------------------------
    # Construct-and-select mechanism (default)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_generation_mode(mode: Optional[str]) -> str:
        """Validate/normalize a generation-mode string.

        Unknown or empty values fall back to the default ('construct_select')
        with a warning, so a config typo can never silently disable generation.
        """
        if mode in VALID_GENERATION_MODES:
            return mode
        if mode:
            logger.warning(
                "Unknown generation_mode '%s' — falling back to "
                "'construct_select'. Valid modes: %s",
                mode, VALID_GENERATION_MODES,
            )
        return "construct_select"

    def _generate_steered(self, profile: BenchmarkProfile,
                          progress_callback=None) -> List[OCLConstraint]:
        """Profile-guided complexity steering with component-wise measured feedback.

        Builds a ``SteeringSpec`` from the profile (suite size, family quota,
        per-class bounds, and per-component complexity profile boxes), runs the
        steered generator, and returns the flat ``List[OCLConstraint]`` contract
        (polarity/UNSAT and verification are applied downstream, unchanged).
        Complexity profiles come from ``profile.complexity.complexity_profiles``;
        if absent, a single box-free profile is used (family/class distribution
        only).
        """
        # Lazy import: steered_generation imports this module.
        from .steered_generation import (
            SteeredGenerator, SteeringSpec, Box, validate_complexity_profiles,
        )

        q = profile.quantities
        cx = profile.complexity
        raw_profiles = getattr(cx, "complexity_profiles", None)
        # Reject non-settable (engine-reported) components and malformed ranges
        # before any box is built, so the user gets a clear, early error.
        validate_complexity_profiles(raw_profiles)
        boxes = []
        if raw_profiles:
            for i, pr in enumerate(raw_profiles):
                ranges = {k: (float(v[0]), float(v[1]))
                          for k, v in (pr.get("ranges") or {}).items()}
                boxes.append((Box(ranges, pr.get("label", f"p{i}")),
                              float(pr.get("pct", 0))))
        if not boxes:
            boxes = [(Box({}, "any"), 100.0)]

        spec = SteeringSpec(
            n=q.invariants,
            sat_ratio=1.0,  # polarity/UNSAT handled downstream by the controller
            family_quota=dict(q.families_pct or {}),
            per_class=(q.per_class_min or 0, q.per_class_max or 10 ** 6),
            profiles=boxes,
            weights=get_complexity_weights(),
            policy=getattr(cx, "on_infeasible", "report") or "report",
        )

        print(f"\n=== STEERED GENERATION START ===")
        print(f"Target: {spec.n} invariants | families={spec.family_quota} | "
              f"profiles={[(b.label, pct) for b, pct in boxes]}")

        gen = SteeredGenerator(self, spec)
        gen.profile()
        suite, report = gen.run()
        constraints = []
        for t in suite:
            c = t[0]
            # Tag each constraint with the complexity profile it was generated
            # for (e.g. easy/medium/difficult), stripping any "+actuator"
            # provenance suffix so downstream difficulty reflects the user tiers.
            label = str(t[5] or "").split("+")[0]
            if label and label not in ("skip", "any"):
                try:
                    c.metadata["profile"] = label
                except Exception:
                    pass
            constraints.append(c)

        # honest summary
        from collections import Counter
        fam_ct = Counter(t[1] for t in suite)
        box_ct = Counter(t[5] for t in suite)
        gaps = [r for r in report if r and r[0] in
                ("unreached", "unfilled", "below_min", "support_infeasible")]
        print(f"Filled {len(constraints)}/{spec.n}  families={dict(fam_ct)}  "
              f"profiles={dict(box_ct)}")
        if gaps:
            print(f"Feasibility gaps reported (not padded): {len(gaps)}")
            for g in gaps[:8]:
                print(f"   - {g}")
        print(f"=== STEERED GENERATION COMPLETE ===\n")

        self.steered_report = self._build_steered_report(spec, suite, report)
        if progress_callback:
            progress_callback(len(constraints), spec.n, 1.0)
        return constraints

    def _build_steered_report(self, spec, suite, report) -> dict:
        """Assemble a JSON-serialisable deviation report for the steered run:
        realised distributions, reachable envelope, derived components, actuator
        usage, and feasibility gaps."""
        from collections import Counter

        def safe(x):
            if isinstance(x, (str, int, float, bool)) or x is None:
                return x
            if isinstance(x, (list, tuple, set)):
                return [safe(i) for i in x]
            if isinstance(x, dict):
                return {str(k): safe(v) for k, v in x.items()}
            return str(x)

        envelope, derived = {}, {}
        unreachable, gaps = [], []
        actuators = Counter()
        for r in report:
            if not r:
                continue
            kind = r[0]
            if kind == "reachable_envelope":
                envelope = {k: [round(v[0], 3), round(v[1], 3)] for k, v in r[1].items()}
            elif kind == "reported_derived_components":
                derived = {k: [round(v[0], 3), round(v[1], 3)] for k, v in r[1].items()}
            elif kind == "box_unreachable":
                unreachable.append({"box": r[1], "blocking": list(r[2]),
                                    "reason": r[3] if len(r) > 3 else None})
            elif kind == "actuator":
                actuators[r[1]] += 1
            else:
                gaps.append({"type": kind, "info": safe(list(r[1:]))})
        total = len(suite)
        n_sat = sum(1 for t in suite if t[2] == "sat")
        return {
            "target_size": spec.n,
            "generated": total,
            "requested_sat_ratio": spec.sat_ratio,
            "realised_sat_ratio": round(n_sat / total, 3) if total else 0.0,
            "family_distribution": dict(Counter(t[1] for t in suite)),
            "profile_distribution": dict(Counter(str(t[5]).split("+")[0] for t in suite)),
            "actuator_fires": dict(actuators),
            "reachable_envelope": envelope,
            "reported_derived_components": derived,
            "unreachable_boxes": unreachable,
            "feasibility_gaps": gaps,
        }

    def _generate_construct_select(self, profile: BenchmarkProfile,
                                   progress_callback=None) -> List[OCLConstraint]:
        """Construct-and-select benchmark generation (default mechanism).

        Phases
        ------
        1-3  CONSTRUCT a candidate pool from the pattern library (the
             building-block vocabulary) with NO complexity steering, reusing
             the engine's parameter generation, semantic validation, and
             diversity filtering.
        --   MEASURE each candidate's exact (Structural, Computational) with
             the complexity calculator and tier on the measured TC.  The
             dependency dimension is deferred to the final suite — RUC is
             emergent and never a per-constraint target.
        4    STRATIFIED SELECT: fill each difficulty tier's quota exactly,
             spreading picks across the measured complexity range for
             diversity.  A tier the pool cannot satisfy is reported as a
             feasibility shortfall — never padded with off-target constraints.
        5    SUITE DEPENDENCY: report RUC measured on the FINAL selection.

        Returns a flat ``List[OCLConstraint]`` (same contract as the legacy
        path), so downstream enrichment/verification is unaffected.
        """
        weights = get_complexity_weights()
        total = profile.quantities.invariants

        print(f"\n=== CONSTRUCT-AND-SELECT GENERATION START ===")
        print(f"Target: {total} invariants")
        print(f"Difficulty mix: {profile.complexity.tc_difficulty_mix}")

        # Phases 1-3: construct + measure the candidate pool
        pool = self._construct_pool(profile, weights, progress_callback)
        buckets: Dict[str, List[_PooledConstraint]] = defaultdict(list)
        for pc in pool:
            buckets[pc.tier].append(pc)

        hist = {t: len(buckets.get(t, [])) for t in TIER_ORDER}
        print(f"\nPhase 1-3  constructed & MEASURED pool: {len(pool)} unique candidates")
        print(f"           pool tier histogram: " +
              "  ".join(f"{t}={hist[t]}" for t in TIER_ORDER))

        # Phase 4: stratified selection against per-tier quotas
        quotas = self._tier_quotas(profile.complexity.tc_difficulty_mix, total)
        selected, shortfall = self._stratified_select(buckets, quotas)

        sel_by_tier: Dict[str, int] = defaultdict(int)
        for pc in selected:
            sel_by_tier[pc.tier] += 1

        print(f"\nPhase 4    stratified selection (fill exact per-tier quotas)")
        print(f"   {'tier':<9}{'target':>7}{'pool':>6}{'selected':>10}{'shortfall':>11}")
        total_short = 0
        for t in TIER_ORDER:
            want = quotas.get(t, 0)
            got = sel_by_tier.get(t, 0)
            short = shortfall.get(t, 0)
            total_short += short
            print(f"   {t:<9}{want:>7}{hist[t]:>6}{got:>10}{short:>11}")
        print(f"   {'TOTAL':<9}{sum(quotas.values()):>7}{len(pool):>6}"
              f"{len(selected):>10}{total_short:>11}")
        if total_short == 0:
            print(f"\n   ==> EXACT MATCH: every per-tier quota met by construction + selection.")
        else:
            print(f"\n   ==> FEASIBILITY GAP: {total_short} slot(s) unfilled "
                  f"(pool cannot supply them) — reported honestly, NOT padded.")

        final = [pc.constraint for pc in selected]

        # Phase 5: suite-level dependency (RUC) measured on the FINAL selection
        self._report_suite_dependency(final)

        if progress_callback:
            progress_callback(len(final), total, 1.0)

        print(f"\n=== GENERATION COMPLETE ===")
        print(f"Total generated: {len(final)} / {total} requested")
        self._print_failure_stats()
        return final

    def _construct_pool(self, profile: BenchmarkProfile, weights: 'ComplexityWeights',
                        progress_callback=None) -> List['_PooledConstraint']:
        """Over-generate a diverse candidate pool and measure each candidate.

        The pattern library is the building-block vocabulary.  For every
        applicable (pattern, context) pair we synthesise up to ``K`` distinct
        candidates using the engine's normal parameter generation + semantic
        validation, deduplicate by OCL text, reject near-duplicates (AST
        similarity), and record the MEASURED (Structural, Computational, TC,
        tier).  No TC steering is applied — complexity is observed, not chased.
        """
        K = 3                       # distinct candidates to keep per (pattern, context)
        per_combo_tries = K * 4     # extra tries to absorb dedup/diversity rejects
        max_total_attempts = max(8000, 60 * profile.quantities.invariants)

        patterns = [p for p in self.all_patterns if p.id not in self.EXCLUDE_LIST]
        if profile.library.enabled:
            patterns = [p for p in patterns if p.id in profile.library.enabled]

        contexts = list(self.metamodel.get_class_names())

        # Applicable (context, pattern) combos, shuffled so the pool is not
        # biased toward early classes/patterns.
        combos = [(ctx, p) for ctx in contexts for p in patterns
                  if self._is_pattern_applicable(p, ctx)]
        random.shuffle(combos)

        pool: List[_PooledConstraint] = []
        pool_constraints: List[OCLConstraint] = []
        seen_ocl: Set[str] = set()
        attempts = 0

        for ctx, pattern in combos:
            if attempts >= max_total_attempts:
                break
            produced = 0
            tries = 0
            while (produced < K and tries < per_combo_tries
                   and attempts < max_total_attempts):
                tries += 1
                attempts += 1
                try:
                    params = self._gen_params(pattern, ctx)
                    c = self.generator.generate(pattern.id, ctx, params)
                except Exception:
                    self.pattern_fail_counts[(pattern.id, ctx)] += 1
                    continue
                if c is None or not getattr(c, 'ocl', None):
                    continue
                if c.ocl in seen_ocl:
                    continue
                if not self._is_diverse(c, pool_constraints, profile):
                    continue
                # accept + MEASURE (dependency deferred -> all_constraints=None)
                seen_ocl.add(c.ocl)
                pool_constraints.append(c)
                self.pattern_success_counts[(pattern.id, ctx)] += 1
                res = compute_total_complexity(
                    c.ocl, metamodel=self.metamodel, context_class=ctx,
                    all_constraints=None, weights=weights,
                )
                pool.append(_PooledConstraint(
                    constraint=c,
                    structural=res.structural_total,
                    computational=res.computational_total,
                    tc=res.tc,
                    tier=tc_to_difficulty_label(res.tc),
                ))
                produced += 1
                if progress_callback and len(pool) % 25 == 0:
                    progress_callback(min(len(pool), profile.quantities.invariants),
                                      profile.quantities.invariants, 0.5)

        return pool

    def _tier_quotas(self, mix: Dict[str, int], total: int) -> Dict[str, int]:
        """Translate a difficulty mix (tier -> percentage) into exact integer
        per-tier quotas summing to ``total``.  Robust to percentages that do
        not sum to exactly 100; rounding drift is absorbed by the largest-share
        tier."""
        shares = {t: max(0, mix.get(t, 0)) for t in TIER_ORDER}
        denom = sum(shares.values()) or 1
        q = {t: int(round(total * shares[t] / denom)) for t in TIER_ORDER}
        drift = total - sum(q.values())
        if drift != 0:
            top = max(TIER_ORDER, key=lambda t: shares[t])
            q[top] = max(0, q[top] + drift)
        return q

    def _pick_spread(self, bucket: List['_PooledConstraint'], k: int) -> List['_PooledConstraint']:
        """Pick ``k`` candidates spread across a bucket's measured complexity
        range (sorted by TC, then Structural, then Computational) so a tier is
        not filled with near-identical constraints."""
        if k <= 0 or not bucket:
            return []
        if k >= len(bucket):
            return list(bucket)
        ordered = sorted(bucket, key=lambda pc: (pc.tc, pc.structural, pc.computational))
        if k == 1:
            return [ordered[len(ordered) // 2]]
        picked: List[_PooledConstraint] = []
        used: Set[int] = set()
        for i in range(k):
            idx = round(i * (len(ordered) - 1) / (k - 1))
            while idx in used and idx < len(ordered) - 1:
                idx += 1
            while idx in used and idx > 0:
                idx -= 1
            used.add(idx)
            picked.append(ordered[idx])
        return picked

    def _stratified_select(self, buckets: Dict[str, List['_PooledConstraint']],
                           quotas: Dict[str, int]
                           ) -> Tuple[List['_PooledConstraint'], Dict[str, int]]:
        """Fill each tier's quota from its bucket via spread-picking; return the
        selected candidates and the per-tier shortfall (0 where fully filled)."""
        selected: List[_PooledConstraint] = []
        shortfall: Dict[str, int] = {}
        for tier in TIER_ORDER:
            want = quotas.get(tier, 0)
            have = buckets.get(tier, [])
            take = self._pick_spread(have, min(want, len(have)))
            selected.extend(take)
            shortfall[tier] = max(0, want - len(take))
        return selected, shortfall

    def _report_suite_dependency(self, final: List[OCLConstraint]) -> None:
        """Measure RUC (Reused Constraint Count) on the FINAL selected suite.

        Dependency is emergent and suite-level: measured against the
        constraints actually shipped, never steered per-constraint or measured
        against the throwaway pool.  Reporting only — it does not alter the
        selection.
        """
        if not final:
            return
        total_reuse = 0
        sharing = 0
        for c in final:
            ruc = compute_ruc(c.ocl, c.context, final)
            if ruc > 0:
                sharing += 1
                total_reuse += ruc
        print(f"\nPhase 5    suite-level dependency (RUC) on the {len(final)} "
              f"selected constraints:")
        print(f"           total reuse links={total_reuse}, "
              f"sharing navigations={sharing}/{len(final)}")

    # ------------------------------------------------------------------
    # VGCR refinement hook
    # ------------------------------------------------------------------

    def generate_refined(
        self,
        failed_constraint: 'OCLConstraint',
        generator_state: 'GeneratorState',
    ) -> Optional['OCLConstraint']:
        """
        Generate a refined candidate after a VGCR property-check failure.

        Called by the VGCR loop when a candidate fails q1–q5.  Instead of
        asking the LLM for a new constraint, this method re-samples from
        the existing pattern library while respecting the blacklists and
        weight adjustments recorded in *generator_state*.

        Strategy:
          1. Try a different pattern for the same context class.
          2. If no alternative pattern works, try the same pattern on a
             different context class.
          3. If nothing works, return None (slot unfilled).

        Args:
            failed_constraint: The candidate that failed property checks.
            generator_state: Current blacklists and weight adjustments
                             from previous failures.

        Returns:
            A new OCLConstraint, or None if no alternative can be found.
        """
        context = failed_constraint.context
        failed_pid = failed_constraint.pattern_id
        max_attempts = 15  # cap to avoid infinite loops

        for attempt in range(max_attempts):
            # Pick a pattern, applying weight modifiers from generator state
            try:
                pattern = self._vgcr_weighted_sample(
                    context, failed_pid, generator_state
                )
            except Exception:
                pattern = None

            if pattern is None:
                # No suitable pattern found for this context — try another class
                alt_context = self._pick_alternative_context(
                    context, generator_state
                )
                if alt_context is None:
                    return None
                context = alt_context
                continue

            # Check blacklist before generating
            if generator_state.is_blacklisted(pattern.id, context):
                continue

            # Check applicability
            if not self._is_pattern_applicable(pattern, context):
                continue

            # Generate the constraint
            try:
                params = self._gen_params(pattern, context)

                # Check binding-level blacklist
                if generator_state.is_blacklisted(
                    pattern.id, context, params
                ):
                    continue

                c = self.generator.generate(pattern.id, context, params)

                # Basic diversity check against recent constraints
                # (We can't check against the full suite here — the VGCR
                # loop handles that via property checks.)
                if c is not None:
                    logger.debug(
                        f"  generate_refined: produced "
                        f"'{pattern.id}@{context}' on attempt {attempt}"
                    )
                    return c
            except Exception as e:
                logger.debug(
                    f"  generate_refined: attempt {attempt} failed — "
                    f"{type(e).__name__}: {str(e)[:60]}"
                )
                continue

        return None

    def _vgcr_weighted_sample(
        self,
        context: str,
        avoid_pattern_id: str,
        generator_state: 'GeneratorState',
    ):
        """
        Sample a pattern for VGCR refinement, respecting generator state.

        Avoids the failed pattern, applies weight modifiers, and skips
        blacklisted (pattern, context) pairs. The replacement is kept in the
        SAME family as the failed constraint so refinement does not drift the
        requested family distribution (and cannot inject a zero-quota family
        such as 'conditional' as a substitute for, e.g., a cardinality slot).
        """
        def _fam(p):
            return classify_family(p.id, getattr(p.category, "value", str(p.category)))

        failed_pat = next((p for p in self.all_patterns if p.id == avoid_pattern_id), None)
        failed_family = _fam(failed_pat) if failed_pat is not None else None

        candidates = [
            p for p in self.all_patterns
            if p.id not in self.EXCLUDE_LIST
            and p.id != avoid_pattern_id
            and not generator_state.is_blacklisted(p.id, context)
            and (failed_family is None or _fam(p) == failed_family)
        ]

        if not candidates:
            return None

        # Build weights incorporating generator state modifiers
        weights = []
        for p in candidates:
            base_w = 1.0
            # Apply generator state weight modifier
            mod = generator_state.get_weight_modifier(p.id)
            weights.append(base_w * mod)

        total_w = sum(weights)
        if total_w <= 0:
            return None

        # Weighted random selection
        import random
        r = random.random() * total_w
        cumulative = 0.0
        for p, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return p

        return candidates[-1]  # fallback

    def _pick_alternative_context(
        self,
        avoid_context: str,
        generator_state: 'GeneratorState',
    ) -> Optional[str]:
        """Pick a different context class for refinement."""
        import random
        all_classes = [
            c for c in self.metamodel.get_class_names()
            if c != avoid_context
        ]
        if not all_classes:
            return None
        random.shuffle(all_classes)
        return all_classes[0]

    def _plan_families(self, profile: BenchmarkProfile) -> Dict[str, int]:
        """Convert family percentages to counts."""
        total = profile.quantities.invariants
        pct = profile.quantities.families_pct
        s = sum(max(0, int(v)) for v in pct.values()) or 1
        weights = {k: max(0, int(v)) / s for k, v in pct.items()}
        counts = {k: int(round(total * weights.get(k, 0))) for k in FAMILY_KEYS}
        
        # Adjust rounding
        diff = total - sum(counts.values())
        if diff != 0:
            top = max(FAMILY_KEYS, key=lambda k: weights.get(k, 0))
            counts[top] = max(0, counts[top] + diff)
        return counts
    
    def _get_enabled_patterns(self, family: str, profile: BenchmarkProfile):
        """Get enabled patterns for family."""
        all_in_family = self.patterns_by_family.get(family, [])
        if not profile.library.enabled:
            return all_in_family
        return [p for p in all_in_family if p.id in profile.library.enabled]
    
    def _is_pattern_applicable(self, pattern, context: str) -> bool:
        """Quick check if pattern is applicable to the given context class."""
        # Special applicability checks for known problematic patterns
        if pattern.id == 'acyclicity':
            # Require a self-association so closure stays in same type
            self_assocs = [
                a for a in self.metamodel.get_associations_for(context)
                if a.target_class == context
            ]
            if not self_assocs:
                return False
        if pattern.id in ['association_exists', 'type_check']:
            # Require at least one single-valued association
            if not self.metamodel.get_single_associations(context):
                return False
        # Navigation patterns resolve their target from a single/collection
        # association at instantiation, so the generic empty-options gate (which
        # probes with no dependency values) cannot see it. Decide here instead.
        if pattern.id == 'simple_navigation':
            # Template 'self.assoc.attr' is a boolean invariant only when attr is
            # boolean, so require a single assoc reaching a boolean attribute.
            return any(
                any(self._normalize_type(at.type) == 'boolean'
                    for at in self.metamodel.get_attributes_for(a.target_class))
                for a in self.metamodel.get_single_associations(context)
            )
        if pattern.id == 'navigation_chain':
            # Boolean path synthesised over single-valued associations (1-2 hops).
            return self._can_navigate(context)
        if pattern.id == 'collection_navigation':
            # Quantified predicate synthesised over a collection's target class.
            return any(
                self._has_usable_attr(a.target_class)
                for a in self.metamodel.get_collection_associations(context)
            )
        if pattern.id in _QUANT_COLLECTION_KIND:
            # forAll/exists/any/one/select/collect over a collection association: the
            # condition/attribute is synthesised at instantiation. Applicable iff at
            # least one collection reaches a target that can support the body — the
            # SAME viable set _gen_params draws the collection from, so a pass here
            # guarantees a successful instantiation (no false positives).
            return bool(self._viable_collections(context, _QUANT_COLLECTION_KIND[pattern.id]))
        if pattern.id == 'collection_collectNested':
            # Require a collection association whose target has a collection association,
            # AND both association names must pass the conservative domain guard (Fix 2).
            found_valid_two_hop = False
            for assoc in self.metamodel.get_collection_associations(context):
                nested = self.metamodel.get_collection_associations(assoc.target_class)
                if nested and self._association_names_compatible(
                    assoc.ref_name, nested[0].ref_name
                ):
                    found_valid_two_hop = True
                    break
            if not found_valid_two_hop:
                return False

        for param in pattern.parameters:
            if not param.required:
                continue
            options = param.get_options_for_context(self.metamodel, context, {})
            if not options and param.default is None:
                return False
        return True

    def _weighted_sample(self, patterns, profile: BenchmarkProfile, context: str = None):
        """Sample pattern with weights using: base * universal * solver * relevance * history."""
        logger.debug(f"      _weighted_sample() - Input: {len(patterns)} patterns")
        # C: Use centralized exclude-list
        logger.debug(f"      Applying exclude-list ({len(self.EXCLUDE_LIST)} patterns)...")
        
        # Universal patterns that work with ANY model (71 total)
        UNIVERSAL_PATTERNS = {
            # Basic null checks
            'attribute_not_null_simple',
            # Comparisons
            'two_attributes_not_equal',
            # Numeric
            'numeric_positive', 'numeric_non_negative', 'numeric_bounded',
            'numeric_greater_than_value', 'numeric_less_than_value', 'numeric_even', 'numeric_odd',
            'numeric_multiple_of', 'numeric_sum_constraint', 'numeric_difference_constraint',
            'numeric_abs_bounded', 'numeric_product_constraint',
            # String
            'string_min_length', 'string_max_length', 'string_exact_length',
            'string_contains_substring', 'string_starts_with', 'string_to_upper_equals',
            'attribute_value_in_set',
            # Collection
            'collection_not_empty_simple', 'collection_has_size',
            'collection_size_range', 'collection_not_empty_check', 'collection_min_size',
            'collection_max_size', 'collection_empty_check',
            # Boolean
            'boolean_is_true', 'boolean_is_false',
            # Logic
            'implies_simple', 'implies_reverse',
            'all_attributes_defined', 'three_attributes_defined',
            # Advanced collections (10 new)
            'collection_including', 'collection_excluding', 'collection_sortedBy', 'collection_sum',
            'collection_isUnique_attr', 'collection_first', 'collection_last', 'collection_any_match',
            'collection_collectNested',
            # Conditionals (2 new)
            'conditional_if_then_else', 'conditional_value_selection',
            # Collection conversions & sequence ops (5 new)
            'collection_asSet', 'collection_asSequence', 'collection_at_index', 'collection_indexOf'
        }
        
        # Filter out excluded patterns
        filtered_patterns = [p for p in patterns if p.id not in self.EXCLUDE_LIST]
        excluded_count = len(patterns) - len(filtered_patterns)
        if excluded_count > 0:
            logger.debug(f"      Filtered out {excluded_count} blacklisted patterns")
            excluded_in_input = [p.id for p in patterns if p.id in self.EXCLUDE_LIST]
            logger.debug(f"      Excluded patterns in input: {excluded_in_input}")
        logger.debug(f"      After exclude-list: {len(filtered_patterns)} patterns remaining")
        
        # Filter out patterns with invalid templates (if validator available)
        if self.pattern_template_validator:
            valid_patterns = []
            for p in filtered_patterns:
                template = getattr(p, 'template', '')
                if template:
                    is_valid, reason = self.pattern_template_validator(p.id, template)
                    if is_valid:
                        valid_patterns.append(p)
                else:
                    # No template, include it
                    valid_patterns.append(p)
            filtered_patterns = valid_patterns
        
        if not filtered_patterns:
            # Fallback: use original if all filtered out (shouldn't happen)
            filtered_patterns = patterns
        
        weights = []
        for p in filtered_patterns:
            # (1) User preference: explicit pattern weights (family mix is handled upstream)
            w_base = profile.library.weights.get(p.id, 1.0)

            # (2) Universal applicability boost
            w_gen = 5.0 if p.id in UNIVERSAL_PATTERNS else 1.0

            # (3) Solver-friendly boost (only when verification is enabled)
            w_solver = 2.0 if (self.verification_enabled and p.id in self.SMT_VERIFIED_PATTERNS) else 1.0

            # (4) Relevance to the class (soft applicability score)
            w_rel = self._compute_relevance_weight(p, context) if context else 1.0

            # (5) Run-time adaptation from success/failure history
            w_hist = 1.0
            if context:
                key = (p.id, context)
                success = self.pattern_success_counts.get(key, 0)
                failure = self.pattern_fail_counts.get(key, 0)
                if success + failure >= 3:
                    success_rate = success / (success + failure)
                    if success_rate > 0.7:
                        w_hist = 1.5
                    elif success_rate < 0.3:
                        w_hist = 0.3

            w = w_base * w_gen * w_solver * w_rel * w_hist

            # TIER 3: Boost semantically suggested patterns (3x boost)
            if context and hasattr(self, 'pattern_suggester'):
                try:
                    suggestions = self.pattern_suggester.suggest_for_class(context)
                    suggested_ids = {s.pattern_id for s in suggestions if s.priority in ['critical', 'high']}
                    if p.id in suggested_ids:
                        w *= 3.0
                except Exception:
                    pass

            # COMPLEXITY STEERING: steer pattern selection toward target TC range
            # Use pattern.complexity (1-5) as a proxy for expected TC
            if hasattr(self, '_coverage') and self._coverage and self._coverage.tc_scores:
                avg_tc = self._coverage.avg_tc()
                min_tc = profile.complexity.min_tc
                max_tc = profile.complexity.max_tc
                pattern_complexity = getattr(p, 'complexity', 1)
                if avg_tc < min_tc:
                    # Below target: boost high-complexity patterns
                    if pattern_complexity >= 3:
                        w *= 2.0
                    elif pattern_complexity <= 1:
                        w *= 0.5
                elif avg_tc > max_tc:
                    # Above target: boost low-complexity patterns
                    if pattern_complexity <= 2:
                        w *= 2.0
                    elif pattern_complexity >= 4:
                        w *= 0.5

            # PER-DIMENSION STEERING: steer toward per-dimension target ranges
            if hasattr(self, '_coverage') and self._coverage:
                pattern_complexity = getattr(p, 'complexity', 1)
                cc = profile.complexity

                # Structural steering
                if (cc.structural_enabled and cc.structural_target_min is not None
                        and cc.structural_target_max is not None
                        and self._coverage.structural_scores):
                    avg_s = self._coverage.avg_structural()
                    if avg_s < cc.structural_target_min:
                        # Need more structural complexity: boost navigation-heavy patterns
                        if pattern_complexity >= 3:
                            w *= 1.5
                        elif pattern_complexity <= 1:
                            w *= 0.7
                    elif avg_s > cc.structural_target_max:
                        if pattern_complexity <= 2:
                            w *= 1.5
                        elif pattern_complexity >= 4:
                            w *= 0.7

                # Computational steering
                if (cc.computational_enabled and cc.computational_target_min is not None
                        and cc.computational_target_max is not None
                        and self._coverage.computational_scores):
                    avg_c = self._coverage.avg_computational()
                    if avg_c < cc.computational_target_min:
                        # Need more computational complexity: boost quantifier/iteration patterns
                        if pattern_complexity >= 3:
                            w *= 1.5
                        elif pattern_complexity <= 1:
                            w *= 0.7
                    elif avg_c > cc.computational_target_max:
                        if pattern_complexity <= 2:
                            w *= 1.5
                        elif pattern_complexity >= 4:
                            w *= 0.7

            weights.append(w)

        # If relevance weighting zeroed out candidates, fall back to uniform weights
        if context:
            filtered = [(p, w) for p, w in zip(filtered_patterns, weights) if w > 0]
            if filtered:
                filtered_patterns, weights = zip(*filtered)
                filtered_patterns = list(filtered_patterns)
                weights = list(weights)
            else:
                weights = [1.0 for _ in filtered_patterns]
        else:
            if all(w == 0 for w in weights):
                weights = [1.0 for _ in filtered_patterns]
        
        total_w = sum(weights) or 1
        probs = [w / total_w for w in weights]
        return random.choices(filtered_patterns, weights=probs, k=1)[0]

    def _compute_relevance_weight(self, pattern, context: str) -> float:
        """Soft relevance weight based on availability of required parameters."""
        if not context:
            return 1.0

        required = [p for p in pattern.parameters if getattr(p, 'required', False)]
        if not required:
            return 1.0

        scores = []
        for param in required:
            options = param.get_options_for_context(self.metamodel, context, {})
            if (not options) and param.default is None:
                return 0.0  # Not applicable to this class
            if options:
                cap = 5
                scores.append(min(len(options), cap) / cap)
            else:
                scores.append(1.0)

        return sum(scores) / max(1, len(scores))
    
    def _select_context(self, class_quota, profile, coverage: CoverageState) -> Optional[str]:
        """Select context class with room, preferring under-covered and high-complexity classes."""
        classes = self.metamodel.get_class_names()
        max_cap = profile.quantities.per_class_max
        
        # TIER 3: Prioritize high-complexity classes for better constraint coverage
        if profile.redundancy.novelty_boost and hasattr(self, 'structure_analyzer'):
            candidates = [c for c in classes if class_quota[c] < max_cap]
            if candidates:
                # Prefer classes not yet used
                unused = [c for c in candidates if c not in coverage.classes_used]
                if unused:
                    candidates = unused
                
                # Weight selection by complexity (higher complexity = higher chance)
                try:
                    candidates_with_complexity = []
                    for c in candidates:
                        metrics = self.structure_analyzer.analyze_class_complexity(c)
                        if metrics:
                            candidates_with_complexity.append((c, metrics['complexity_score']))
                        else:
                            candidates_with_complexity.append((c, 1.0))
                    
                    # Weight by complexity
                    if candidates_with_complexity:
                        total_complexity = sum(score for _, score in candidates_with_complexity) or 1
                        weights = [max(1.0, score) / total_complexity for _, score in candidates_with_complexity]
                        return random.choices([c for c, _ in candidates_with_complexity], weights=weights, k=1)[0]
                except Exception:
                    pass
                
                return random.choice(candidates)
        
        # Fallback: Random with cap
        for _ in range(10):
            c = random.choice(classes)
            if class_quota[c] < max_cap:
                return c
        return None
    
    def _gen_params(self, pattern, context: str) -> Dict:
        """Generate parameters for pattern.
        
        Raises:
            ValueError: If required parameters cannot be populated or semantic validation fails
        """
        def _number_default_for(name: str) -> int:
            """Generate a reasonable numeric default based on parameter name."""
            if name in ['max_length', 'min_length', 'length']:
                if name == 'max_length':
                    return random.randint(10, 100)
                if name == 'min_length':
                    return random.randint(1, 20)
                return random.randint(5, 50)
            if name in ['index', 'position']:
                return random.randint(1, 5)
            if name in ['start', 'substring_start']:
                return random.randint(1, 5)
            if name in ['end', 'substring_end']:
                # End must be > start if start exists
                start_val = params.get('start', params.get('substring_start', 1))
                return random.randint(start_val + 1, start_val + 10)
            if name in ['prefix_length', 'suffix_length']:
                return random.randint(2, 10)
            if name == 'divisor':
                return random.randint(2, 9)   # >= 2: 'mod 1 = 0' is a tautology
            if name == 'size' or 'size' in name.lower():
                return random.randint(1, 3)
            if name in ['threshold', 'value', 'min_value', 'max_value']:
                return random.randint(1, 100)
            return random.randint(0, 10)

        logger.debug(f"    _gen_params() - Pattern: {pattern.id}, Context: {context}")
        logger.debug(f"    Pattern has {len(pattern.parameters)} parameters")
        params = {}
        for param in pattern.parameters:
            options = param.get_options_for_context(self.metamodel, context, params)

            # Override options for patterns requiring special association handling
            if pattern.id in ['association_exists', 'type_check'] and param.options == 'associations':
                options = [a.ref_name for a in self.metamodel.get_single_associations(context)]

            if pattern.id == 'type_check' and param.name == 'type':
                assoc_name = params.get('attribute')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [assoc.target_class]
                else:
                    options = []

            if pattern.id == 'acyclicity' and param.options == 'associations':
                self_assocs = [
                    a.ref_name for a in self.metamodel.get_associations_for(context)
                    if a.target_class == context
                ]
                options = self_assocs

            if pattern.id == 'collection_collectNested' and param.name == 'nested_collection':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [a.ref_name for a in self.metamodel.get_collection_associations(assoc.target_class)]
                else:
                    options = []

            if pattern.id == 'sum_operation' and param.name == 'attribute':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    # Prefer boolean attributes for select(...) predicate
                    options = [
                        a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                        if self._normalize_type(a.type) == 'boolean'
                    ]
                else:
                    options = []

            if pattern.id == 'collection_sum' and param.name == 'numeric_attribute':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                        if self._normalize_type(a.type) == 'numeric'
                    ]
                else:
                    options = []

            if pattern.id == 'difference_operation' and param.name == 'collection2':
                assoc_name = params.get('collection1')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    # A set op against the SAME collection is a no-op (X - X = empty,
                    # X u X = X): require a DISTINCT collection of the same target type.
                    options = [
                        a.ref_name for a in self.metamodel.get_collection_associations(context)
                        if a.target_class == assoc.target_class and a.ref_name != assoc_name
                    ]
                else:
                    options = []

            if pattern.id in ['symmetricDifference', 'union_operation', 'intersection_operation', 'includesAll_excludesAll'] and param.name == 'collection2':
                assoc_name = params.get('collection1')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    # Distinct collection of the same target type — self-ops are no-ops.
                    options = [
                        a.ref_name for a in self.metamodel.get_collection_associations(context)
                        if a.target_class == assoc.target_class and a.ref_name != assoc_name
                    ]
                else:
                    options = []

            if pattern.id == 'collection_including' and param.name == 'element':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.ref_name for a in self.metamodel.get_single_associations(context)
                        if a.target_class == assoc.target_class
                    ]
                else:
                    options = []

            if pattern.id == 'flatten_operation' and param.name == 'nested_collection':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.ref_name for a in self.metamodel.get_collection_associations(assoc.target_class)
                    ]
                else:
                    options = []

            if pattern.id == 'division_modulo' and param.name == 'attr1':
                options = [
                    a.name for a in self.metamodel.get_attributes_for(context)
                    if self._is_integer_type(a.type)
                ]

            # 'mod' is an Integer-only OCL op — never apply it to EDouble/Real
            # attributes (fixes estimatedCost/lateFee/discount mod ...).
            if pattern.id in ['numeric_even', 'numeric_odd', 'numeric_multiple_of'] \
                    and param.name == 'numeric_attribute':
                options = [
                    a.name for a in self.metamodel.get_attributes_for(context)
                    if self._is_integer_type(a.type)
                ]

            # String character ops (toLowerCase/toUpperCase) are meaningless on
            # date-role attributes (timestamp, expiry, ...); restrict to plain text.
            if pattern.id in ['string_to_lower_equals', 'string_to_upper_equals'] \
                    and param.name == 'string_attribute':
                options = [
                    a.name for a in self.metamodel.get_attributes_for(context)
                    if self._normalize_type(a.type) == 'string'
                    and not any(k in a.name.lower() for k in
                                ('date', 'time', 'expiry', 'expire', 'created', 'updated', 'at'))
                ]

            if pattern.id in ['sortedBy', 'collection_sortedBy'] and param.name == 'attribute':
                assoc_name = params.get('collection')
                assoc = self._get_association(context, assoc_name) if assoc_name else None
                if assoc:
                    options = [
                        a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                        if self._normalize_type(a.type) in ['numeric', 'string']
                    ]
                else:
                    options = []

            # 'self.assoc.attr' is a valid boolean invariant only for a boolean
            # target attribute, so restrict the resolved options accordingly.
            if pattern.id == 'simple_navigation' and param.name == 'attribute':
                assoc = self._get_association(context, params.get('association'))
                options = [
                    a.name for a in self.metamodel.get_attributes_for(assoc.target_class)
                    if self._normalize_type(a.type) == 'boolean'
                ] if assoc else []

            # Navigation patterns: synthesise free-form expression parameters from
            # the metamodel. These params carry no option list and no default, so
            # without this they would fail the generic "required" check below.
            if pattern.id == 'navigation_chain' and param.name == 'navigation_path':
                expr = self._synth_navigation_path(context)
                if expr is None:
                    raise ValueError("Parameter validation failed: Navigation Path is required")
                params[param.name] = expr
                continue
            if pattern.id == 'collection_navigation' and param.name == 'operation':
                # Restrict to quantifiers so the synthesised body stays a boolean
                # constraint (select/collect would yield a non-boolean expression).
                options = ['forAll', 'exists']
            if pattern.id == 'collection_navigation' and param.name == 'nested':
                pred = self._synth_nested_predicate(context, params)
                if pred is None:
                    raise ValueError("Parameter validation failed: Nested Expression is required")
                params[param.name] = pred
                continue

            # Quantified collection patterns (forAll/exists/any/one/select/collect):
            # restrict the collection to one whose target can support the synthesised
            # body, then synthesise the iteration condition from that target. The
            # collection filter is what makes applicable == instantiable everywhere.
            if pattern.id in _QUANT_COLLECTION_KIND and param.name == 'collection':
                options = self._viable_collections(context, _QUANT_COLLECTION_KIND[pattern.id])
            if pattern.id in ('forall_nested', 'exists_constraint', 'any_operation',
                              'one_operation', 'select_operation') and param.name == 'condition':
                assoc = self._get_association(context, params.get('collection'))
                # forAll/select are encoded only for numeric `var.attr OP <number>`.
                numeric_only = pattern.id in ('forall_nested', 'select_operation')
                cond = self._attr_predicate(
                    assoc.target_class, var=params.get('iterator', 'x'),
                    numeric_only=numeric_only,
                ) if assoc else None
                if cond is None:
                    raise ValueError("Parameter validation failed: Condition is required")
                params[param.name] = cond
                continue

            # FILTER: Date filter DISABLED to maximize constraint generation
            # Dates are typed as EString and have separate encoder support
            # if options and param.options == 'string_attributes':
            #     filtered_string_options = []
            #     for opt in options:
            #         opt_lower = opt.lower()
            #         if any(x in opt_lower for x in ['date', 'time', 'at', 'when', 'timestamp']):
            #             continue
            #         filtered_string_options.append(opt)
            #     if filtered_string_options:
            #         options = filtered_string_options
            
            if options:
                # Avoid null-checks on string attributes already constrained by string patterns
                if pattern.id in ['attribute_null_check', 'null_check'] and param.name == 'attribute':
                    blocked = self._string_constrained_attrs.get(context, set())
                    if blocked:
                        options = [opt for opt in options if opt not in blocked]
                        if not options:
                            raise ValueError(
                                "Parameter validation failed: no eligible attributes for null-check "
                                "(string attributes already constrained)"
                            )
                # Special handling for patterns that compare two things
                # Ensure they're different to avoid tautologies (e.g., self.x = self.x)
                if param.name in ['second_attribute', 'second_string', 'second_numeric', 'second_attr', 'attribute2', 'attr2', 'third_attribute', 'third_numeric']:
                    # Get the "first" parameter name (attr2 -> attr1 covers numeric_comparison)
                    first_param_name = param.name.replace('second_', 'first_').replace('third_', 'first_').replace('attribute2', 'attribute1').replace('attr2', 'attr1').replace('_attr', '')
                    if 'attribute' in first_param_name and first_param_name not in params:
                        first_param_name = 'attribute'  # Fallback
                    
                    if first_param_name in params:
                        # For third parameter, also check against second parameter
                        excluded_values = [params[first_param_name]]
                        if 'third' in param.name:
                            second_param_name = param.name.replace('third_', 'second_')
                            if second_param_name in params:
                                excluded_values.append(params[second_param_name])
                        
                        # Filter out excluded values
                        options = [opt for opt in options if opt not in excluded_values]
                        
                        # Additional validation: filter out attributes with incompatible types
                        first_attr = params[first_param_name]
                        first_type = self._get_attribute_type(context, first_attr)
                        filtered_options = []
                        for opt in options:
                            opt_type = self._get_attribute_type(context, opt)
                            if first_type and opt_type and not self._types_compatible(first_type, opt_type):
                                continue
                            if not self._attribute_names_compatible(first_attr, opt, context=context):
                                continue

                            filtered_options.append(opt)
                        
                        # Apply filtered options (even if empty to force failure)
                        options = filtered_options
                        
                        # TIER 2: Apply semantic filtering for attribute pairs
                        if hasattr(self, 'semantic_validator') and self.semantic_validator:
                            options = self._filter_semantically_valid_options(
                                pattern.id, first_param_name, params[first_param_name], 
                                param.name, options, context
                            )
                        
                        if not options:
                            # If no valid option available, fail
                            raise ValueError(f"Parameter validation failed: {param.label} must differ from {first_param_name} and be type-compatible")
                
                params[param.name] = random.choice(options)
            elif param.required and param.default is not None:
                # Required parameter with no options but has default (e.g., iterator variable)
                params[param.name] = param.default
            elif param.required:
                # No options/default; try to synthesize required numeric values
                t = getattr(param.type, "value", str(param.type))
                if t == "number":
                    params[param.name] = _number_default_for(param.name)
                else:
                    # Cannot populate required parameter - pattern not applicable to this context
                    raise ValueError(f"Parameter validation failed: {param.label} is required")
            else:
                # Optional parameter - use defaults
                t = getattr(param.type, "value", str(param.type))
                if t == "number":
                    params[param.name] = _number_default_for(param.name)
                elif t == "boolean":
                    params[param.name] = random.choice(["true", "false"])
                elif t == "text":
                    # Honour the declared default (e.g. select_operation's
                    # result_operation '->size() >= 1'); only fall back otherwise.
                    params[param.name] = param.default if param.default is not None else "value"
                else:
                    # Use default if available, otherwise fail for missing metamodel-dependent params
                    if param.default is not None:
                        params[param.name] = param.default
                    elif param.options in ['string_attributes', 'numeric_attributes', 'attributes', 
                                          'collection_associations', 'single_associations']:
                        # Metamodel-dependent parameter with no options - pattern not applicable
                        raise ValueError(f"Parameter '{param.label}' requires {param.options} but none found for {context}")
                    else:
                        # Generic parameter - use empty string as fallback
                        params[param.name] = ""
        
        # Universal self-comparison check (catches all patterns)
        logger.debug(f"    _gen_params() - Running self-comparison check...")
        self._check_self_comparison(pattern.id, params)
        logger.debug(f"    _gen_params() - Self-comparison check passed")

        # Type compatibility check for attribute pairs
        logger.debug(f"    _gen_params() - Running attribute type compatibility check...")
        self._validate_attribute_pairs(context, pattern.id, params)
        logger.debug(f"    _gen_params() - Attribute type compatibility check passed")
        
        # Pattern-specific parameter validation
        logger.debug(f"    _gen_params() - Running parameter validation...")
        self._validate_pattern_params(pattern.id, params)
        logger.debug(f"    _gen_params() - Parameter validation passed")
        
        # TIER 2: Final semantic validation of complete parameter set
        if hasattr(self, 'semantic_validator') and self.semantic_validator:
            is_valid, reason = self.semantic_validator.validate_parameters(pattern.id, context, params)
            if not is_valid:
                raise ValueError(f"Semantic validation failed: {reason}")
        
        return params

    def _track_string_constraints(self, pattern, params: Dict, context: str) -> None:
        """Track string attributes already constrained by string patterns."""
        category = getattr(pattern.category, "value", str(pattern.category))
        is_string_pattern = category == "string" or pattern.id.startswith("string_")
        if not is_string_pattern:
            return

        for param in pattern.parameters:
            if param.options == "string_attributes":
                attr = params.get(param.name)
                if attr:
                    self._string_constrained_attrs[context].add(attr)
            elif param.name in ("attribute", "string_attribute", "first_string", "second_string"):
                attr = params.get(param.name)
                if attr:
                    attr_type = self._get_attribute_type(context, attr)
                    if attr_type and self._normalize_type(attr_type) == "string":
                        self._string_constrained_attrs[context].add(attr)
    
    def _check_self_comparison(self, pattern_id: str, params: Dict):
        """Check for self-comparison tautologies across all attribute parameters.
        
        Raises:
            ValueError: If same attribute used in comparison
        """
        logger.debug(f"      _check_self_comparison() - Pattern: {pattern_id}")
        logger.debug(f"      Parameters: {params}")
        
        # Collect all attribute parameter values
        attr_params = {}
        for key, value in params.items():
            # Look for attribute parameters (not numeric values or operators)
            if 'attribute' in key.lower() or 'attr' in key:
                # Skip if it's a boolean/operator value
                if value not in ['true', 'false', None, ''] and not isinstance(value, (int, float)):
                    attr_params[key] = value
        
        logger.debug(f"      Found {len(attr_params)} attribute parameters: {attr_params}")
        
        # Check for duplicates
        if len(attr_params) >= 2:
            values = list(attr_params.values())
            unique_values = set(values)
            logger.debug(f"      Checking for duplicates: {len(values)} values, {len(unique_values)} unique")
            if len(unique_values) < len(values):
                # Found duplicate - identify which
                for i, val in enumerate(values):
                    if values.count(val) > 1:
                        logger.debug(f"      Self-comparison detected: attribute '{val}' used multiple times")
                        raise ValueError(f"Self-comparison detected: attribute '{val}' used multiple times in {pattern_id}")
        else:
            logger.debug(f"      No duplicate check needed (only {len(attr_params)} attribute params)")

    def _validate_attribute_pairs(self, context: str, pattern_id: str, params: Dict):
        """Ensure attribute pairs are type- and name-compatible.

        Handles three cases:
          (a) Both attributes in the context class  → full LLM + type check
          (b) One attribute cross-class, one in ctx → cross-class type + keyword check
          (c) Both attributes cross-class           → logged and skipped (path unknown)

        Raises:
            ValueError: If attribute pairs are incompatible
        """
        attr_names = {a.name for a in self.metamodel.get_attributes_for(context)}
        pairs = [
            ('first_attribute', 'second_attribute'),
            ('attribute1', 'attribute2'),
            ('first_attr', 'second_attr'),
            ('first_numeric', 'second_numeric'),
            ('first_string', 'second_string'),
            ('first_attribute', 'third_attribute'),
            ('second_attribute', 'third_attribute'),
            ('first_numeric', 'third_numeric'),
            ('second_numeric', 'third_numeric'),
        ]

        for left_key, right_key in pairs:
            if left_key not in params or right_key not in params:
                continue
            left = params[left_key]
            right = params[right_key]

            left_in_ctx  = left  in attr_names
            right_in_ctx = right in attr_names

            # ── Case (c): both cross-class — cannot validate without path info ──
            if not left_in_ctx and not right_in_ctx:
                logger.warning(
                    "[cross-class skip] Both '%s' and '%s' are outside context "
                    "'%s' in pattern '%s' — semantic validation bypassed. "
                    "Path resolution not yet implemented for this case.",
                    left, right, context, pattern_id
                )
                continue

            # ── Case (a): both within context class — original validation ────
            if left_in_ctx and right_in_ctx:
                left_type  = self._get_attribute_type(context, left)
                right_type = self._get_attribute_type(context, right)
                if left_type and right_type and not self._types_compatible(left_type, right_type):
                    raise ValueError(
                        f"Parameter validation failed: {left_key} and {right_key} "
                        f"must be type-compatible"
                    )
                if not self._attribute_names_compatible(left, right, context=context):
                    raise ValueError(
                        f"Parameter validation failed: {left_key} and {right_key} "
                        f"must be semantically compatible"
                    )
                continue

            # ── Case (b): one cross-class, one in context ─────────────────────
            ctx_attr   = left  if left_in_ctx  else right
            cross_attr = right if left_in_ctx  else left

            ctx_type   = self._get_attribute_type(context, ctx_attr)
            cross_type = self._resolve_cross_attr_type_multihop(context, cross_attr, params)

            # Type check (skip if type cannot be resolved — conservative admission)
            if ctx_type and cross_type and not self._types_compatible(ctx_type, cross_type):
                raise ValueError(
                    f"Parameter validation failed: cross-class attributes "
                    f"'{ctx_attr}' ({ctx_type}) and '{cross_attr}' ({cross_type}) "
                    f"are type-incompatible"
                )

            # Semantic check — use domain-family classifier so that
            # same-family cross-class pairs (e.g. money vs money) are admitted
            # and only genuinely incompatible pairs (e.g. money vs temporal)
            # are rejected.  _cross_class_semantics_compatible() falls back to
            # the keyword heuristic when the import is unavailable.
            if not self._cross_class_semantics_compatible(ctx_attr, cross_attr):
                raise ValueError(
                    f"Parameter validation failed: cross-class attributes "
                    f"'{ctx_attr}' and '{cross_attr}' are semantically incompatible"
                )
    
    def _validate_pattern_params(self, pattern_id: str, params: Dict):
        """Validate pattern-specific parameter constraints.
        
        Raises:
            ValueError: If parameters don't meet pattern-specific constraints
        """
        logger.debug(f"      _validate_pattern_params() - Pattern: {pattern_id}")
        logger.debug(f"      Parameters to validate: {params}")
        
        # String length patterns
        if pattern_id in ['string_max_length', 'string_min_length', 'string_exact_length',
                         'string_not_empty', 'string_operation']:
            logger.debug(f"      Checking string length constraints...")
            if 'max_length' in params and params['max_length'] < 1:
                logger.debug(f"      Invalid max_length: {params['max_length']}")
                raise ValueError(f"max_length must be at least 1, got {params['max_length']}")
            if 'min_length' in params and params['min_length'] < 1:
                logger.debug(f"      Invalid min_length: {params['min_length']}")
                raise ValueError(f"min_length must be at least 1, got {params['min_length']}")
            if 'length' in params and params['length'] < 1:
                logger.debug(f"      Invalid length: {params['length']}")
                raise ValueError(f"length must be at least 1, got {params['length']}")
            logger.debug(f"      String length constraints OK")
        
        # Substring patterns - ensure valid ranges
        if pattern_id == 'string_starts_with' and 'prefix_length' in params:
            logger.debug(f"      Checking prefix_length for string_starts_with...")
            # For substring(1, prefix_length), prefix_length must be > 1 to extract characters
            if params['prefix_length'] <= 1:
                logger.debug(f"      Invalid prefix_length: {params['prefix_length']} (must be > 1)")
                raise ValueError(f"prefix_length ({params['prefix_length']}) must be > 1 to extract characters")
            logger.debug(f"      prefix_length OK: {params['prefix_length']}")
        
        if pattern_id in ['string_operation'] and 'start' in params and 'end' in params:
            logger.debug(f"      Checking substring start/end...")
            if params['start'] >= params['end']:
                logger.debug(f"      Invalid substring range: start={params['start']}, end={params['end']}")
                raise ValueError(f"substring start ({params['start']}) must be less than end ({params['end']})")
            logger.debug(f"      Substring range OK: start={params['start']}, end={params['end']}")
        
        # Numeric range patterns - ensure min < max
        if pattern_id in ['numeric_bounded', 'range_constraint']:
            if 'min_value' in params and 'max_value' in params:
                if params['min_value'] >= params['max_value']:
                    raise ValueError(f"min_value ({params['min_value']}) must be less than max_value ({params['max_value']})")
        
        # Collection size patterns
        if pattern_id in ['size_constraint', 'collection_min_size', 'collection_max_size', 
                         'collection_has_size', 'collection_size_range']:
            if 'size' in params and params['size'] < 0:
                raise ValueError(f"size must be non-negative, got {params['size']}")
            if 'min_size' in params and params['min_size'] < 0:
                raise ValueError(f"min_size must be non-negative, got {params['min_size']}")
            if 'max_size' in params and params['max_size'] < 0:
                raise ValueError(f"max_size must be non-negative, got {params['max_size']}")
            
            # Ensure min_size <= max_size for range patterns
            if 'min_size' in params and 'max_size' in params:
                if params['min_size'] > params['max_size']:
                    raise ValueError(f"min_size ({params['min_size']}) must not exceed max_size ({params['max_size']})")
        
        # Modulo/divisor patterns - divisor must be >= 2 ('x mod 1 = 0' is always true)
        if pattern_id in ['numeric_even', 'numeric_odd', 'numeric_multiple_of', 'division_modulo']:
            if 'divisor' in params and params['divisor'] < 2:
                raise ValueError(f"divisor must be >= 2, got {params['divisor']}")
    
    def _is_diverse(self, candidate: OCLConstraint, existing: List[OCLConstraint], profile: BenchmarkProfile) -> bool:
        """Check if candidate is diverse enough."""
        threshold = profile.redundancy.similarity_threshold
        for e in existing[-min(20, len(existing)):]:  # check last 20
            if similarity(candidate, e) > threshold:
                return False
        return True
    
    def _filter_semantically_valid_options(self, pattern_id: str, first_param: str, 
                                            first_value: str, second_param: str, 
                                            options: List[str], context: str) -> List[str]:
        """Filter options to keep only semantically valid combinations.
        
        Args:
            pattern_id: Pattern identifier
            first_param: Name of first parameter
            first_value: Value of first parameter
            second_param: Name of second parameter
            options: Available options for second parameter
            context: Context class name
            
        Returns:
            Filtered list of valid options
        """
        if not self.semantic_validator:
            return options
        
        # For two-attribute patterns, filter based on semantic compatibility
        if pattern_id in ['two_attributes_equal', 'two_attributes_not_equal', 'numeric_comparison']:
            from semantic_rules import is_valid_equality_pair
            
            valid_options = []
            for option in options:
                # Check if this pair makes semantic sense
                if is_valid_equality_pair(first_value, option, context):
                    valid_options.append(option)
            
            # If all options filtered out, return at least one to avoid complete failure
            # (Better to generate one weird constraint than fail completely)
            if not valid_options and options:
                return [options[0]]  # Fallback: return first option
            
            return valid_options if valid_options else options
        
        return options
    
    def _pattern_for_deficit(self, target_name: str, profile: BenchmarkProfile):
        """Find pattern that addresses a coverage deficit."""
        # C: Use centralized exclude-list
        available_patterns = [p for p in self.all_patterns if p.id not in self.EXCLUDE_LIST]
        if not available_patterns:
            available_patterns = self.all_patterns  # Fallback
        
        # Parse target
        if target_name.startswith("op:"):
            op = target_name.split(":")[1]
            # Find patterns that use this operator
            for p in available_patterns:
                if op.lower() in p.template.lower():
                    return p
        elif target_name.startswith("hops:"):
            # Prefer navigation patterns
            nav_patterns = [p for p in self.patterns_by_family.get("navigation", []) if p.id not in self.EXCLUDE_LIST]
            if nav_patterns:
                return random.choice(nav_patterns)
        elif target_name.startswith("depth:"):
            # Prefer quantified patterns
            quant_patterns = [p for p in self.patterns_by_family.get("quantified", []) if p.id not in self.EXCLUDE_LIST]
            if quant_patterns:
                return random.choice(quant_patterns)
        
        return random.choice(available_patterns) if available_patterns else None
    
    def _prune_redundant(self, constraints: List[OCLConstraint], profile: BenchmarkProfile) -> List[OCLConstraint]:
        """Remove redundant constraints (greedy mode)."""
        if profile.redundancy.implication_mode == "greedy":
            # Simple greedy: remove high-similarity pairs
            kept = []
            for c in constraints:
                if all(similarity(c, k) < 0.95 for k in kept):
                    kept.append(c)
            return kept
        return constraints
    
    def _print_failure_stats(self):
        """A: Print pattern failure statistics for debugging and optimization."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not self.pattern_fail_counts:
            return
        
        print(f"\nPattern Success/Failure Statistics:")
        
        # Aggregate by pattern across all contexts
        pattern_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
        
        for (pattern_id, context), count in self.pattern_fail_counts.items():
            pattern_stats[pattern_id]['failure'] += count
        
        for (pattern_id, context), count in self.pattern_success_counts.items():
            pattern_stats[pattern_id]['success'] += count
        
        # Sort by failure count
        sorted_patterns = sorted(pattern_stats.items(), 
                                key=lambda x: x[1]['failure'], 
                                reverse=True)
        
        print("\nTop 10 patterns by failure count:")
        for pattern_id, stats in sorted_patterns[:10]:
            success = stats['success']
            failure = stats['failure']
            total = success + failure
            if total > 0:
                success_rate = success / total * 100
                print(f"  {pattern_id}: {failure} failures, {success} successes ({success_rate:.1f}% success rate)")
    
    def generate_sat_unsat_pairs(self, profile: BenchmarkProfile, num_pairs: int = 10) -> List[Tuple[OCLConstraint, OCLConstraint]]:
        """B: Generate paired SAT/UNSAT constraints for evaluation.
        
        For each pair:
        - First constraint is likely SAT (lenient, single condition)
        - Second constraint adds conflicting condition to make it UNSAT
        
        Args:
            profile: Benchmark configuration
            num_pairs: Number of SAT/UNSAT pairs to generate
            
        Returns:
            List of (sat_constraint, unsat_constraint) tuples
        """
        pairs = []
        contexts = self.metamodel.get_class_names()
        
        for _ in range(num_pairs):
            context = random.choice(contexts)
            
            # Generate base SAT constraint using SAT-likely pattern
            sat_patterns = [p for p in self.all_patterns 
                          if p.id in self.SAT_LIKELY_PATTERNS and p.id not in self.EXCLUDE_LIST]
            
            if not sat_patterns:
                continue
            
            sat_pattern = random.choice(sat_patterns)
            
            try:
                params = self._gen_params(sat_pattern, context)
                sat_constraint = self.generator.generate(sat_pattern.id, context, params)
                
                # Try to generate conflicting UNSAT constraint
                # Use UNSAT-likely patterns or add conflicting bounds
                unsat_patterns = [p for p in self.all_patterns 
                                if p.id in self.UNSAT_LIKELY_PATTERNS and p.id not in self.EXCLUDE_LIST]
                
                if unsat_patterns:
                    unsat_pattern = random.choice(unsat_patterns)
                    unsat_params = self._gen_params(unsat_pattern, context)
                    unsat_constraint = self.generator.generate(unsat_pattern.id, context, unsat_params)
                    
                    pairs.append((sat_constraint, unsat_constraint))
            
            except Exception:
                continue
        
        return pairs
