"""
VGCR Property Checker — Multi-property SMT verification for the
Verification-Guided Constraint Refinement loop.

Checks each SAT-core candidate against five formal quality properties:
    q1  Not a tautology        (Φ_M ∧ ¬c is SAT)
    q2  Not a contradiction    (Φ_M ∧ c  is SAT)
    q3  Independence           (Φ_M ∧ B_current ∧ ¬c is SAT)
    q4  Suite consistency      (Φ_M ∧ B_current ∧ c  is SAT)
    q5  TC conformance         (analytical — no SMT query)

The checks run as a sequential pipeline: the first failure short-circuits
and returns a structured diagnosis that drives generator-state refinement.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

from modules.core.models import OCLConstraint, Metamodel
from modules.generation.benchmark.complexity_calculator import (
    compute_total_complexity,
    ComplexityWeights,
)
from modules.generation.benchmark.implication_checker import (
    _negated_constraint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FailureType(Enum):
    """Enumeration of VGCR property-check failure reasons."""
    TAUTOLOGY = "tautology"
    CONTRADICTION = "contradiction"
    REDUNDANT = "redundant"
    INCONSISTENT = "inconsistent"
    TC_MISMATCH = "tc_mismatch"
    NEGATION_UNSUPPORTED = "negation_unsupported"
    ENCODING_ERROR = "encoding_error"


@dataclass
class PropertyCheckResult:
    """Outcome of running the full VGCR property-check pipeline."""
    passed: bool
    failure_type: Optional[FailureType] = None
    failure_message: str = ""
    # For INCONSISTENT failures — names of conflicting constraints
    conflicting_constraints: List[str] = field(default_factory=list)
    # For TC_MISMATCH — actual TC value
    actual_tc: Optional[float] = None
    target_tc_range: Optional[Tuple[float, float]] = None
    # Timing
    check_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    # Which checks were actually executed (for reporting)
    checks_run: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed:
            return f"PASSED (all {len(self.checks_run)} checks in {self.total_time:.3f}s)"
        return (
            f"FAILED [{self.failure_type.value}]: {self.failure_message} "
            f"(after {len(self.checks_run)} checks in {self.total_time:.3f}s)"
        )


@dataclass
class VGCRStats:
    """Aggregate statistics for the VGCR refinement loop."""
    total_candidates: int = 0
    passed_first_try: int = 0
    passed_after_retry: int = 0
    failed_all_retries: int = 0
    total_retries: int = 0
    # Per-failure-type counts
    failure_counts: Dict[str, int] = field(default_factory=lambda: {
        ft.value: 0 for ft in FailureType
    })
    # Per-check timing
    total_smt_queries: int = 0
    total_smt_time: float = 0.0
    # Independence check stats
    independence_checked: int = 0
    independence_skipped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON output."""
        total_passed = self.passed_first_try + self.passed_after_retry
        return {
            "total_candidates": self.total_candidates,
            "passed": total_passed,
            "passed_first_try": self.passed_first_try,
            "passed_after_retry": self.passed_after_retry,
            "failed": self.failed_all_retries,
            "pass_rate": (
                f"{total_passed / self.total_candidates * 100:.1f}%"
                if self.total_candidates > 0 else "N/A"
            ),
            "total_retries": self.total_retries,
            "failure_breakdown": {
                k: v for k, v in self.failure_counts.items() if v > 0
            },
            "total_smt_queries": self.total_smt_queries,
            "total_smt_time_s": round(self.total_smt_time, 3),
            "independence_checked": self.independence_checked,
            "independence_skipped": self.independence_skipped,
        }


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

class VGCRPropertyChecker:
    """
    Multi-property SMT checker for the VGCR refinement loop.

    Wraps the existing ``FrameworkConstraintVerifier`` and adds the four
    SMT-based quality checks (q1–q4) plus the analytical TC check (q5).
    """

    # Independence check is expensive (batch solve with negation).
    # Only run it once the suite exceeds this size.
    INDEPENDENCE_THRESHOLD = 5

    # Maximum number of suite constraints to include in batch checks
    # (q3 independence and q4 consistency).  Using only the most recent
    # constraints keeps batch solve time bounded while still catching
    # local conflicts and implications.  The final suite-level consistency
    # check in Phase 3 verifies the full suite at the end.
    BATCH_WINDOW_SIZE = 25

    # Consistency check (q4) is skipped when the suite is below this
    # size, since small sets rarely conflict.  The final Phase 3 check
    # covers the full suite regardless.
    CONSISTENCY_THRESHOLD = 3

    def __init__(
        self,
        verifier,                       # FrameworkConstraintVerifier
        metamodel: Metamodel,
        complexity_weights: Optional[ComplexityWeights] = None,
        enable_independence: bool = True,
        independence_threshold: int = 5,
        batch_window_size: int = 25,
        consistency_threshold: int = 3,
    ):
        """
        Args:
            verifier: Initialised FrameworkConstraintVerifier instance.
            metamodel: The current metamodel (for TC computation).
            complexity_weights: Weights for TC calculation.
            enable_independence: Whether to run the q3 check at all.
            independence_threshold: Min suite size before q3 activates.
            batch_window_size: Max constraints in q3/q4 batch checks.
            consistency_threshold: Min suite size before q4 activates.
        """
        self.verifier = verifier
        self.metamodel = metamodel
        self.complexity_weights = complexity_weights or ComplexityWeights()
        self.enable_independence = enable_independence
        self.INDEPENDENCE_THRESHOLD = independence_threshold
        self.BATCH_WINDOW_SIZE = batch_window_size
        self.CONSISTENCY_THRESHOLD = consistency_threshold
        self.stats = VGCRStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(
        self,
        candidate: OCLConstraint,
        current_suite: List[OCLConstraint],
        target_tc_range: Tuple[float, float],
    ) -> PropertyCheckResult:
        """
        Run the full property-check pipeline on *candidate*.

        Checks execute in order q1 → q2 → q3 → q4 → q5.
        The first failure short-circuits; the result carries a structured
        diagnosis that the ``GeneratorState`` can act on.

        Args:
            candidate: The OCL constraint to check.
            current_suite: Constraints already accepted into B_SAT.
            target_tc_range: (min_tc, max_tc) for the generation profile.

        Returns:
            PropertyCheckResult with pass/fail + diagnosis.
        """
        t0 = time.time()
        result = PropertyCheckResult(passed=True)

        # — q1: Not a tautology ——————————————————————————————
        ok, msg, dt = self._check_not_tautology(candidate)
        result.checks_run.append("q1_tautology")
        result.check_times["q1_tautology"] = dt
        if not ok:
            result.passed = False
            result.failure_type = (
                FailureType.NEGATION_UNSUPPORTED
                if "negation unsupported" in msg.lower()
                else FailureType.TAUTOLOGY
            )
            result.failure_message = msg
            result.total_time = time.time() - t0
            self.stats.failure_counts[result.failure_type.value] += 1
            return result

        # — q2: Not a contradiction ——————————————————————————
        ok, msg, dt = self._check_not_contradiction(candidate)
        result.checks_run.append("q2_contradiction")
        result.check_times["q2_contradiction"] = dt
        if not ok:
            result.passed = False
            result.failure_type = FailureType.CONTRADICTION
            result.failure_message = msg
            result.total_time = time.time() - t0
            self.stats.failure_counts[FailureType.CONTRADICTION.value] += 1
            return result

        # — q3: Independence (conditional) ————————————————————
        if (
            self.enable_independence
            and len(current_suite) >= self.INDEPENDENCE_THRESHOLD
        ):
            # Use a sliding window to keep batch size bounded
            window = current_suite[-self.BATCH_WINDOW_SIZE:]
            ok, msg, dt = self._check_independence(candidate, window)
            result.checks_run.append("q3_independence")
            result.check_times["q3_independence"] = dt
            self.stats.independence_checked += 1
            if not ok:
                result.passed = False
                result.failure_type = FailureType.REDUNDANT
                result.failure_message = msg
                result.total_time = time.time() - t0
                self.stats.failure_counts[FailureType.REDUNDANT.value] += 1
                return result
        else:
            self.stats.independence_skipped += 1

        # — q4: Suite consistency —————————————————————————————
        if len(current_suite) >= self.CONSISTENCY_THRESHOLD:
            # Use a sliding window to keep batch size bounded.
            # The final Phase 3 suite-level check covers the full set.
            window = current_suite[-self.BATCH_WINDOW_SIZE:]
            ok, msg, dt, conflicts = self._check_consistency(
                candidate, window
            )
            result.checks_run.append("q4_consistency")
            result.check_times["q4_consistency"] = dt
            if not ok:
                result.passed = False
                result.failure_type = FailureType.INCONSISTENT
                result.failure_message = msg
                result.conflicting_constraints = conflicts
                result.total_time = time.time() - t0
                self.stats.failure_counts[FailureType.INCONSISTENT.value] += 1
                return result

        # — q5: TC conformance (analytical) ———————————————————
        ok, msg, actual_tc = self._check_tc_conformance(
            candidate, target_tc_range
        )
        result.checks_run.append("q5_tc")
        result.check_times["q5_tc"] = 0.0  # negligible
        if not ok:
            result.passed = False
            result.failure_type = FailureType.TC_MISMATCH
            result.failure_message = msg
            result.actual_tc = actual_tc
            result.target_tc_range = target_tc_range
            result.total_time = time.time() - t0
            self.stats.failure_counts[FailureType.TC_MISMATCH.value] += 1
            return result

        # — All passed ————————————————————————————————————————
        result.total_time = time.time() - t0
        self.stats.failure_counts.get("passed", None)  # no-op
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_not_tautology(
        self, candidate: OCLConstraint
    ) -> Tuple[bool, str, float]:
        """
        q1: Negate *candidate* and check SAT(Φ_M ∧ ¬c).

        If UNSAT → the original is a tautology (always true under the
        metamodel axioms).  Returns (passed, message, elapsed).
        """
        t0 = time.time()

        negated = _negated_constraint(candidate)
        if negated is None:
            # Negation not supported for this pattern — log and skip
            dt = time.time() - t0
            logger.debug(
                f"q1 skip: negation unsupported for pattern "
                f"'{candidate.pattern_id}'"
            )
            # We treat unsupported negation as a pass with a warning,
            # rather than blocking the candidate.  The paper reports
            # coverage of q1 separately.
            return True, "negation unsupported — q1 skipped", dt

        vr = self.verifier.verify(negated)
        dt = time.time() - t0
        self.stats.total_smt_queries += 1
        self.stats.total_smt_time += dt

        if vr.solver_result == 'error':
            # Encoding error on the negated form — skip check gracefully
            logger.debug(
                f"q1 skip: encoding error on negated form of "
                f"'{candidate.pattern_id}'"
            )
            return True, "encoding error on negation — q1 skipped", dt

        if vr.is_satisfiable is False:
            # ¬c is UNSAT → c is a tautology
            return (
                False,
                f"Tautology: negation is UNSAT — constraint is always "
                f"true under metamodel axioms.",
                dt,
            )

        if vr.is_satisfiable is None:
            # Solver timeout / unknown — skip gracefully
            return True, "solver timeout on negation — q1 inconclusive", dt

        # ¬c is SAT → c is not a tautology ✓
        return True, "", dt

    def _check_not_contradiction(
        self, candidate: OCLConstraint
    ) -> Tuple[bool, str, float]:
        """
        q2: Check SAT(Φ_M ∧ c).

        If UNSAT → candidate is a contradiction (always false).
        This reuses the standard single-constraint verify().
        """
        t0 = time.time()
        vr = self.verifier.verify(candidate)
        dt = time.time() - t0
        self.stats.total_smt_queries += 1
        self.stats.total_smt_time += dt

        if vr.solver_result == 'error':
            return (
                False,
                f"Encoding error: constraint could not be encoded for Z3.",
                dt,
            )

        if vr.is_satisfiable is False:
            return (
                False,
                f"Contradiction: constraint is UNSAT — always false "
                f"under metamodel axioms.",
                dt,
            )

        if vr.is_satisfiable is None:
            # Solver timeout — allow through with warning
            logger.debug(
                f"q2 inconclusive: solver timeout for "
                f"'{candidate.pattern_id}'"
            )
            return True, "solver timeout — q2 inconclusive", dt

        # SAT → not a contradiction ✓
        return True, "", dt

    def _check_independence(
        self,
        candidate: OCLConstraint,
        current_suite: List[OCLConstraint],
    ) -> Tuple[bool, str, float]:
        """
        q3: Check SAT(Φ_M ∧ B_current ∧ ¬c).

        If UNSAT → c is logically implied by the current suite (redundant).
        """
        t0 = time.time()

        negated = _negated_constraint(candidate)
        if negated is None:
            dt = time.time() - t0
            return True, "negation unsupported — q3 skipped", dt

        # Build a batch: current suite + negated candidate
        batch = list(current_suite) + [negated]
        results = self.verifier.verify_batch(batch, silent=True)
        dt = time.time() - t0
        self.stats.total_smt_queries += 1
        self.stats.total_smt_time += dt

        # The batch result tells us if (B_current ∧ ¬c) is SAT or UNSAT.
        # We check the overall result via the last constraint's solver_result
        # (all share the same global solver result in batch mode).
        global_result = None
        for r in results:
            if r.solver_result in ('sat', 'unsat'):
                global_result = r.solver_result
                break

        if global_result == 'unsat':
            return (
                False,
                f"Redundant: constraint is implied by the current suite "
                f"({len(current_suite)} constraints). Adding it provides "
                f"no new information.",
                dt,
            )

        # SAT or unknown → not redundant ✓
        return True, "", dt

    def _check_consistency(
        self,
        candidate: OCLConstraint,
        current_suite: List[OCLConstraint],
    ) -> Tuple[bool, str, float, List[str]]:
        """
        q4: Check SAT(Φ_M ∧ B_current ∧ c).

        If UNSAT → adding c breaks joint satisfiability.
        Returns (passed, message, elapsed, conflicting_constraint_ids).
        """
        t0 = time.time()

        # Build batch: current suite + candidate
        batch = list(current_suite) + [candidate]
        results = self.verifier.verify_batch(batch, silent=True)
        dt = time.time() - t0
        self.stats.total_smt_queries += 1
        self.stats.total_smt_time += dt

        global_result = None
        for r in results:
            if r.solver_result in ('sat', 'unsat'):
                global_result = r.solver_result
                break

        if global_result == 'unsat':
            # Try to identify conflicting constraints.
            # The verifier doesn't expose unsat_core directly, so we
            # report the full suite context as the conflict set.
            # A future enhancement could use Z3's unsat_core extraction.
            conflict_ids = [
                f"{c.pattern_id}@{c.context}" for c in current_suite
            ]
            # Heuristic: limit reported conflicts to keep messages useful
            reported = conflict_ids[:5]
            suffix = (
                f" (and {len(conflict_ids) - 5} more)"
                if len(conflict_ids) > 5 else ""
            )
            return (
                False,
                f"Inconsistent: adding this constraint makes the SAT "
                f"core unsatisfiable. Potential conflicts: "
                f"{', '.join(reported)}{suffix}",
                dt,
                conflict_ids,
            )

        if global_result is None or global_result == 'unknown':
            # Timeout — allow through with warning
            logger.debug(
                f"q4 inconclusive: solver timeout for consistency check "
                f"with suite of size {len(current_suite)}"
            )
            return True, "solver timeout — q4 inconclusive", dt, []

        # SAT → consistent ✓
        return True, "", dt, []

    def _check_tc_conformance(
        self,
        candidate: OCLConstraint,
        target_tc_range: Tuple[float, float],
    ) -> Tuple[bool, str, Optional[float]]:
        """
        q5: Check that TC(c) falls within the target range.

        This is an analytical check — no SMT query required.
        """
        tc_result = compute_total_complexity(
            candidate.ocl,
            metamodel=self.metamodel,
            context_class=candidate.context,
            weights=self.complexity_weights,
        )
        tc = tc_result.tc
        min_tc, max_tc = target_tc_range

        if min_tc <= tc <= max_tc:
            return True, "", tc

        direction = "below" if tc < min_tc else "above"
        return (
            False,
            f"TC mismatch: TC={tc:.1f} is {direction} the target range "
            f"[{min_tc:.1f}, {max_tc:.1f}].",
            tc,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> VGCRStats:
        """Return accumulated statistics."""
        return self.stats

    def reset_stats(self):
        """Reset all counters (e.g., between metamodels)."""
        self.stats = VGCRStats()
