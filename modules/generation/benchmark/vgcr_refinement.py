"""
VGCR Refinement Loop — Generator-driven refinement for SAT-core construction.

When a candidate constraint fails a property check, the loop does NOT ask the
LLM to synthesise a replacement.  Instead, it updates a deterministic
*generator state* (blacklists, weight adjustments) and derives a new candidate
from the existing pattern library.

This keeps the refinement formally grounded and avoids introducing
nondeterministic LLM generation into the verified SAT-core pipeline.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Any

from modules.core.models import OCLConstraint, Metamodel
from modules.verification.vgcr_verifier import (
    VGCRPropertyChecker,
    PropertyCheckResult,
    FailureType,
    VGCRStats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator state — deterministic refinement rules
# ---------------------------------------------------------------------------

@dataclass
class GeneratorState:
    """
    Tracks blacklists and weight adjustments that guide generator refinement.

    When a candidate fails a VGCR property check, the failure type determines
    which entries are added here.  The engine's ``generate_refined()`` method
    consults this state to produce a different candidate without free-form
    LLM involvement.
    """

    # (pattern_id, context) combinations to avoid
    blacklisted_pattern_context: Set[Tuple[str, str]] = field(
        default_factory=set
    )

    # Specific (pattern_id, context, param_key, param_value) bindings to skip
    blacklisted_bindings: Set[Tuple[str, str, str, str]] = field(
        default_factory=set
    )

    # Pattern-level weight multipliers (pattern_id → float).
    # Values < 1.0 down-weight; 0.0 effectively disables.
    pattern_weight_mods: Dict[str, float] = field(default_factory=dict)

    # Feature families to prioritise (family → priority boost multiplier)
    family_priority_boosts: Dict[str, float] = field(default_factory=dict)

    # Record of all applied refinement actions (for logging / paper stats)
    actions_log: List[Dict[str, Any]] = field(default_factory=list)

    def apply_failure(
        self,
        failure_type: FailureType,
        constraint: OCLConstraint,
        conflicting_constraints: Optional[List[str]] = None,
        actual_tc: Optional[float] = None,
        target_tc_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Update generator state based on a structured failure diagnosis.

        Each failure type maps to a specific, deterministic refinement action.
        No LLM is involved.

        Args:
            failure_type: Which property check failed.
            constraint: The failed candidate.
            conflicting_constraints: For INCONSISTENT — IDs of conflicts.
            actual_tc: For TC_MISMATCH — the computed TC value.
            target_tc_range: For TC_MISMATCH — the desired (min, max).
        """
        pid = constraint.pattern_id
        ctx = constraint.context
        action = {"failure": failure_type.value, "pattern": pid, "context": ctx}

        if failure_type == FailureType.TAUTOLOGY:
            # The pattern produced a trivially true constraint.
            # Down-weight the pattern so the engine tries others first.
            current = self.pattern_weight_mods.get(pid, 1.0)
            self.pattern_weight_mods[pid] = current * 0.3
            action["action"] = "down_weight_pattern"
            action["new_weight"] = self.pattern_weight_mods[pid]

        elif failure_type == FailureType.CONTRADICTION:
            # The pattern produced an always-false constraint.
            # Blacklist this specific (pattern, context) combination —
            # the pattern may work on other classes.
            self.blacklisted_pattern_context.add((pid, ctx))
            action["action"] = "blacklist_pattern_context"

        elif failure_type == FailureType.REDUNDANT:
            # The constraint is implied by the existing suite.
            # Blacklist the exact pattern+context combo and boost
            # under-represented feature families.
            self.blacklisted_pattern_context.add((pid, ctx))
            action["action"] = "blacklist_and_diversify"

        elif failure_type == FailureType.INCONSISTENT:
            # Adding the constraint breaks joint satisfiability.
            # Blacklist the specific binding that caused the conflict.
            for key, val in constraint.parameters.items():
                val_str = str(val) if val is not None else ""
                self.blacklisted_bindings.add((pid, ctx, key, val_str))
            # Also mildly down-weight the pattern
            current = self.pattern_weight_mods.get(pid, 1.0)
            self.pattern_weight_mods[pid] = current * 0.5
            action["action"] = "blacklist_bindings_and_downweight"
            if conflicting_constraints:
                action["conflicts"] = conflicting_constraints[:3]

        elif failure_type == FailureType.TC_MISMATCH:
            # TC is outside the target range.
            # This doesn't blacklist anything — instead, the engine should
            # pick a pattern of different inherent complexity.
            if actual_tc is not None and target_tc_range is not None:
                min_tc, max_tc = target_tc_range
                if actual_tc < min_tc:
                    action["action"] = "need_higher_complexity"
                    action["direction"] = "up"
                else:
                    action["action"] = "need_lower_complexity"
                    action["direction"] = "down"
            else:
                action["action"] = "tc_adjust"

        elif failure_type == FailureType.NEGATION_UNSUPPORTED:
            # q1 was skipped — no state change needed
            action["action"] = "no_change"

        elif failure_type == FailureType.ENCODING_ERROR:
            # Constraint couldn't be encoded for Z3 at all
            self.blacklisted_pattern_context.add((pid, ctx))
            action["action"] = "blacklist_encoding_error"

        self.actions_log.append(action)

    def is_blacklisted(
        self, pattern_id: str, context: str, params: Dict[str, Any] = None
    ) -> bool:
        """Check if a (pattern, context, params) combo is blacklisted."""
        if (pattern_id, context) in self.blacklisted_pattern_context:
            return True

        if params and self.blacklisted_bindings:
            for key, val in params.items():
                val_str = str(val) if val is not None else ""
                if (pattern_id, context, key, val_str) in self.blacklisted_bindings:
                    return True

        return False

    def get_weight_modifier(self, pattern_id: str) -> float:
        """Return the weight multiplier for a pattern (default 1.0)."""
        return self.pattern_weight_mods.get(pattern_id, 1.0)

    def get_refinement_summary(self) -> Dict[str, Any]:
        """Summarise all refinement actions for logging / paper output."""
        action_counts: Dict[str, int] = {}
        for entry in self.actions_log:
            a = entry.get("action", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1
        return {
            "total_refinements": len(self.actions_log),
            "blacklisted_pattern_contexts": len(self.blacklisted_pattern_context),
            "blacklisted_bindings": len(self.blacklisted_bindings),
            "weight_modifications": len(self.pattern_weight_mods),
            "action_breakdown": action_counts,
        }


# ---------------------------------------------------------------------------
# VGCR Loop
# ---------------------------------------------------------------------------

class VGCRLoop:
    """
    Generator-driven refinement loop for SAT-core construction.

    For each candidate constraint from the template-based generator:
      1. Run the full property-check pipeline (q1–q5).
      2. If it passes → add to the verified SAT core.
      3. If it fails → update the generator state and ask the engine
         for a refined candidate (up to ``max_retries`` times).
      4. If all retries exhausted → leave the slot unfilled.

    The engine's ``generate_refined()`` method is responsible for
    producing an alternative candidate that avoids the failure.  It
    consults the ``GeneratorState`` blacklists and weight adjustments
    to make a different pattern/binding/context choice.
    """

    def __init__(
        self,
        engine,                          # BenchmarkEngineV2
        checker: VGCRPropertyChecker,
        metamodel: Metamodel,
        max_retries: int = 3,
    ):
        """
        Args:
            engine: The BenchmarkEngineV2 instance (for generate_refined).
            checker: Initialised VGCRPropertyChecker.
            metamodel: Current metamodel.
            max_retries: Maximum refinement attempts per candidate slot.
        """
        self.engine = engine
        self.checker = checker
        self.metamodel = metamodel
        self.max_retries = max_retries
        self.state = GeneratorState()
        self.stats = VGCRStats()

    def refine_suite(
        self,
        candidates: List[OCLConstraint],
        target_tc_range: Tuple[float, float],
        progress_callback=None,
    ) -> Tuple[List[OCLConstraint], Dict[str, Any]]:
        """
        Run the VGCR refinement loop over a list of candidates.

        Args:
            candidates: Raw SAT candidates from template-based generation.
            target_tc_range: (min_tc, max_tc) for TC conformance checks.
            progress_callback: Optional (accepted, total, score) callback.

        Returns:
            (verified_suite, stats_dict) where verified_suite is the
            jointly satisfiable, property-checked SAT core and stats_dict
            contains all refinement statistics for the paper.
        """
        t0 = time.time()
        verified: List[OCLConstraint] = []

        self.stats.total_candidates = len(candidates)
        logger.info(
            f"VGCR: starting refinement loop for {len(candidates)} "
            f"candidates (max_retries={self.max_retries})"
        )

        for idx, candidate in enumerate(candidates):
            result = self._refine_single(
                candidate, verified, target_tc_range, slot_idx=idx
            )

            if result is not None:
                verified.append(result)
                if progress_callback:
                    progress_callback(
                        len(verified), len(candidates),
                        len(verified) / max(1, idx + 1)
                    )

        total_time = time.time() - t0

        # Merge checker stats into our stats
        checker_stats = self.checker.get_stats()
        self.stats.total_smt_queries = checker_stats.total_smt_queries
        self.stats.total_smt_time = checker_stats.total_smt_time
        self.stats.independence_checked = checker_stats.independence_checked
        self.stats.independence_skipped = checker_stats.independence_skipped
        self.stats.failure_counts = dict(checker_stats.failure_counts)

        stats_dict = self.stats.to_dict()
        stats_dict["total_time_s"] = round(total_time, 3)
        stats_dict["generator_refinement"] = self.state.get_refinement_summary()

        logger.info(
            f"VGCR: complete — "
            f"{self.stats.passed_first_try + self.stats.passed_after_retry}/"
            f"{self.stats.total_candidates} passed "
            f"({self.stats.failed_all_retries} unfilled) "
            f"in {total_time:.1f}s"
        )

        return verified, stats_dict

    def _refine_single(
        self,
        candidate: OCLConstraint,
        current_suite: List[OCLConstraint],
        tc_range: Tuple[float, float],
        slot_idx: int = 0,
    ) -> Optional[OCLConstraint]:
        """
        Try to get one constraint through all property checks,
        with up to ``max_retries`` generator-driven refinement attempts.

        Returns the accepted constraint, or None if all retries exhausted.
        """
        c = candidate

        for attempt in range(self.max_retries + 1):  # attempt 0 = original
            # Skip if this (pattern, context) is already blacklisted
            if self.state.is_blacklisted(
                c.pattern_id, c.context, c.parameters
            ):
                logger.debug(
                    f"  Slot {slot_idx}: attempt {attempt} skipped — "
                    f"'{c.pattern_id}@{c.context}' is blacklisted"
                )
                # Try to get a refined candidate for the remaining attempts
                if attempt < self.max_retries:
                    c = self._get_refined_candidate(c, current_suite)
                    if c is None:
                        break
                    self.stats.total_retries += 1
                    continue
                else:
                    break

            # Run property checks
            result = self.checker.check_all(c, current_suite, tc_range)

            if result.passed:
                # ✅ Accepted
                if attempt == 0:
                    self.stats.passed_first_try += 1
                else:
                    self.stats.passed_after_retry += 1

                logger.debug(
                    f"  Slot {slot_idx}: ACCEPTED on attempt {attempt} — "
                    f"'{c.pattern_id}@{c.context}' ({result.summary})"
                )
                return c

            # ❌ Failed — apply refinement
            logger.debug(
                f"  Slot {slot_idx}: attempt {attempt} FAILED — "
                f"{result.summary}"
            )

            self.state.apply_failure(
                result.failure_type,
                c,
                conflicting_constraints=result.conflicting_constraints,
                actual_tc=result.actual_tc,
                target_tc_range=result.target_tc_range,
            )

            # Try to get a refined candidate
            if attempt < self.max_retries:
                c = self._get_refined_candidate(c, current_suite)
                if c is None:
                    logger.debug(
                        f"  Slot {slot_idx}: engine returned no refined "
                        f"candidate — giving up"
                    )
                    break
                self.stats.total_retries += 1
            # else: final attempt exhausted

        # All retries failed
        self.stats.failed_all_retries += 1
        logger.debug(f"  Slot {slot_idx}: UNFILLED after {self.max_retries} retries")
        return None

    def _get_refined_candidate(
        self,
        failed: OCLConstraint,
        current_suite: List[OCLConstraint],
    ) -> Optional[OCLConstraint]:
        """
        Ask the engine for a refined candidate using the current
        generator state.

        The engine's ``generate_refined()`` method is responsible for
        consulting ``self.state`` to avoid blacklisted patterns/bindings
        and adjusting weights.

        If the engine doesn't have ``generate_refined``, fall back to
        returning None (slot unfilled).
        """
        if not hasattr(self.engine, 'generate_refined'):
            logger.debug(
                "  Engine does not support generate_refined() — "
                "refinement unavailable"
            )
            return None

        try:
            refined = self.engine.generate_refined(
                failed_constraint=failed,
                generator_state=self.state,
            )
            return refined
        except Exception as e:
            logger.debug(f"  generate_refined() raised: {e}")
            return None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return full statistics dictionary."""
        return self.stats.to_dict()

    def get_generator_state(self) -> GeneratorState:
        """Return the current generator state (for inspection / testing)."""
        return self.state
