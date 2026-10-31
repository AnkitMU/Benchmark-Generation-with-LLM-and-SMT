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


@dataclass
class SlotRetryState:
    """
    Per-slot retry bookkeeping: which pattern ids are ruled out for the
    remaining retries of ONE slot (``_refine_single``).

    ``GeneratorState`` is suite-wide and only down-weights a pattern after a
    tautology, which the box-preserving steered regeneration ignores, so the
    engine could re-propose the same tautological template for the same slot
    until the retry budget was gone (e.g. ``association_exists`` on a
    mandatory end, 33 proposals for one slot).  This state makes the switch
    explicit: a pattern is excluded for the slot

      * as soon as it fails q1 (tautology) -- the template, not the binding,
        is the problem, so the replacement must come from a DIFFERENT pattern
        of the same family and complexity box; and
      * once the same (pattern, context) pair has failed ``REPEAT_LIMIT``
        times for the slot for any reason (duplicate, contradiction, ...).

    The excluded ids are handed to ``engine.generate_refined(exclude_patterns=)``
    on every subsequent retry of the slot.  No solver involved.
    """

    REPEAT_LIMIT = 2

    excluded_patterns: Set[str] = field(default_factory=set)
    fail_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def record_failure(
        self,
        pattern_id: str,
        context: str,
        failure_type: Optional[FailureType] = None,
    ) -> bool:
        """Record one rejected candidate for the slot; return True if its
        pattern is (now) excluded for the slot's remaining retries."""
        key = (pattern_id, context)
        self.fail_counts[key] = self.fail_counts.get(key, 0) + 1
        if failure_type == FailureType.TAUTOLOGY \
                or self.fail_counts[key] >= self.REPEAT_LIMIT:
            self.excluded_patterns.add(pattern_id)
        return pattern_id in self.excluded_patterns

    def is_excluded(self, pattern_id: str) -> bool:
        return pattern_id in self.excluded_patterns


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
        is_duplicate=None,
    ):
        """
        Args:
            engine: The BenchmarkEngineV2 instance (for generate_refined).
            checker: Initialised VGCRPropertyChecker.
            metamodel: Current metamodel.
            max_retries: Maximum refinement attempts per candidate slot.
            is_duplicate: Optional ``(candidate, suite) -> bool`` near-duplicate
                test. A candidate it flags is never accepted: the slot is
                regenerated like any failed slot, so deduplication happens
                INSIDE the verified fill loop rather than deleting constraints
                after verification.
        """
        self.engine = engine
        self.checker = checker
        self.metamodel = metamodel
        self.max_retries = max_retries
        self.is_duplicate = is_duplicate
        self.state = GeneratorState()
        self.stats = VGCRStats()
        self.duplicates_rejected = 0
        # Slots that ended unfilled: the last failed candidate stands for the
        # slot (its family and box are what a later refill must honour).
        self.unfilled_slots: List[OCLConstraint] = []
        self.unfilled_log: List[Dict[str, Any]] = []
        self.coverage_log: List[Dict[str, Any]] = []

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
        stats_dict["duplicates_rejected"] = self.duplicates_rejected
        stats_dict["unfilled_slots"] = list(self.unfilled_log)
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
        # Slot-level pattern exclusions (tautology => switch pattern; repeated
        # failure of the same (pattern, context) pair => switch pattern).
        slot_state = SlotRetryState()

        for attempt in range(self.max_retries + 1):  # attempt 0 = original
            # Skip if this (pattern, context) is already blacklisted
            if self.state.is_blacklisted(
                c.pattern_id, c.context, c.parameters
            ):
                logger.debug(
                    f"  Slot {slot_idx}: attempt {attempt} skipped — "
                    f"'{c.pattern_id}@{c.context}' is blacklisted"
                )
                slot_state.record_failure(c.pattern_id, c.context)
                # Try to get a refined candidate for the remaining attempts
                if attempt < self.max_retries:
                    nxt = self._get_refined_candidate(c, current_suite, slot_state)
                    if nxt is None:
                        break
                    c = nxt
                    self.stats.total_retries += 1
                    continue
                else:
                    break

            # Near-duplicate gate (no SMT): a candidate too similar to an
            # already-verified constraint is sent back through regeneration.
            if self.is_duplicate is not None and self.is_duplicate(c, current_suite):
                self.duplicates_rejected += 1
                logger.debug(
                    f"  Slot {slot_idx}: attempt {attempt} near-duplicate — "
                    f"'{c.pattern_id}@{c.context}'"
                )
                slot_state.record_failure(c.pattern_id, c.context)
                if attempt < self.max_retries:
                    nxt = self._get_refined_candidate(c, current_suite, slot_state)
                    if nxt is None:
                        break
                    c = nxt
                    self.stats.total_retries += 1
                    continue
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
            if slot_state.record_failure(c.pattern_id, c.context, result.failure_type):
                logger.debug(
                    f"  Slot {slot_idx}: pattern '{c.pattern_id}' excluded for "
                    f"the remaining retries ({result.failure_type.value})"
                )

            # Try to get a refined candidate
            if attempt < self.max_retries:
                nxt = self._get_refined_candidate(c, current_suite, slot_state)
                if nxt is None:
                    logger.debug(
                        f"  Slot {slot_idx}: engine returned no refined "
                        f"candidate — giving up"
                    )
                    break
                c = nxt
                self.stats.total_retries += 1
            # else: final attempt exhausted

        # All retries failed: the slot stays unfilled and is REPORTED, exactly
        # as an unreached slot is during generation. It is never padded with a
        # replacement outside the slot's complexity box.
        self.stats.failed_all_retries += 1
        self.unfilled_slots.append(c)
        meta = getattr(c, "metadata", None) or {}
        self.unfilled_log.append({
            "slot": slot_idx,
            "last_pattern": c.pattern_id,
            "last_context": c.context,
            "profile": meta.get("profile"),
            "excluded_patterns": sorted(slot_state.excluded_patterns),
        })
        logger.debug(f"  Slot {slot_idx}: UNFILLED after {self.max_retries} retries")
        return None

    def _get_refined_candidate(
        self,
        failed: OCLConstraint,
        current_suite: List[OCLConstraint],
        slot_state: Optional[SlotRetryState] = None,
    ) -> Optional[OCLConstraint]:
        """
        Ask the engine for a refined candidate using the current
        generator state (and the slot's pattern exclusions, if any).

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
            excluded = slot_state.excluded_patterns if slot_state is not None else None
            return self._call_engine(failed, current_suite, exclude_patterns=excluded)
        except Exception as e:
            logger.debug(f"  generate_refined() raised: {e}")
            return None

    def _call_engine(
        self,
        failed: OCLConstraint,
        current_suite: List[OCLConstraint],
        only_classes: Optional[List[str]] = None,
        exclude_patterns: Optional[Set[str]] = None,
    ) -> Optional[OCLConstraint]:
        """Ask the engine for a replacement, passing the near-duplicate veto,
        an optional class restriction and the slot's pattern exclusions when
        the engine supports them."""
        kwargs: Dict[str, Any] = {}
        if self.is_duplicate is not None:
            dup = self.is_duplicate
            kwargs["accept"] = lambda nc: not dup(nc, current_suite)
        if only_classes is not None:
            kwargs["only_classes"] = list(only_classes)
        if exclude_patterns:
            kwargs["exclude_patterns"] = set(exclude_patterns)
        # Pass only the keyword arguments this engine's generate_refined()
        # accepts (checked once per call via its signature) so a TypeError
        # raised INSIDE the engine is never mistaken for a signature mismatch
        # and silently retried without the slot's exclusions.
        import inspect
        try:
            params = inspect.signature(self.engine.generate_refined).parameters
            accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            accepted = {k: (accepts_any or k in params) for k in kwargs}
        except (TypeError, ValueError):
            accepted = {k: True for k in kwargs}
        if only_classes is not None and not accepted.get("only_classes", True):
            return None  # engine cannot honour the class restriction
        for key, ok in accepted.items():
            if not ok:
                logger.debug(f"engine.generate_refined() does not accept '{key}'; dropping it")
                kwargs.pop(key)
        return self.engine.generate_refined(
            failed_constraint=failed, generator_state=self.state, **kwargs
        )

    # ------------------------------------------------------------------
    # Coverage repair (post-refinement)
    # ------------------------------------------------------------------

    def repair_coverage(
        self,
        verified: List[OCLConstraint],
        tc_range: Tuple[float, float],
        classes: List[str],
        min_per_class: int = 1,
        max_checks_per_class: int = 6,
    ) -> Tuple[List[OCLConstraint], List[Dict[str, Any]]]:
        """Refill classes that refinement left below ``min_per_class``.

        Mirrors the generation-time minima repair, but every replacement is
        property-checked before it is committed:

          1. an UNFILLED slot is filled on the deficit class (same family and
             box as the slot) -- this adds coverage and restores a lost slot;
          2. otherwise a DONOR from a class above the minimum is swapped for a
             same-family, same-box constraint on the deficit class, so suite
             size and tier/family quotas are unchanged.

        A class that cannot be repaired is reported, never padded.
        """
        counts: Dict[str, int] = {}
        for c in verified:
            counts[c.context] = counts.get(c.context, 0) + 1

        for X in classes:
            while counts.get(X, 0) < min_per_class:
                checks = 0
                done = False

                # 1) fill an unfilled slot on X
                for slot_c in list(self.unfilled_slots):
                    if checks >= max_checks_per_class:
                        break
                    cand = self._call_engine(slot_c, verified, only_classes=[X])
                    if cand is None or cand.context != X:
                        continue
                    checks += 1
                    if self.checker.check_all(cand, verified, tc_range).passed:
                        verified.append(cand)
                        # unfilled_slots and unfilled_log are index-aligned
                        k = next(j for j, u in enumerate(self.unfilled_slots) if u is slot_c)
                        self.unfilled_slots.pop(k)
                        self.unfilled_log.pop(k)
                        counts[X] = counts.get(X, 0) + 1
                        self.coverage_log.append({
                            "class": X, "action": "filled_unfilled_slot",
                            "pattern": cand.pattern_id,
                            "profile": (cand.metadata or {}).get("profile"),
                        })
                        done = True
                        break

                # 2) donor swap preserving the donor's family and box
                if not done:
                    for i, donor in enumerate(verified):
                        if checks >= max_checks_per_class:
                            break
                        Y = donor.context
                        if Y == X or counts.get(Y, 0) <= min_per_class:
                            continue
                        cand = self._call_engine(
                            donor, verified[:i] + verified[i + 1:], only_classes=[X]
                        )
                        if cand is None or cand.context != X:
                            continue
                        checks += 1
                        rest = verified[:i] + verified[i + 1:]
                        if self.checker.check_all(cand, rest, tc_range).passed:
                            verified[i] = cand
                            counts[Y] -= 1
                            counts[X] = counts.get(X, 0) + 1
                            self.coverage_log.append({
                                "class": X, "action": "donor_swap", "donor_class": Y,
                                "pattern": cand.pattern_id,
                                "profile": (cand.metadata or {}).get("profile"),
                            })
                            done = True
                            break

                if not done:
                    self.coverage_log.append({
                        "class": X, "action": "unrepaired",
                        "have": counts.get(X, 0), "min": min_per_class,
                    })
                    break

        return verified, list(self.coverage_log)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return full statistics dictionary."""
        return self.stats.to_dict()

    def get_generator_state(self) -> GeneratorState:
        """Return the current generator state (for inspection / testing)."""
        return self.state
