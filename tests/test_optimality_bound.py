"""Tests for the outer-assignment optimality bound (max-flow b-matching).

Run directly (``python3 tests/test_optimality_bound.py``) or via pytest.
"""
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.generation.benchmark.optimality_bound import (
    max_fill_assignment, max_fill_upper_bound,
)
from modules.generation.benchmark.steered_generation import (
    Box, Slot, SteeredGenerator, SteeringSpec, COMPONENTS,
)


# ---------------------------------------------------------------------------
# max_fill_upper_bound: hand-computable instances
# ---------------------------------------------------------------------------

def test_all_slots_fillable():
    # 3 slots, each supported by both classes, ample capacity.
    assert max_fill_upper_bound(
        [{"A", "B"}, {"A", "B"}, {"A", "B"}], {"A": 5, "B": 5}) == 3


def test_capacity_binds():
    # 4 slots all supported only by A with ceiling 2 -> optimum 2.
    assert max_fill_upper_bound(
        [{"A"}, {"A"}, {"A"}, {"A"}], {"A": 2, "B": 2}) == 2


def test_unsupported_slot_contributes_zero():
    # Middle slot has no supporting class.
    assert max_fill_upper_bound(
        [{"A"}, set(), {"B"}], {"A": 1, "B": 1}) == 2


def test_matching_forces_spread():
    # Slot 0 supported by A only, slot 1 by A or B, U=1: the optimum (2)
    # requires slot 1 to take B -- a greedy that gives A to slot 1 first
    # would strand slot 0, which is exactly what the bound must see past.
    assert max_fill_upper_bound([{"A"}, {"A", "B"}], {"A": 1, "B": 1}) == 2


def test_zero_capacity_class_ignored():
    assert max_fill_upper_bound([{"A"}, {"A"}], {"A": 0}) == 0


def test_empty_instance():
    assert max_fill_upper_bound([], {"A": 3}) == 0


# ---------------------------------------------------------------------------
# max_fill_assignment: the witness matching itself
# ---------------------------------------------------------------------------

def _check_witness(slot_classes, caps, expected_value):
    value, assign = max_fill_assignment(slot_classes, caps)
    assert value == expected_value
    assert len(assign) == len(slot_classes)
    used: Dict[str, int] = {}
    for i, c in enumerate(assign):
        if c is None:
            continue
        assert c in slot_classes[i], f"slot {i} assigned unsupported class {c}"
        used[c] = used.get(c, 0) + 1
    for c, n in used.items():
        assert n <= caps[c], f"class {c} over capacity: {n} > {caps[c]}"
    assert sum(1 for c in assign if c is not None) == value
    return assign


def test_witness_matches_value_and_respects_support():
    _check_witness([{"A"}, {"A", "B"}, {"B"}], {"A": 1, "B": 1}, 2)


def test_witness_forces_spread():
    # The optimal witness MUST route slot 1 to B so slot 0 can take A.
    assign = _check_witness([{"A"}, {"A", "B"}], {"A": 1, "B": 1}, 2)
    assert assign == ["A", "B"]


def test_witness_leaves_infeasible_slot_unassigned():
    assign = _check_witness([{"A"}, set(), {"A"}], {"A": 2}, 2)
    assert assign[1] is None


# ---------------------------------------------------------------------------
# SteeredGenerator.assignment_upper_bound: stubbed engine, no profiling run
# ---------------------------------------------------------------------------

class _Cat:
    value = "cardinality"


class _Pattern:
    def __init__(self, pid):
        self.id = pid
        self.category = _Cat()


class _Metamodel:
    def get_class_names(self):
        return ["A", "B"]


class _Engine:
    metamodel = _Metamodel()
    all_patterns = [_Pattern("p1")]
    EXCLUDE_LIST = set()
    UNSAT_LIKELY_PATTERNS = set()


def _vec(**over):
    v = {c: 0.0 for c in COMPONENTS}
    v.update(over)
    return v


def test_generator_bound_respects_class_ceiling():
    spec = SteeringSpec(n=4, per_class=(0, 1))
    gen = SteeredGenerator(_Engine(), spec)
    # Hand-planted profile: pattern p1 reaches wno=2 on both classes.
    gen.fam_of = {"p1": "cardinality"}
    gen.prof = {("p1", "A"): [_vec(wno=2.0)], ("p1", "B"): [_vec(wno=2.0)]}
    gen.splittable = {}
    box = Box({"wno": (1.0, 3.0)}, "easy")
    slots = [Slot("cardinality", "sat", box) for _ in range(4)]
    # 4 slots, 2 supporting classes, ceiling U=1 -> certified optimum is 2.
    assert gen.assignment_upper_bound(slots) == 2


def test_generator_bound_unreachable_box():
    spec = SteeringSpec(n=2, per_class=(0, 5))
    gen = SteeredGenerator(_Engine(), spec)
    gen.fam_of = {"p1": "cardinality"}
    gen.prof = {("p1", "A"): [_vec(wno=2.0)]}
    gen.splittable = {}
    # tcc is only raisable via the cast actuator, which needs a collection
    # association; the stub metamodel has none, so a box mandating tcc >= 1
    # is unreachable and the bound must be 0.
    box = Box({"tcc": (1.0, 2.0)}, "typed")
    slots = [Slot("cardinality", "sat", box) for _ in range(2)]
    assert gen.assignment_upper_bound(slots) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failures else 0)
