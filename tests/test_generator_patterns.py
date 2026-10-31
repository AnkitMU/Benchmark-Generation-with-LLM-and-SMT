"""Generator-side tests for existence / navigation patterns and the VGCR
pattern-switching retry logic.

Background: ``context Reservation inv: self.customer <> null`` is a tautology
when ``Reservation.customer`` is a mandatory [1..1] end, and VGCR's q1
correctly rejects it.  The replacement generator used to re-propose the same
``association_exists`` template for the same slot until the retry budget was
exhausted (CarRental: 14 of 100 slots left empty, 136 tautology events).

  (a) ``association_exists`` binds only OPTIONAL ([0..1]) single-valued ends;
  (b) ``navigation_consistency`` binds ``self.vehicle.branch = self.branch``
      on Rental (the only well-typed triple there);
  (c) the VGCR retry path excludes a pattern for the slot after a tautology,
      and after the same (pattern, context) pair failed twice for any reason
      -- unit-tested on the state/selection logic, no solver involved.

Run from the repository root with ``python3 -m pytest tests/ -q``.
"""
import contextlib
import io
import random

import pytest

from modules.core.models import OCLConstraint, Parameter, ParameterType
from modules.generation.benchmark.engine_v2 import BenchmarkEngineV2, classify_family
from modules.generation.benchmark.vgcr_refinement import (
    GeneratorState, SlotRetryState, VGCRLoop,
)
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.verification.vgcr_verifier import FailureType, PropertyCheckResult

CAR_RENTAL = 'models/car_rental.xmi'

# Single-valued ends of models/car_rental.xmi by multiplicity lower bound.
MANDATORY_ENDS = {
    ('Vehicle', 'branch'), ('Rental', 'vehicle'), ('Rental', 'customer'),
    ('License', 'customer'), ('Customer', 'license'), ('Reservation', 'customer'),
    ('Reservation', 'branch'), ('Rental', 'branch'), ('Branch', 'company'),
    ('Payment', 'rental'),
}
OPTIONAL_ENDS = {('Rental', 'payment'), ('Reservation', 'vehicle')}


@pytest.fixture(scope='module')
def car_rental():
    return MetamodelExtractor(CAR_RENTAL).get_metamodel()


@pytest.fixture(scope='module')
def engine(car_rental):
    with contextlib.redirect_stdout(io.StringIO()):
        return BenchmarkEngineV2(car_rental, enable_semantic_validation=True,
                                 verification_enabled=True, generation_mode='steered')


def pattern(engine, pid):
    p = next((p for p in engine.all_patterns if p.id == pid), None)
    assert p is not None, f"pattern {pid} not in the library"
    return p


def body(constraint):
    return constraint.ocl.split('inv:', 1)[1].strip()


# ── (a) association_exists never binds a mandatory end ───────────────────────

def test_metamodel_lower_bounds_match_expectation(car_rental):
    """Sanity: the XMI really carries the multiplicities the tests rely on."""
    ends = {(a.source_class, a.ref_name): a.lower
            for cls in car_rental.get_class_names()
            for a in car_rental.get_single_associations(cls)}
    assert set(ends) == MANDATORY_ENDS | OPTIONAL_ENDS
    assert all(ends[e] == 1 for e in MANDATORY_ENDS)
    assert all(ends[e] == 0 for e in OPTIONAL_ENDS)


def test_optional_associations_option_source(car_rental):
    """The 'optional_associations' option source resolves to [0..1] ends only."""
    param = Parameter(name='association', label='Association', type=ParameterType.SELECT,
                      options='optional_associations')
    resolved = {(cls, ref) for cls in car_rental.get_class_names()
                for ref in param.get_options_for_context(car_rental, cls)}
    assert resolved == OPTIONAL_ENDS


def test_association_exists_declares_optional_associations(engine):
    p = pattern(engine, 'association_exists')
    assert p.template == 'self.{association} <> null'
    assert [prm.options for prm in p.parameters] == ['optional_associations']


def test_association_exists_never_binds_a_mandatory_end(engine, car_rental):
    random.seed(7)
    bound = set()
    for cls in car_rental.get_class_names():
        p = pattern(engine, 'association_exists')
        applicable = engine._is_pattern_applicable(p, cls)
        has_optional = any(cls == c for c, _ in OPTIONAL_ENDS)
        assert applicable == has_optional, cls
        for _ in range(40):
            try:
                params = engine._gen_params(p, cls)
            except ValueError:
                assert not applicable, f"{cls}: applicable but binding failed"
                continue
            assert applicable, f"{cls}: bound although not applicable"
            bound.add((cls, params['association']))
            c = engine.generator.generate(p.id, cls, params)
            assert body(c) == f"self.{params['association']} <> null"
    assert bound == OPTIONAL_ENDS
    assert not (bound & MANDATORY_ENDS)


# ── (b) navigation_consistency on Rental ─────────────────────────────────────

def test_navigation_consistency_binds_vehicle_branch_on_rental(engine):
    random.seed(11)
    p = pattern(engine, 'navigation_consistency')
    assert engine._is_pattern_applicable(p, 'Rental')
    seen = set()
    for _ in range(30):
        params = engine._gen_params(p, 'Rental')
        c = engine.generator.generate(p.id, 'Rental', params)
        seen.add(body(c))
    assert seen == {'self.vehicle.branch = self.branch'}


def test_navigation_consistency_applicability_matches_binding(engine, car_rental):
    """applicable == instantiable on every class (no false positives)."""
    p = pattern(engine, 'navigation_consistency')
    for cls in car_rental.get_class_names():
        triples = engine._navigation_consistency_triples(cls)
        assert engine._is_pattern_applicable(p, cls) == bool(triples), cls
        for a, b, c in triples:
            # both paths must reach the same class and the direct end differs
            a_end = car_rental.get_class(cls).get_association(a)
            b_end = car_rental.get_class(a_end.target_class).get_association(b)
            c_end = car_rental.get_class(cls).get_association(c)
            assert b_end.target_class == c_end.target_class and c != a
    assert engine._navigation_consistency_triples('Rental') == [('vehicle', 'branch', 'branch')]


def test_conditional_existence_binds_flag_and_optional_end(engine):
    random.seed(3)
    p = pattern(engine, 'conditional_existence')
    seen = set()
    for _ in range(20):
        params = engine._gen_params(p, 'Rental')
        seen.add(body(engine.generator.generate(p.id, 'Rental', params)))
    assert seen == {'self.isCompleted implies self.payment <> null'}
    # Vehicle has boolean attributes but no optional single-valued end
    assert not engine._is_pattern_applicable(p, 'Vehicle')


def test_new_patterns_are_navigation_family():
    assert classify_family('navigation_consistency', 'navigation') == 'navigation'
    assert classify_family('conditional_existence', 'navigation') == 'navigation'
    # the if-then-else family is untouched
    assert classify_family('conditional_if_then_else', 'advanced') == 'conditional'


# ── (c) VGCR retry path: pattern switching ───────────────────────────────────

def test_slot_state_excludes_pattern_after_tautology():
    st = SlotRetryState()
    assert st.record_failure('association_exists', 'Reservation', FailureType.TAUTOLOGY)
    assert st.is_excluded('association_exists')
    assert st.excluded_patterns == {'association_exists'}


def test_slot_state_excludes_pair_after_two_failures_of_any_kind():
    st = SlotRetryState()
    assert not st.record_failure('p', 'A', FailureType.CONTRADICTION)
    assert not st.is_excluded('p')
    # a different context does not count toward the same pair
    assert not st.record_failure('p', 'B', FailureType.INCONSISTENT)
    assert not st.is_excluded('p')
    # second failure of the SAME (pattern, context) pair, any reason
    assert st.record_failure('p', 'A')
    assert st.is_excluded('p')


def mk(pid, ctx='Reservation', ocl=None):
    return OCLConstraint(ocl=ocl or f"context {ctx}\ninv: self.x <> null",
                         pattern_id=pid, pattern_name=pid, context=ctx,
                         parameters={}, confidence=1.0, timestamp='', metadata={})


class _FakeChecker:
    """No solver: fails a chosen pattern with a chosen failure type."""

    def __init__(self, failing):
        self.failing = failing  # pattern_id -> FailureType

    def check_all(self, candidate, current_suite, tc_range):
        ft = self.failing.get(candidate.pattern_id)
        if ft is None:
            return PropertyCheckResult(passed=True)
        return PropertyCheckResult(passed=False, failure_type=ft,
                                   failure_message=ft.value)


class _FakeEngine:
    """Re-proposes the first pattern until it is excluded, then the second;
    records the exclusion set it was handed on every call."""

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.calls = []

    def generate_refined(self, failed_constraint, generator_state,
                         accept=None, only_classes=None, exclude_patterns=None):
        excluded = set(exclude_patterns or ())
        self.calls.append(excluded)
        pid = self.second if self.first in excluded else self.first
        return mk(pid, failed_constraint.context)


def test_vgcr_retry_switches_pattern_after_tautology():
    engine = _FakeEngine('association_exists', 'navigation_consistency')
    checker = _FakeChecker({'association_exists': FailureType.TAUTOLOGY})
    loop = VGCRLoop(engine=engine, checker=checker, metamodel=None, max_retries=15)

    result = loop._refine_single(mk('association_exists'), [], (0.0, 1e4), slot_idx=0)

    assert result is not None and result.pattern_id == 'navigation_consistency'
    # exactly one retry, and it was asked to exclude the tautological pattern
    assert engine.calls == [{'association_exists'}]
    assert loop.stats.passed_after_retry == 1
    assert loop.stats.failed_all_retries == 0


def test_vgcr_retry_switches_pattern_after_repeated_failure():
    # TC_MISMATCH changes no generator state, so without the slot rule the
    # engine would keep re-proposing 'p1' on the same context.
    engine = _FakeEngine('p1', 'p2')
    checker = _FakeChecker({'p1': FailureType.TC_MISMATCH})
    loop = VGCRLoop(engine=engine, checker=checker, metamodel=None, max_retries=15)

    result = loop._refine_single(mk('p1', 'Rental'), [], (0.0, 1e4), slot_idx=3)

    assert result is not None and result.pattern_id == 'p2'
    assert engine.calls == [set(), {'p1'}]


def test_vgcr_unfilled_slot_reports_exclusions():
    engine = _FakeEngine('p1', 'p2')
    checker = _FakeChecker({'p1': FailureType.TAUTOLOGY, 'p2': FailureType.TAUTOLOGY})
    loop = VGCRLoop(engine=engine, checker=checker, metamodel=None, max_retries=2)

    assert loop._refine_single(mk('p1'), [], (0.0, 1e4), slot_idx=5) is None
    assert loop.unfilled_log[-1]['excluded_patterns'] == ['p1', 'p2']


def test_engine_weighted_sample_honours_exclusions(engine):
    """The legacy (box-free) replacement sampler skips excluded ids too."""
    random.seed(5)
    nav = {p.id for p in engine.all_patterns
           if classify_family(p.id, p.category.value) == 'navigation'}
    excluded = nav - {'navigation_consistency'}
    for _ in range(20):
        p = engine._vgcr_weighted_sample('Rental', 'navigation_consistency', GeneratorState())
        assert p is None or p.id not in {'navigation_consistency'}
        p = engine._vgcr_weighted_sample('Rental', 'simple_navigation', GeneratorState(),
                                         exclude_patterns=excluded)
        assert p is not None and p.id == 'navigation_consistency'
