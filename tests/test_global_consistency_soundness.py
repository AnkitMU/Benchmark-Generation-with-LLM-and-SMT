"""Soundness tests for the joint (global) consistency check.

Background: the pipeline's Phase B sends the individually-satisfiable
constraints of a benchmark together to Z3 through
``FrameworkConstraintVerifier.check_consistency`` /
``GenericGlobalConsistencyChecker``.  It used to report the Branch/Customer
pair below as consistent (with a witness that violated the Branch invariant)
because (1) the two ends of a bidirectional association were independent free
variables, (2) collection-end multiplicities were not enforced and (3) the
body of ``->exists`` was ignored.  USE 6.0.0's Model Validator reports the pair
UNSATISFIABLE at scope 6 with at least one instance per class.
"""
import contextlib
import io
import textwrap

import pytest

from modules.core.models import OCLConstraint
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.verification.framework_verifier import FrameworkConstraintVerifier

CAR_RENTAL = 'models/car_rental.xmi'
SCOPE = 6


def mk(ctx, ocl, pattern_id='x'):
    """Build a constraint with a pattern id the legacy encoders do not know, so
    the result depends on the whole-expression encoder alone."""
    return OCLConstraint(ocl=f"context {ctx}\ninv: {ocl}", pattern_id=pattern_id,
                         pattern_name=pattern_id, context=ctx, parameters={},
                         confidence=1.0, timestamp='', metadata={})


@pytest.fixture(scope='module')
def car_rental_verifier():
    mm = MetamodelExtractor(CAR_RENTAL).get_metamodel()
    with contextlib.redirect_stdout(io.StringIO()):
        v = FrameworkConstraintVerifier(mm, CAR_RENTAL, scope_per_class=SCOPE,
                                        timeout_ms=15000, batch_timeout_ms=120000)
    assert v.framework_available, v.unavailable_reason
    return v


def check(verifier, constraints):
    with contextlib.redirect_stdout(io.StringIO()):
        return verifier.check_consistency(constraints)


# ── the reported pair ────────────────────────────────────────────────────────

BRANCH_EXISTS = ('Branch', 'self.reservations->exists(x | x.estimatedCost >= 10)')
CUSTOMER_FORALL_LT10 = ('Customer', '(self.rentalCount > 0) and (self.reservations->forAll(v | v.estimatedCost < 10))')
CUSTOMER_FORALL_LT100 = ('Customer', '(self.rentalCount > 0) and (self.reservations->forAll(v | v.estimatedCost < 100))')


def test_branch_customer_pair_is_inconsistent(car_rental_verifier):
    """Every Reservation has exactly one branch and one customer (eOpposites),
    so the reservation with estimatedCost >= 10 that every Branch needs must
    appear in its customer's reservations, where every cost is < 10."""
    result = check(car_rental_verifier, [mk(*BRANCH_EXISTS), mk(*CUSTOMER_FORALL_LT10)])
    assert result['solver_result'] == 'unsat', result
    assert result['consistent'] is False
    assert result['verified'] is True
    assert result['unencoded'] == []
    assert result['legacy_encoded'] == []


def test_branch_customer_control_pair_is_consistent(car_rental_verifier):
    """Same shape with a non-conflicting bound stays consistent."""
    result = check(car_rental_verifier, [mk(*BRANCH_EXISTS), mk(*CUSTOMER_FORALL_LT100)])
    assert result['solver_result'] == 'sat', result
    assert result['consistent'] is True
    assert result['verified'] is True
    assert result['unencoded'] == []


# ── the individual root causes ───────────────────────────────────────────────

def test_exists_body_is_encoded(car_rental_verifier):
    """``->exists(x | body)`` must constrain the chosen element, not just require
    a non-empty collection."""
    result = check(car_rental_verifier, [
        mk(*BRANCH_EXISTS),
        mk('Reservation', 'self.estimatedCost < 10'),
    ])
    assert result['solver_result'] == 'unsat', result


def test_optional_opposite_end_is_coupled(car_rental_verifier):
    """Vehicle.reservations (*) is the eOpposite of Reservation.vehicle (0..1).
    If no reservation has a vehicle, no vehicle can have reservations."""
    result = check(car_rental_verifier, [
        mk('Vehicle', 'self.reservations->size() >= 1'),
        mk('Reservation', 'self.vehicle = null'),
    ])
    assert result['solver_result'] == 'unsat', result
    # and the coupling is not over-constraining: a vehicle per reservation is fine
    result = check(car_rental_verifier, [
        mk('Vehicle', 'self.reservations->size() >= 1'),
        mk('Reservation', 'self.vehicle <> null'),
    ])
    assert result['solver_result'] == 'sat', result


def test_required_opposite_end_is_coupled(car_rental_verifier):
    """Branch.reservations (*) <-> Reservation.branch (1): the branch a
    reservation points to must be the branch that contains it."""
    result = check(car_rental_verifier, [
        mk('Branch', 'self.reservations->forAll(r | r.branch = self)'),
        mk('Branch', 'self.reservations->notEmpty()'),
    ])
    assert result['solver_result'] == 'sat', result
    result = check(car_rental_verifier, [
        mk('Branch', 'self.reservations->exists(r | r.branch <> self)'),
    ])
    assert result['solver_result'] == 'unsat', result


TINY_XMI = textwrap.dedent('''\
    <?xml version="1.0" encoding="UTF-8"?>
    <ecore:EPackage xmlns:xmi="http://www.omg.org/XMI"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xmlns:ecore="http://www.eclipse.org/emf/2002/Ecore"
        name="tiny" nsURI="http://example.org/tiny" nsPrefix="tiny">
      <eClassifiers xsi:type="ecore:EClass" name="A">
        <eStructuralFeatures xsi:type="ecore:EAttribute" name="n"
          eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt"/>
        <eStructuralFeatures xsi:type="ecore:EReference" name="bs" upperBound="3"
          eType="#//B" eOpposite="#//B/a"/>
        <eStructuralFeatures xsi:type="ecore:EReference" name="cs" lowerBound="2" upperBound="-1"
          eType="#//C" eOpposite="#//C/a"/>
      </eClassifiers>
      <eClassifiers xsi:type="ecore:EClass" name="B">
        <eStructuralFeatures xsi:type="ecore:EReference" name="a" lowerBound="1"
          eType="#//A" eOpposite="#//A/bs"/>
      </eClassifiers>
      <eClassifiers xsi:type="ecore:EClass" name="C">
        <eStructuralFeatures xsi:type="ecore:EReference" name="a" lowerBound="1"
          eType="#//A" eOpposite="#//A/cs"/>
      </eClassifiers>
    </ecore:EPackage>
''')


@pytest.fixture(scope='module')
def tiny_verifier(tmp_path_factory):
    xmi = tmp_path_factory.mktemp('tiny') / 'tiny.xmi'
    xmi.write_text(TINY_XMI)
    mm = MetamodelExtractor(str(xmi)).get_metamodel()
    with contextlib.redirect_stdout(io.StringIO()):
        v = FrameworkConstraintVerifier(mm, str(xmi), scope_per_class=SCOPE,
                                        timeout_ms=15000, batch_timeout_ms=60000)
    assert v.framework_available, v.unavailable_reason
    return v


def test_collection_end_upper_bound(tiny_verifier):
    """A.bs has upperBound=3: a collection of 4 is impossible, 3 is fine."""
    assert check(tiny_verifier, [mk('A', 'self.bs->size() >= 4')])['solver_result'] == 'unsat'
    assert check(tiny_verifier, [mk('A', 'self.bs->size() = 3')])['solver_result'] == 'sat'


def test_collection_end_lower_bound(tiny_verifier):
    """A.cs has lowerBound=2."""
    assert check(tiny_verifier, [mk('A', 'self.cs->size() < 2')])['solver_result'] == 'unsat'
    assert check(tiny_verifier, [mk('A', 'self.cs->size() = 2')])['solver_result'] == 'sat'


# ── reporting ────────────────────────────────────────────────────────────────

def test_unencodable_constraint_is_not_reported_consistent(car_rental_verifier):
    """A constraint neither encoder can handle must not vanish into a 'sat'."""
    result = check(car_rental_verifier, [
        mk(*BRANCH_EXISTS),
        mk('Branch', 'self.reservations->sortedBy(x | x.estimatedCost)->first().estimatedCost > 5'),
    ])
    assert result['consistent'] is not True
    assert result['verified'] is False
    assert len(result['unencoded']) == 1


def test_pattern_id_does_not_change_the_verdict(car_rental_verifier):
    """The verdict depends on the OCL text, not on the generator's pattern id."""
    pair = [mk(*BRANCH_EXISTS, pattern_id='exists_constraint'),
            mk(*CUSTOMER_FORALL_LT10, pattern_id='implies_simple')]
    assert check(car_rental_verifier, pair)['solver_result'] == 'unsat'
