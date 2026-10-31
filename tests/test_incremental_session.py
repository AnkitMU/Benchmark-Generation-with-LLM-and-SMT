"""Incremental Z3 session: unsat cores name the conflicting constraints, a
consistent set stays SAT, and a candidate tested with push/pop leaves no trace."""
import io, contextlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.core.models import OCLConstraint
from modules.verification.framework_verifier import FrameworkConstraintVerifier


def _mk(ctx, body, pid="x"):
    return OCLConstraint(ocl=f"context {ctx}\ninv: {body}", pattern_id=pid, pattern_name=pid,
                         context=ctx, parameters={}, confidence=1.0, timestamp="", metadata={})


BRANCH = _mk("Branch", "self.reservations->exists(x | x.estimatedCost >= 10)", "exists_constraint")
CUSTOMER_BAD = _mk("Customer", "(self.rentalCount > 0) and (self.reservations->forAll(v | v.estimatedCost < 10))", "collection_navigation")
CUSTOMER_OK = _mk("Customer", "(self.rentalCount > 0) and (self.reservations->forAll(v | v.estimatedCost < 100))", "collection_navigation")
VEHICLE = _mk("Vehicle", "self.branch <> null", "association_exists")


@pytest.fixture(scope="module")
def verifier():
    with contextlib.redirect_stdout(io.StringIO()):
        mm = MetamodelExtractor("models/car_rental.xmi").get_metamodel()
        return FrameworkConstraintVerifier(mm, "models/car_rental.xmi", scope_per_class=6,
                                           timeout_ms=15000, batch_timeout_ms=60000)


def test_core_names_the_conflict(verifier):
    res = verifier.check_consistency_with_core([VEHICLE, BRANCH, CUSTOMER_BAD])
    assert res["solver_result"] == "unsat"
    assert set(res["core"]) == {1, 2}, res   # the vehicle constraint is not part of the conflict


def test_consistent_set_is_sat(verifier):
    res = verifier.check_consistency_with_core([VEHICLE, BRANCH, CUSTOMER_OK])
    assert res["solver_result"] == "sat" and res["core"] == []


def test_candidate_is_retracted_after_rejection(verifier):
    s = verifier.open_session()
    assert s.add_permanent(verifier.constraint_to_dict(BRANCH, 0)) == "encoded"
    assert s.try_candidate(verifier.constraint_to_dict(CUSTOMER_BAD, 1))[0] == "unsat"
    assert s.check() == "sat"                       # the rejected candidate left nothing behind
    assert s.try_candidate(verifier.constraint_to_dict(CUSTOMER_OK, 1))[0] == "sat"
    assert s.asserted == ["exists_constraint__0"]
