"""The VGCR non-triviality check must only reject constraints that no model can
violate. Existence checks on optional ends are genuine constraints; on mandatory
ends they are tautologies. Uses the real Z3 verifier on models/car_rental.xmi."""
import io, contextlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.core.models import OCLConstraint
from modules.verification.framework_verifier import FrameworkConstraintVerifier


def _mk(ctx, body, pid):
    return OCLConstraint(ocl=f"context {ctx}\ninv: {body}", pattern_id=pid, pattern_name=pid,
                         context=ctx, parameters={}, confidence=1.0, timestamp="", metadata={})


@pytest.fixture(scope="module")
def verifier():
    with contextlib.redirect_stdout(io.StringIO()):
        mm = MetamodelExtractor("models/car_rental.xmi").get_metamodel()
        return FrameworkConstraintVerifier(mm, "models/car_rental.xmi", scope_per_class=6,
                                           timeout_ms=15000, batch_timeout_ms=60000)


def _negation_sat(verifier, ctx, negated_body):
    with contextlib.redirect_stdout(io.StringIO()):
        r = verifier.verify(_mk(ctx, negated_body, "association_exists"),
                            scope_overrides={ctx: 1}, required_classes={ctx})
    return r.is_satisfiable


def test_optional_end_existence_is_not_a_tautology(verifier):
    # Rental.payment is [0..1]: a rental without payment is a legal model.
    assert _negation_sat(verifier, "Rental", "self.payment = null") is True


def test_mandatory_end_existence_is_a_tautology(verifier):
    # Reservation.customer is [1..1]: no legal model has a reservation without customer.
    assert _negation_sat(verifier, "Reservation", "self.customer = null") is False
