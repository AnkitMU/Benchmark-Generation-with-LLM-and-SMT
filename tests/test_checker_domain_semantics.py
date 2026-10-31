"""The checker must decide constraints by the metamodel's semantics only:
no invented numeric bounds, no name-based value guesses, enums over their
literals. Each case was a wrong verdict before the fix."""
import io, contextlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.core.models import OCLConstraint
from modules.verification.framework_verifier import FrameworkConstraintVerifier

_cache = {}
def _verifier(xmi):
    if xmi not in _cache:
        with contextlib.redirect_stdout(io.StringIO()):
            mm = MetamodelExtractor(xmi).get_metamodel()
            _cache[xmi] = FrameworkConstraintVerifier(mm, xmi, scope_per_class=6, timeout_ms=30000, batch_timeout_ms=30000)
    return _cache[xmi]

def verdict(xmi, ctx, body):
    v = _verifier(xmi)
    oc = OCLConstraint(ocl=f"context {ctx}\ninv: {body}", pattern_id="t", pattern_name="t", context=ctx,
                       parameters={}, confidence=1.0, timestamp="", metadata={})
    with contextlib.redirect_stdout(io.StringIO()):
        s = v.open_session(timeout_ms=30000, required_classes={ctx})
        assert s.add_permanent(v.constraint_to_dict(oc, 0)) == "encoded"
        return s.check()

CAR, LIB, UNI = "models/car_rental.xmi", "models/library_system.xmi", "models/university_system.xmi"

@pytest.mark.parametrize("xmi,ctx,body,expected", [
    (CAR, "Customer", "self.balance < 0", "sat"),                   # Real was forced >= 0
    (CAR, "Customer", "self.age < -2000", "sat"),                   # Integer was forced >= -1000
    (CAR, "Customer", "self.age > 200", "sat"),                     # 'age' was forced into [0,150]
    (LIB, "Book", "self.pages > 500", "sat"),                       # 'pages' contains 'age'
    (CAR, "Vehicle", "self.tankLevel > 100", "sat"),                # 'level' was forced into [0,100]
    (CAR, "Rental", "self.startDate >= self.endDate", "sat"),       # date ordering was assumed
    (CAR, "Vehicle", "self.status = VehicleStatus::Rented", "sat"),
    (CAR, "Vehicle", "self.status <> VehicleStatus::Available and self.status <> VehicleStatus::Rented "
                     "and self.status <> VehicleStatus::Maintenance", "unsat"),   # enum was unbounded
    (UNI, "Program", "self.level = ProgramLevel::PhD", "sat"),      # enum named 'level' is not a percentage
])
def test_domain_semantics(xmi, ctx, body, expected):
    assert verdict(xmi, ctx, body) == expected

def test_literal_codes_are_process_independent():
    v = _verifier(CAR)
    import zlib
    assert v.checker._string_to_int("Rented") == zlib.crc32(b"Rented") % 1_000_000
