"""The seed portfolio must decide the recorded hard instance that a single
Z3 run often cannot, and must return the same verdict as the instance has
(USE and every successful seed agree it is satisfiable)."""
import io, contextlib, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.core.models import OCLConstraint
from modules.verification.framework_verifier import FrameworkConstraintVerifier

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "carrental_hard_joint_instance_sat.json"


def test_portfolio_decides_hard_instance():
    data = json.load(open(FIXTURE))["constraints"]
    cs = [OCLConstraint(ocl=d["ocl"], pattern_id=d["pattern_id"], pattern_name=d["pattern_name"],
                        context=d["context"], parameters={}, confidence=1.0, timestamp="", metadata={}) for d in data]
    with contextlib.redirect_stdout(io.StringIO()):
        mm = MetamodelExtractor("models/car_rental.xmi").get_metamodel()
        v = FrameworkConstraintVerifier(mm, "models/car_rental.xmi", scope_per_class=6, timeout_ms=15000,
                                        batch_timeout_ms=180000, portfolio_seeds=4, portfolio_first_try_ms=1000)
        t0 = time.time()
        res = v.check_consistency_with_core(cs)
    assert res["solver_result"] == "sat"
    assert time.time() - t0 < 180
