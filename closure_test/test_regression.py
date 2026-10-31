#!/usr/bin/env python3
"""Regression guard: the closure change must not alter verdicts for ordinary
constraints.  Re-verifies a sample of delivered constraints from the paper_v3
CarRental suite (``constraints_sat.json`` / ``constraints_unsat.json``) through
``FrameworkConstraintVerifier`` and compares the Z3 verdict with the file's
label, also reporting how many took the legacy fallback path.

Run from the repository root:
    python3 closure_test/test_regression.py [N_PER_LABEL]
"""
from __future__ import annotations
import json, sys, io, contextlib, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor  # noqa: E402
from modules.verification.framework_verifier import FrameworkConstraintVerifier  # noqa: E402
from modules.core.models import OCLConstraint  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
suite = ROOT / "benchmarks" / "paper_v3" / "with_vgcr" / "CarRental"
xmi = str(ROOT / "models" / "car_rental.xmi")
rng = random.Random(1)


def load(name):
    d = json.loads((suite / name).read_text())
    return d["constraints"] if isinstance(d, dict) else d


sample = [(c, "sat") for c in rng.sample(load("constraints_sat.json"), N)] + \
         [(c, "unsat") for c in rng.sample(load("constraints_unsat.json"), N)]

with contextlib.redirect_stdout(io.StringIO()):
    mm = MetamodelExtractor(xmi).get_metamodel()
    v = FrameworkConstraintVerifier(mm, xmi, scope_per_class=4, timeout_ms=30000)

mismatch, fallbacks = 0, 0
for c, label in sample:
    oc = OCLConstraint(ocl=c["ocl"], pattern_id=c.get("pattern_id", "generic"),
                       pattern_name=c.get("pattern_name", "generic"), context=c["context"],
                       parameters=c.get("parameters", {}))
    with contextlib.redirect_stdout(io.StringIO()):
        r = v.verify(oc)
    enc = next(iter(v.checker.constraint_encoding.values()), None)
    if enc != "generic":
        fallbacks += 1
    if r.solver_result != label:
        mismatch += 1
        print(f"MISMATCH [{label}] {c['context']}: {c['ocl'].splitlines()[-1].strip()}\n"
              f"   got {r.solver_result} ({enc}; {v.checker.generic_fallbacks})")
print(f"{len(sample)} constraints: {mismatch} verdict mismatches, {fallbacks} legacy fallbacks")
sys.exit(1 if mismatch else 0)
