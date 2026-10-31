#!/usr/bin/env python3
"""Generator-side check for the closure patterns.

Instantiates ``closure_operation`` and ``acyclicity`` repeatedly on the
self-associative ``org.xmi`` model exactly as the engine does (applicability
gate, ``_gen_params`` binding, ``OCLGenerator``), then pushes every instance
through the VGCR property checks (q1 tautology, q2 contradiction, q4
consistency against the growing suite) and the Z3 verifier, reporting the
verdict and whether the exact (``generic``) encoder handled it.

Run from the repository root:
    python3 closure_test/test_generation.py
"""
from __future__ import annotations
import sys, io, contextlib, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor  # noqa: E402
from modules.generation.benchmark.engine_v2 import BenchmarkEngineV2  # noqa: E402
from modules.verification.framework_verifier import FrameworkConstraintVerifier  # noqa: E402
from modules.verification.vgcr_verifier import VGCRPropertyChecker  # noqa: E402

XMI = str(Path(__file__).with_name("org.xmi"))

with contextlib.redirect_stdout(io.StringIO()):
    mm = MetamodelExtractor(XMI).get_metamodel()
    eng = BenchmarkEngineV2(mm, enable_semantic_validation=False)
    verifier = FrameworkConstraintVerifier(mm, XMI, scope_per_class=4, timeout_ms=30000)
    vgcr = VGCRPropertyChecker(verifier, mm, enable_independence=False)

pats = {p.id: p for p in eng.all_patterns}
assert "closure_operation" not in eng.EXCLUDE_LIST
assert eng._is_pattern_applicable(pats["closure_operation"], "Employee")
assert not eng._is_pattern_applicable(pats["closure_operation"], "Department")

suite, seen, rows = [], set(), []
random.seed(3)
for pid in ("closure_operation", "acyclicity"):
    for _ in range(12):
        params = eng._gen_params(pats[pid], "Employee")
        c = eng.generator.generate(pid, "Employee", params)
        key = c.ocl.strip()
        if key in seen:
            continue
        seen.add(key)
        with contextlib.redirect_stdout(io.StringIO()):
            r = vgcr.check_all(c, suite, (0.0, 1e9))
            v = verifier.verify(c)
        enc = next(iter(verifier.checker.constraint_encoding.values()), None)
        rows.append((pid, key.splitlines()[-1].strip(), v.solver_result, enc,
                     "pass" if r.passed else r.failure_type.value))
        if r.passed and v.solver_result == "sat":
            suite.append(c)

for row in rows:
    print(f"{row[0]:18s} {row[2]:5s} {row[3]:8s} {row[4]:14s} {row[1]}")
print()
print("distinct instances:", len(rows),
      "| verdicts:", dict(Counter(r[2] for r in rows)),
      "| encoding:", dict(Counter(r[3] for r in rows)),
      "| VGCR:", dict(Counter(r[4] for r in rows)))
bad = [r for r in rows if r[3] != "generic" or r[2] not in ("sat", "unsat")]
print("ALL PASS" if not bad else f"FAILURES: {bad}")
sys.exit(1 if bad else 0)
