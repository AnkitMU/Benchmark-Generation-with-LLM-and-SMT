#!/usr/bin/env python3
"""Checks for the bounded ``->closure()`` encoding in the exact OCL encoder.

Part 1 — differential test against brute-force reachability.
    For random graphs on k slots the step relation is pinned with equalities,
    the encoder's closure value is evaluated in the resulting Z3 model, and the
    member set is compared with a Python BFS (source elements plus everything
    reachable from them, which is the OCL / USE semantics).  Exercised bodies:
    many-valued ``reports``, single-valued ``manager``, a two-hop body
    ``v.manager.manager`` and a filtered body ``v.reports->select(...)``.
    ``->size()`` and ``->includes(self)`` of the closure are checked as whole
    invariant formulas on top of the member set.

Part 2 — verdicts for whole invariants through the checker's normal entry
    point (``verify_all_constraints``): acyclicity, a forced cycle, their
    contradiction, and interaction with ``allInstances()``.

Run from the repository root:
    python3 closure_test/test_closure.py
"""
from __future__ import annotations
import random, sys, types, io, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "hybrid-ssr-ocl-full-extended" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

# Import the checker without executing super_encoder/__init__.py (it pulls in
# torch / sentence-transformers); same trick as modules/verification/framework_verifier.py.
stub = types.ModuleType("ssr_ocl.super_encoder")
stub.__path__ = [str(SRC / "ssr_ocl" / "super_encoder")]
stub.__package__ = "ssr_ocl.super_encoder"
sys.modules.setdefault("ssr_ocl.super_encoder", stub)

from z3 import Solver, sat, unsat, Not, is_true  # noqa: E402
from ssr_ocl.super_encoder.generic_global_consistency_checker import GenericGlobalConsistencyChecker  # noqa: E402
from ssr_ocl.super_encoder.generic_ocl_expression_encoder import OCLParser, ObjSet, _TRUE  # noqa: E402

XMI = str(Path(__file__).with_name("org.xmi"))
K = 4
SCOPE = {"nEmployee": K, "nDepartment": 1}


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def bfs(adj, start):
    """Slots reachable from *start* by one or more steps."""
    seen, frontier = set(), [j for j in range(K) if adj[start][j]]
    while frontier:
        x = frontier.pop()
        if x in seen:
            continue
        seen.add(x)
        frontier.extend(j for j in range(K) if adj[x][j])
    return seen


def pin_many(sv, adj):
    R = sv["Employee.reports"]
    fs = [sv["Employee_presence"][i] for i in range(K)]
    for i in range(K):
        for j in range(K):
            fs.append(R[i][j] if adj[i][j] else Not(R[i][j]))
    return fs


def pin_single(sv, succ):
    ref, present = sv["Employee.manager"], sv["Employee.manager_present"]
    fs = [sv["Employee_presence"][i] for i in range(K)]
    for i in range(K):
        if succ[i] is None:
            fs.append(Not(present[i]))
        else:
            fs += [present[i], ref[i] == succ[i]]
    return fs


def model_of(pins):
    s = Solver()
    s.add(*pins)
    assert s.check() == sat
    return s.model()


def members(enc, sv, expr, i, m):
    """Member slots of *expr* (an object collection) evaluated for self = slot i."""
    enc.shared_vars, enc.scope, enc.solver = sv, SCOPE, None
    v = enc._eval(OCLParser(expr).parse(), {"self": ObjSet([(_TRUE, "Employee", i)], single=True)})
    assert isinstance(v, ObjSet)
    return {y for g, _, y in v.elems if is_true(m.eval(g, model_completion=True))}


def determined(pins, f):
    """True/False when *f* is forced by the pins; raises if it is not."""
    s = Solver(); s.add(*pins); s.add(f)
    r = s.check()
    s2 = Solver(); s2.add(*pins); s2.add(Not(f))
    r2 = s2.check()
    assert (r, r2) in ((sat, unsat), (unsat, sat)), (r, r2)
    return r == sat


def check_family(enc, sv, pins, closure_expr, rel, label, failures):
    m = model_of(pins)
    n = 0
    for i in range(K):
        exp = bfs(rel, i)
        got = members(enc, sv, closure_expr, i, m)
        n += 1
        if got != exp:
            failures.append((label, "members", i, sorted(exp), sorted(got)))
        inv_size = enc.encode_invariant("Employee", f"context Employee inv: {closure_expr}->size() = {len(exp)}",
                                        sv, SCOPE, None)[i]
        inv_incl = enc.encode_invariant("Employee", f"context Employee inv: {closure_expr}->includes(self)",
                                        sv, SCOPE, None)[i]
        n += 2
        if not determined(pins, inv_size):
            failures.append((label, "size", i, len(exp)))
        if determined(pins, inv_incl) != (i in exp):
            failures.append((label, "includes(self)", i, i in exp))
    return n


def part1(checker):
    enc = checker.generic_encoder
    rng = random.Random(7)
    failures, checks = [], 0

    for _ in range(50):                                   # many-valued reports
        adj = [[rng.random() < 0.35 for _ in range(K)] for _ in range(K)]
        sv = quiet(checker._create_shared_variables, SCOPE)
        checks += check_family(enc, sv, pin_many(sv, adj),
                               "self.reports->closure(e | e.reports)", adj, "many", failures)

    for _ in range(50):                                   # single-valued manager
        succ = [rng.choice([None] + list(range(K))) for _ in range(K)]
        adj = [[succ[i] == j for j in range(K)] for i in range(K)]
        sv = quiet(checker._create_shared_variables, SCOPE)
        checks += check_family(enc, sv, pin_single(sv, succ),
                               "self.manager->closure(m | m.manager)", adj, "single", failures)

    for _ in range(40):                                   # two-hop body
        succ = [rng.choice([None] + list(range(K))) for _ in range(K)]
        two = [[any(succ[i] == l and succ[l] == j for l in range(K)) for j in range(K)] for i in range(K)]
        sv = quiet(checker._create_shared_variables, SCOPE)
        checks += check_family(enc, sv, pin_single(sv, succ),
                               "self.manager.manager->closure(m | m.manager.manager)", two, "twohop", failures)

    for _ in range(40):                                   # filtered body
        adj = [[rng.random() < 0.4 for _ in range(K)] for _ in range(K)]
        sal = [rng.choice([5, 20]) for _ in range(K)]
        fadj = [[adj[i][j] and sal[j] > 10 for j in range(K)] for i in range(K)]
        sv = quiet(checker._create_shared_variables, SCOPE)
        pins = pin_many(sv, adj) + [sv["Employee.salary"][i] == sal[i] for i in range(K)]
        checks += check_family(enc, sv, pins,
                               "self.reports->select(r | r.salary > 10)->closure(e | e.reports->select(r | r.salary > 10))",
                               fadj, "filtered", failures)

    for f in failures[:10]:
        print("FAIL", f)
    print(f"Part 1: {checks} differential checks, {len(failures)} failures")
    return not failures


def verdict(checker, texts, scope=SCOPE):
    cs = [{"name": f"c{i}", "pattern": "generic", "context": "Employee", "text": t}
          for i, t in enumerate(texts)]
    res, _ = quiet(checker.verify_all_constraints, cs, scope)
    enc = {n: checker.constraint_encoding.get(n) for n in (c["name"] for c in cs)}
    return res, enc, dict(checker.generic_fallbacks)


def part2(checker):
    ok = True
    acyc_m = "context Employee inv: self.manager->closure(m | m.manager)->excludes(self)"
    cyc_m = "context Employee inv: self.manager->closure(m | m.manager)->includes(self)"
    acyc_r = "context Employee inv: self.reports->closure(e | e.reports)->excludes(self)"
    cases = [
        ([acyc_m], "sat"),                                       # acyclic management
        ([cyc_m], "sat"),                                        # everyone on a cycle (a ring)
        ([acyc_m, cyc_m], "unsat"),                              # contradiction
        ([acyc_r, "context Employee inv: self.reports->notEmpty()"], "unsat"),   # acyclic + total: no finite model
        ([acyc_r, "context Employee inv: self.manager->isEmpty() implies "
                  "self.reports->closure(e | e.reports)->size() = Employee.allInstances()->size() - 1"], "sat"),
        ([f"context Employee inv: self.reports->closure(e | e.reports)->size() > {K}"], "unsat"),  # exceeds scope
        # manager chain must be consistent with reports: manager's closure over reports contains self
        (["context Employee inv: self.manager->notEmpty() implies "
          "self.manager.reports->closure(e | e.reports)->includes(self)"], "sat"),
    ]
    for texts, expected in cases:
        res, enc, fb = verdict(checker, texts)
        good = res == expected and all(v == "generic" for v in enc.values())
        ok &= good
        print(("ok  " if good else "FAIL"), expected, "->", res, enc, fb or "")
    return ok


def main():
    checker = quiet(GenericGlobalConsistencyChecker, xmi_file=XMI, rich_instances=False,
                    timeout_ms=60000, show_raw_values=False)
    p1 = part1(checker)
    p2 = part2(checker)
    print("ALL PASS" if (p1 and p2) else "FAILURES")
    sys.exit(0 if (p1 and p2) else 1)


if __name__ == "__main__":
    main()
