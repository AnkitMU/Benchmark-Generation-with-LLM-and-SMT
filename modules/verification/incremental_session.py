"""
Incremental Z3 session over the framework's global consistency checker.

The checker (GenericGlobalConsistencyChecker) builds a fresh solver, shared
variables and domain axioms for every call of ``verify_all_constraints``.  The
VGCR loop and the compatibility pass call it hundreds of times per model with
an ever-growing set, which re-solves the same problem from scratch each time.

This session reuses the checker's own building blocks (shared variables,
domain axioms, per-constraint encoders) on ONE solver:

* accepted constraints are asserted permanently, each under a tracking
  literal ``C_<name>`` so an UNSAT verdict yields an unsat core of names;
* a candidate is tested under ``push``/``pop`` so a rejected candidate leaves
  no trace and the learned clauses of the accepted set carry over.

Nothing in the checker file is modified; encoders write into a collecting
proxy whose formulas are then asserted (tracked) on the real solver.
"""
from __future__ import annotations
import contextlib
import os
from typing import Dict, List, Optional, Tuple

import z3


class _CollectingSolver:
    """Stands in for a z3.Solver while an encoder runs: records every formula
    passed to ``add`` and forwards anything else to the real solver."""

    def __init__(self, real: z3.Solver):
        self._real = real
        self.formulas: List = []

    def add(self, *formulas):
        for f in formulas:
            if isinstance(f, (list, tuple)):
                self.formulas.extend(f)
            else:
                self.formulas.append(f)

    def append(self, *formulas):  # some encoders use append
        self.add(*formulas)

    def __getattr__(self, name):
        return getattr(self._real, name)


class IncrementalConsistencySession:
    def __init__(self, checker, scope: Dict, timeout_ms: Optional[int] = None,
                 required_classes: Optional[set] = None):
        self.checker = checker
        self.scope = scope
        self.solver = z3.Solver()
        if timeout_ms:
            self.solver.set("timeout", int(timeout_ms))
        # Same per-run reset as verify_all_constraints()
        checker.constraint_status = {}
        checker.encoding_errors = []
        checker.constraint_tags = {}
        checker.constraint_encoding = {}
        checker.generic_fallbacks = {}
        previous = getattr(checker, 'required_classes', None)
        checker.required_classes = required_classes
        try:
            with self._quiet():
                self.shared_vars = checker._create_shared_variables(scope)
                checker._add_domain_constraints(self.solver, self.shared_vars, scope)
        finally:
            checker.required_classes = previous
        self.asserted: List[str] = []      # names asserted permanently, in order
        self.tags: Dict[str, z3.BoolRef] = {}
        self.status: Dict[str, str] = {}   # name -> 'encoded' | 'error'

    @staticmethod
    @contextlib.contextmanager
    def _quiet():
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            yield

    def _encode(self, cdict: Dict) -> List:
        proxy = _CollectingSolver(self.solver)
        with self._quiet():
            self.checker._encode_constraint_by_pattern_tracked(
                proxy, self.shared_vars, self.scope,
                cdict['name'], cdict.get('pattern', 'unknown'), cdict.get('context', ''), cdict.get('text', ''))
        return proxy.formulas

    def _assert_tracked(self, cdict: Dict) -> str:
        name = cdict['name']
        try:
            formulas = self._encode(cdict)
        except Exception as e:  # unencodable constraint: recorded, not asserted
            self.status[name] = 'error'
            self.checker.encoding_errors.append({'name': name, 'error': str(e)})
            return 'error'
        tag = z3.Bool(f"C_{name}")
        self.solver.assert_and_track(z3.And(*formulas) if formulas else z3.BoolVal(True), tag)
        self.tags[name] = tag
        self.status[name] = 'encoded'
        return 'encoded'

    def add_permanent(self, cdict: Dict) -> str:
        """Assert a constraint for the rest of the session. Returns 'encoded' or 'error'."""
        st = self._assert_tracked(cdict)
        if st == 'encoded':
            self.asserted.append(cdict['name'])
        return st

    def check(self) -> str:
        r = self.solver.check()
        return 'sat' if r == z3.sat else ('unsat' if r == z3.unsat else 'unknown')

    def unsat_core_names(self) -> List[str]:
        try:
            return [str(t)[2:] for t in self.solver.unsat_core() if str(t).startswith('C_')]
        except z3.Z3Exception:
            return []

    def try_candidate(self, cdict: Dict) -> Tuple[str, List[str]]:
        """Test one candidate against everything asserted so far, then retract it.
        Returns (verdict, core_names); verdict is 'sat', 'unsat', 'unknown' or 'error'."""
        self.solver.push()
        try:
            if self._assert_tracked(cdict) == 'error':
                return 'error', []
            verdict = self.check()
            core = self.unsat_core_names() if verdict == 'unsat' else []
            return verdict, core
        finally:
            self.solver.pop()
            self.tags.pop(cdict['name'], None)
