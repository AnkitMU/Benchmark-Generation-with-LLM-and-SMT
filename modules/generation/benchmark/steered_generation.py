"""
Profile-Guided Complexity Steering with Component-Wise Measured Feedback (alg:ccs).

In-architecture realization. Complexity is always MEASURED off the generated OCL
via ``compute_total_complexity``; acceptance is gated on the measured nine-component
vector. The fixed-template pattern library exposes two steering actuators, which
this module uses in place of literal nav-depth/nesting/#vars knobs:

  * directed binding selection  -> re-drawn parameters on each attempt
  * profile-guided pattern+class switching for template-fixed components

Polarity is realized by restricting UNSAT slots to UNSAT-prone patterns; the
*intended* polarity is verified downstream by VGCR (deviations are reported).

The nine components (RUC/dependency is suite-level and handled by VGCR, not here):
  structural leaves : nnr_c, wnc, dn_ca, vrc, wno, wnm
  computational     : oc, tcc, cic
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .engine_v2 import BenchmarkEngineV2, classify_family
from .complexity_calculator import compute_total_complexity
from .metadata_enricher import get_complexity_weights

STRUCT: Tuple[str, ...] = ("nnr_c", "wnc", "dn_ca", "vrc", "wno", "wnm")
COMPUT: Tuple[str, ...] = ("oc", "tcc", "cic")
COMPONENTS: Tuple[str, ...] = STRUCT + COMPUT

Vec = Dict[str, float]
SKIP_BOX = object()  # sentinel for the "skip" policy active box (accept any in-family)


# ──────────────────────────────────────────────────────────────────────────
# Specification
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Box:
    """Per-component target ranges; components absent from ``ranges`` are free."""
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    label: str = ""

    def contains(self, v: Vec) -> bool:
        return all(lo <= v[c] <= hi for c, (lo, hi) in self.ranges.items())

    def mid(self, c: str) -> float:
        lo, hi = self.ranges[c]
        return 0.5 * (lo + hi)

    def width(self, c: str) -> float:
        lo, hi = self.ranges[c]
        return max(1.0, hi - lo)


@dataclass
class Slot:
    family: str
    polarity: str          # "sat" | "unsat"
    box: Box


@dataclass
class SteeringSpec:
    n: int
    sat_ratio: float = 0.8
    family_quota: Dict[str, float] = field(default_factory=dict)      # family -> %
    per_class: Tuple[int, int] = (0, 10 ** 6)                         # (L, U)
    profiles: List[Tuple[Box, float]] = field(default_factory=list)   # (box, %)
    weights: Optional[Any] = None
    # budgets / knobs
    K: int = 12            # profiling samples per (pattern, class)
    T: int = 24            # tries per slot
    T0: int = 6            # stall limit before switching pair
    lam: float = 0.25      # diversity penalty
    mu: float = 0.5        # class-demand weight (skip mode)
    eps: float = 2.0       # near-box tolerance (normalized distance)
    h: int = 5             # selection breadth (sample among best-h)
    deltas: Optional[Dict[str, float]] = None  # relax widths per component
    policy: str = "report"  # report | relax | skip


def _largest_remainder(weights: Dict[Any, float], total: int) -> Dict[Any, int]:
    """Integer allocation summing exactly to ``total`` (largest-remainder method)."""
    keys = list(weights.keys())
    pos = {k: max(0.0, weights[k]) for k in keys}
    s = sum(pos.values()) or 1.0
    raw = {k: total * pos[k] / s for k in keys}
    out = {k: int(raw[k]) for k in keys}
    rem = total - sum(out.values())
    order = sorted(keys, key=lambda k: raw[k] - out[k], reverse=True)
    for k in order[:max(0, rem)]:
        out[k] += 1
    return out


# ──────────────────────────────────────────────────────────────────────────
# Steered generator
# ──────────────────────────────────────────────────────────────────────────
class SteeredGenerator:
    def __init__(self, engine: BenchmarkEngineV2, spec: SteeringSpec):
        self.eng = engine
        self.mm = engine.metamodel
        self.spec = spec
        self.w = spec.weights or get_complexity_weights()
        self.w_s = float(getattr(self.w, "structural_weight", 1.0))
        self.w_c = float(getattr(self.w, "computational_weight", 1.0))
        self.pat_by_id = {p.id: p for p in engine.all_patterns
                          if p.id not in engine.EXCLUDE_LIST}
        self.unsat_ids: Set[str] = set(getattr(engine, "UNSAT_LIKELY_PATTERNS", set()))
        self.classes: List[str] = list(self.mm.get_class_names())
        self.prof: Dict[Tuple[str, str], List[Vec]] = {}
        self.fam_of: Dict[str, str] = {}

    # ---- engine primitives -------------------------------------------------
    def _family(self, pid: str) -> str:
        p = self.pat_by_id[pid]
        return classify_family(pid, getattr(p.category, "value", str(p.category)))

    def _applicable(self, pid: str, cls: str) -> bool:
        try:
            return bool(self.eng._is_pattern_applicable(self.pat_by_id[pid], cls))
        except Exception:
            return True

    def _instantiate(self, pid: str, cls: str):
        p = self.pat_by_id[pid]
        try:
            params = self.eng._gen_params(p, cls)
            c = self.eng.generator.generate(pid, cls, params)
        except Exception:
            return None
        if not c or not getattr(c, "ocl", None):
            return None
        return c

    def _measure(self, ocl: str, cls: str) -> Vec:
        r = compute_total_complexity(ocl, metamodel=self.mm, context_class=cls,
                                     all_constraints=None, weights=self.w)
        return {k: float(getattr(r, k)) for k in COMPONENTS}

    # ---- Phase 0: profiling ------------------------------------------------
    def profile(self) -> None:
        for pid in self.pat_by_id:
            self.fam_of[pid] = self._family(pid)
            for cls in self.classes:
                if not self._applicable(pid, cls):
                    continue
                seen: Set[str] = set()
                vecs: List[Vec] = []
                for _ in range(self.spec.K):
                    c = self._instantiate(pid, cls)
                    if c is None or c.ocl in seen:
                        continue
                    seen.add(c.ocl)
                    vecs.append(self._measure(c.ocl, cls))
                if vecs:
                    self.prof[(pid, cls)] = vecs

    # ---- distances / gaps --------------------------------------------------
    def _dist(self, box: Box, v: Vec) -> float:
        s = 0.0
        for c, (lo, hi) in box.ranges.items():
            rho = self.w_s if c in STRUCT else self.w_c
            s += rho * abs(v[c] - 0.5 * (lo + hi)) / max(1.0, hi - lo)
        return s

    def _gaps(self, box: Box, v: Vec) -> Dict[str, float]:
        """Normalized signed per-component gaps; empty dict => in box."""
        g: Dict[str, float] = {}
        for c, (lo, hi) in box.ranges.items():
            if v[c] < lo:
                g[c] = (lo - v[c]) / max(1.0, hi - lo)
            elif v[c] > hi:
                g[c] = (hi - v[c]) / max(1.0, hi - lo)
        return g

    def _norm(self, g: Dict[str, float]) -> float:
        return sum((self.w_s if c in STRUCT else self.w_c) * abs(val)
                   for c, val in g.items())

    def _reachable(self, pid: str, cls: str, box: Box) -> Set[str]:
        vecs = self.prof.get((pid, cls), [])
        return {c for c, (lo, hi) in box.ranges.items()
                if any(lo <= v[c] <= hi for v in vecs)}

    # ---- Phase 1: SelectPair ----------------------------------------------
    def _eligible(self, family: str, polarity: str, excl: Set[Tuple[str, str]]):
        # Polarity is a SOFT preference, not a hard filter: UNSAT-prone patterns are
        # favoured for unsat slots via the score, but any in-family pattern may fill
        # the slot as an *intended*-unsat candidate (verified later by VGCR).
        for (pid, cls), vecs in self.prof.items():
            if (pid, cls) in excl or self.fam_of[pid] != family:
                continue
            yield pid, cls, vecs

    def _polarity_bonus(self, pid: str, polarity: str) -> float:
        """Negative score bonus that biases unsat slots toward UNSAT-prone patterns."""
        if polarity == "unsat" and pid in self.unsat_ids:
            return -1.0
        return 0.0

    def select_pair(self, family: str, polarity: str, box, excl,
                    cap, need, use) -> Optional[Tuple[str, str]]:
        has_cap = lambda cls: cap.get(cls, 0) > 0

        if box is SKIP_BOX:  # box-free selection (skip policy)
            cands = [(pid, cls) for pid, cls, _ in self._eligible(family, polarity, excl)
                     if has_cap(cls)]
            if not cands:
                return None
            need_c = [pc for pc in cands if need.get(pc[1], 0) > 0]
            pool = need_c or cands
            pool.sort(key=lambda pc: self.spec.lam * use.get(pc[0], 0)
                      - self.spec.mu * need.get(pc[1], 0)
                      + self._polarity_bonus(pc[0], polarity))
            return random.choice(pool[:max(1, min(len(pool), self.spec.h))])

        c_in: List[Tuple[str, str, float]] = []
        c_near: List[Tuple[str, str, float]] = []
        for pid, cls, vecs in self._eligible(family, polarity, excl):
            if not has_cap(cls):
                continue
            dmin = min(self._dist(box, v) for v in vecs)
            if any(box.contains(v) for v in vecs):
                c_in.append((pid, cls, dmin))
            elif dmin <= self.spec.eps:
                c_near.append((pid, cls, dmin))
        cands = c_in or c_near
        if not cands:
            return None
        need_c = [t for t in cands if need.get(t[1], 0) > 0]
        pool = need_c or cands
        pool.sort(key=lambda t: t[2] + self.spec.lam * use.get(t[0], 0)
                  + self._polarity_bonus(t[0], polarity))
        pid, cls, _ = random.choice(pool[:max(1, min(len(pool), self.spec.h))])
        return pid, cls

    # ---- ApplyFeedback (in-architecture): redraw vs switch -----------------
    def _exhausted(self, pid: str, cls: str, box: Box, gaps: Dict[str, float]) -> bool:
        """True when the most-violated component cannot be moved into the box by
        re-drawing this pattern's bindings (it is template-fixed out of range)."""
        if not gaps:
            return False
        jstar = max(gaps, key=lambda c: (self.w_s if c in STRUCT else self.w_c) * abs(gaps[c]))
        return jstar not in self._reachable(pid, cls, box)

    # ---- ApplyPolicy -------------------------------------------------------
    def apply_policy(self, box: Box):
        pol = self.spec.policy
        if pol == "report":
            return None
        if pol == "skip":
            return SKIP_BOX
        # relax: widen each range by <= delta_j, clamp lower bound at 0, keep integer
        deltas = self.spec.deltas or {c: 1.0 for c in COMPONENTS}
        new = {}
        for c, (lo, hi) in box.ranges.items():
            d = deltas.get(c, 1.0)
            new[c] = (max(0.0, round(lo - d)), round(hi + d))
        return Box(new, label=box.label + "+relax")

    # ---- Phase 0: Compile (marginal-preserving joint slots) ----------------
    def _compile(self, fam_q, prof_q, n_sat) -> List[Slot]:
        fam_bag: List[str] = []
        for fam, q in fam_q.items():
            fam_bag += [fam] * q
        box_bag: List[Box] = []
        for idx, q in prof_q.items():
            box_bag += [self.spec.profiles[idx][0]] * q
        pol_bag = ["sat"] * n_sat + ["unsat"] * (self.spec.n - n_sat)
        random.shuffle(fam_bag); random.shuffle(box_bag); random.shuffle(pol_bag)
        slots = [Slot(f, s, b) for f, s, b in zip(fam_bag, pol_bag, box_bag)]
        # scarcity-first: fewest supporting (p,X) pairs first
        slots.sort(key=lambda sl: self._support(sl))
        return slots

    def _support(self, slot: Slot) -> int:
        n = 0
        for pid, cls, vecs in self._eligible(slot.family, slot.polarity, set()):
            if any(slot.box.contains(v) for v in vecs):
                n += 1
        return n

    def _support_feasible(self, fam_q, prof_q, n_sat) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        L, U = self.spec.per_class
        nc = len(self.classes)
        if L * nc > self.spec.n:
            reasons.append(f"per-class min {L} x {nc} classes > N={self.spec.n}")
        if U * nc < self.spec.n:
            reasons.append(f"per-class max {U} x {nc} classes < N={self.spec.n}")
        # at least one supporting pair per (family, box) that has quota
        for fam, fq in fam_q.items():
            if fq == 0:
                continue
            for idx, pq in prof_q.items():
                if pq == 0:
                    continue
                box = self.spec.profiles[idx][0]
                ok = any(any(box.contains(v) for v in vecs)
                         for pid, cls, vecs in self._eligible(fam, "sat", set()))
                if not ok:
                    reasons.append(f"no SAT pattern supports family '{fam}' x profile '{box.label or idx}'")
        return (len(reasons) == 0), reasons

    # ---- per-slot fill loop ------------------------------------------------
    def _fill_slot(self, slot, cap, got, need, use, H, A, suite, report) -> bool:
        box = slot.box
        excl: Set[Tuple[str, str]] = set()
        pair = self.select_pair(slot.family, slot.polarity, box, excl, cap, need, use)
        if pair is None:
            box2 = self.apply_policy(box)
            if box2 is None:
                report.append(("unfilled", slot.family, slot.polarity, box.label))
                return False
            box = box2
            report.append(("policy", self.spec.policy, slot.family, box.label))
            pair = self.select_pair(slot.family, slot.polarity, box, excl, cap, need, use)
            if pair is None:
                report.append(("unfilled", slot.family, slot.polarity, box.label))
                return False

        pid, cls = pair
        stall = dup = 0
        g_prev = float("inf")
        active_box = box  # Box or SKIP_BOX
        for _ in range(self.spec.T):
            c = self._instantiate(pid, cls)
            switch = False
            if c is None or c.ocl in A:
                # failed draw or duplicate of an already-attempted candidate:
                # count as lack of progress so deterministic patterns switch out.
                dup += 1
                switch = dup >= self.spec.T0
            else:
                A.add(c.ocl)
                dup = 0
                v = self._measure(c.ocl, cls)
                in_box = (active_box is SKIP_BOX) or active_box.contains(v)
                if in_box and c.ocl not in H:
                    blabel = "skip" if active_box is SKIP_BOX else active_box.label
                    suite.append((c, slot.family, slot.polarity, cls, v, blabel))
                    H.add(c.ocl)
                    cap[cls] -= 1
                    got[cls] += 1
                    need[cls] = max(0, need[cls] - 1)
                    use[pid] = use.get(pid, 0) + 1
                    return True
                if active_box is not SKIP_BOX:
                    gaps = self._gaps(active_box, v)
                    gn = self._norm(gaps)
                    stall = stall + 1 if gn >= g_prev else 0
                    g_prev = gn
                    switch = self._exhausted(pid, cls, active_box, gaps) or stall >= self.spec.T0
            if switch:
                excl.add((pid, cls))
                nxt = self.select_pair(slot.family, slot.polarity, active_box,
                                       excl, cap, need, use)
                if nxt is None:
                    break
                pid, cls = nxt
                stall = dup = 0
                g_prev = float("inf")
        report.append(("unreached", slot.family, slot.polarity,
                       box.label if box is not SKIP_BOX else "skip"))
        return False

    # ---- Phase 6: minima repair by replacement -----------------------------
    def _repair_minima(self, got, need, cap, use, H, suite, report):
        L, _ = self.spec.per_class
        for X in self.classes:
            if got.get(X, 0) >= L:
                continue
            # find a constraint of an over-min class Y to replace
            for i, (c, fam, pol, cls_y, v, blabel) in enumerate(suite):
                if cls_y == X or got.get(cls_y, 0) <= L:
                    continue
                # regenerate the same (family, polarity) slot for X (box best-effort)
                pair = self.select_pair(fam, pol, SKIP_BOX, set(), {X: 1}, {X: 0}, use)
                if pair is None:
                    continue
                pid, _ = pair
                newc = self._instantiate(pid, X)
                if newc is None or newc.ocl in H:
                    continue
                # swap
                H.discard(c.ocl); H.add(newc.ocl)
                got[cls_y] -= 1; got[X] += 1
                cap[cls_y] += 1; cap[X] -= 1
                need[X] = max(0, need[X] - 1)
                suite[i] = (newc, fam, pol, X, self._measure(newc.ocl, X), "repair")
                break
            if got.get(X, 0) < L:
                report.append(("below_min", X, got.get(X, 0), L))

    # ---- entry point -------------------------------------------------------
    def run(self) -> Tuple[List[Any], List[Any]]:
        spec = self.spec
        report: List[Any] = []
        fam_q = _largest_remainder(spec.family_quota, spec.n)
        prof_q = _largest_remainder({i: pct for i, (_b, pct) in enumerate(spec.profiles)}, spec.n)
        n_sat = round(spec.sat_ratio * spec.n)

        ok, reasons = self._support_feasible(fam_q, prof_q, n_sat)
        if not ok:
            report.append(("support_infeasible", reasons))

        slots = self._compile(fam_q, prof_q, n_sat)
        L, U = spec.per_class
        cap = {c: U for c in self.classes}
        got = {c: 0 for c in self.classes}
        need = {c: L for c in self.classes}
        use: Dict[str, int] = {}
        suite: List[Any] = []
        H: Set[str] = set()
        A: Set[str] = set()

        for slot in slots:
            self._fill_slot(slot, cap, got, need, use, H, A, suite, report)
        self._repair_minima(got, need, cap, use, H, suite, report)
        return suite, report
