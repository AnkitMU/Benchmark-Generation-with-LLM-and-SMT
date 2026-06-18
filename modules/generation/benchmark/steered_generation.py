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
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .engine_v2 import BenchmarkEngineV2, classify_family
from .complexity_calculator import compute_total_complexity
from .metadata_enricher import get_complexity_weights

STRUCT: Tuple[str, ...] = ("nnr_c", "wnc", "dn_ca", "vrc", "wno", "wnm")
COMPUT: Tuple[str, ...] = ("oc", "tcc", "cic")
COMPONENTS: Tuple[str, ...] = STRUCT + COMPUT

# The five components a user may set in a complexity profile's ``ranges``.
# These are the independently controllable axes (measurement-grounded): the
# others are either redundant or coupled to these and must not be user-pinned.
USER_SETTABLE: Tuple[str, ...] = ("nnr_c", "dn_ca", "wno", "cic", "tcc")
# The four engine-reported components: present in the measured nine-vector and
# in the deviation report, but NOT accepted as input.
DERIVED_REPORTED: Tuple[str, ...] = ("oc", "wnc", "vrc", "wnm")
REPORTED_COMPONENTS: Tuple[str, ...] = DERIVED_REPORTED  # public alias
_DERIVED_REASON: Dict[str, str] = {
    "oc":  "equals wno under the current operator weights",
    "wnc": "follows from nnr_c and dn_ca",
    "vrc": "follows from cic (iterators bind variables)",
    "wnm": "follows from wno and cic",
}


def validate_complexity_profiles(profiles: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Validate user-supplied ``complexity_profiles`` against the five-knob model.

    Only the five independently controllable components may appear in a
    profile's ``ranges``:

        nnr_c  navigation breadth   dn_ca  navigation depth
        wno    operator load        cic    collection iteration
        tcc    type conversions

    The four derived components (oc, wnc, vrc, wnm) are engine-reported and
    are rejected as input. Each range must be ``[lo, hi]`` with
    ``0 <= lo <= hi``. Shares (``pct``) should sum to 100 (a non-fatal warning
    is emitted otherwise; the engine normalises by largest remainder).

    Returns ``profiles`` unchanged on success; raises :class:`ValueError` with
    an actionable message otherwise.
    """
    if not profiles:
        return profiles
    total_pct = 0.0
    for i, pr in enumerate(profiles):
        label = pr.get("label") if isinstance(pr, dict) else None
        where = f"complexity_profiles[{i}]" + (f" ('{label}')" if label else "")
        if not isinstance(pr, dict):
            raise ValueError(f"{where}: each profile must be a mapping with 'pct' and 'ranges'")

        pct = pr.get("pct", pr.get("share"))
        if pct is None:
            raise ValueError(f"{where}: missing 'pct' (share of the suite)")
        try:
            total_pct += float(pct)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: 'pct' must be a number, got {pct!r}")

        ranges = pr.get("ranges") or {}
        if not isinstance(ranges, dict):
            raise ValueError(f"{where}: 'ranges' must be a mapping component -> [lo, hi]")
        for comp, rng in ranges.items():
            if comp in DERIVED_REPORTED:
                raise ValueError(
                    f"{where}: '{comp}' is engine-reported, not user-settable "
                    f"({_DERIVED_REASON.get(comp, 'coupled to the settable components')}). "
                    f"Set only {list(USER_SETTABLE)}."
                )
            if comp not in USER_SETTABLE:
                raise ValueError(
                    f"{where}: unknown complexity component '{comp}'. "
                    f"Settable components are {list(USER_SETTABLE)}."
                )
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                raise ValueError(f"{where}: range for '{comp}' must be [lo, hi], got {rng!r}")
            try:
                lo, hi = float(rng[0]), float(rng[1])
            except (TypeError, ValueError):
                raise ValueError(f"{where}: range bounds for '{comp}' must be numbers, got {rng!r}")
            if lo < 0 or hi < 0:
                raise ValueError(f"{where}: range for '{comp}' must be non-negative, got {rng!r}")
            if lo > hi:
                raise ValueError(f"{where}: range for '{comp}' has lo > hi: {rng!r}")

    if abs(total_pct - 100.0) > 1e-6:
        warnings.warn(
            f"complexity_profiles 'pct' values sum to {total_pct:g}, not 100; "
            f"shares will be normalised.",
            stacklevel=2,
        )
    return profiles

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
        # Footprints (built during profile()): per (pattern, class) the measured
        # per-component envelope, and the capability set of components the pair
        # can drive above zero. These index the discrete pattern space by box.
        self.fp: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
        self.cap: Dict[Tuple[str, str], Set[str]] = {}
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
                    fp: Dict[str, Tuple[float, float]] = {}
                    cap: Set[str] = set()
                    for comp in COMPONENTS:
                        col = [v[comp] for v in vecs]
                        fp[comp] = (min(col), max(col))
                        if max(col) > 0.0:
                            cap.add(comp)
                    self.fp[(pid, cls)] = fp
                    self.cap[(pid, cls)] = cap

    # ---- Footprints: reachability envelope & box feasibility ---------------
    def reachable_envelope(self) -> Dict[str, Tuple[float, float]]:
        """Union per-component ``[min, max]`` over all profiled pairs: the region
        of the nine-component space the library can actually reach under the
        current binding sampling. A user box outside this envelope is infeasible."""
        env: Dict[str, Tuple[float, float]] = {}
        for fp in self.fp.values():
            for comp, (lo, hi) in fp.items():
                if comp in env:
                    env[comp] = (min(env[comp][0], lo), max(env[comp][1], hi))
                else:
                    env[comp] = (lo, hi)
        return env

    @staticmethod
    def _supports(fp: Dict[str, Tuple[float, float]], cap: Set[str], box: Box) -> bool:
        """True if this pair's footprint can satisfy every constrained component of
        ``box``: the footprint range must intersect the target, and a positive
        lower bound requires the component to be reachable (in ``cap``)."""
        for comp, (blo, bhi) in box.ranges.items():
            flo, fhi = fp.get(comp, (0.0, 0.0))
            if fhi < blo or flo > bhi:          # footprint disjoint from target range
                return False
            if blo > 0.0 and comp not in cap:   # prerequisite construct absent
                return False
        return True

    def supported_pairs(self, box: Box) -> List[Tuple[str, str]]:
        """Profiled (pattern, class) pairs whose footprint can reach ``box``."""
        return [key for key, fp in self.fp.items()
                if self._supports(fp, self.cap.get(key, set()), box)]

    def validate_boxes(self, boxes: List[Tuple[Box, float]]) -> List[Any]:
        """Feasibility of each requested box against the reachable footprints.
        Emits a deviation entry for any box no pair can reach, naming the
        blocking components (or all components when the conflict is joint)."""
        env = self.reachable_envelope()
        any_cap = set().union(*self.cap.values()) if self.cap else set()
        out: List[Any] = []
        for box, _share in boxes:
            if not box.ranges or self.supported_pairs(box):
                continue
            blocking: List[str] = []
            for comp, (blo, bhi) in box.ranges.items():
                elo, ehi = env.get(comp, (0.0, 0.0))
                if ehi < blo or elo > bhi or (blo > 0.0 and comp not in any_cap):
                    blocking.append(comp)
            # empty `blocking` => each component is individually reachable but no
            # single pair reaches them jointly.
            out.append(("box_unreachable", box.label,
                        blocking or sorted(box.ranges), "joint" if not blocking else "component"))
        return out

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
            if any(self._box_reachable(v, box, cls) for v in vecs):
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

    # ---- Actuators: structural transforms layered on a base candidate -------
    # The fixed templates reach only a small region of the nine-component space;
    # an actuator rewrites a base candidate to move a deficient component toward
    # its target. CONJOIN (implemented) raises operator/navigation load
    # (wno, oc, nnr_c, wnc, wnm) by AND-ing a second in-class invariant onto the
    # base. The structural actuators -- forAll-wrap (raises cic, vrc) and
    # cast-insert (raises tcc) -- are specified for future work and currently
    # return None, so the loop falls back to binding re-draw / pattern switching
    # for those components. Polarity, redundancy, and joint consistency of any
    # actuated candidate are checked downstream by VGCR.
    # Components each actuator raises.
    _RAISE_BY_CONJOIN: Tuple[str, ...] = ("wno", "oc", "nnr_c", "wnc", "wnm")
    _RAISE_BY_FORALL:  Tuple[str, ...] = ("cic", "vrc")
    _RAISE_BY_CAST:    Tuple[str, ...] = ("tcc",)
    _RAISE_BY_DEEPNAV: Tuple[str, ...] = ("dn_ca",)
    _NUMERIC_HINTS = ("int", "real", "float", "double", "long", "number")

    def _collection_assocs(self, cls: str):
        try:
            return list(self.mm.get_collection_associations(cls))
        except Exception:
            return []

    def _single_assocs(self, cls: str):
        try:
            return list(self.mm.get_single_associations(cls))
        except Exception:
            return []

    def _attrs(self, cls: str):
        try:
            return list(self.mm.get_attributes_for(cls))
        except Exception:
            return []

    def _actuators_for(self, cls: str) -> Set[str]:
        """Components raisable by an actuator available for this class."""
        raisable = set(self._RAISE_BY_CONJOIN)        # conjoin is always available
        if self._collection_assocs(cls):              # forAll-wrap / cast need a collection
            raisable |= set(self._RAISE_BY_FORALL) | set(self._RAISE_BY_CAST)
        if self._single_assocs(cls):                  # deepen-navigation needs a single assoc
            raisable |= set(self._RAISE_BY_DEEPNAV)
        return raisable

    def _box_reachable(self, v: Vec, box: Box, cls: str) -> bool:
        """True if ``v`` is in ``box`` or can be RAISED into it by an available
        actuator (below the lower bound only on raisable components; nothing above
        the upper bound, since no lowering actuator exists yet). Used by selection
        so a pair whose base draws sit below a box is still eligible."""
        raisable = self._actuators_for(cls)
        for comp, (lo, hi) in box.ranges.items():
            if v[comp] > hi:
                return False
            if v[comp] < lo and comp not in raisable:
                return False
        return True

    def _conjoin_with(self, c, cls: str, term: str):
        """Build ``(base) and (term)`` as a new OCLConstraint, or None."""
        from dataclasses import replace
        from .complexity_calculator import _get_body
        base = _get_body(c.ocl)
        if not term or term == base:
            return None
        return replace(c, ocl=f"context {cls}\ninv: ({base}) and ({term})")

    def _conjoin(self, c1, cls: str):
        """Conjoin a second distinct in-class invariant (raises operator/nav load)."""
        from .complexity_calculator import _get_body
        applicable = [pid for pid in self.pat_by_id if self._applicable(pid, cls)]
        random.shuffle(applicable)
        for pid in applicable[:8]:
            c2 = self._instantiate(pid, cls)
            if c2 is None:
                continue
            r = self._conjoin_with(c1, cls, _get_body(c2.ocl))
            if r is not None:
                return r
        return None

    def _forall_term(self, cls: str) -> Optional[str]:
        """A boolean ``forAll`` term over a collection association (raises cic, vrc,
        wnm), formed by re-binding a generated element-type invariant to the
        iterator variable. None if ``cls`` has no collection association."""
        from .complexity_calculator import _get_body
        assocs = self._collection_assocs(cls)
        random.shuffle(assocs)
        for a in assocs[:6]:
            ref = getattr(a, "ref_name", None)
            tgt = getattr(a, "target_class", None)
            if not ref or not tgt:
                continue
            pids = [pid for pid in self.pat_by_id if self._applicable(pid, tgt)]
            random.shuffle(pids)
            for pid in pids[:6]:
                c2 = self._instantiate(pid, tgt)
                if c2 is None:
                    continue
                body = _get_body(c2.ocl)
                if not body or "self" not in body:
                    continue
                body_v = re.sub(r"\bself\b", "v", body)
                return f"self.{ref}->forAll(v | {body_v})"
        return None

    def _cast_term(self, cls: str) -> Optional[str]:
        """A boolean term using a type-conversion op over a collection association
        (raises tcc via asSet). None if ``cls`` has no collection association."""
        for a in self._collection_assocs(cls):
            ref = getattr(a, "ref_name", None)
            if ref:
                return f"self.{ref}->asSet()->size() >= 0"
        return None

    def _deepnav_term(self, cls: str) -> Optional[str]:
        """A boolean term over a length-2 (sometimes 3) navigation chain (raises
        dn_ca, and nnr_c/wnc), built from single-valued associations plus an
        attribute of the reached class. None if no such chain exists."""
        singles = self._single_assocs(cls)
        random.shuffle(singles)
        for a in singles[:6]:
            ref1 = getattr(a, "ref_name", None)
            t1 = getattr(a, "target_class", None)
            if not ref1 or not t1:
                continue
            prefix, tgt = f"self.{ref1}", t1                    # depth 2
            for b in self._single_assocs(t1)[:3]:               # optionally extend to depth 3
                ref2 = getattr(b, "ref_name", None)
                t2 = getattr(b, "target_class", None)
                if ref2 and t2 and random.random() < 0.5:
                    prefix, tgt = f"self.{ref1}.{ref2}", t2
                    break
            attrs = self._attrs(tgt)
            random.shuffle(attrs)
            for at in attrs[:6]:
                an = getattr(at, "name", None)
                if not an:
                    continue
                ty = str(getattr(at, "type", "")).lower()
                if any(k in ty for k in self._NUMERIC_HINTS):
                    return f"{prefix}.{an} >= 0"
                return f"{prefix}.{an} = {prefix}.{an}"          # type-agnostic, still deepens
        return None

    def _actuate(self, c, cls: str, gaps: Dict[str, float]):
        """Apply the actuator for a deficient (below-target) component: deepen-nav
        for dn_ca, forAll-wrap for cic/vrc, cast for tcc, conjoin for operator/
        navigation load. Returns a new OCLConstraint or None."""
        below = {comp for comp, g in gaps.items() if g > 0.0}
        if below & set(self._RAISE_BY_DEEPNAV):
            t = self._deepnav_term(cls)
            if t:
                return self._conjoin_with(c, cls, t)
        if below & set(self._RAISE_BY_FORALL):
            t = self._forall_term(cls)
            if t:
                return self._conjoin_with(c, cls, t)
        if below & set(self._RAISE_BY_CAST):
            t = self._cast_term(cls)
            if t:
                return self._conjoin_with(c, cls, t)
        if below & set(self._RAISE_BY_CONJOIN):
            return self._conjoin(c, cls)
        return None

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
            if any(self._box_reachable(v, slot.box, cls) for v in vecs):
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
                ok = any(any(self._box_reachable(v, box, cls) for v in vecs)
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
        act_budget = 3            # max conjoin-actuator attempts for this slot
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
                    # Actuator: if the candidate is BELOW target on operator/
                    # navigation load, conjoin it with another in-class invariant
                    # to raise that load, then re-measure and accept if in box.
                    if act_budget > 0:
                        ac = self._actuate(c, cls, gaps)
                        if ac is not None:
                            act_budget -= 1
                            if ac.ocl not in A and ac.ocl not in H:
                                A.add(ac.ocl)
                                v2 = self._measure(ac.ocl, cls)
                                if active_box.contains(v2):
                                    suite.append((ac, slot.family, slot.polarity, cls,
                                                  v2, active_box.label + "+conjoin"))
                                    H.add(ac.ocl)
                                    cap[cls] -= 1
                                    got[cls] += 1
                                    need[cls] = max(0, need[cls] - 1)
                                    use[pid] = use.get(pid, 0) + 1
                                    report.append(("actuator", "conjoin", cls, active_box.label))
                                    return True
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
        # Footprint feasibility (Phase 2): record the library's reachable envelope
        # and flag any requested box no profiled pattern can reach.
        report.append(("reachable_envelope", self.reachable_envelope()))
        report.extend(self.validate_boxes(spec.profiles))
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

        # Surface the four engine-reported (derived) components explicitly:
        # they are MEASURED, never user-set. Report their realised [min, max]
        # across the accepted suite so the deviation report shows what the
        # coupled components came out to be.
        derived: Dict[str, Tuple[float, float]] = {}
        for comp in DERIVED_REPORTED:
            vals = [t[4][comp] for t in suite
                    if len(t) > 4 and isinstance(t[4], dict) and comp in t[4]]
            if vals:
                derived[comp] = (min(vals), max(vals))
        if derived:
            report.append(("reported_derived_components", derived))

        return suite, report
