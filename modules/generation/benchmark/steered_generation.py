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
        # Whether a pair's draws are top-level conjunctions (=> the lowering
        # actuator can reduce an over-target component by dropping a conjunct).
        self.splittable: Dict[Tuple[str, str], bool] = {}
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
                from .complexity_calculator import _get_body
                seen: Set[str] = set()
                vecs: List[Vec] = []
                splittable = False
                for _ in range(self.spec.K):
                    c = self._instantiate(pid, cls)
                    if c is None or c.ocl in seen:
                        continue
                    seen.add(c.ocl)
                    vecs.append(self._measure(c.ocl, cls))
                    if not splittable and len(self._split_top_and(_get_body(c.ocl))) >= 2:
                        splittable = True
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
                    self.splittable[(pid, cls)] = splittable

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
            split = self.splittable.get((pid, cls), False)
            if any(self._box_reachable(v, box, cls, split) for v in vecs):
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
    # Max actuators stacked onto one candidate. A box that is below on several axes
    # at once (e.g. 'difficult' needs depth + iteration + operator load together)
    # cannot be reached by a single actuator, so _actuate composes up to this many.
    _MAX_COMPOSE: int = 4

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

    def _box_reachable(self, v: Vec, box: Box, cls: str, lowerable: bool = False) -> bool:
        """True if ``v`` is in ``box`` or can be moved into it by an available
        actuator: RAISED (below the lower bound only on raisable components), or
        LOWERED (above the upper bound only when ``lowerable`` -- the pair produces
        conjunctions a drop can reduce). Used by selection so a pair whose base
        draws sit just outside a box is still eligible."""
        raisable = self._actuators_for(cls)
        for comp, (lo, hi) in box.ranges.items():
            if v[comp] > hi and not lowerable:
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
        """A LIGHT boolean ``forAll`` term over a collection association: a single
        scalar predicate on a target attribute (``self.coll->forAll(v | v.attr >= 0)``).
        It raises cic by exactly one iteration level with minimal nnr_c/wno, so the
        compositional actuator can nudge a candidate into a moderate box without
        overshooting -- re-binding a *full* generated invariant here instead spikes
        cic to 3 and blows past the upper bounds (e.g. the 'difficult' box). None if
        ``cls`` has no collection whose target exposes a usable attribute."""
        assocs = self._collection_assocs(cls)
        random.shuffle(assocs)
        for a in assocs[:6]:
            ref = getattr(a, "ref_name", None)
            tgt = getattr(a, "target_class", None)
            if not ref or not tgt:
                continue
            pred = self.eng._attr_predicate(tgt, var="v")
            if pred:
                return f"self.{ref}->forAll(v | {pred})"
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

    @staticmethod
    def _split_top_and(body: str) -> List[str]:
        """Split a boolean body on top-level ' and ' (outside parentheses)."""
        parts: List[str] = []
        depth = 0
        start = 0
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch in "([":
                depth += 1
                i += 1
            elif ch in ")]":
                depth -= 1
                i += 1
            elif depth == 0:
                m = re.match(r"\s+and\s+", body[i:])
                if m:
                    parts.append(body[start:i].strip())
                    i += m.end()
                    start = i
                else:
                    i += 1
            else:
                i += 1
        parts.append(body[start:].strip())
        return [p for p in parts if p]

    def _drop_conjunct(self, c, cls: str, over: Set[str]):
        """Lowering actuator: if ``c``'s body is a top-level conjunction, drop the
        conjunct whose removal most reduces the over-target components, keeping a
        valid invariant. Returns a new OCLConstraint or None."""
        from dataclasses import replace
        from .complexity_calculator import _get_body
        parts = self._split_top_and(_get_body(c.ocl))
        if len(parts) < 2:
            return None
        best = None
        best_score = None
        for i in range(len(parts)):
            rest = parts[:i] + parts[i + 1:]
            nb = " and ".join(f"({p})" for p in rest)
            cand = replace(c, ocl=f"context {cls}\ninv: {nb}")
            score = sum(self._measure(cand.ocl, cls).get(comp, 0.0) for comp in over)
            if best_score is None or score < best_score:
                best_score = score
                best = cand
        return best

    # ---- Per-context joint-consistency (generation-time) -------------------
    @staticmethod
    def _strip_wrap(s: str) -> str:
        """Strip balanced WRAPPING parens (from `_split_top_and` conjuncts) without
        eating a trailing method-call '()'. '(self.x->isEmpty())' -> 'self.x->isEmpty()'."""
        s = s.strip()
        while len(s) >= 2 and s[0] == '(' and s[-1] == ')':
            depth, wrapped = 0, True
            for i, ch in enumerate(s):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and i != len(s) - 1:
                        wrapped = False
                        break
            if not wrapped:
                break
            s = s[1:-1].strip()
        return s

    @staticmethod
    def _bool_claim(conjunct: str):
        """If ``conjunct`` forces a boolean nav path true/false, return (path, bool).
        Recognises 'self.x = true/false', 'self.x <> true/false', and a bare boolean
        navigation 'self.x' (asserts true). None otherwise."""
        c = SteeredGenerator._strip_wrap(conjunct)
        m = re.match(r'self\.([\w.]+)\s*(=|<>)\s*(true|false)$', c)
        if m:
            val = (m.group(3) == 'true')
            if m.group(2) == '<>':
                val = not val
            return (m.group(1), val)
        if re.match(r'self\.[\w.]+$', c):     # bare boolean nav => asserts true
            return (c[5:], True)
        return None

    @staticmethod
    def _emptiness_claim(conjunct: str):
        """If ``conjunct`` forces a collection nav path empty/non-empty, return
        (path, 'EMPTY'|'NONEMPTY'). None otherwise."""
        m = re.match(r'self\.([\w.]+)->(.+)', SteeredGenerator._strip_wrap(conjunct))
        if not m:
            return None
        path, rest = m.group(1), m.group(2)
        if rest.startswith('isEmpty()'):
            return (path, 'EMPTY')
        if re.match(r'(notEmpty\(\)|exists\(|any\(|one\()', rest):
            return (path, 'NONEMPTY')
        # 'size()' appears at the start of rest for a direct 'self.coll->size()' and
        # after '->' for 'self.coll->collect(...)->size()'; match both.
        sm = re.search(r'(?:^|->)size\(\)\s*(>=|<=|>|<|=)\s*(\d+)', rest)
        if sm:
            op, n = sm.group(1), int(sm.group(2))
            if op == '>':
                return (path, 'NONEMPTY')                 # size > n (>= 0) -> non-empty
            if op == '>=':
                return (path, 'NONEMPTY') if n >= 1 else None
            if op == '=':
                return (path, 'EMPTY' if n == 0 else 'NONEMPTY')
            if op == '<':
                return (path, 'EMPTY') if n <= 1 else None   # size < 1 -> empty
            if op == '<=':
                return (path, 'EMPTY') if n == 0 else None   # size <= 0 -> empty
        # 'collect(...)->sum() >= N' (N > 0) requires a non-empty source collection.
        if re.search(r'->sum\(\)\s*(>=|>)\s*[1-9]', rest):
            return (path, 'NONEMPTY')
        return None

    @staticmethod
    def _mod_claim(conjunct: str):
        """If ``conjunct`` pins an attribute's PARITY via an even modulus, return
        (path, 'ODD'|'EVEN'). 'x mod 2 = 1' => ODD; 'x mod 4 = 0' => EVEN. An odd
        modulus does not pin parity, so returns None there."""
        m = re.match(r'self\.([\w.]+)\s+mod\s+(\d+)\s*=\s*(\d+)$',
                     SteeredGenerator._strip_wrap(conjunct))
        if m:
            mod, rem = int(m.group(2)), int(m.group(3))
            if mod % 2 == 0:                          # even modulus pins parity
                return (m.group(1), 'ODD' if rem % 2 else 'EVEN')
        return None

    _ALL_SIGNS = frozenset({'+', '-', '0'})
    _OP_SIGNS = {'>': frozenset({'+'}), '>=': frozenset({'+', '0'}),
                 '<': frozenset({'-'}), '<=': frozenset({'-', '0'}),
                 '=': frozenset({'0'}), '<>': frozenset({'+', '-'})}

    @staticmethod
    def _order_signs(conjunct: str):
        """Ordering between two attributes, as the allowed signs of (a - b). E.g.
        'self.a > self.b' -> ('a|b', {'+'}); 'self.a - self.b >= 10' -> ('a|b', {'+'}).
        The pair is normalised (alphabetical) with the sign-set flipped if swapped, so
        contradictory orderings on the same pair intersect to the empty set."""
        s = SteeredGenerator._strip_wrap(conjunct)
        m = re.match(r'self\.(\w+)\s*(>=|<=|<>|>|<|=)\s*self\.(\w+)$', s)
        if m:
            a, b, signs = m.group(1), m.group(3), set(SteeredGenerator._OP_SIGNS[m.group(2)])
        else:
            m = re.match(r'self\.(\w+)\s*-\s*self\.(\w+)\s*(>=|>)\s*(\d+)$', s)
            if not m:
                return None
            a, b, op, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
            signs = {'+'} if (k > 0 or op == '>') else {'+', '0'}   # a-b (>=k>0 | >k>=0) => a>b
        if a > b:
            flip = {'+': '-', '-': '+', '0': '0'}
            a, b, signs = b, a, {flip[x] for x in signs}
        return (f"{a}|{b}", frozenset(signs))

    @staticmethod
    def _quant_claim(conjunct: str):
        """forAll/exists/any over a collection with a BOOLEAN element body. Returns
        (coll, attr, 'A'|'E', boolval): 'A' = universal (forAll), 'E' = existential
        (exists/any). e.g. 'self.posts->forAll(x | x.isPublic <> true)' ->
        ('posts','isPublic','A',False)."""
        s = SteeredGenerator._strip_wrap(conjunct)
        s = re.sub(r'\s*<>\s*null$', '', s)          # 'any(...) <> null' tail
        m = re.match(
            r'self\.(\w+)->(forAll|exists|any)\(\s*(\w+)\s*\|\s*\3\.(\w+)\s*(=|<>)\s*(true|false)\s*\)$',
            s)
        if not m:
            return None
        val = (m.group(6) == 'true')
        if m.group(5) == '<>':
            val = not val
        return (m.group(1), m.group(4), 'A' if m.group(2) == 'forAll' else 'E', val)

    @staticmethod
    def _check_register(cls: str, ocl: str, claims: dict) -> bool:
        """Per-context joint-consistency gate (single source of truth, used by the
        generator at fill time and by the controller's post-generation pass). Extracts
        the claims a constraint makes on ``cls`` — boolean polarity, collection
        emptiness, mod parity, attribute ordering, and boolean quantifier bodies — and
        checks them against ``claims`` (mutated in place). Returns False (recording
        nothing) on any contradiction: a boolean forced true AND false, a collection
        forced empty AND non-empty, a number forced odd AND even, an ordering whose
        sign-set intersects to empty (a>b AND a<=b), or a collection whose elements are
        forced by forAll to one boolean value while exists requires the other
        (forAll(attr=false) vs exists(attr=true)). On no conflict, records and True."""
        from .complexity_calculator import _get_body
        eq: Dict[Tuple[str, str], Any] = {}        # exact-value claims
        orders: Dict[Tuple[str, str], frozenset] = {}  # ordering sign-sets
        quants = []                                # (coll, attr, 'A'|'E', boolval)
        for cj in SteeredGenerator._split_top_and(_get_body(ocl)):
            for tag, fn in (('b:', SteeredGenerator._bool_claim),
                            ('e:', SteeredGenerator._emptiness_claim),
                            ('p:', SteeredGenerator._mod_claim)):
                r = fn(cj)
                if r:
                    eq[(cls, tag + r[0])] = r[1]
            o = SteeredGenerator._order_signs(cj)
            if o:
                k = (cls, 'o:' + o[0])
                orders[k] = orders.get(k, SteeredGenerator._ALL_SIGNS) & o[1]
            q = SteeredGenerator._quant_claim(cj)
            if q:
                quants.append(q)
        for k, v in eq.items():
            if k in claims and claims[k] != v:
                return False
        merged = {}
        for k, sg in orders.items():
            inter = claims.get(k, SteeredGenerator._ALL_SIGNS) & sg
            if not inter:
                return False
            merged[k] = inter
        # quantifier-body conflicts: forAll pins every element's bool value; exists
        # requires at least one element of the other value -> contradiction.
        qa_new, qe_new = {}, {}
        for coll, attr, kind, val in quants:
            ka, ke = (cls, f'qa:{coll}.{attr}'), (cls, f'qe:{coll}.{attr}')
            if kind == 'A':
                if claims.get(ka, val) != val:
                    return False                      # forAll true AND forAll false
                if any(ev != val for ev in claims.get(ke, ())):
                    return False                      # forAll W vs exists V!=W
                qa_new[ka] = val
            else:
                if claims.get(ka, val) != val:
                    return False                      # exists V vs forAll W!=V
                qe_new.setdefault(ke, set()).add(val)
        claims.update(eq)
        claims.update(merged)
        claims.update(qa_new)
        for ke, vals in qe_new.items():
            claims[ke] = set(claims.get(ke, set())) | vals
        return True

    def _register_claims(self, cls: str, ocl: str) -> bool:
        """Gen-time wrapper over the shared consistency gate (uses self.claims)."""
        return self._check_register(cls, ocl, self.claims)

    def _raise_once(self, c, cls: str, box: Box, below: Set[str]):
        """Apply the first actuator whose result does NOT overshoot any box ceiling
        and that targets a still-below component, returning ``(new_constraint, kind)``
        or None. Order is deliberate: forAll-wrap FIRST -- its scalar body raises cic
        AND dn_ca together (one collection navigation), so the 'difficult' box is hit
        without a separate deepen-nav, whose +nnr_c quantum would breach the ceiling.
        Then deepen-nav (dn_ca), cast (tcc), conjoin (operator/navigation load). A
        candidate that overshoots is skipped so the next actuator is tried."""
        attempts = []
        if below & set(self._RAISE_BY_FORALL):
            t = self._forall_term(cls)
            attempts.append((self._conjoin_with(c, cls, t) if t else None, "forall"))
        if below & set(self._RAISE_BY_DEEPNAV):
            t = self._deepnav_term(cls)
            attempts.append((self._conjoin_with(c, cls, t) if t else None, "deepnav"))
        if below & set(self._RAISE_BY_CAST):
            t = self._cast_term(cls)
            attempts.append((self._conjoin_with(c, cls, t) if t else None, "cast"))
        if below & set(self._RAISE_BY_CONJOIN):
            attempts.append((self._conjoin(c, cls), "conjoin"))
        for r, kind in attempts:
            if r is None:
                continue
            v = self._measure(r.ocl, cls)
            if not any(v[comp] > hi for comp, (lo, hi) in box.ranges.items()):
                return r, kind
        return None

    def _actuate(self, c, cls: str, box: Box, gaps: Dict[str, float]):
        """Move ``c`` toward ``box``, returning ``(new_constraint, kind)`` or None.
        Overshoot -> lower (drop a conjunct). Otherwise RAISE by COMPOSING actuators:
        apply one for a still-below component, re-measure, and stack the next onto the
        result, until ``c`` is in ``box`` or no non-overshooting actuator applies. A
        single actuator cannot close a gap that is below on several axes at once (the
        'difficult' box needs depth + iteration + operator load together), so up to
        ``_MAX_COMPOSE`` are composed; ``kind`` is the '+'-joined chain."""
        above = {comp for comp, g in gaps.items() if g < 0.0}
        if above:                                       # over-target: lower
            r = self._drop_conjunct(c, cls, above)
            if r is not None:
                return r, "drop"
        # A box that MANDATES iteration (cic lower bound > 0) is best built forAll-FIRST
        # from a minimal seed: bolting the heavy forAll onto an already-complex base
        # overshoots nnr_c/wno (the base may already sit near their ceilings), so cic
        # can never be raised. gaps['cic'] > 0 means the base has too little iteration.
        if gaps.get("cic", 0.0) > 0.0:
            seed = self._iteration_seed(c, cls, box)
            if seed is not None:
                return seed
        cur, kinds = c, []
        for _ in range(self._MAX_COMPOSE):
            v = self._measure(cur.ocl, cls)
            if box.contains(v):
                break
            below = {comp for comp, g in self._gaps(box, v).items() if g > 0.0}
            if not below:
                break
            step = self._raise_once(cur, cls, box, below)
            if step is None:
                break
            cur, kinds = step[0], kinds + [step[1]]
        if not kinds:
            return None
        return cur, "+".join(kinds)

    def _iteration_seed(self, c, cls: str, box: Box):
        """Build an iteration constraint forAll-FIRST for a box that requires cic>=1.
        A scalar-bodied forAll over a collection supplies cic AND dn_ca cheaply; a
        minimal scalar arithmetic conjunct (and optional top-up conjuncts) raise
        wno/nnr_c into range. Returns ``(constraint, kind)`` in-box, or None. This
        replaces actuating an already-complex base, whose intrinsic nnr_c/wno leaves
        no headroom for the forAll's cost (e.g. arithmetic x difficult)."""
        from dataclasses import replace
        fa = self._forall_term(cls)
        if not fa:
            return None
        pred = self.eng._attr_predicate(cls)            # 'attr >= 0' (no prefix)
        expr = f"(self.{pred}) and ({fa})" if pred else fa
        cur = replace(c, ocl=f"context {cls}\ninv: {expr}")
        kinds = ["forall"]
        for _ in range(self._MAX_COMPOSE):
            v = self._measure(cur.ocl, cls)
            if box.contains(v):
                return cur, "+".join(kinds)
            below = {comp for comp, g in self._gaps(box, v).items() if g > 0.0}
            if not below:
                break
            step = self._raise_once(cur, cls, box, below)
            if step is None:
                break
            cur, kinds = step[0], kinds + [step[1]]
        v = self._measure(cur.ocl, cls)
        return (cur, "+".join(kinds)) if box.contains(v) else None

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
            split = self.splittable.get((pid, cls), False)
            if any(self._box_reachable(v, slot.box, cls, split) for v in vecs):
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
                ok = any(any(self._box_reachable(v, box, cls, self.splittable.get((pid, cls), False))
                             for v in vecs)
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
                if in_box and c.ocl not in H and self._register_claims(cls, c.ocl):
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
                        res = self._actuate(c, cls, active_box, gaps)
                        if res is not None:
                            ac, kind = res
                            act_budget -= 1
                            if ac.ocl not in A and ac.ocl not in H:
                                A.add(ac.ocl)
                                v2 = self._measure(ac.ocl, cls)
                                if active_box.contains(v2) and self._register_claims(cls, ac.ocl):
                                    suite.append((ac, slot.family, slot.polarity, cls,
                                                  v2, active_box.label + "+" + kind))
                                    H.add(ac.ocl)
                                    cap[cls] -= 1
                                    got[cls] += 1
                                    need[cls] = max(0, need[cls] - 1)
                                    use[pid] = use.get(pid, 0) + 1
                                    # one report row per composed actuator -> clean histogram
                                    for k in kind.split("+"):
                                        report.append(("actuator", k, cls, active_box.label))
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
        # Per-context claims accumulated from accepted constraints, used to reject a
        # candidate that would make the context's invariant set jointly UNSAT.
        self.claims: Dict[Tuple[str, str], Any] = {}

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
