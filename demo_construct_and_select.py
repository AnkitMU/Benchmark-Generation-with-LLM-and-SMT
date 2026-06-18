#!/usr/bin/env python3
"""
Standalone DEMO of the construct-and-select benchmark generation mechanism.

This file does NOT modify the existing engine. It reuses the project's real
`core.models.Metamodel` and `complexity_calculator` (as the measuring tape)
to demonstrate the proposed mechanism end-to-end:

  Phase 1  CONSTRUCT constraints from building-block RECIPES (templates x params)
           -- no TC steering, no running-average nudging.
  Phase 2  MEASURE each constructed constraint EXACTLY with complexity_calculator.
  Phase 3  FILE each under its MEASURED difficulty tier (Str, Comp, Dep, TC).
  Phase 4  STRATIFIED SELECTION: fill the user's per-tier quotas exactly from the
           measured pool; report any shortfall HONESTLY (no random padding).

Run:  PYTHONPATH=. python3 demo_construct_and_select.py
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, cycle
from typing import Dict, Iterator, List, Tuple

from modules.core.models import Metamodel, Class, Attribute, Association
from modules.generation.benchmark.complexity_calculator import (
    ComplexityWeights, compute_total_complexity, compute_ruc, tc_to_difficulty_label,
)

TIERS = ["trivial", "easy", "medium", "hard", "expert"]


# --------------------------------------------------------------------------
# A small but real metamodel (CarRental domain)
# --------------------------------------------------------------------------
def build_metamodel() -> Metamodel:
    def I(n): return Attribute(n, "Integer")
    def R(n): return Attribute(n, "Real")
    def B(n): return Attribute(n, "Boolean")
    def S(n): return Attribute(n, "String")

    customer = Class("Customer",
        attributes=[I("age"), I("loyaltyScore"), I("riskScore"), S("name")],
        associations=[
            Association("c_rentals", "rentals", "Customer", "Rental", lower=0, upper=-1),
            Association("c_cars", "cars", "Customer", "Car", lower=0, upper=-1),
            Association("c_primary", "primaryCar", "Customer", "Car", lower=1, upper=1),
        ])
    rental = Class("Rental",
        attributes=[I("startMileage"), I("endMileage"), I("days"), R("cost"), B("paid")],
        associations=[
            Association("r_car", "car", "Rental", "Car", lower=1, upper=1),
        ])
    car = Class("Car",
        attributes=[I("year"), I("mileage"), I("seats"), R("price")],
        associations=[
            Association("car_services", "services", "Car", "Service", lower=0, upper=-1),
        ])
    service = Class("Service",
        attributes=[R("cost"), I("durationHours")],
        associations=[])
    return Metamodel(classes={c.name: c for c in [customer, rental, car, service]})


# --------------------------------------------------------------------------
# Recipe -> OCL constructor (templates x building-block params).
# Each generator yields (template_name, recipe_desc, ocl_body).
# Building blocks vary: #comparisons, nav depth, #iterators, nesting.
# --------------------------------------------------------------------------
def numeric_attrs(mm, cls):
    return [a.name for a in mm.get_attributes_for(cls) if a.type in ("Integer", "Real")]


def bool_attrs(mm, cls):
    return [a.name for a in mm.get_attributes_for(cls) if a.type == "Boolean"]


_C = cycle([0, 1, 18, 100, 1900])


def gen_recipes(mm, ctx) -> Iterator[Tuple[str, str, str]]:
    nattrs = numeric_attrs(mm, ctx)
    singles = mm.get_single_associations(ctx)
    colls = mm.get_collection_associations(ctx)

    # flat comparisons vs constants: 1..4 comparisons (low complexity)
    for k in range(1, 5):
        if nattrs:
            picks = [nattrs[i % len(nattrs)] for i in range(k)]
            body = " and ".join(f"self.{a} > {next(_C)}" for a in picks)
            yield ("flat_const", f"{k}x cmp(const)", body)

    # flat attribute-attribute comparisons: 1..3 DISTINCT pairs
    pairs = list(combinations(nattrs, 2))
    for k in range(1, 4):
        chosen = pairs[:k]
        if chosen:
            terms = [f"self.{a} > self.{b}" for a, b in chosen]
            yield ("flat_pair", f"{len(terms)}x cmp(attr-attr)", " and ".join(terms))

    # single-navigation comparisons (depth 2): 1..2 cmp through each single assoc
    for sa in singles:
        tn = numeric_attrs(mm, sa.target_class)
        for k in range(1, 3):
            if tn:
                picks = [tn[i % len(tn)] for i in range(k)]
                body = " and ".join(f"self.{sa.ref_name}.{a} > {next(_C)}" for a in picks)
                yield ("single_nav", f"nav:{sa.ref_name} {k}x cmp", body)

    # forAll over a collection: 1..3 DISTINCT predicates (one iterator)
    for ca in colls:
        tn = numeric_attrs(mm, ca.target_class)
        atoms = [f"v.{a} < v.{b}" for a, b in combinations(tn, 2)] + [f"v.{a} > 0" for a in tn]
        for preds in range(1, 4):
            chosen = atoms[:preds]
            if len(chosen) == preds:
                body = f"self.{ca.ref_name}->forAll(v | {' and '.join(chosen)})"
                yield ("forAll", f"forAll:{ca.ref_name} {preds}preds", body)

    # exists over a collection (one iterator, cheap predicate)
    for ca in colls:
        bn = bool_attrs(mm, ca.target_class)
        tn = numeric_attrs(mm, ca.target_class)
        pred = f"v.{bn[0]}" if bn else (f"v.{tn[0]} > 0" if tn else None)
        if pred:
            yield ("exists", f"exists:{ca.ref_name}", f"self.{ca.ref_name}->exists(v | {pred})")

    # select -> forAll (two chained iterators)
    for ca in colls:
        tn = numeric_attrs(mm, ca.target_class)
        if len(tn) >= 2:
            body = (f"self.{ca.ref_name}->select(v | v.{tn[0]} > 0)"
                    f"->forAll(v | v.{tn[0]} < v.{tn[1]})")
            yield ("select_forAll", f"select->forAll:{ca.ref_name}", body)

    # nested iterators (depth 2): collection whose target has its own collection
    for ca in colls:
        for cb in mm.get_collection_associations(ca.target_class):
            itn = numeric_attrs(mm, cb.target_class)
            if itn:
                yield ("nested",
                       f"forAll:{ca.ref_name}/exists:{cb.ref_name}",
                       f"self.{ca.ref_name}->forAll(a | a.{cb.ref_name}->exists(b | b.{itn[0]} > 0))")
                if len(itn) >= 2:
                    yield ("nested+",
                           f"forAll:{ca.ref_name}/exists:{cb.ref_name} +pred",
                           f"self.{ca.ref_name}->forAll(a | a.{cb.ref_name}->"
                           f"exists(b | b.{itn[0]} > 0 and b.{itn[1]} < 100))")


# --------------------------------------------------------------------------
# Pool: construct + measure
# --------------------------------------------------------------------------
@dataclass
class Built:
    ocl: str
    context: str
    template: str
    recipe: str
    structural: float = 0.0
    computational: float = 0.0
    dependency: float = 0.0
    tc: float = 0.0
    tier: str = ""


def construct_pool(mm) -> List[Built]:
    seen, pool = set(), []
    for ctx in mm.get_class_names():
        for tname, rdesc, body in gen_recipes(mm, ctx):
            ocl = f"context {ctx} inv: {body}"
            if ocl in seen:
                continue
            seen.add(ocl)
            pool.append(Built(ocl=ocl, context=ctx, template=tname, recipe=rdesc))
    return pool


def measure_pool(mm, pool: List[Built], weights: ComplexityWeights):
    # Tier on the SUITE-INDEPENDENT complexity (Str + Comp). Dependency (RUC)
    # is intentionally deferred: it is a property of the FINAL suite, not the
    # throwaway pool, so we pass all_constraints=None here (RUC = 0).
    for b in pool:
        res = compute_total_complexity(b.ocl, metamodel=mm, context_class=b.context,
                                       all_constraints=None, weights=weights)
        b.structural = round(res.structural_total, 2)
        b.computational = round(res.computational_total, 2)
        b.dependency = 0.0                       # set later, on the selected suite
        b.tc = round(res.tc, 2)                  # = w_s*Str + w_c*Comp (Dep deferred)
        b.tier = tc_to_difficulty_label(res.tc)


def measure_suite_dependency(suite: List[Built]) -> int:
    """RUC is emergent: measure each constraint's reuse against the SELECTED suite."""
    total = 0
    for b in suite:
        r = compute_ruc(b.ocl, b.context, suite)
        b.dependency = float(r)
        total += r
    return total


# --------------------------------------------------------------------------
# Phase 4: stratified selection
# --------------------------------------------------------------------------
def quotas(mix: Dict[str, int], N: int) -> Dict[str, int]:
    q = {t: round(N * mix.get(t, 0) / 100) for t in TIERS}
    drift = N - sum(q.values())
    if drift:
        biggest = max(TIERS, key=lambda t: mix.get(t, 0))
        q[biggest] += drift
    return q


def pick_spread(bucket: List[Built], k: int) -> List[Built]:
    """Pick k items spread across the bucket's complexity range (diversity)."""
    if k <= 0:
        return []
    if k >= len(bucket):
        return list(bucket)
    s = sorted(bucket, key=lambda b: (b.tc, b.structural, b.computational))
    if k == 1:
        return [s[len(s) // 2]]
    idx = sorted(set(round(i * (len(s) - 1) / (k - 1)) for i in range(k)))
    i = 0
    while len(idx) < k:           # top up if rounding collided
        if i not in idx:
            idx.append(i)
        i += 1
    return [s[j] for j in sorted(idx)[:k]]


def stratified_select(pool, q):
    buckets = {t: [b for b in pool if b.tier == t] for t in TIERS}
    selected, shortfall = {}, {}
    for t in TIERS:
        take = pick_spread(buckets[t], q[t])
        selected[t] = take
        shortfall[t] = max(0, q[t] - len(take))
    return buckets, selected, shortfall


# --------------------------------------------------------------------------
# Demo driver + report
# --------------------------------------------------------------------------
def main():
    mm = build_metamodel()
    weights = ComplexityWeights()

    N = 20
    mix = {"trivial": 10, "easy": 25, "medium": 30, "hard": 25, "expert": 10}

    print("=" * 70)
    print(" CONSTRUCT-AND-SELECT BENCHMARK GENERATION  —  DEMO")
    print("=" * 70)

    print("\nMetamodel (CarRental):")
    for cn in mm.get_class_names():
        attrs = ", ".join(a.name for a in mm.get_attributes_for(cn))
        assocs = ", ".join(f"{a.ref_name}{'*' if a.is_collection else ''}->{a.target_class}"
                           for a in mm.get_associations_for(cn))
        print(f"  {cn:9} attrs[{attrs}]  assoc[{assocs or '-'}]")

    print(f"\nUser spec:   N={N}   difficulty_mix={mix}")
    print("Generation:  building-block recipes (templates) — NO TC steering.")
    print("Tiering:     on measured Str+Comp (Dep deferred to the final suite).")

    # Phase 1-3
    pool = construct_pool(mm)
    measure_pool(mm, pool, weights)
    hist = {t: sum(1 for b in pool if b.tier == t) for t in TIERS}
    print(f"\nPhase 1-3   constructed & MEASURED pool: {len(pool)} unique constraints")
    print("            pool tier histogram:  " + "   ".join(f"{t}={hist[t]}" for t in TIERS))

    # Phase 4
    q = quotas(mix, N)
    buckets, selected, shortfall = stratified_select(pool, q)

    print("\nPhase 4     stratified selection (fill exact per-tier quotas)")
    print(f"   {'tier':8}{'target':>7}{'pool':>6}{'selected':>10}{'shortfall':>11}")
    tot_t = tot_p = tot_s = tot_sh = 0
    for t in TIERS:
        sel = len(selected[t])
        tot_t += q[t]; tot_p += len(buckets[t]); tot_s += sel; tot_sh += shortfall[t]
        print(f"   {t:8}{q[t]:>7}{len(buckets[t]):>6}{sel:>10}{shortfall[t]:>11}")
    print(f"   {'TOTAL':8}{tot_t:>7}{tot_p:>6}{tot_s:>10}{tot_sh:>11}")

    if tot_sh == 0:
        print("\n   ==> EXACT MATCH: every per-tier quota met by construction + selection.")
    else:
        print(f"\n   ==> {tot_sh} slot(s) UNFILLABLE on this metamodel — reported as an")
        print("       honest feasibility gap (NOT padded with random constraints).")

    # Phase 5: dependency (RUC) is emergent — measure it on the FINAL selected
    # suite (not the over-generated pool).
    suite = [b for t in TIERS for b in selected[t]]
    total_dep = measure_suite_dependency(suite)
    reused = sum(1 for b in suite if b.dependency > 0)
    print(f"\nPhase 5     suite-level dependency (RUC) measured on the {len(suite)} selected")
    print(f"            constraints: total reuse links={total_dep}, "
          f"sharing navigations={reused}/{len(suite)}")

    print("\nSample SELECTED constraint per tier (complexity is MEASURED, not steered):")
    for t in TIERS:
        if selected[t]:
            b = selected[t][0]
            print(f"\n  [{t}]  Str={b.structural}  Comp={b.computational}  "
                  f"TC(Str+Comp)={b.tc}  suiteDep={b.dependency}")
            print(f"        recipe = {b.template} ({b.recipe})")
            print(f"        {b.ocl}")
        else:
            print(f"\n  [{t}]  (no constraint at this tier — feasibility gap)")

    # Show the (Str, Comp) spread that produced the tiers — the real control axes
    print("\nPer-tier measured (Str, Comp) ranges in the selected suite:")
    for t in TIERS:
        if selected[t]:
            ss = [b.structural for b in selected[t]]
            cc = [b.computational for b in selected[t]]
            print(f"  {t:8} Str[{min(ss):.1f}-{max(ss):.1f}]  Comp[{min(cc):.1f}-{max(cc):.1f}]")


if __name__ == "__main__":
    main()
