#!/usr/bin/env python3
"""
Compute the headline evaluation numbers from benchmark report.json files.

Usage:
    python summarize_paper_run.py benchmarks/paper/with_vgcr [benchmarks/paper/no_vgcr]

Reads every <root>/<Model>/report.json and prints a per-model table plus
the aggregates quoted in the paper:
  * semantic admissibility: same-type attribute pairs rejected / candidate
  * complexity fidelity: delivered SAT constraints inside their requested box
  * VGCR replacement rate: refilled / (refilled + unfilled) failed slots
  * mean delivered suite size, with and without VGCR
Also writes <root>/paper_summary.json for each root.
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def load_root(root: Path):
    rows = []
    for rp in sorted(root.glob("*/report.json")):
        r = json.loads(rp.read_text())
        fs = r.get("final_suite") or {}
        sa = fs.get("slot_accounting") or {}
        cf = fs.get("complexity_fidelity_sat") or {}
        sem = r.get("semantic_admissibility") or {}
        opt = r.get("optimality") or {}
        rows.append({
            "model": r.get("model", rp.parent.name),
            "complete": bool(fs),
            "requested": r.get("target_size"),
            "generated": r.get("generated"),
            "delivered": fs.get("delivered"),
            "sat": fs.get("sat"),
            "unsat": fs.get("unsat"),
            "sat_in_box": cf.get("in_box"),
            "sat_boxed": cf.get("boxed_constraints"),
            "vgcr_enabled": r.get("vgcr_enabled"),
            "vgcr_refilled": sa.get("vgcr_regenerated"),
            "vgcr_unfilled": sa.get("vgcr_unfilled"),
            "dedup_removed": sa.get("dedup_removed"),
            "dedup_refilled": sa.get("dedup_refilled"),
            "dedup_unfilled": sa.get("dedup_unfilled"),
            "sem_candidate_pairs": sem.get("candidate_pairs"),
            "sem_comparable": sem.get("comparable_pairs"),
            "sem_rejected": sem.get("rejected_pairs"),
            "sem_source": sem.get("source"),
            "opt_bound": opt.get("assignment_upper_bound"),
            "opt_ratio": opt.get("ratio"),
        })
    return rows


def _sum(rows, k):
    vals = [r[k] for r in rows if r.get(k) is not None]
    return sum(vals) if vals else None


def aggregate(rows):
    comp = [r for r in rows if r["complete"]]
    n = len(comp)
    agg = {"models": len(rows), "complete_models": n}
    if n:
        agg["mean_delivered"] = round(sum(r["delivered"] for r in comp) / n, 1)
        agg["total_delivered"] = _sum(comp, "delivered")
        agg["total_sat"] = _sum(comp, "sat")
        agg["total_unsat"] = _sum(comp, "unsat")
        ib, bx = _sum(comp, "sat_in_box"), _sum(comp, "sat_boxed")
        agg["sat_in_box"] = ib; agg["sat_boxed"] = bx
        agg["sat_in_box_rate"] = round(ib / bx, 4) if bx else None
        rf, uf = _sum(comp, "vgcr_refilled") or 0, _sum(comp, "vgcr_unfilled") or 0
        agg["vgcr_failed_slots"] = rf + uf
        agg["vgcr_refilled"] = rf
        agg["vgcr_replacement_rate_pct"] = round(100 * rf / (rf + uf), 1) if rf + uf else None
        # Combined replacement rate: every candidate that failed a check
        # (duplicate/redundant OR a VGCR verification failure) and was
        # replaced by a same-family, same-level candidate.
        drm, drf = _sum(comp, "dedup_removed") or 0, _sum(comp, "dedup_refilled") or 0
        agg["dedup_removed"] = drm; agg["dedup_refilled"] = drf
        agg["failed_candidates_total"] = rf + uf + drm
        agg["failed_candidates_replaced"] = rf + drf
        agg["replacement_rate_all_failures_pct"] = (
            round(100 * (rf + drf) / (rf + uf + drm), 1) if (rf + uf + drm) else None)
        cand, rej = _sum(rows, "sem_candidate_pairs"), _sum(rows, "sem_rejected")
        agg["sem_candidate_pairs"] = cand; agg["sem_rejected_pairs"] = rej
        agg["sem_rejection_rate_pct"] = round(100 * rej / cand, 1) if cand else None
        ratios = [r["opt_ratio"] for r in comp if r["opt_ratio"] is not None]
        agg["mean_optimality_ratio"] = round(sum(ratios) / len(ratios), 3) if ratios else None
    return agg


def print_table(root, rows):
    print(f"\n== {root} ==")
    hdr = f"{'model':20} {'deliv':>5} {'sat':>4} {'unsat':>5} {'inbox':>7} {'vgcr r/u':>9} {'dedup rm/rf/uf':>15} {'pairs rej/cand':>15} {'N*':>4} {'ratio':>6}"
    print(hdr)
    for r in rows:
        if not r["complete"]:
            print(f"{r['model']:20} (incomplete: generated={r['generated']})"); continue
        print(f"{r['model']:20} {r['delivered']:>5} {r['sat']:>4} {r['unsat']:>5} "
              f"{str(r['sat_in_box'])+'/'+str(r['sat_boxed']):>7} "
              f"{str(r['vgcr_refilled'])+'/'+str(r['vgcr_unfilled']):>9} "
              f"{str(r['dedup_removed'])+'/'+str(r['dedup_refilled'])+'/'+str(r['dedup_unfilled']):>15} "
              f"{str(r['sem_rejected'])+'/'+str(r['sem_candidate_pairs']):>15} "
              f"{str(r['opt_bound']):>4} {str(r['opt_ratio']):>6}")


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 1
    results = {}
    for a in argv[1:]:
        root = Path(a)
        rows = load_root(root)
        agg = aggregate(rows)
        print_table(root, rows)
        print("aggregate:", json.dumps(agg, indent=1))
        (root / "paper_summary.json").write_text(json.dumps({"root": str(root), "models": rows, "aggregate": agg}, indent=2))
        results[str(root)] = agg
    if len(results) == 2:
        (a, b) = list(results.values())
        print("\n== headline ==")
        print(f"mean delivered: {a.get('mean_delivered')} vs {b.get('mean_delivered')}")
        print(f"VGCR replacement rate (verification failures only): {a.get('vgcr_replacement_rate_pct')}% of {a.get('vgcr_failed_slots')} failed slots")
        print(f"Replacement rate (verification + redundancy failures): {a.get('replacement_rate_all_failures_pct')}% "
              f"({a.get('failed_candidates_replaced')} of {a.get('failed_candidates_total')})")
        print(f"semantic rejection: {a.get('sem_rejected_pairs')} of {a.get('sem_candidate_pairs')} pairs ({a.get('sem_rejection_rate_pct')}%)")
        print(f"SAT in-box: {a.get('sat_in_box')}/{a.get('sat_boxed')} ({a.get('sat_in_box_rate')})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
