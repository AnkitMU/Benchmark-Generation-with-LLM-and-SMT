#!/usr/bin/env python3
"""Run all 10 metamodels through the steered benchmark pipeline and emit the
results LaTeX table.

Usage (from the repo root, same environment as generate_benchmark_suite.py —
needs z3, and ollama running if SEMANTIC is left on):

    python make_results_table.py                 # run all 10 models, print LaTeX
    python make_results_table.py > results.tex   # save the table
    python make_results_table.py --no-semantic   # skip the LLM semantic filter
    python make_results_table.py --reuse         # don't re-run; build table from
                                                 # existing benchmarks_table/ output

Column mapping (each value is read from the per-model run, not hard-coded):
    Cls    number of classes in the metamodel
    Attr   total number of attributes across all classes
    Gen    constraints generated that entered VGCR   (vgcr_stats.total_candidates)
    Verif  constraints that passed VGCR verification (vgcr_stats.passed)
    Taut   tautologies caught       (failure_breakdown.tautology)
    Cont   contradictions caught    (failure_breakdown.contradiction)
    Redn   redundant caught         (failure_breakdown.redundant)
    Inco   inconsistent caught      (failure_breakdown.inconsistent)
    Retry  total VGCR refinement retries (vgcr_stats.total_retries)
    Time   wall-clock seconds for the whole per-model run (measured here)
"""
import os
import sys
import json
import time
import copy
import subprocess

import yaml  # PyYAML (already a dependency of the suite config loader)

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor  # noqa: E402

# (display name, xmi path) in the table's row order.
MODELS = [
    ("CarRental",          "models/car_rental.xmi"),
    ("FlightBooking",      "models/flight_booking.xmi"),
    ("SocialMedia",        "models/social_media.xmi"),
    ("BankSystem",         "models/bank.xmi"),
    ("LibrarySystem",      "models/library_system.xmi"),
    ("EcommerceSystem",    "models/ecommerce_system.xmi"),
    ("UniversitySystem",   "models/university_system.xmi"),
    ("ManufacturingPlant", "models/manufacturing_plant.xmi"),
    ("HospitalSystem",     "models/hospital.xmi"),
    ("IoTSensorNetwork",   "models/iotsensornetwork.xmi"),
]

BASE_CONFIG = os.path.join(REPO, "examples/example_suite_steered.yaml")
OUT_ROOT = os.path.join(REPO, "benchmarks_table")
NO_SEMANTIC = "--no-semantic" in sys.argv
REUSE = "--reuse" in sys.argv


def metamodel_size(xmi):
    mm = MetamodelExtractor(os.path.join(REPO, xmi)).get_metamodel()
    classes = mm.get_class_names()
    attrs = sum(len(mm.get_attributes_for(c)) for c in classes)
    return len(classes), attrs


def _summary_path(name, suite_name):
    return os.path.join(OUT_ROOT, name, f"{suite_name}_summary.json")


def run_model(name, xmi):
    """Run one model end-to-end, return (stats_dict, suite_name, elapsed_seconds)."""
    base = yaml.safe_load(open(BASE_CONFIG))
    suite_name = base.get("suite_name", "OCL-Steered-FiveKnob")
    cfg = copy.deepcopy(base)
    cfg["models"] = [{
        "xmi": xmi,
        "name": name,
        "profiles": base["models"][0]["profiles"],   # reuse the base profile(s)
    }]
    cfg["output_root"] = os.path.join("benchmarks_table", name) + os.sep
    tmp = os.path.join("/tmp", f"results_cfg_{name}.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    elapsed = 0.0
    if not REUSE:
        cmd = [sys.executable, "generate_benchmark_suite.py", "--config", tmp, "--quiet"]
        if NO_SEMANTIC:
            cmd.append("--no-semantic")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            sys.stderr.write(f"  [warn] {name}: exit {proc.returncode}\n")
            sys.stderr.write("  " + (proc.stderr or proc.stdout or "")[-500:] + "\n")
    return suite_name, elapsed


def collect_stats(name, suite_name, elapsed):
    s = dict(gen=0, verif=0, taut=0, cont=0, redn=0, inco=0, retry=0, time=elapsed)
    try:
        d = json.load(open(_summary_path(name, suite_name)))
        v = d["models"][0]["profiles"][0].get("vgcr_stats", {}) or {}
        fb = v.get("failure_breakdown", {}) or {}
        s.update(
            gen=v.get("total_candidates", 0),
            verif=v.get("passed", 0),
            taut=fb.get("tautology", 0),
            cont=fb.get("contradiction", 0),
            redn=fb.get("redundant", 0),
            inco=fb.get("inconsistent", 0),
            retry=v.get("total_retries", 0),
        )
    except Exception as e:  # pragma: no cover - defensive
        sys.stderr.write(f"  [warn] {name}: could not read stats ({e})\n")
    return s


def emit_latex(rows):
    # rows: list of (name, cls, attr, gen, verif, taut, cont, redn, inco, retry, time)
    n = len(rows)
    def col(i):
        return [r[i] for r in rows]
    def avg(i):
        return sum(col(i)) / n if n else 0.0

    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\begin{tabular}{l rr rr rrrr rr}")
    out.append(r"\toprule")
    out.append(r"\textbf{Model} & \textbf{Cls} & \textbf{Attr} "
               r"& \textbf{Gen} & \textbf{Verif} "
               r"& \textbf{Taut} & \textbf{Cont} & \textbf{Redn} & \textbf{Inco} "
               r"& \textbf{Retry} & \textbf{Time\,(s)} \\")
    out.append(r"\midrule")
    for (name, cls, attr, gen, verif, taut, cont, redn, inco, retry, t) in rows:
        out.append(
            f"{name:<18} & {cls:>2d} & {attr:>3d}  & {gen:>3d} & {verif:>3d}  "
            f"& {taut:>2d} & {cont:>2d} & {redn:>2d} & {inco:>2d}  "
            f"& {retry:>3d} & {t:>6.0f} \\\\"
        )
    out.append(r"\midrule")
    out.append(
        r"\textbf{Average}   & \textbf{%.1f} & \textbf{%.1f}"
        r"  & \textbf{%.1f} & \textbf{%.1f}"
        r"  & \textbf{%.1f} & \textbf{%.1f} & \textbf{%.1f} & \textbf{%.1f}"
        r"  & \textbf{%.1f} & \textbf{%.0f} \\"
        % (avg(1), avg(2), avg(3), avg(4), avg(5), avg(6), avg(7), avg(8), avg(9), avg(10))
    )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\caption{Generation and VGCR verification statistics across ten metamodels.}")
    out.append(r"\label{tab:results}")
    out.append(r"\end{table}")
    print("\n".join(out))


def main():
    rows = []
    for name, xmi in MODELS:
        sys.stderr.write(f"[{len(rows)+1}/{len(MODELS)}] {name} ...\n")
        sys.stderr.flush()
        cls, attr = metamodel_size(xmi)
        suite_name, elapsed = run_model(name, xmi)
        s = collect_stats(name, suite_name, elapsed)
        rows.append((name, cls, attr, s["gen"], s["verif"],
                     s["taut"], s["cont"], s["redn"], s["inco"], s["retry"], s["time"]))
        sys.stderr.write(
            f"      Cls={cls} Attr={attr} Gen={s['gen']} Verif={s['verif']} "
            f"Taut={s['taut']} Cont={s['cont']} Redn={s['redn']} Inco={s['inco']} "
            f"Retry={s['retry']} Time={s['time']:.0f}s\n"
        )
        sys.stderr.flush()
    sys.stderr.write("\n=== LaTeX table ===\n")
    emit_latex(rows)


if __name__ == "__main__":
    main()
