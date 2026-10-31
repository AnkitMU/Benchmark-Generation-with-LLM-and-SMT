#!/usr/bin/env python3
"""Build the paper's per-model evaluation table (LaTeX) from a with-VGCR root, a
no-VGCR root, and a JSON of per-model log statistics (tautology/contradiction/
inconsistency events, near-duplicate vetoes, wall-clock seconds, Phase B verdict).

Usage: python3 experiments/make_paper_table.py benchmarks/paper_v2/with_vgcr benchmarks/paper_v2/no_vgcr <logstats.json>
"""
import json, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor

vg, nv, stats = Path(sys.argv[1]), Path(sys.argv[2]), json.load(open(sys.argv[3]))
cfg = yaml.safe_load(open('examples/paper_ten_models_vgcr.yaml'))
xmi = {m['name']: m['xmi'] for m in cfg['models']}
order = ['CarRental', 'FlightBooking', 'SocialMedia', 'BankSystem', 'LibrarySystem', 'EcommerceSystem', 'UniversitySystem', 'ManufacturingPlant', 'HospitalSystem', 'IoTSensorNetwork']
rows = []
for n in order:
    r = json.load(open(vg / n / 'report.json')); fs = r['final_suite']; sa = fs['slot_accounting']
    mm = MetamodelExtractor(xmi[n]).get_metamodel()
    names = list(mm.classes.keys()) if isinstance(mm.classes, dict) else [c.name for c in mm.classes]
    nvp = nv / n / 'report.json'
    nvd = json.load(open(nvp))['final_suite']['delivered'] if nvp.exists() and 'final_suite' in json.load(open(nvp)) else None
    st = stats.get(n, {})
    rows.append(dict(model=n, cls=len(names), attr=sum(len(mm.get_attributes_for(c)) for c in names), pairs=r['semantic_admissibility']['candidate_pairs'],
                     filled=r['generated'], gen=fs['delivered'], novgcr=nvd, sat=fs['sat'], unsat=fs['unsat'],
                     taut=st.get('taut', 0), contr=st.get('contr', 0), inco=st.get('inconsistent', st.get('inco', 0)),
                     dup=sa['dedup_removed'] + st.get('veto', 0), time=st.get('time'), phaseB=st.get('phaseB', '?'),
                     inbox=fs['complexity_fidelity_sat']['in_box_rate']))
keys = ['cls', 'attr', 'pairs', 'filled', 'gen', 'novgcr', 'sat', 'unsat', 'taut', 'contr', 'inco', 'dup', 'time']
def cell(r, k):
    v = r[k]
    if v is None: return '--'
    return f"{v:.0f}" if k == 'time' else f"{v}"
L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l rrr rrr rr rrrr r}", r"\toprule",
     r"\textbf{Model} & \textbf{Class} & \textbf{Attr} & \textbf{Pairs} & \textbf{Filled} & \textbf{Gen} & \textbf{Gen$^{-}$} & \textbf{SAT} & \textbf{UNSAT} & \textbf{Taut} & \textbf{Contr} & \textbf{Inco} & \textbf{Dup} & \textbf{Time\,(s)} \\", r"\midrule"]
w = max(len(r['model']) for r in rows)
for r in rows:
    mark = r"$^{\dagger}$" if r['phaseB'] != 'globally consistent' else ''
    L.append(f"{(r['model'] + mark):<{w + 12}} & " + " & ".join(f"{cell(r, k):>5}" for k in keys) + r" \\")
avg = {k: (sum(r[k] for r in rows if r[k] is not None) / max(1, sum(1 for r in rows if r[k] is not None))) for k in keys}
L += [r"\midrule", r"\textbf{Average} & " + " & ".join(rf"\textbf{{{avg[k]:.1f}}}" for k in keys) + r" \\", r"\bottomrule", r"\end{tabular}"]
L.append(r"""\caption{Evaluation across $10$ metamodels ($100$ constraints requested per model, seed $42$). Class/Attr\,=\,classes and attributes in the metamodel; Pairs\,=\,same-typed attribute pairs analysed by the semantic analysis; Filled\,=\,slots filled by generation, out of $100$; Gen\,=\,constraints delivered with VGCR; Gen$^{-}$\,=\,constraints delivered with VGCR disabled, where candidates that fail verification are discarded; SAT/UNSAT\,=\,satisfiable and unsatisfiable constraints in the delivered suite; Taut/Contr/Inco\,=\,tautologies, contradictions, and joint inconsistencies caught by VGCR, counted per failed check across retries; Dup\,=\,near-duplicates removed before verification or rejected inside VGCR; Time\,=\,wall-clock generation and verification time per model with VGCR. All delivered SAT constraints lie inside their requested complexity level. $^{\dagger}$Joint consistency of the SAT constraints was not decided within the solver time limit at scope~6 (including the seed portfolio); each of its SAT constraints is individually verified.}""")
L += [r"\label{tab:models}", r"\end{table}"]
print("\n".join(L))
tot = {k: sum(r[k] for r in rows if r[k] is not None) for k in ('gen', 'novgcr', 'sat', 'unsat', 'taut', 'contr', 'inco', 'dup')}
print("% totals:", tot, "| in-box rates:", sorted({r['inbox'] for r in rows}), "| phaseB:", {r['model']: r['phaseB'] for r in rows if r['phaseB'] != 'globally consistent'})
