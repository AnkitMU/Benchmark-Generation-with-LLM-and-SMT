#!/usr/bin/env python3
"""
Validate a delivered benchmark suite with the USE tool (UML-based Specification
Environment) and its Kodkod Model Validator, and compare the verdicts with the
suite's SAT/UNSAT labels.

Usage:
    python3 experiments/use_comparison/run_use_comparison.py \
        --use-home ~/Downloads/use-6.0.0 --model models/car_rental.xmi \
        --suite benchmarks/paper/with_vgcr/CarRental --out experiments/use_comparison/results/CarRental \
        [--scope 6] [--real-as-integer] [--core]

Steps: (1) generate a .use specification from the XMI; (2) write one spec per
constraint (USE spells `a mod b` as `a.mod(b)`); (3) write Model Validator
configurations with explicit integer value sets drawn from the suite's literals,
`scope` instances per class, and at least one instance of the context class;
(4) run every individual validation plus a joint validation of all transformable
SAT constraints (all classes populated) in ONE USE session; (5) parse verdicts,
unsupported operations and times; (6) optionally (--core) compute a minimal
jointly-unsatisfiable subset by sequential elimination.

Known limits of Model Validator 5.2.0 observed with USE 6.0.0: string operations
(size on strings, toLower/toUpper, indexOf) cannot be transformed; Real-typed
attributes make every comparison trivially unsatisfiable. --real-as-integer
retypes Real attributes as Integer, which preserves verdicts when no constraint
uses division or fractional literals (the script checks and reports this).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TYPES = {'estring': 'String', 'eint': 'Integer', 'elong': 'Integer', 'edouble': 'Real', 'efloat': 'Real', 'eboolean': 'Boolean'}
STRING_OPS = re.compile(r"\.(size|toLowerCase|toUpperCase|toLower|toUpper|indexOf|concat|substring|at|characters)\(")
# USE 6.0.0 / Model Validator 5.2.0 treats ->any(...) as undefined whenever more than
# one element satisfies the predicate (verified on a 3-invariant probe), so a joint
# validation that includes such constraints can be spuriously UNSATISFIABLE. They are
# validated individually but excluded from the joint run and reported separately.
ANY_OP = re.compile(r"->any\(")
# Attribute or role names that are USE keywords are suffixed with "_" in the
# specification and in every constraint body (`self.model` -> `self.model_`).
USE_KEYWORDS = {"model", "class", "end", "association", "attributes", "operations", "constraints",
                "context", "inv", "pre", "post", "enum", "abstract", "composition", "aggregation",
                "between", "role", "ordered", "statemachines", "let", "in", "if", "then", "else", "endif"}
def safe_name(n): return n + "_" if n in USE_KEYWORDS else n
def rename_in_body(body, names):
    for n in names: body = re.sub(rf"\.{n}\b(?!\()", f".{n}_", body)
    return body


def load_metamodel(xmi):
    from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
    mm = MetamodelExtractor(xmi).get_metamodel()
    names = list(mm.classes.keys()) if isinstance(mm.classes, dict) else [c.name for c in mm.classes]
    return mm, names


def use_spec(mm, names, xmi, model_name, real_as_integer):
    enums, cur = {}, None
    for line in open(xmi):
        m = re.search(r'<eClassifiers xsi:type="ecore:EEnum" name="(\w+)"', line)
        if m: cur = m.group(1); enums[cur] = []
        m = re.search(r'<eLiterals name="(\w+)"', line)
        if m and cur: enums[cur].append(m.group(1))
    out = [f"model {model_name}", ""]
    for e, lits in enums.items(): out.append(f"enum {e} {{{', '.join(lits)}}}")
    out.append("")
    for n in names:
        out += [f"class {n}", "attributes"]
        for a in mm.get_attributes_for(n):
            t = TYPES.get(a.type.lower(), a.type)
            if real_as_integer and t == 'Real': t = 'Integer'
            out.append(f"  {safe_name(a.name)} : {t}")
        out += ["end", ""]
    def mult(lo, up): return f"[{lo}]" if (lo == up and up != -1) else f"[{lo}..{'*' if up == -1 else up}]"
    seen, assocs = set(), []
    for a in mm.get_all_associations():
        if a.opposite_ref:
            opp_cls, opp_ref = a.opposite_ref.split('/')[-2:]
            key = frozenset([(a.source_class, a.ref_name), (opp_cls, opp_ref)])
            if key in seen: continue
            seen.add(key)
            b = next(x for x in mm.get_all_associations() if x.source_class == opp_cls and x.ref_name == opp_ref)
            back_role, back_mult = safe_name(b.ref_name), mult(b.lower, b.upper)
        else:
            # Unidirectional reference: USE needs a second end. Give it a unique
            # generated role name and an unbounded multiplicity, which adds no
            # constraint beyond what the Ecore reference expresses.
            key = (a.source_class, a.ref_name)
            if key in seen: continue
            seen.add(key)
            back_role, back_mult = f"{a.source_class[0].lower()}{a.source_class[1:]}_{a.ref_name}_src", "[0..*]"
        name = f"{a.source_class}_{a.ref_name}"; assocs.append(name)
        out += [f"association {name} between", f"  {a.source_class}{back_mult} role {back_role}",
                f"  {a.target_class}{mult(a.lower, a.upper)} role {safe_name(a.ref_name)}", "end", ""]
    return "\n".join(out), assocs


def to_use_ocl(body):
    return re.sub(r"(self\.\w+) mod (\d+)", r"\1.mod(\2)", body)


def literal_domain(bodies):
    lits = set()
    for b in bodies:
        lits.update(int(x) for x in re.findall(r"(?<![\w.])-?\d+(?![\w.])", b))
    # Contiguous small range (covers mod/multiple-of constraints such as
    # quantity.mod(7) = 0 and quantity.mod(9) = 0 with 0 < quantity < 96), plus
    # the immediate neighbourhood of every larger literal. Keeping the domain
    # small matters: Kodkod translation time grows quickly with the atom count.
    top = min(max(lits | {25}), 128)
    ints = set(range(-1, top + 2))
    for v in lits:
        if v > top: ints |= {v - 1, v, v + 1}
    return sorted(ints)


def properties(classes, assocs, ints, scope, context=None, joint=False):
    L = ["[default]", "Integer = Set{" + ",".join(map(str, ints)) + "}", "String_min = 0", "String_max = 6", ""]
    for c in classes: L += [f"{c}_min = {1 if (joint or c == context) else 0}", f"{c}_max = {scope}"]
    for a in assocs: L += [f"{a}_min = 0", f"{a}_max = {scope * scope}"]
    return "\n".join(L) + "\n"


def run_use(use_home, spec, commands, log_path):
    """Run one headless USE session over a command file; return the full log."""
    cmd_path = log_path.with_suffix('.cmd'); cmd_path.write_text("\n".join(commands) + "\nquit\n")
    out = subprocess.run([f"{use_home}/bin/use", "-nogui", "-nr", spec, str(cmd_path)],
                         capture_output=True, text=True, timeout=3600).stdout
    log_path.write_text(out); return out


def parse_runs(log, cmd_name):
    """Split a session log at each `open` echo and parse the validator verdict that follows."""
    runs = {}
    chunks = re.split(rf"{re.escape(cmd_name)}> open (\S+)", log)[1:]
    for i in range(0, len(chunks), 2):
        spec_path, body = chunks[i], chunks[i + 1]
        v = re.search(r"INFO: (TRIVIALLY_UNSATISFIABLE|UNSATISFIABLE|SATISFIABLE)", body)
        t = re.search(r"Translation time \(Kodkod to SAT\): (\d+) ms; Solving time: (\d+) ms", body)
        err = re.search(r"OCL operation (\w+) is not supported", body)
        runs[spec_path] = dict(verdict=v.group(1) if v else None, trans_ms=int(t.group(1)) if t else None,
                               solve_ms=int(t.group(2)) if t else None, unsupported=err.group(1) if err else None)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-home', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--suite', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--scope', type=int, default=6); ap.add_argument('--real-as-integer', action='store_true')
    ap.add_argument('--core', action='store_true', help='compute a minimal jointly-unsatisfiable subset (slow)')
    a = ap.parse_args()
    out = Path(a.out); (out / 'inv').mkdir(parents=True, exist_ok=True)
    mm, names = load_metamodel(a.model)
    model_name = Path(a.suite).name
    spec, assocs = use_spec(mm, names, a.model, model_name, a.real_as_integer)
    renamed = sorted({x.name for n in names for x in mm.get_attributes_for(n) if x.name in USE_KEYWORDS}
                     | {x.ref_name for x in mm.get_all_associations() if x.ref_name in USE_KEYWORDS})
    (out / f'{model_name}.use').write_text(spec)
    rows = [json.loads(l) for l in open(Path(a.suite) / 'manifest.jsonl')]
    idx = []
    for c in rows:
        body = rename_in_body(to_use_ocl(c['ocl'].split('inv:', 1)[1].strip()), renamed); name = f"c{c['constraint_id']:03d}"
        (out / 'inv' / f'{name}.use').write_text(spec + f"\nconstraints\ncontext {c['context']} inv {name}: {body}\n")
        idx.append(dict(name=name, context=c['context'], label=c['solver_result'], level=c['meta_profile'],
                        pattern=c['pattern_id'], ocl=body, string_op=bool(STRING_OPS.search(body)), any_op=bool(ANY_OP.search(body))))
    risky = [c['name'] for c in idx if re.search(r"/|\.div\(|\d+\.\d+", c['ocl'])]
    ints = literal_domain([c['ocl'] for c in idx])
    for cls in names: (out / f'ctx_{cls}.properties').write_text(properties(names, assocs, ints, a.scope, context=cls))
    (out / 'joint.properties').write_text(properties(names, assocs, ints, a.scope, joint=True))
    sat_ok = [c for c in idx if c['label'] == 'SAT' and not c['string_op'] and not c['any_op']]
    (out / 'joint.use').write_text(spec + "\nconstraints\n" + "\n".join(f"context {c['context']} inv {c['name']}: {c['ocl']}" for c in sat_ok) + "\n")
    os.chdir(out)
    cmds = ["mv -config bitwidth := 12"]
    for c in idx: cmds += [f"open inv/{c['name']}.use", f"mv -validate ctx_{c['context']}.properties"]
    cmds += ["open joint.use", "mv -validate joint.properties"]
    runs = parse_runs(run_use(a.use_home, f'{model_name}.use', cmds, Path('batch.log')), 'batch.cmd')
    results = []
    for c in idx:
        r = dict(runs.get(f"inv/{c['name']}.use", {})); r.update(c)
        if r.get('unsupported'): r['outcome'] = 'unsupported'
        elif r.get('verdict') is None: r['outcome'] = 'no-verdict'
        elif c['any_op']: r['outcome'] = 'any-semantics'   # USE treats any() with >1 candidate as undefined
        else:
            ul = 'SAT' if r['verdict'] == 'SATISFIABLE' else 'UNSAT'
            r['outcome'] = 'agree' if ul == c['label'] else 'disagree'
        results.append(r)
    joint = dict(runs.get('joint.use', {})); joint['n_invariants'] = len(sat_ok)
    core = None
    if a.core and joint.get('verdict') and 'UNSAT' in joint['verdict']:
        keep = list(sat_ok); step = 0
        def verdict(cs):
            Path('core_try.use').write_text(spec + "\nconstraints\n" + "\n".join(f"context {c['context']} inv {c['name']}: {c['ocl']}" for c in cs) + "\n")
            log = run_use(a.use_home, f'{model_name}.use', ["mv -config bitwidth := 12", "open core_try.use", "mv -validate joint.properties"], Path('core_step.log'))
            return parse_runs(log, 'core_step.cmd').get('core_try.use', {}).get('verdict')
        for c in list(sat_ok):
            trial = [x for x in keep if x['name'] != c['name']]
            if 'UNSAT' in (verdict(trial) or ''): keep = trial
        core = [dict(name=c['name'], context=c['context'], ocl=c['ocl']) for c in keep]
    def ms(r): return (r.get('trans_ms') or 0) + (r.get('solve_ms') or 0)
    summary = dict(model=model_name, scope=a.scope, real_as_integer=a.real_as_integer, n=len(idx),
                   parsed=len(idx), outcomes=dict(Counter(f"{r['label']}/{r['outcome']}" for r in results)),
                   unsupported_ops=dict(Counter(r['unsupported'] for r in results if r.get('unsupported'))),
                   string_constraints=sum(1 for c in idx if c['string_op']),
                   any_constraints_excluded_from_joint=sum(1 for c in idx if c['label'] == 'SAT' and c['any_op'] and not c['string_op']),
                   any_constraints=sum(1 for c in idx if c['any_op']),
                   string_constraints_unsupported=sum(1 for r in results if r['string_op'] and r.get('unsupported')),
                   risky_for_real_as_integer=risky, joint=joint, core=core,
                   time_ms_by_level={lvl: dict(n=len(x), mean=round(statistics.mean(x)), max=max(x)) for lvl in ('easy', 'medium', 'difficult')
                                     for x in [[ms(r) for r in results if r['label'] == 'SAT' and r['level'].startswith(lvl) and r['outcome'] != 'unsupported']] if x})
    json.dump(dict(summary=summary, results=results), open('results.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
