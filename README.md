# Automatic Benchmark Generation Framework for OCL (Object Constraint Language)

This framework generates **complexity-steered, solver-verified OCL benchmark suites** from
UML/Ecore metamodels. Given a metamodel and a target *complexity profile*, it produces diverse
OCL invariants whose **measured** complexity lands inside user-specified ranges, keeps the SAT
subset **jointly satisfiable**, and confirms every SAT/UNSAT label with the **Z3** SMT solver.

## Features

- **Profile-guided complexity steering** — you set per-component complexity ranges (a "box") for
  each difficulty tier (easy / medium / difficult). The generator *measures* each candidate's
  complexity and steers it into the target box using structural actuators
  (forAll-wrap, deepen-navigation, conjoin, cast, drop-conjunct).
- **Nine measured complexity components** — structural (NNR-C, WNC, DN-CA, VRC, WNO, WNM) and
  computational (OC, TCC, CIC). Five are user-controllable (`nnr_c`, `dn_ca`, `wno`, `cic`, `tcc`);
  the other four are derived and reported.
- **Pattern-library generation** — 113 OCL patterns across families
  (cardinality, arithmetic, navigation, quantified, conditional, string, type-checks).
- **Joint consistency** — a per-context consistency gate keeps the SAT subset co-satisfiable. It
  catches boolean-polarity (`x = true` vs `x = false`), collection-emptiness (`isEmpty` vs
  `size() >= 2`), mod-parity (`mod 2 = 1` vs `mod 4 = 0`), numeric-ordering (`a - b >= k` vs
  `a <= b`), and quantifier-body (`exists(p=true)` vs `forAll(p=false)`) contradictions — both at
  generation time and as a post-VGCR backstop.
- **VGCR verification & refinement** — each candidate is SMT-checked (individual status, joint
  consistency, non-triviality, redundancy, complexity conformance); rejected candidates are
  regenerated up to `R` times, guided by accumulated failures.
- **SAT & UNSAT** — SAT constraints are solver-verified; UNSAT constraints are produced by
  Z3-confirmed mutation to hit the requested ratio.
- **Metadata & deduplication** — operators, navigation/quantifier depth, difficulty, and total
  complexity; structural (tree-edit) similarity dedup plus semantic clustering.
- **Semantic admissibility (optional)** — an LLM filter (`phi4-mini` via Ollama) checks
  attribute compatibility and semantic plausibility.
- **Export** — `.ocl`, `.json`, `manifest.jsonl`, and a per-model deviation `report.json`.

## How a run flows

1. **Profile** each applicable `(pattern, class)` pair — measure its reachable complexity footprint.
2. **Compile slots** — allocate `(family × difficulty × polarity)` slots to meet the requested
   size, SAT ratio, and per-class bounds; record any infeasible cells in the deviation report.
3. **Generate** — for each slot, draw/actuate a candidate until its measured complexity enters the
   target box, rejecting any that would break per-context joint consistency.
4. **Repair** class-minimum deficits with same-family replacements.
5. **VGCR refine** — SMT-verify the suite, regenerate rejected constraints (up to `max_retries`),
   and prune jointly-inconsistent ones.
6. **Inject UNSAT**, **deduplicate**, **verify** (Z3 individual + global consistency under a
   bounded scope), and **export**.

## Quick start

```bash
# the full verifier stack requires Python 3.10+
python3.10 -m venv venv
source venv/bin/activate

# keep installer tooling current so pip can resolve macOS wheels
python -m pip install --upgrade pip setuptools wheel

# install deps
python -m pip install -r requirements.txt

# (optional) start the semantic filter backend, or run with --no-semantic
#   ollama serve  &  ollama pull phi4-mini

# run the steered example suite (CarRental, 100 constraints)
python generate_benchmark_suite.py --config examples/example_suite_steered.yaml
```

Outputs are written to `benchmarks/<ModelName>/`.

Useful flags: `--no-semantic` (skip the LLM filter / Ollama), `--no-vgcr` (skip refinement),
`--quiet`, `--verbose`.

## Configuration

The primary example is `examples/example_suite_steered.yaml`. Key knobs:

```yaml
generation_mode: steered
models:
  - xmi: "models/car_rental.xmi"
    name: "CarRental"
    profiles:
      - name: balanced
        constraints: 100              # requested suite size N
        sat_ratio: 0.8                # SAT fraction (r)
        unsat_ratio: 0.2
        per_class_min: 3              # L: min constraints per class
        per_class_max: 50             # U
        similarity_threshold: 0.95    # structural (tree-edit) dedup cutoff; 1.0 = off
        on_infeasible: relax          # report | relax | skip

        # pattern family mix (must sum to 100)
        families_pct: { cardinality: 30, arithmetic: 25, navigation: 25, quantified: 20 }

        # per-difficulty complexity boxes over the FIVE controllable components
        complexity_profiles:
          - { label: easy,      pct: 40, ranges: { nnr_c: [1,3], dn_ca: [1,2], wno: [0,4],  cic: [0,0], tcc: [0,0] } }
          - { label: medium,    pct: 35, ranges: { nnr_c: [2,5], dn_ca: [1,3], wno: [3,8],  cic: [0,1], tcc: [0,1] } }
          - { label: difficult, pct: 25, ranges: { nnr_c: [3,7], dn_ca: [2,4], wno: [6,12], cic: [1,3], tcc: [0,2] } }

        operator_weights: { ... }     # Measure() operator weights

vgcr:         { enable: true, max_retries: 9 }      # R: regeneration retries per rejected constraint
verification: { enable: true, scope_per_class: 6 }  # bounded SMT scope (instances per class)
semantic:     { enable: true, model: phi4-mini }
output_root:  "benchmarks/"
```

A single-profile model writes straight to `benchmarks/<ModelName>/`; multi-profile models keep a
`<ModelName>/<profile>/` subfolder. The legacy `examples/example_suite.yaml` uses the older
non-steered `construct_select` mode.

## Output files

A run writes to `benchmarks/<ModelName>/`:

- `constraints.ocl` / `constraints.json` — the full suite
- `constraints_sat.ocl` / `constraints_sat.json` — SAT subset
- `constraints_unsat.ocl` / `constraints_unsat.json` — UNSAT subset
- `manifest.jsonl` — one record per constraint (pattern, operators, difficulty, depth, solver result, …)
- `report.json` — steered deviation report (reachable envelope, family/profile distribution,
  actuator usage, feasibility gaps)
- `<suite>_summary.json` — suite-level statistics

## Reproducing the results table

`make_results_table.py` runs all 10 bundled metamodels and emits a LaTeX results table (classes,
attributes, generated, verified, VGCR tautology/contradiction/redundant/inconsistent breakdown,
retries, and wall-clock time):

```bash
python make_results_table.py > results.tex            # run all 10 models
python make_results_table.py --no-semantic --reuse    # faster / build table from existing output
```

## Notes

- The embedded hybrid verifier package targets Python `>=3.10`.
- `z3-solver` is pinned below `4.14` to prefer wheel-backed macOS installs and avoid local source builds.
- `numpy` is pinned below `2` to stay compatible with the `torch` stack used by the verifier and semantic modules.
- Solver verification time scales with the number of constraints: the SAT subset is checked both
  individually and for global consistency under a bounded scope (`scope_per_class`).

---

## GenAI Disclosure & Usage

This project acknowledges the use of Generative AI (GenAI) in its development and debugging:

- **Development** — GenAI tools were used for code optimization, documentation assistance, and
  refining the natural-language patterns within the OCL generation engine.
- **Verification** — No constraint is added to a benchmark without passing the SMT verification
  pipeline. Every SAT/UNSAT label is confirmed by the Z3 solver, ensuring the ground truth is
  mathematically sound and not subject to AI hallucinations.
