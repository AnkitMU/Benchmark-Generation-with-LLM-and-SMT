# Automatic Benchmark Generation Framework for OCL(Object Constraint Language)

Our framework generates OCL benchmarks from UML/Ecore metamodels. It generates feature driven diverse OCL constraints which are solver verified (Z3 SMT) through a verification pipeline.

## Feature of Framework
- Generate OCL constraints from a pattern library based on user Configuration
- Create SAT and UNSAT constraints
- Add metadata (operators, difficulty, depth, etc.)
-  Remove deduplicate and analyze constraints.
- verify entire benchmark using SMT solver (Z3)
- Export JSON/JSONL and OCL files

## Quick start

```bash
# full verifier stack requires Python 3.10+
python3.10 -m venv venv
source venv/bin/activate

# keep installer tooling current so pip can resolve macOS wheels
python -m pip install --upgrade pip setuptools wheel

# install deps
python -m pip install -r requirements.txt

# run example suite
python generate_benchmark_suite.py --config examples/example_suite.yaml
```

Outputs are written to `benchmarks/`.

## Repository layout
- `generate_benchmark_suite.py` — entry point; drives generation, verification and export from a YAML config
- `modules/` — generation (pattern engine, composer), semantic analysis, synthesis, verification (VGCR) and utilities
- `hybrid-ssr-ocl-full-extended/` — embedded hybrid OCL→SMT verifier package (`ssr_ocl`), including the trained pattern classifier under `models/adapted_model/`
- `models/` — the ten UML/Ecore XMI metamodels used in the paper
- `examples/` — ready-to-run suite configs (`example_suite.yaml`, `paper_ten_models_*.yaml`, `one_model_*.yaml`)
- `templates/` — OCL pattern library
- `config/` — semantic rules and business-logic profiles
- `benchmarks/` — generated suites; `paper/`, `paper_v2/`, `paper_v3/` are the runs reported in the paper (with and without VGCR)
- `benchmarks_table/` — an earlier steered-generation suite kept for reference
- `experiments/` — parallel runner, paper-table builder and the USE-based oracle comparison (`use_comparison/`)
- `cache/` and `logs/` are created at runtime and are not tracked

Notes:
- The embedded hybrid verifier package targets Python `>=3.10`.
- `z3-solver` is pinned below `4.14` to prefer wheel-backed macOS installs and avoid local source builds.
- `numpy` is pinned below `2` to stay compatible with the current `torch` stack used by the verifier and semantic modules.

## Configuration
Edit a YAML file like `examples/example_suite.yaml` to control:
- models (XMI paths)
- number of constraints
- SAT/UNSAT ratio
- pattern family mix
- verification options

## Output files
Typical run produces:
- `constraints.ocl`
- `constraints.json`
- `constraints_sat.ocl` / `constraints_unsat.ocl`
- `manifest.jsonl`
- summary JSON


- Solver verification is slower(affected by number of constraints to be verified) but gives verification labels (SAT/UNSAT).
- Adavance features (similarity, implication checks, metadata_label) are enabled by default in the config file (example_suite.yaml).

---
## GenAI Disclosure & Usage
This project acknowledges the use of Generative AI (GenAI) in its development and debugging:

Development: GenAI tools were utilized for code optimization, documentation assistance and refining the natural language patterns within the OCL generation engine.

Verification: No constraint is added to a benchmark without passing the SMT verification pipeline. Every SAT/UNSAT label is confirmed by the Z3 solver, ensuring the ground truth is mathematically sound and not subject to AI hallucinations.
