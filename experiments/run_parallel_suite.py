#!/usr/bin/env python3
"""
Run a multi-model suite configuration with one process per model.

The models of a suite are independent, so running them concurrently makes the
whole suite take roughly as long as its slowest model. Each model gets its own
configuration (same profile, same verification settings, same output root) and
its own log; the per-model summaries are merged into one suite summary at the end.

Usage:
    python experiments/run_parallel_suite.py --config examples/paper_ten_models_vgcr.yaml \
        [--jobs 4] [--output-root benchmarks/paper_v3/with_vgcr/] [--portfolio-seeds 4] \
        [--portfolio-first-try-ms 20000] [--logs logs/parallel_run]

Keep --jobs x (1 + portfolio seeds) near the core count: parallel processes compete
for CPU, and checks close to their time limit can otherwise time out.
"""
from __future__ import annotations
import argparse, copy, json, subprocess, sys, time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--output-root')
    ap.add_argument('--portfolio-seeds', type=int)
    ap.add_argument('--portfolio-first-try-ms', type=int)
    ap.add_argument('--logs', default='logs/parallel_run')
    ap.add_argument('--models', nargs='*', help='subset of model names to run')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.output_root: cfg['output_root'] = a.output_root
    ver = cfg.setdefault('verification', {})
    if a.portfolio_seeds is not None: ver['portfolio_seeds'] = a.portfolio_seeds
    if a.portfolio_first_try_ms is not None: ver['portfolio_first_try_ms'] = a.portfolio_first_try_ms
    models = [m for m in cfg['models'] if not a.models or m['name'] in a.models]
    logs = Path(a.logs); logs.mkdir(parents=True, exist_ok=True)

    queue, running, done = list(models), {}, {}
    t_start = time.time()
    while queue or running:
        while queue and len(running) < a.jobs:
            m = queue.pop(0)
            one = copy.deepcopy(cfg); one['models'] = [m]; one['suite_name'] = f"{cfg['suite_name']}__{m['name']}"
            cpath = logs / f"{m['name']}.yaml"; yaml.safe_dump(one, open(cpath, 'w'), sort_keys=False, width=100)
            log = open(logs / f"{m['name']}.log", 'w')
            running[m['name']] = (subprocess.Popen([sys.executable, '-u', 'generate_benchmark_suite.py', '--config', str(cpath)],
                                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), time.time(), log)
            print(f"[{time.time()-t_start:7.0f}s] started {m['name']}", flush=True)
        for name, (p, t0, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); done[name] = dict(exit=p.returncode, seconds=round(time.time() - t0, 1))
                del running[name]
                print(f"[{time.time()-t_start:7.0f}s] finished {name} exit={p.returncode} in {done[name]['seconds']}s", flush=True)
        time.sleep(2)

    # merge per-model summaries into one suite summary
    out_root = ROOT / cfg['output_root']
    merged = {'suite_name': cfg['suite_name'], 'models': [], 'runs': done, 'wall_seconds': round(time.time() - t_start, 1)}
    for m in models:
        f = out_root / f"{cfg['suite_name']}__{m['name']}_summary.json"
        if f.exists():
            merged['models'].extend(json.load(open(f)).get('models', []))
    (out_root / f"{cfg['suite_name']}_summary.json").write_text(json.dumps(merged, indent=2))
    print(f"all done in {merged['wall_seconds']}s; summary: {out_root / (cfg['suite_name'] + '_summary.json')}")
    return 0 if all(r['exit'] == 0 for r in done.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
