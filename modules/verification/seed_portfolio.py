"""
Seed portfolio for joint-consistency checks.

Z3's search on our larger consistency instances is highly seed-sensitive: on
the recorded hard instance (tests/fixtures/carrental_hard_joint_instance_sat.json)
seeds 0 and 3 find a model in ~12 s while seeds 1, 2, 4 and 5 run past 90 s.
Every definite verdict is trustworthy (a model proves consistency, an unsat
core proves a conflict), so running the same check under several seeds in
separate processes and taking the FIRST definite answer is sound and turns
seed bad luck into a short wait.

Workers rebuild the verifier from the XMI path (Z3 objects cannot be shared
between processes) and receive constraints as plain dicts.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional


def _worker(seed: int, xmi_path: str, scope_per_class: int, timeout_ms: int,
            constraints: List[Dict], out) -> None:
    import contextlib, io, os, sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import z3
        z3.set_param('smt.random_seed', seed)
        z3.set_param('sat.random_seed', seed)
        from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
        from modules.core.models import OCLConstraint
        from modules.verification.framework_verifier import FrameworkConstraintVerifier
        with contextlib.redirect_stdout(io.StringIO()):
            mm = MetamodelExtractor(xmi_path).get_metamodel()
            v = FrameworkConstraintVerifier(mm, xmi_path, scope_per_class=scope_per_class,
                                            timeout_ms=timeout_ms, batch_timeout_ms=timeout_ms)
            v.portfolio_seeds = 0   # never recurse
            cs = [OCLConstraint(ocl=d['ocl'], pattern_id=d['pattern_id'], pattern_name=d['pattern_name'],
                                context=d['context'], parameters={}, confidence=1.0, timestamp='', metadata={})
                  for d in constraints]
            res = v.check_consistency_with_core(cs, timeout_ms=timeout_ms)
        out.put((seed, res))
    except Exception as e:  # report, never hang the parent
        out.put((seed, {'solver_result': 'unknown', 'core': [], 'unencoded': [], 'error': f'{type(e).__name__}: {e}'}))


def run_portfolio(xmi_path: str, scope_per_class: int, constraints: List[Dict],
                  seeds: List[int], timeout_ms: int) -> Optional[Dict]:
    """Run one consistency check per seed in separate processes; return the
    first definite ('sat'/'unsat') result with the winning seed recorded under
    'seed', or None if every worker ends without a definite verdict.

    Workers are plain subprocesses (`python -m modules.verification.seed_portfolio`),
    so this works whatever started the caller (script, stdin, notebook)."""
    import json, os, subprocess, sys, tempfile
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump({'xmi_path': os.path.abspath(xmi_path), 'scope_per_class': scope_per_class,
                   'timeout_ms': timeout_ms, 'constraints': constraints}, fh)
        job = fh.name
    procs = {}
    try:
        for seed in seeds:
            procs[seed] = subprocess.Popen(
                [sys.executable, '-m', 'modules.verification.seed_portfolio', job, str(seed)],
                cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        deadline = time.time() + timeout_ms / 1000.0 + 120   # + worker start-up
        while procs and time.time() < deadline:
            for seed, p in list(procs.items()):
                if p.poll() is None:
                    continue
                out = p.stdout.read() if p.stdout else ''
                del procs[seed]
                for line in out.splitlines()[::-1]:
                    if line.startswith('PORTFOLIO_RESULT '):
                        res = json.loads(line[len('PORTFOLIO_RESULT '):])
                        if res.get('solver_result') in ('sat', 'unsat'):
                            return dict(res, seed=seed)
                        break
            time.sleep(0.2)
        return None
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        try:
            os.unlink(job)
        except OSError:
            pass


def _main(argv: List[str]) -> int:
    import json
    job = json.load(open(argv[0])); seed = int(argv[1])

    class _Out:
        def put(self, item):
            print('PORTFOLIO_RESULT ' + json.dumps(item[1]), flush=True)
    _worker(seed, job['xmi_path'], job['scope_per_class'], job['timeout_ms'], job['constraints'], _Out())
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_main(sys.argv[1:]))
