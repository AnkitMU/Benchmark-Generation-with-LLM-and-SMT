#!/usr/bin/env python3
"""
OCL Benchmark Suite Generator v3.0
===================================
Generates machine-verified OCL benchmark suites from UML/Ecore metamodels.

Three integrated approaches:
  1. LLM-guided semantic admissibility (pre-generation filter)
  2. Complexity-controlled pattern-based generation
  3. VGCR — Verification-Guided Constraint Refinement (inline verification)

Usage:
  python generate_benchmark_suite.py --config examples/example_suite.yaml
  python generate_benchmark_suite.py --config suite.yaml --no-vgcr
  python generate_benchmark_suite.py --config suite.yaml --vgcr-retries 5
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress TensorFlow/gRPC issues in conda environments (macOS Apple Silicon)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("no_grpc_proxy", "localhost,127.0.0.1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from modules.generation.benchmark.suite_config import BenchmarkSuite
from modules.generation.benchmark.suite_controller_enhanced import EnhancedSuiteController
from modules.generation.benchmark.engine_v2 import VALID_GENERATION_MODES

VERSION = "3.0"


def main():
    parser = argparse.ArgumentParser(
        description="Generate machine-verified OCL benchmark suites with "
                    "semantic admissibility filtering and VGCR quality assurance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with full pipeline (LLM + VGCR + SMT)
  python generate_benchmark_suite.py --config examples/example_suite.yaml

  # Disable VGCR (post-hoc verification only, like ECMFA baseline)
  python generate_benchmark_suite.py --config suite.yaml --no-vgcr

  # Custom VGCR retries
  python generate_benchmark_suite.py --config suite.yaml --vgcr-retries 5

  # Disable semantic filter (SMT-only mode)
  python generate_benchmark_suite.py --config suite.yaml --no-semantic

  # Force a generation mechanism for the whole run (A/B the mechanisms)
  python generate_benchmark_suite.py --config suite.yaml --generation-mode construct_select
  python generate_benchmark_suite.py --config suite.yaml --generation-mode legacy

  # Silent mode
  python generate_benchmark_suite.py --config suite.yaml --quiet
        """
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        type=str,
        help='Path to suite configuration YAML file'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output (minimal logging)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (default)'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging (very detailed)'
    )

    parser.add_argument(
        '--no-research-features',
        action='store_true',
        help='Disable advanced features (metadata, UNSAT, similarity, etc.)'
    )

    # --- VGCR flags ---
    vgcr_group = parser.add_argument_group('VGCR options',
        'Verification-Guided Constraint Refinement settings')

    vgcr_group.add_argument(
        '--no-vgcr',
        action='store_true',
        help='Disable VGCR refinement loop (fall back to post-hoc verification only)'
    )

    vgcr_group.add_argument(
        '--vgcr-retries',
        type=int,
        default=None,
        metavar='N',
        help='Max refinement retries per constraint slot (default: 3, from config)'
    )

    vgcr_group.add_argument(
        '--no-independence',
        action='store_true',
        help='Disable q3 independence check in VGCR (faster, weaker guarantees)'
    )

    # --- Semantic filter flags ---
    sem_group = parser.add_argument_group('Semantic filter options',
        'LLM-guided semantic admissibility settings')

    sem_group.add_argument(
        '--no-semantic',
        action='store_true',
        help='Disable LLM-based semantic admissibility filter'
    )

    # --- Generation mechanism flags ---
    gen_group = parser.add_argument_group('Generation options',
        'Constraint generation mechanism settings')

    gen_group.add_argument(
        '--generation-mode',
        choices=list(VALID_GENERATION_MODES),
        default=None,
        metavar='MODE',
        help='Force the generation mechanism for the whole run, overriding the '
             'suite default AND any per-profile generation_mode in the YAML. '
             "'construct_select' = over-generate a pool, measure exact "
             'complexity, then stratified-select to fill each tier quota exactly; '
             "'legacy' = older TC-steering path. Default: use the YAML config."
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'OCL Benchmark Generator v{VERSION} '
                f'(with Semantic Admissibility + VGCR)'
    )

    args = parser.parse_args()

    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"OCL Benchmark Suite Generator v{VERSION}")
    print("Semantic Admissibility + VGCR + SMT Verification")
    print(f"{'='*70}\n")
    print(f"Config: {args.config}")

    # Set verbosity
    verbose = not args.quiet if not args.verbose else True
    debug = args.debug
    enable_research = not args.no_research_features

    if debug:
        print("Debug logging ENABLED")
    if not enable_research:
        print("Advanced features DISABLED")

    try:
        # Load suite configuration
        suite = BenchmarkSuite.from_yaml(str(config_path))

        # --- Apply CLI overrides to suite config ---

        # VGCR overrides
        if args.no_vgcr:
            suite.vgcr.enable = False
        if args.vgcr_retries is not None:
            suite.vgcr.max_retries = args.vgcr_retries
        if args.no_independence:
            suite.vgcr.enable_independence_check = False

        # Semantic filter overrides
        if args.no_semantic:
            suite.semantic.enable = False

        # Generation-mechanism override: a CLI --generation-mode forces the
        # whole run, overriding both the suite default and any per-profile
        # generation_mode in the YAML (so the forced mode applies uniformly).
        if args.generation_mode:
            suite.generation_mode = args.generation_mode
            for m in suite.models:
                for p in m.profiles:
                    p.generation_mode = None

        # Print configuration summary
        print(f"Loaded suite: {suite.suite_name} v{suite.version}")
        print(f"   Models: {len(suite.models)}")
        total_profiles = sum(len(m.profiles) for m in suite.models)
        print(f"   Profiles: {total_profiles}")
        print(f"   Output: {suite.output_root}")
        print(f"   Generation Mode:   {suite.generation_mode}"
              f"{' (forced via CLI)' if args.generation_mode else ' (from config)'}")
        print(f"   Advanced Features: {'ENABLED' if enable_research else 'DISABLED'}")
        print(f"   Semantic Filter:   {'ENABLED' if suite.semantic.enable else 'DISABLED'}"
              f"{' (' + suite.semantic.model + ')' if suite.semantic.enable else ''}")
        print(f"   VGCR:              {'ENABLED' if suite.vgcr.enable else 'DISABLED'}"
              f"{' (retries=' + str(suite.vgcr.max_retries) + ')' if suite.vgcr.enable else ''}")
        if suite.vgcr.enable:
            print(f"     Independence (q3): {'ENABLED' if suite.vgcr.enable_independence_check else 'DISABLED'}")
        print()

        # Run generation with enhanced controller
        controller = EnhancedSuiteController(
            suite,
            verbose=verbose,
            debug=debug,
            enable_research_features=enable_research
        )
        stats = controller.generate_suite()

        # === Summary ===
        print(f"\n{'='*70}")
        print("SUITE GENERATION COMPLETE")
        print(f"{'='*70}")

        total = stats['total_constraints']
        if total > 0:
            print(f"Total Constraints: {total}")
            print(f"  Valid: {stats['total_valid']}")
            print(f"  SAT:   {stats['total_sat']} "
                  f"({stats['total_sat']/total*100:.1f}%)")
            print(f"  UNSAT: {stats['total_unsat']} "
                  f"({stats['total_unsat']/total*100:.1f}%)")
            if stats['total_unknown'] > 0:
                print(f"  Unknown: {stats['total_unknown']}")
        else:
            print("No constraints generated.")

        # --- VGCR summary ---
        _print_vgcr_summary(stats)

        print(f"\nOutputs saved to: {suite.output_root}")

        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)


def _print_vgcr_summary(stats: dict):
    """Print VGCR statistics if available in any profile."""
    # Collect VGCR stats across all models → profiles
    all_vgcr = []
    for model_stats in stats.get('models', []):
        model_name = model_stats.get('model', '?')
        for profile_stats in model_stats.get('profiles', []):
            vgcr = profile_stats.get('vgcr_stats')
            if vgcr:
                all_vgcr.append((model_name, vgcr))

    if not all_vgcr:
        return

    print(f"\n{'─'*70}")
    print("VGCR Quality Assurance Summary")
    print(f"{'─'*70}")

    totals = {
        'candidates': 0, 'passed': 0, 'retries': 0,
        'smt_queries': 0, 'smt_time': 0.0,
        'tautology': 0, 'contradiction': 0, 'redundant': 0,
        'inconsistent': 0, 'tc_mismatch': 0,
    }

    for model_name, vgcr in all_vgcr:
        n_in = vgcr.get('total_candidates', 0)
        n_pass = vgcr.get('passed', 0)
        rate = vgcr.get('pass_rate', '?')
        fb = vgcr.get('failure_breakdown', {})

        print(f"  {model_name:25s}  {n_in}→{n_pass}  pass={rate}  "
              f"taut={fb.get('tautology',0)} "
              f"contr={fb.get('contradiction',0)} "
              f"redun={fb.get('redundant',0)} "
              f"incon={fb.get('inconsistent',0)} "
              f"tc={fb.get('tc_mismatch',0)}")

        totals['candidates'] += n_in
        totals['passed'] += n_pass
        totals['retries'] += vgcr.get('total_retries', 0)
        totals['smt_queries'] += vgcr.get('total_smt_queries', 0)
        totals['smt_time'] += vgcr.get('total_smt_time_s', 0.0)
        for key in ('tautology', 'contradiction', 'redundant',
                     'inconsistent', 'tc_mismatch'):
            totals[key] += fb.get(key, 0)

    if len(all_vgcr) > 1:
        t = totals
        total_violations = (t['tautology'] + t['contradiction'] +
                           t['redundant'] + t['inconsistent'] +
                           t['tc_mismatch'])
        pass_pct = (t['passed'] / t['candidates'] * 100
                    if t['candidates'] > 0 else 0)
        print(f"  {'─'*65}")
        print(f"  {'TOTAL':25s}  {t['candidates']}→{t['passed']}  "
              f"pass={pass_pct:.1f}%  "
              f"taut={t['tautology']} contr={t['contradiction']} "
              f"redun={t['redundant']} incon={t['inconsistent']} "
              f"tc={t['tc_mismatch']}")
        print(f"\n  Violations detected & repaired: {total_violations}")
        print(f"  Total retries: {t['retries']}")
        print(f"  SMT queries: {t['smt_queries']}  "
              f"({t['smt_time']:.1f}s total)")


if __name__ == '__main__':
    main()
