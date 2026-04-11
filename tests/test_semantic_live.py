"""
Live test for LLM Semantic Analysis with Phi-4-mini via Ollama.

Usage:
    # Test with default model (IOTsensornetwork)
    python tests/test_semantic_live.py

    # Test with a specific model
    python tests/test_semantic_live.py models/hospital.xmi

    # Test with phi4 instead of phi4-mini
    python tests/test_semantic_live.py models/hospital.xmi --model phi4

    # Force re-analysis (ignore cache)
    python tests/test_semantic_live.py --no-cache

Prerequisites:
    1. Ollama must be running: ollama serve
    2. Model must be pulled: ollama pull phi4-mini
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.semantic.metamodel.xmi_extractor import extract_metamodel
from modules.semantic.llm_semantic_analyzer import LLMSemanticAnalyzer

# ── Configuration ──────────────────────────────────────────
DEFAULT_XMI = "models/iotsensornetwork.xmi"
DEFAULT_MODEL = "phi4-mini"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "semantic_results"


def setup_logging():
    """Configure logging to see LLM analysis progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )


def print_results(matrix, metamodel):
    """Print a detailed summary of semantic analysis results."""
    print("\n" + "=" * 70)
    print("SEMANTIC ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Model used: {matrix.model_used}")
    print(f"Generated at: {matrix.generated_at}")
    print(f"Total classes analyzed: {len(matrix.class_profiles)}")

    total_comparable = 0
    total_incomparable = 0
    total_analyzed = 0
    total_skipped = 0

    for cls_name, profile in sorted(matrix.class_profiles.items()):
        comp = len(profile.comparable_pairs)
        incomp = len(profile.incomparable_pairs)
        total_comparable += comp
        total_incomparable += incomp

        if profile.analyzed:
            total_analyzed += 1
        else:
            total_skipped += 1

        status = "LLM-analyzed" if profile.analyzed else "heuristic-fallback"
        print(f"\n  {cls_name} [{status}]")

        if profile.comparable_pairs:
            print(f"    Comparable pairs ({comp}):")
            for a, b in sorted(profile.comparable_pairs):
                print(f"      + {a} vs {b}")

        if profile.incomparable_pairs:
            print(f"    Incomparable pairs ({incomp}):")
            for a, b in sorted(profile.incomparable_pairs):
                print(f"      - {a} vs {b}")

        if not profile.comparable_pairs and not profile.incomparable_pairs:
            print(f"    (no same-type attribute pairs)")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  LLM-analyzed classes: {total_analyzed}")
    print(f"  Heuristic-fallback classes: {total_skipped}")
    print(f"  Total comparable pairs: {total_comparable}")
    print(f"  Total incomparable pairs: {total_incomparable}")
    print(f"{'=' * 70}")


def save_results(matrix, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(matrix.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Live test for LLM Semantic Analysis")
    parser.add_argument("xmi", nargs="?", default=DEFAULT_XMI,
                        help=f"Path to XMI model file (default: {DEFAULT_XMI})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-analysis, ignore cached results")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate",
                        help="Ollama API endpoint")
    args = parser.parse_args()

    setup_logging()

    # Resolve XMI path
    xmi_path = Path(args.xmi)
    if not xmi_path.is_absolute():
        xmi_path = PROJECT_ROOT / xmi_path
    if not xmi_path.exists():
        print(f"ERROR: XMI file not found: {xmi_path}")
        print(f"\nAvailable models:")
        for f in sorted((PROJECT_ROOT / "models").glob("*.xmi")):
            print(f"  {f.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    model_name = xmi_path.stem

    print(f"{'=' * 70}")
    print(f"LLM Semantic Analysis — Live Test")
    print(f"{'=' * 70}")
    print(f"  XMI file:  {xmi_path.relative_to(PROJECT_ROOT)}")
    print(f"  LLM model: {args.model}")
    print(f"  Cache:     {'disabled' if args.no_cache else 'enabled'}")
    print(f"{'=' * 70}")

    # Step 1: Extract metamodel
    print(f"\n[1/3] Extracting metamodel from {model_name}.xmi ...")
    metamodel = extract_metamodel(str(xmi_path))

    class_count = len(metamodel.classes) if isinstance(metamodel.classes, dict) else len(list(metamodel.classes))
    print(f"  Found {class_count} classes")

    # Step 2: Run semantic analysis
    print(f"\n[2/3] Running semantic analysis with {args.model} ...")
    analyzer = LLMSemanticAnalyzer(
        model=args.model,
        ollama_url=args.ollama_url,
        use_cache=not args.no_cache,
    )
    matrix = analyzer.analyze_metamodel(metamodel, model_name=model_name)

    # Step 3: Display and save results
    print(f"\n[3/3] Results:")
    print_results(matrix, metamodel)

    output_file = OUTPUT_DIR / f"{model_name}_{args.model.replace('-', '_')}_results.json"
    save_results(matrix, output_file)

    print(f"\nDone! You can inspect the JSON at: {output_file}")


if __name__ == "__main__":
    main()
