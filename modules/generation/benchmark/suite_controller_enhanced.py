"""
Features Integrated:
1. Metadata Enrichment (operators, depth, difficulty)
2. Manifest Generation (ML-friendly JSONL format)
3. Structural Similarity (tree-edit deduplication)
4. Semantic Similarity (clustering)
5. Implication Checking (constraint relationships)
6. UNSAT Generation (negative examples via mutation)
"""
import os
# Force the torch-only transformers backend BEFORE any heavy import. The semantic
# similarity step (sentence-transformers) otherwise lets `transformers` eagerly
# import TensorFlow, whose native init deadlocks on some machines and hangs the run
# at "Computing semantic similarity..." with `[mutex.cc:452] RAW: Lock blocking @cache`.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from modules.core.models import Metamodel, OCLConstraint
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.generation.benchmark.engine_v2 import BenchmarkEngineV2
from modules.generation.benchmark.bench_config import BenchmarkProfile, QuantitiesConfig
from modules.verification.framework_verifier import FrameworkConstraintVerifier
from modules.utils.logger import FrameworkLogger, get_logger

# Advanced features (function-based modules)
from modules.generation.benchmark import metadata_enricher
from modules.generation.benchmark import manifest_generator
from modules.generation.benchmark import constraint_similarity
from modules.generation.benchmark import implication_checker
from modules.generation.benchmark import unsat_generator

from .suite_config import BenchmarkSuite, ModelSpec, ProfileSpec, get_difficulty_profile


def _fmt_duration(seconds: float) -> str:
    """Human-readable wall-clock: '9.5s', '2m 15s', '1h 03m 20s'."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {sec:02d}s"


class EnhancedSuiteController:
    """
    Enhanced controller with all advanced features integrated.
    
    Enhancements:
    - Comprehensive logging at every step
    - Metadata enrichment with operators/depth/difficulty
    - UNSAT generation via mutation strategies
    - Structural (tree-edit) similarity for deduplication
    - Semantic similarity for clustering
    - Implication checking for relationships
    - Manifest.jsonl generation for ML pipelines
    """

    # Greedy compatibility pruning re-runs a batch solve per SAT candidate, but only
    # AFTER the cheap quick check detects a conflict. Generation-time consistency now
    # prevents most contradictions (boolean polarity, collection emptiness) at the
    # source, so the greedy rarely fires; this covers normal suite sizes (up to ~100
    # SAT) so numeric infeasibilities (e.g. staffCount/vehicleCount inequality chains)
    # are caught and pruned rather than deferred to a gate the encoder passes vacuously.
    MAX_GREEDY_COMPATIBILITY_CONSTRAINTS = 100
    
    def __init__(
        self,
        suite: BenchmarkSuite,
        verbose: bool = True,
        debug: bool = False,
        enable_research_features: bool = True
    ):
        """
        Initialize enhanced suite controller.
        
        Args:
            suite: Benchmark suite specification
            verbose: Enable verbose logging
            debug: Enable debug logging
            enable_research_features: Enable all 6 advanced features
        """
        self.suite = suite
        self.enable_research_features = enable_research_features
        self.logger = FrameworkLogger(verbose=verbose, debug=debug, name="SuiteController")
        
        self.output_root = Path(suite.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Per-model semantic admissibility counts (candidate same-type pairs,
        # approved, rejected). Persisted into each report.json so the
        # headline 'rejects X of Y pairs' figure is reproducible from artifacts.
        self._semantic_summary: Dict[str, Dict] = {}
        
        # Get git commit if not provided
        if suite.git_commit is None:
            suite.git_commit = self._get_git_commit()
        
        # Initialize advanced modules
        if self.enable_research_features:
            self.logger.step("Initializing advanced features...")
            self._init_research_modules()
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                self.logger.debug(f"Git commit: {commit[:8]}")
                return commit
        except Exception as e:
            self.logger.debug(f"Could not get git commit: {e}")
        return None
    
    def _init_research_modules(self):
        """Initialize all research feature modules."""
        self.logger.indent()
        
        try:
            self.logger.debug("Verifying metadata_enricher module...")
            assert hasattr(metadata_enricher, 'enrich_constraint_metadata')
            self.logger.success("metadata_enricher ready")
            
            self.logger.debug("Verifying manifest_generator module...")
            assert hasattr(manifest_generator, 'generate_manifest')
            self.logger.success("manifest_generator ready")
            
            self.logger.debug("Verifying constraint_similarity module...")
            assert hasattr(constraint_similarity, 'ast_similarity')
            assert hasattr(constraint_similarity, 'compute_embeddings_batch')
            self.logger.success("constraint_similarity ready")
            
            self.logger.debug("Verifying implication_checker module...")
            assert hasattr(implication_checker, 'check_syntactic_implication')
            self.logger.success("implication_checker ready")
            
            self.logger.debug("Verifying unsat_generator module...")
            assert hasattr(unsat_generator, 'generate_mixed_sat_unsat_set')
            self.logger.success("unsat_generator ready")
            
            self.logger.success("All 6 advanced modules verified")
            
        except Exception as e:
            self.logger.error(f"Failed to verify advanced modules: {e}")
            self.logger.warning("Advanced features will be disabled")
            self.enable_research_features = False
        
        self.logger.dedent()
    
    def generate_suite(self) -> Dict:
        """
        Generate complete benchmark suite with all advanced features.
        
        Returns:
            Suite statistics dictionary
        """
        suite_t0 = time.monotonic()
        self.logger.section(f"BENCHMARK SUITE GENERATION: {self.suite.suite_name}")
        self.logger.info(f"Version: {self.suite.version}")
        self.logger.info(f"Framework Version: {self.suite.framework_version}")
        self.logger.info(f"Output: {self.output_root}")
        self.logger.info(f"Advanced Features: {'ENABLED' if self.enable_research_features else 'DISABLED'}")

        suite_stats = {
            'suite_name': self.suite.suite_name,
            'version': self.suite.version,
            'generated_at': datetime.now().isoformat(),
            'research_features_enabled': self.enable_research_features,
            'models': [],
            'total_constraints': 0,
            'total_sat': 0,
            'total_unsat': 0,
            'total_valid': 0,
            'total_unknown': 0,
            'total_generation_time_s': 0.0,
        }
        
        # Process each model
        for model_idx, model_spec in enumerate(self.suite.models, 1):
            self.logger.section(
                f"MODEL {model_idx}/{len(self.suite.models)}: {model_spec.name}",
                char="-"
            )
            self.logger.info(f"XMI: {model_spec.xmi}")
            
            model_stats = self._process_model(model_spec, model_idx)
            suite_stats['models'].append(model_stats)
            
            # Aggregate statistics
            suite_stats['total_constraints'] += model_stats['total_constraints']
            suite_stats['total_valid'] += model_stats['total_valid']
            suite_stats['total_sat'] += model_stats['total_sat']
            suite_stats['total_unsat'] += model_stats['total_unsat']
            suite_stats['total_unknown'] += model_stats['total_unknown']
        
        # Total wall-clock for the whole suite (recorded before saving so it lands
        # in the summary JSON too).
        suite_stats['total_generation_time_s'] = round(time.monotonic() - suite_t0, 1)

        # Save suite summary
        summary_path = self.output_root / f"{self.suite.suite_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(suite_stats, f, indent=2)

        self.logger.success(f"Suite summary saved to: {summary_path}")

        # The final summary (banner, totals, VGCR table) is printed once by the
        # CLI entry point, which also prints in --quiet mode.
        return suite_stats
    
    def _process_model(self, model_spec: ModelSpec, model_idx: int) -> Dict:
        """Process single model with all its profiles."""
        model_stats = {
            'model': model_spec.name,
            'xmi': model_spec.xmi,
            'profiles': [],
            'total_constraints': 0,
            'total_valid': 0,
            'total_sat': 0,
            'total_unsat': 0,
            'total_unknown': 0
        }
        
        try:
            # Load metamodel
            self.logger.step("Loading metamodel from XMI...")
            self.logger.indent()
            
            extractor = MetamodelExtractor(model_spec.xmi)
            metamodel = extractor.get_metamodel()
            
            self.logger.success(f"Loaded {len(metamodel.classes)} classes")
            self.logger.debug(f"  - Attributes: {sum(len(c.attributes) for c in metamodel.classes.values())}")
            self.logger.debug(f"  - Associations: {len(metamodel.get_all_associations())}")
            
            self.logger.dedent()
            
            # LLM Semantic Analysis (pre-generation, runs once per metamodel)
            semantic_matrix = None
            sem_cfg = self.suite.semantic
            if sem_cfg.enable:
              try:
                from modules.semantic.llm_semantic_analyzer import LLMSemanticAnalyzer
                self.logger.step(f"Running LLM semantic analysis ({sem_cfg.model})...")
                # Ollama generate endpoint: append /api/generate if user provided base URL
                ollama_gen_url = sem_cfg.ollama_url
                if not ollama_gen_url.endswith("/api/generate"):
                    ollama_gen_url = ollama_gen_url.rstrip("/") + "/api/generate"
                analyzer = LLMSemanticAnalyzer(
                    model=sem_cfg.model,
                    ollama_url=ollama_gen_url,
                    use_cache=sem_cfg.use_cache,
                )
                semantic_matrix = analyzer.analyze_metamodel(metamodel, model_name=model_spec.name)
                n_classes = len(semantic_matrix.class_profiles)
                if n_classes > 0:
                    total_comp = sum(len(p.comparable_pairs) for p in semantic_matrix.class_profiles.values())
                    total_incomp = sum(len(p.incomparable_pairs) for p in semantic_matrix.class_profiles.values())
                    # State provenance and the denominator, so '0 comparable'
                    # reads as a verdict on N candidate pairs, not as a failure.
                    cov = analyzer.coverage_summary(semantic_matrix, metamodel)
                    source = (
                        f"from cache, generated {semantic_matrix.generated_at}"
                        if getattr(semantic_matrix, "from_cache", False) else "fresh analysis"
                    )
                    self.logger.success(
                        f"Semantic analysis complete ({source}): {n_classes} classes — "
                        f"{cov['llm_classes']} LLM-analysed, {cov['heuristic_classes']} heuristic"
                    )
                    self.logger.info(
                        f"  {total_comp} of {cov['candidate_pairs']} same-type attribute pairs "
                        f"approved as comparable, {total_incomp} explicitly rejected"
                    )
                    self._semantic_summary[model_spec.name] = {
                        "source": "llm" + ("+cache" if getattr(semantic_matrix, "from_cache", False) else ""),
                        "llm_model": sem_cfg.model,
                        "generated_at": getattr(semantic_matrix, "generated_at", ""),
                        "classes": n_classes,
                        "llm_classes": cov['llm_classes'],
                        "heuristic_classes": cov['heuristic_classes'],
                        "incomplete_classes": cov['incomplete_classes'],
                        "candidate_pairs": cov['candidate_pairs'],
                        "comparable_pairs": total_comp,
                        "explicitly_incomparable_pairs": total_incomp,
                        # Everything not approved is not generated: rules + LLM + heuristics.
                        "rejected_pairs": cov['candidate_pairs'] - total_comp,
                    }
                    if cov['incomplete_classes']:
                        self.logger.warning(
                            f"  Semantic matrix is incomplete for {cov['incomplete_classes']} "
                            f"classes that need LLM analysis — heuristics decide for them"
                        )
                else:
                    self.logger.warning("Semantic analysis returned empty matrix — using heuristic fallback")
                    self._semantic_summary[model_spec.name] = {"source": "heuristic", "reason": "empty_matrix"}
              except Exception as e:
                self.logger.warning(f"LLM semantic analysis failed: {e} — using heuristic fallback")
                self._semantic_summary[model_spec.name] = {"source": "heuristic", "reason": f"error: {e}"}
            else:
                self.logger.info("Semantic analysis disabled in config — using heuristic fallback")
                self._semantic_summary[model_spec.name] = {"source": "heuristic", "reason": "disabled"}

            # Initialize engine and verifier
            self.logger.step("Initializing generation engine...")
            engine = BenchmarkEngineV2(
                metamodel,
                verification_enabled=self.suite.verification.enable,
                semantic_matrix=semantic_matrix,
                generation_mode=self.suite.generation_mode,
            )
            self.logger.success(f"Engine initialized (generation_mode={engine.generation_mode})")
            
            verifier = None
            if self.suite.verification.enable:
                self.logger.step("Initializing constraint verifier...")
                verifier = FrameworkConstraintVerifier(
                    metamodel, model_spec.xmi,
                    scope_per_class=self.suite.verification.scope_per_class,
                    timeout_ms=self.suite.verification.per_constraint_timeout_ms,
                    batch_timeout_ms=self.suite.verification.batch_timeout_ms,
                    portfolio_seeds=getattr(self.suite.verification, 'portfolio_seeds', 0),
                    portfolio_first_try_ms=getattr(self.suite.verification, 'portfolio_first_try_ms', 60000),
                )
                self.logger.success(
                    "Verifier initialized"
                    + (f" (seed portfolio: {verifier.portfolio_seeds} seeds after "
                       f"{verifier.portfolio_first_try_ms // 1000}s)" if verifier.portfolio_seeds else "")
                )
            
            # Process each profile
            for prof_idx, prof_spec in enumerate(model_spec.profiles, 1):
                self.logger.subsection(
                    f"PROFILE {prof_idx}/{len(model_spec.profiles)}: {prof_spec.name}",
                    char="·"
                )
                
                prof_stats = self._generate_profile(
                    model_spec, prof_spec, engine, verifier, metamodel
                )
                
                model_stats['profiles'].append(prof_stats)
                model_stats['total_constraints'] += prof_stats['constraint_count']
                model_stats['total_valid'] += prof_stats.get('valid_count', 0)
                model_stats['total_sat'] += prof_stats.get('sat_count', 0)
                model_stats['total_unsat'] += prof_stats.get('unsat_count', 0)
                model_stats['total_unknown'] += prof_stats.get('unknown_count', 0)
        
        except Exception as e:
            self.logger.error(f"Failed to process model: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            model_stats['error'] = str(e)
        
        return model_stats
    
    def _generate_profile(
        self,
        model_spec: ModelSpec,
        prof_spec: ProfileSpec,
        engine: BenchmarkEngineV2,
        verifier: Optional[FrameworkConstraintVerifier],
        metamodel: Metamodel
    ) -> Dict:
        """Generate constraints for a single profile with advanced features."""
        profile_t0 = time.monotonic()
        # Set random seed
        random.seed(prof_spec.seed)
        self.logger.debug(f"Random seed set to: {prof_spec.seed}")
        
        # Build profile
        profile = self._build_profile_from_spec(prof_spec, metamodel)
        
        # Create output directory. A single-profile model writes straight to
        # <output_root>/<model> (e.g. benchmarks/CarRental); models with multiple
        # profiles keep a per-profile subdir (<model>/<profile>) so they don't
        # overwrite each other.
        if len(model_spec.profiles) == 1:
            output_dir = self.output_root / model_spec.name
        else:
            output_dir = self.output_root / model_spec.name / prof_spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Output directory: {output_dir}")
        
        prof_stats = {
            'profile': prof_spec.name,
            'seed': prof_spec.seed,
            'complexity': prof_spec.complexity_profile,
            'output_dir': str(output_dir),
            'research_features_applied': []
        }
        
        try:
            # ============================================================
            # STEP 1: Generate base SAT constraints
            # ============================================================
            # Apply per-profile generation-mode override (falls back to the
            # suite-level default when the profile does not specify one).
            engine.generation_mode = engine._normalize_generation_mode(
                prof_spec.generation_mode or self.suite.generation_mode
            )
            self.logger.step(
                f"Generating {prof_spec.constraints} base constraints "
                f"(mode={engine.generation_mode})..."
            )
            self.logger.indent()

            constraints = engine.generate(profile, progress_callback=None)
            prof_stats['constraint_count'] = len(constraints)
            prof_stats['initial_sat_count'] = len(constraints)

            # Persist the steered deviation report (envelope, feasibility gaps,
            # actuator usage, derived components) next to the benchmark.
            steered_report = getattr(engine, "steered_report", None)
            if steered_report is not None:
                report_obj = dict(steered_report)
                report_obj["model"] = model_spec.name
                report_obj["profile"] = prof_spec.name
                report_obj["semantic_admissibility"] = self._semantic_summary.get(model_spec.name)
                report_path = output_dir / "report.json"
                with open(report_path, "w") as f:
                    json.dump(report_obj, f, indent=2)
                self.logger.info(f"Steered report saved to {report_path}")

            self.logger.success(f"Generated {len(constraints)} base constraints")
            self.logger.dedent()
            
            # ============================================================
            # STEP 2: Metadata Enrichment (TC-based complexity metrics)
            # Always enabled — core complexity metrics for benchmark quality
            # ============================================================
            if True:  # Always run enrichment (complexity metrics are core, not research-only)
                self.logger.step("Applying metadata enrichment (complexity metrics)...")
                self.logger.indent()

                # Pass 1: Enrich all constraints (without RUC — needs full set)
                enriched_constraints = []
                for i, constraint in enumerate(constraints):
                    enriched = metadata_enricher.enrich_constraint_metadata(
                        constraint, metamodel=metamodel
                    )
                    enriched_constraints.append(enriched)

                # Pass 2: Recompute with full constraint list for RUC (dependency metric)
                for i, constraint in enumerate(enriched_constraints):
                    enriched = metadata_enricher.enrich_constraint_metadata(
                        constraint, metamodel=metamodel, all_constraints=enriched_constraints
                    )
                    enriched_constraints[i] = enriched

                    if i < 3:  # Show first 3 for debugging
                        cm = enriched.metadata.get('complexity_metrics', {})
                        self.logger.debug(f"  Constraint {i+1}:")
                        self.logger.debug(f"    - Operators: {enriched.metadata.get('operators_used', [])}")
                        self.logger.debug(f"    - Difficulty: {enriched.metadata.get('difficulty', 'unknown')}")
                        self.logger.debug(f"    - TC: {cm.get('tc', 0.0)}")

                constraints = enriched_constraints

                # Log TC distribution
                tc_scores = [c.metadata.get('complexity_metrics', {}).get('tc', 0.0)
                             for c in constraints]
                if tc_scores:
                    avg_tc = sum(tc_scores) / len(tc_scores)
                    min_tc = min(tc_scores)
                    max_tc = max(tc_scores)
                    self.logger.info(f"TC distribution: min={min_tc:.1f}, avg={avg_tc:.1f}, max={max_tc:.1f}")

                prof_stats['research_features_applied'].append('metadata_enrichment')
                self.logger.success(f"Enriched {len(constraints)} constraints with complexity metrics")
                self.logger.dedent()

            # Snapshot per-class coverage BEFORE any refinement drop (VGCR / STEP
            # 3.4 / 3.5), so the final audit (STEP 7.9) can flag classes that met
            # the per-class minimum here but were dropped below it by refinement.
            from collections import Counter as _Counter
            pre_refine_counts = _Counter(
                getattr(c, "context", None) for c in constraints
            )

            # ============================================================
            # STEP 2.4: Structural deduplication WITH refill (pre-verification)
            # ============================================================
            # Deduplicating after verification silently deletes slots and breaks
            # the tier/family quotas. Instead, near-duplicates are removed HERE
            # and their slots are sent back through the fill loop (same family,
            # same complexity box); VGCR then keeps later replacements distinct.
            dedup_threshold = self._dedup_threshold(prof_spec)
            if self.enable_research_features:
                self.logger.step("Deduplicating candidates (tree-edit) with slot refill...")
                self.logger.indent()
                constraints, dedup_info = self._dedup_and_refill(
                    constraints, engine, dedup_threshold, metamodel
                )
                prof_stats['dedup'] = dedup_info
                if dedup_info['removed']:
                    self.logger.info(
                        f"Near-duplicates (>{dedup_threshold*100:.0f}% similar): "
                        f"{dedup_info['removed']} removed, "
                        f"{dedup_info['refilled']} refilled in the same family and box, "
                        f"{len(dedup_info['unfilled'])} left unfilled (reported)"
                    )
                else:
                    self.logger.success(
                        f"No duplicates found (threshold: {dedup_threshold*100:.0f}%)"
                    )
                self.logger.dedent()

            # ============================================================
            # STEP 2.5: VGCR Refinement Loop (SAT-core construction)
            # ============================================================
            vgcr_enabled = (
                getattr(self.suite, 'vgcr', None)
                and self.suite.vgcr.enable
                and verifier
                and getattr(verifier, 'framework_available', False)
            )
            if vgcr_enabled:
                self.logger.step("Running VGCR refinement loop (SAT-core construction)...")
                self.logger.indent()

                try:
                    from modules.verification.vgcr_verifier import VGCRPropertyChecker
                    from modules.generation.benchmark.vgcr_refinement import VGCRLoop
                    from modules.generation.benchmark.complexity_calculator import ComplexityWeights
                    from modules.generation.benchmark.metadata_enricher import get_complexity_weights

                    # Determine TC range from profile
                    tc_range_cfg = (
                        self.suite.vgcr.tc_range_override
                        or getattr(prof_spec, 'target_tc_range', None)
                        or {"min": 3.0, "max": 25.0}
                    )
                    tc_range = (
                        tc_range_cfg.get("min", 3.0),
                        tc_range_cfg.get("max", 25.0),
                    )

                    # Initialise property checker
                    checker = VGCRPropertyChecker(
                        verifier=verifier,
                        metamodel=metamodel,
                        complexity_weights=get_complexity_weights(),
                        enable_independence=self.suite.vgcr.enable_independence_check,
                        independence_threshold=self.suite.vgcr.independence_threshold,
                    )

                    # Initialise VGCR loop
                    vgcr_loop = VGCRLoop(
                        engine=engine,
                        checker=checker,
                        metamodel=metamodel,
                        max_retries=self.suite.vgcr.max_retries,
                        is_duplicate=(
                            (lambda cand, suite_:
                                self._is_near_duplicate(cand, suite_, dedup_threshold))
                            if self.enable_research_features else None
                        ),
                    )

                    # Run refinement
                    before_count = len(constraints)
                    constraints, vgcr_stats = vgcr_loop.refine_suite(
                        constraints, tc_range
                    )

                    # STEP 2.6: refill classes that refinement dropped below the
                    # per-class minimum (verified fill of an unfilled slot, else a
                    # same-family same-box donor swap). Unrepairable classes are
                    # reported by the final audit, never padded.
                    _cov_msg = None
                    _Lmin = prof_spec.per_class_min or 1
                    _now = _Counter(getattr(c, "context", None) for c in constraints)
                    _deficit = [
                        cls for cls, n0 in pre_refine_counts.items()
                        if cls is not None and n0 >= _Lmin and _now.get(cls, 0) < _Lmin
                    ]
                    if _deficit:
                        constraints, _cov_log = vgcr_loop.repair_coverage(
                            constraints, tc_range, _deficit, min_per_class=_Lmin
                        )
                        prof_stats['coverage_repairs'] = _cov_log
                        vgcr_stats['unfilled_slots'] = list(vgcr_loop.unfilled_log)
                        _fixed = [r['class'] for r in _cov_log if r['action'] != 'unrepaired']
                        _open = [r['class'] for r in _cov_log if r['action'] == 'unrepaired']
                        # Logged after the VGCR summary below: the repair runs
                        # after the refinement loop, so it should read that way.
                        _cov_msg = (
                            f"Coverage repair: {len(_deficit)} classes below min={_Lmin} "
                            f"after refinement; repaired {_fixed or 'none'}"
                            + (f", unrepaired {_open}" if _open else "")
                        )
                    after_count = len(constraints)

                    # VGCR may replace failed candidates with freshly-generated
                    # constraints that never went through STEP 2 enrichment, so they
                    # would be written with no difficulty label. Re-enrich any such
                    # constraint (idempotent for those already enriched). Then map
                    # every label onto the steered 3-tier vocabulary so policy/skip
                    # fills and the 5-bucket TC fallback (trivial/hard/expert) do not
                    # mix 'hard' in with the profile labels (easy/medium/difficult).
                    _TIER = {'trivial': 'easy', 'hard': 'difficult', 'expert': 'difficult'}
                    for _c in constraints:
                        if not _c.metadata.get('difficulty'):
                            metadata_enricher.enrich_constraint_metadata(
                                _c, metamodel=metamodel
                            )
                        _d = _c.metadata.get('difficulty')
                        if _d in _TIER:
                            _c.metadata['difficulty'] = _TIER[_d]

                    prof_stats['vgcr_stats'] = vgcr_stats
                    prof_stats['vgcr_accepted'] = after_count
                    prof_stats['vgcr_rejected'] = before_count - after_count
                    prof_stats['research_features_applied'].append('vgcr_refinement')

                    self.logger.success(
                        f"VGCR: {after_count}/{before_count} slots verified — "
                        f"{vgcr_stats.get('passed_first_try', 0)} passed first try, "
                        f"{vgcr_stats.get('passed_after_retry', 0)} regenerated"
                        f"{' in their exact box' if getattr(engine, '_steered_gen', None) is not None else ''}, "
                        f"{len(vgcr_stats.get('unfilled_slots', []))} left unfilled (reported); "
                        f"{vgcr_stats.get('duplicates_rejected', 0)} near-duplicate candidates rejected"
                    )
                    if vgcr_stats.get('failure_breakdown'):
                        self.logger.info(
                            f"  Failure breakdown: {vgcr_stats['failure_breakdown']}"
                        )
                    self.logger.info(
                        f"  SMT queries: {vgcr_stats.get('total_smt_queries', 0)}, "
                        f"  SMT time: {vgcr_stats.get('total_smt_time_s', 0):.1f}s"
                    )
                    if _cov_msg:
                        self.logger.info(_cov_msg)

                except ImportError as e:
                    self.logger.warning(f"VGCR modules not available: {e}")
                    self.logger.warning("Skipping VGCR refinement — using raw candidates")
                except Exception as e:
                    self.logger.error(f"VGCR refinement failed: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    self.logger.warning("Falling back to raw candidates")

                self.logger.dedent()
            else:
                if getattr(self.suite, 'vgcr', None) and self.suite.vgcr.enable:
                    self.logger.info(
                        "VGCR enabled but verifier not available — skipping"
                    )

            # ============================================================
            # STEP 3: UNSAT Generation (Z3-verified, retry loop)
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Generating UNSAT constraints via mutation...")
                self.logger.indent()

                import contextlib, io as _io

                target_unsat_ratio = prof_spec.unsat_ratio or 0.4
                n_total = len(constraints)
                target_unsat = int(n_total * target_unsat_ratio)
                self.logger.debug(
                    f"Target UNSAT ratio: {target_unsat_ratio*100:.0f}% "
                    f"({target_unsat} of {n_total})"
                )

                has_z3 = verifier and getattr(verifier, 'framework_available', False)

                # Shuffle candidate indices so we don't always mutate the same ones
                candidate_indices = list(range(n_total))
                random.shuffle(candidate_indices)

                confirmed_unsat = []          # verified UNSAT constraints
                consumed_indices = set()      # SAT indices consumed by confirmed UNSAT
                mutation_counts = {}          # strategy → count

                for idx in candidate_indices:
                    if len(confirmed_unsat) >= target_unsat:
                        break

                    sat_constraint = constraints[idx]

                    # Try each strategy (order: specific → general fallback)
                    for strategy in unsat_generator.ALL_STRATEGIES:
                        if not strategy.can_apply(sat_constraint, metamodel):
                            continue

                        candidate = strategy.apply(sat_constraint, metamodel)

                        # self_contradiction (P ∧ ¬P) is guaranteed UNSAT by
                        # construction — skip Z3 which may timeout on complex OCL.
                        is_tautological = candidate.metadata.get('mutation') == 'self_contradiction'

                        if has_z3 and not is_tautological:
                            with contextlib.redirect_stdout(_io.StringIO()):
                                is_unsat, msg = unsat_generator.verify_unsat_generation(
                                    candidate, verifier,
                                    original_constraint=sat_constraint
                                )
                            if not is_unsat:
                                self.logger.debug(
                                    f"  Strategy {strategy.get_name()} failed Z3: {msg}"
                                )
                                continue  # try next strategy for same constraint

                        # Confirmed (or no Z3 available — trust the mutation)
                        confirmed_unsat.append(candidate)
                        consumed_indices.add(idx)
                        mut_name = candidate.metadata.get('mutation', 'unknown')
                        mutation_counts[mut_name] = mutation_counts.get(mut_name, 0) + 1
                        break  # move to next SAT candidate

                # Build final set: keep SAT constraints that weren't consumed + confirmed UNSAT
                sat_kept = [c for i, c in enumerate(constraints) if i not in consumed_indices]
                all_constraints = sat_kept + confirmed_unsat

                sat_count = len(sat_kept)
                unsat_count = len(confirmed_unsat)

                self.logger.success(f"Generated {unsat_count} verified UNSAT constraints")
                if unsat_count < target_unsat:
                    self.logger.warning(
                        f"Could only produce {unsat_count}/{target_unsat} UNSAT "
                        f"(exhausted {len(candidate_indices)} candidates)"
                    )
                if mutation_counts:
                    self.logger.debug(f"  Mutation strategies: {mutation_counts}")

                prof_stats['sat_count_after_mutation'] = sat_count
                prof_stats['unsat_count_after_mutation'] = unsat_count
                prof_stats['constraint_count'] = len(all_constraints)
                prof_stats['research_features_applied'].append('unsat_generation')

                self.logger.info(
                    f"Total constraints: {len(all_constraints)} ({sat_count} SAT + {unsat_count} UNSAT)"
                )

                # A mutant is structurally heavier than its parent: re-measure it
                # and give it its own tier (parent tier kept as 'slot_profile').
                unsat_tiers = self._relabel_unsat(confirmed_unsat, engine)
                if unsat_tiers:
                    prof_stats['unsat_measured_tiers'] = unsat_tiers
                    self.logger.info(f"UNSAT mutants re-measured: tiers={unsat_tiers}")
                self.logger.dedent()

                constraints = all_constraints

            # ============================================================
            # STEP 3.4: Structural per-context consistency (cheap, no Z3)
            # ============================================================
            # Drops SAT constraints that make a context's invariant set contradictory
            # on a boolean attribute (forced true AND false) or a collection (forced
            # empty AND non-empty). This is a deterministic backstop for two gaps the
            # Z3 gate below cannot close: (a) per-context contradictions it passes via
            # VACUOUS satisfaction (a model with zero instances of that class), and
            # (b) contradictions re-introduced by VGCR refinement / UNSAT mutation,
            # which bypass the generator's own gen-time consistency tracker.
            try:
                from .steered_generation import SteeredGenerator as _SG
                _claims = {}                          # (context, key) -> polarity / sign-set
                _kept, _dropped = [], 0
                for _c in constraints:
                    if _c.metadata.get('is_unsat', False):
                        _kept.append(_c)               # never touch intended-UNSAT
                        continue
                    if _SG._check_register(_c.context, _c.ocl, _claims):
                        _kept.append(_c)
                    else:
                        _dropped += 1
                if _dropped:
                    self.logger.warning(
                        f"Dropped {_dropped} structurally-contradictory SAT constraints "
                        "(boolean polarity / collection emptiness per context)"
                    )
                    constraints = _kept
                    prof_stats['structural_conflicts_removed'] = _dropped
            except Exception as _e:
                self.logger.debug(f"Structural consistency pass skipped: {_e}")

            # ============================================================
            # STEP 3.5: Constraint Compatibility Checking (Batch Mode)
            # ============================================================
            # NOTE: Runs silently - uses greedy algorithm to find maximal compatible subset
            should_check_global_consistency = bool(
                verifier
                and getattr(verifier, 'framework_available', False)
                and len(constraints) > 0
                and getattr(self.suite.verification, 'check_global_consistency', True)
            )
            if should_check_global_consistency:
                self.logger.step("Checking SAT constraint compatibility...")
                self.logger.indent()
                sat_constraints_only = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                
                if len(sat_constraints_only) > 0:
                    if self._should_attempt_compatibility_pruning(len(sat_constraints_only)):
                        try:
                            self.logger.debug(
                                f"Checking {len(sat_constraints_only)} SAT constraints for global consistency"
                            )

                            # Quick compatibility check (silent). One tracked solver call;
                            # on UNSAT the unsat core names the conflicting constraints.
                            if hasattr(verifier, 'check_consistency_with_core'):
                                compat = verifier.check_consistency_with_core(sat_constraints_only)
                                is_consistent = compat['solver_result'] == 'sat'
                                is_inconsistent = compat['solver_result'] == 'unsat'
                            else:
                                compat_results = verifier.verify_batch(sat_constraints_only, silent=True)
                                is_consistent = any(r.solver_result == 'sat' for r in compat_results)
                                is_inconsistent = any(r.solver_result == 'unsat' for r in compat_results)

                            if is_consistent:
                                self.logger.success("SAT constraint set is globally consistent")
                                prof_stats['consistency_verified'] = True
                            elif not is_inconsistent:
                                # Solver timeout / unknown is NOT evidence of a conflict. Pruning
                                # here would make the delivered suite depend on machine speed.
                                prof_stats['consistency_verified'] = None
                                self.logger.warning(
                                    "SAT compatibility check was inconclusive (solver timeout) — "
                                    "no constraints pruned; the final Phase B check decides"
                                )
                            else:
                                # Model UNSAT - find compatible subset using greedy algorithm
                                if hasattr(verifier, 'check_consistency_with_core'):
                                    compatible_sat = self._find_compatible_subset_by_core(sat_constraints_only, verifier)
                                else:
                                    compatible_sat = self._find_compatible_subset_batch(sat_constraints_only, verifier)

                                if len(compatible_sat) < len(sat_constraints_only):
                                    removed = len(sat_constraints_only) - len(compatible_sat)

                                    # Rebuild constraints list
                                    unsat_constraints_only = [c for c in constraints if c.metadata.get('is_unsat', False)]
                                    constraints = compatible_sat + unsat_constraints_only
                                    prof_stats['constraint_count'] = len(constraints)
                                    prof_stats['conflicts_removed'] = removed
                                    self.logger.warning(
                                        f"Removed {removed} conflicting SAT constraints before deduplication"
                                    )
                                    self.logger.info(
                                        f"Post-pruning total: {len(constraints)} "
                                        f"({len(compatible_sat)} SAT + {len(unsat_constraints_only)} UNSAT)"
                                    )

                                    # Re-check consistency after pruning
                                    recheck = verifier.verify_batch(compatible_sat, silent=True)
                                    is_consistent_after = any(r.solver_result == 'sat' for r in recheck)
                                    prof_stats['consistency_verified'] = is_consistent_after
                                    if is_consistent_after:
                                        self.logger.success("Post-pruning SAT set is globally consistent")
                                    else:
                                        self.logger.warning(
                                            "Post-pruning SAT set is STILL inconsistent — "
                                            "consistency_verified=false will be recorded in output"
                                        )
                                else:
                                    prof_stats['consistency_verified'] = False
                                    self.logger.warning(
                                        "SAT constraint set was inconsistent, but no smaller compatible subset was found"
                                    )
                        except Exception as e:
                            prof_stats['consistency_verified'] = False
                            self.logger.warning(f"SAT compatibility check failed: {e}")
                    else:
                        self.logger.info(
                            "Deferring greedy SAT compatibility pruning to final verification "
                            f"because {len(sat_constraints_only)} SAT constraints exceed the "
                            f"{self.MAX_GREEDY_COMPATIBILITY_CONSTRAINTS}-constraint recovery budget"
                        )
                else:
                    self.logger.success("No SAT constraints to check for compatibility")
                self.logger.dedent()

                # Best-effort gate (save-safe): the greedy pass above already prunes
                # to a consistent subset. If some residual inconsistency remains we
                # still SAVE the pruned benchmark and record consistency_verified=false,
                # rather than aborting to no output (which surprises the user as
                # "nothing was saved"). The flag is recorded in the output metadata.
                if prof_stats.get('consistency_verified') is False:
                    self.logger.warning(
                        "SAT constraints not fully verified consistent after pruning — "
                        "saving the best-effort set with consistency_verified=false recorded."
                    )
            elif verifier and len(constraints) > 0:
                self.logger.info("Skipping SAT compatibility pruning: framework verifier unavailable")

            # ============================================================
            # STEP 4: Structural Similarity (Tree-Edit) — late safety net
            # ============================================================
            # Deduplication proper happens in STEP 2.4 (with refill) and inside
            # VGCR. This pass only catches SAT/SAT duplicates introduced by
            # later steps; anything it removes is a lost slot and is reported
            # as 'late_duplicates_removed' in the final certificate.
            if self.enable_research_features:
                self.logger.step("Checking for late structural duplicates (tree-edit)...")
                self.logger.indent()
                
                before_count = len(constraints)
                
                # Compute pairwise similarities
                similarities = []
                for i in range(len(constraints)):
                    for j in range(i + 1, len(constraints)):
                        sim = constraint_similarity.ast_similarity(constraints[i], constraints[j])
                        similarities.append((i, j, sim))
                
                self.logger.debug(f"Computed {len(similarities)} pairwise similarities")
                
                # Simple deduplication: remove constraints with similarity > threshold.
                # Use 'is not None' so an explicit 0.0 (max dedup) is honoured rather
                # than silently falling back to the default (0.0 is falsy).
                threshold = (prof_spec.similarity_threshold
                             if prof_spec.similarity_threshold is not None else 0.85)
                self.logger.debug(f"Deduplication threshold: {threshold}")
                
                # Mark duplicates (never remove confirmed UNSAT constraints)
                to_remove = set()
                for i, j, sim in similarities:
                    if sim > threshold:
                        # A SAT constraint and an UNSAT mutant are never
                        # duplicates of each other, and UNSAT is protected.
                        if (constraints[i].metadata.get('is_unsat', False)
                                or constraints[j].metadata.get('is_unsat', False)):
                            continue
                        if i in to_remove:
                            continue  # i is already gone; j duplicates nothing kept
                        to_remove.add(j)  # Keep first occurrence, remove second
                
                dedup_constraints = [c for idx, c in enumerate(constraints) if idx not in to_remove]
                
                removed = before_count - len(dedup_constraints)
                if removed > 0:
                    self.logger.warning(
                        f"Removed {removed} late duplicate constraints "
                        f"(>{threshold*100:.0f}% similar) — these slots are lost and reported"
                    )
                    constraints = dedup_constraints
                    prof_stats['constraint_count'] = len(constraints)
                    prof_stats['duplicates_removed'] = removed
                else:
                    self.logger.success(f"No duplicates found (threshold: {threshold*100:.0f}%)")
                
                prof_stats['research_features_applied'].append('ast_similarity')
                self.logger.dedent()
            
            # ============================================================
            # STEP 5: Semantic Similarity & Clustering
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Computing semantic similarity...")
                self.logger.indent()
                
                try:
                    # Cluster constraints by embedding similarity. The clustering
                    # call computes the embeddings internally (a single encode pass),
                    # and is run under a watchdog so a native torch/tokenizers stall
                    # degrades to "skip clustering" instead of hanging the pipeline.
                    clustering_threshold = 0.75  # Constraints with >75% similarity clustered together
                    clusters = constraint_similarity.cluster_by_semantic_similarity_safe(
                        constraints, threshold=clustering_threshold, timeout_s=90.0
                    )

                    num_clusters = len(clusters)
                    self.logger.success(f"Clustered into {num_clusters} semantic groups")
                    
                    # Build cluster assignment map (constraint index -> cluster id)
                    cluster_assignment = {}
                    for cluster_id, cluster_indices in enumerate(clusters):
                        for idx in cluster_indices:
                            cluster_assignment[idx] = cluster_id
                    
                    # Log cluster distribution
                    cluster_sizes = [len(c) for c in clusters]
                    self.logger.debug(f"  Cluster sizes: {cluster_sizes}")
                    
                    # Add cluster info to metadata
                    for idx, constraint in enumerate(constraints):
                        constraint.metadata['semantic_cluster'] = cluster_assignment.get(idx, -1)
                    
                    prof_stats['semantic_clusters'] = num_clusters
                    prof_stats['research_features_applied'].append('semantic_similarity')
                except Exception as e:
                    self.logger.warning(f"Semantic clustering failed: {e}")
                    self.logger.debug("Continuing without semantic clustering")
                
                self.logger.dedent()
            
            # ============================================================
            # STEP 6: Implication Checking
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Checking constraint implications...")
                self.logger.indent()
                
                # Check implications between constraints (syntactic, optionally solver-based)
                implications = {f"{c.pattern_id}_{c.context}": [] for c in constraints}
                impl_count = 0
                use_solver = bool(
                    verifier
                    and self.suite.verification.enable
                    and getattr(self.suite.verification, 'implication_use_z3', False)
                )
                
                relations = implication_checker.find_implications(
                    constraints,
                    use_z3=use_solver,
                    verifier=verifier,
                    same_context_only=True
                )
                
                for i, j, relation in relations:
                    if relation == implication_checker.ImplicationRelation.C1_IMPLIES_C2:
                        implications[f"{constraints[i].pattern_id}_{constraints[i].context}"].append(
                            f"{constraints[j].pattern_id}_{constraints[j].context}"
                        )
                        impl_count += 1
                    elif relation == implication_checker.ImplicationRelation.C2_IMPLIES_C1:
                        implications[f"{constraints[j].pattern_id}_{constraints[j].context}"].append(
                            f"{constraints[i].pattern_id}_{constraints[i].context}"
                        )
                        impl_count += 1
                    elif relation == implication_checker.ImplicationRelation.EQUIVALENT:
                        implications[f"{constraints[i].pattern_id}_{constraints[i].context}"].append(
                            f"{constraints[j].pattern_id}_{constraints[j].context}"
                        )
                        implications[f"{constraints[j].pattern_id}_{constraints[j].context}"].append(
                            f"{constraints[i].pattern_id}_{constraints[i].context}"
                        )
                        impl_count += 2
                
                self.logger.success(f"Found {impl_count} implication relationships")
                
                if impl_count > 0:
                    # Show some examples
                    shown = 0
                    for constraint_id, implies_list in implications.items():
                        if implies_list and shown < 3:
                            self.logger.debug(f"  {constraint_id} implies {len(implies_list)} constraints")
                            shown += 1
                
                # Add implication info to metadata
                for constraint in constraints:
                    constraint_id = f"{constraint.pattern_id}_{constraint.context}"
                    constraint.metadata['implies'] = implications.get(constraint_id, [])
                
                prof_stats['implication_count'] = impl_count
                prof_stats['research_features_applied'].append('implication_checking')
                self.logger.dedent()
            
            # ============================================================
            # STEP 7: Verification — two phases
            #
            # Phase A (individual): each generated SAT constraint is verified
            #   on its own.  Any constraint that is individually UNSAT (i.e.
            #   no model instance satisfies it regardless of other constraints)
            #   is a bad constraint and is removed from the benchmark.
            #
            # Phase B (global consistency): the surviving, individually-SAT
            #   constraints are sent together to Z3.  A satisfying model at
            #   the configured bounded scope proves the benchmark is consistent
            #   per the paper's definition.  If Z3 still returns UNSAT here,
            #   the profile is aborted.
            # ============================================================
            verif_results = []
            if verifier and self.suite.verification.enable:
                framework_available = getattr(verifier, 'framework_available', False)
                self.logger.step(
                    "Verifying constraints with Z3 (individual + global)..."
                    if framework_available
                    else "Recording fallback verification results..."
                )
                self.logger.indent()

                sat_constraints_for_verification = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                unsat_constraints_skipped        = [c for c in constraints if c.metadata.get('is_unsat', False)]

                self.logger.info(
                    f"  SAT to verify: {len(sat_constraints_for_verification)}, "
                    f"intentional UNSAT (skipped): {len(unsat_constraints_skipped)}"
                )

                # ── Phase A: individual satisfiability ───────────────────────
                # Each constraint is checked in isolation with per_constraint
                # timeout.  Individually-UNSAT constraints are generation
                # artefacts; remove them before the global check.
                indiv_results   = []   # one result per sat constraint
                individually_ok = []   # constraints that passed Phase A

                if framework_available:
                    self.logger.info("  Phase A: individual satisfiability check...")
                    for idx, c in enumerate(sat_constraints_for_verification, 1):
                        r = verifier.verify(c)
                        indiv_results.append(r)
                        if r.solver_result == 'sat':
                            individually_ok.append(c)
                        elif r.solver_result == 'unsat':
                            self.logger.debug(
                                f"    [{idx}] individually UNSAT — excluded: "
                                f"{c.pattern_id} on {c.context}"
                            )
                        if idx % 20 == 0 or idx == len(sat_constraints_for_verification):
                            self.logger.debug(
                                f"    Progress: {idx}/{len(sat_constraints_for_verification)} checked, "
                                f"{len(individually_ok)} individually SAT so far"
                            )

                    indiv_removed = len(sat_constraints_for_verification) - len(individually_ok)
                    if indiv_removed:
                        self.logger.warning(
                            f"  Phase A: removed {indiv_removed} individually-UNSAT constraints"
                        )
                    self.logger.info(
                        f"  Phase A done: {len(individually_ok)}/{len(sat_constraints_for_verification)} "
                        f"constraints are individually satisfiable"
                    )

                    # Rebuild constraints list to drop Phase A failures
                    constraints = individually_ok + unsat_constraints_skipped
                    prof_stats['constraint_count'] = len(constraints)
                else:
                    # Framework unavailable — treat all as individually ok
                    individually_ok = sat_constraints_for_verification
                    indiv_results   = [verifier.verify(c) for c in sat_constraints_for_verification]

                # ── Trim verif_results to survivors only ─────────────────────
                # After Phase A, `constraints` was rebuilt to contain only
                # individually_ok + unsat_constraints_skipped.  verif_results
                # must match that list so downstream Steps 7.5+ can zip them
                # without index errors.  Drop results for the removed constraints.
                survivor_ids = {id(c) for c in individually_ok}
                verif_results = [
                    r
                    for c, r in zip(sat_constraints_for_verification, indiv_results)
                    if id(c) in survivor_ids
                ]
                # verif_results now has exactly len(individually_ok) entries,
                # all with solver_result='sat' from Phase A.

                # ── Phase B: global consistency ──────────────────────────────
                # Send the surviving individually-SAT constraints together to
                # Z3.  With adequate bounded scope this should return SAT,
                # proving the benchmark is consistent (paper definition).
                global_consistency = None

                if framework_available and individually_ok:
                    self.logger.info(
                        f"  Phase B: global consistency check "
                        f"(scope={self.suite.verification.scope_per_class} per class, "
                        f"timeout={self.suite.verification.batch_timeout_ms}ms)..."
                    )
                    # Tracked joint check first: one solver call with unsat-core support and,
                    # if configured, the seed portfolio for inconclusive runs. Fall back to the
                    # legacy batch check only when it cannot decide.
                    batch_results = None
                    global_consistency = None
                    if hasattr(verifier, 'check_consistency_with_core'):
                        tracked = verifier.check_consistency_with_core(individually_ok)
                        if tracked['solver_result'] in ('sat', 'unsat') and not tracked['unencoded']:
                            global_consistency = tracked['solver_result'] == 'sat'
                            if tracked.get('seed'):
                                self.logger.info(f"  Phase B decided by seed portfolio (seed {tracked['seed']})")
                            if tracked['core']:
                                prof_stats['phase_b_unsat_core'] = [individually_ok[k].pattern_id + '@' + individually_ok[k].context for k in tracked['core']]
                    tracked_ran = hasattr(verifier, 'check_consistency_with_core') and not tracked['unencoded']
                    if global_consistency is None and not tracked_ran:
                        # The legacy batch check is only a fallback for constraints the tracked
                        # path could not encode; re-running an inconclusive check just doubles the wait.
                        batch_results = verifier.verify_batch(individually_ok, silent=True)
                        global_consistency = self._infer_batch_consistency(batch_results)

                    if global_consistency is True:
                        self.logger.success("  Phase B: globally consistent ✓")
                        # Use batch results so final labels carry the SAT model
                        # (the tracked check keeps the Phase A labels, which are SAT).
                        if batch_results is not None:
                            verif_results = batch_results
                    elif global_consistency is False:
                        self.logger.error(
                            f"  Phase B: UNSAT at scope "
                            f"{self.suite.verification.scope_per_class} — "
                            "consider increasing scope_per_class"
                        )
                        # Keep Phase A 'sat' results — each constraint is still
                        # individually satisfiable; only the joint check failed.
                    else:
                        # Timeout — treat as unknown for global consistency;
                        # keep Phase A 'sat' labels for individual constraints.
                        self.logger.warning(
                            f"  Phase B: timed out at scope "
                            f"{self.suite.verification.scope_per_class} — "
                            "global consistency unknown (try increasing scope_per_class "
                            "or batch_timeout_ms)"
                        )

                # ── Aggregate counts ─────────────────────────────────────────
                valid_count   = sum(1 for r in verif_results if r.is_valid)
                sat_count     = sum(1 for r in verif_results if r.solver_result == 'sat')
                unsat_count   = sum(1 for r in verif_results if r.solver_result == 'unsat')
                unknown_count = sum(1 for r in verif_results if r.solver_result == 'unknown')

                total_unsat_constraints = unsat_count + len(unsat_constraints_skipped)
                verified_count  = len(individually_ok) if framework_available else 0
                fallback_count  = 0 if framework_available else len(individually_ok)

                if global_consistency is not None:
                    prof_stats['consistency_verified'] = global_consistency

                prof_stats.update({
                    'valid_count':             valid_count,
                    'sat_count':               sat_count,
                    'unsat_count':             total_unsat_constraints,
                    'unknown_count':           unknown_count,
                    'verified_count':          verified_count,
                    'fallback_unknown_count':  fallback_count,
                    'skipped_unsat_count':     len(unsat_constraints_skipped),
                    'intentional_unsat_count': len(unsat_constraints_skipped),
                })

                self.logger.success("Verification complete:")
                if framework_available:
                    self.logger.info(
                        f"  Phase A (individual): {len(individually_ok)}/{len(sat_constraints_for_verification)} "
                        f"constraints individually satisfiable"
                    )
                    consistency_label = (
                        "consistent ✓" if global_consistency is True
                        else ("UNSAT — see above" if global_consistency is False
                              else "unknown (timeout)")
                    )
                    self.logger.info(f"  Phase B (global):     {consistency_label}")
                else:
                    self.logger.info("  Verified: 0 SAT constraints (framework unavailable)")
                    self.logger.info(
                        f"  Fallback: {len(individually_ok)} constraints marked Unknown"
                    )
                self.logger.info(f"  Skipped: {len(unsat_constraints_skipped)} UNSAT constraints (intentionally contradictory)")
                self.logger.info(
                    f"  Valid: {valid_count}/{len(individually_ok)} "
                    f"({valid_count / len(individually_ok) * 100:.1f}%)"
                    if individually_ok else "  Valid: 0/0 (N/A)"
                )
                self.logger.info(f"  Result: {sat_count} SAT, {total_unsat_constraints} UNSAT, {unknown_count} Unknown")

                # Abort only when Phase B fails (global consistency violated
                # even after individually-UNSAT constraints were removed).
                if (framework_available and individually_ok
                        and global_consistency is False
                        and getattr(self.suite.verification, 'check_global_consistency', True)):
                    self.logger.error(
                        "Aborting profile — SAT constraints are not globally consistent "
                        f"at scope {self.suite.verification.scope_per_class}. "
                        "Try increasing scope_per_class in the verification config."
                    )
                    prof_stats['status'] = 'skipped'
                    prof_stats['skip_reason'] = 'SAT constraints not globally consistent'
                    self.logger.dedent()
                    return prof_stats
                self.logger.dedent()
            
            # ============================================================
            # STEP 7.5: Exclude UNKNOWN and encoding-error constraints
            # ============================================================
            # Only keep constraints with definitive verification results.
            # - UNKNOWN: solver timed out — no label
            # - error: encoder couldn't parse the pattern — never verified
            if verif_results:
                sat_constraints_for_verification = [
                    c for c in constraints if not c.metadata.get('is_unsat', False)
                ]
                exclude_indices = set()
                for idx, r in enumerate(verif_results):
                    if r.solver_result in ('unknown', 'error'):
                        exclude_indices.add(idx)

                if exclude_indices:
                    exclude_ocls = {
                        sat_constraints_for_verification[i].ocl for i in exclude_indices
                    }
                    before_count = len(constraints)
                    constraints = [
                        c for c in constraints if c.ocl not in exclude_ocls
                    ]
                    verif_results = [
                        r for r in verif_results
                        if r.solver_result not in ('unknown', 'error')
                    ]
                    removed = before_count - len(constraints)
                    prof_stats['unverified_excluded'] = removed
                    prof_stats['constraint_count'] = len(constraints)
                    self.logger.warning(
                        f"Excluded {removed} unverified constraints from final output "
                        f"(UNKNOWN or encoding error)"
                    )

            # ============================================================
            # STEP 7.75: Adjust SAT/UNSAT ratio to match configuration
            # ============================================================
            # Earlier pipeline steps (compatibility pruning, dedup, UNKNOWN
            # filtering) remove SAT constraints but protect UNSAT, so the
            # ratio drifts.  Re-balance here to match the configured ratio.
            if self.enable_research_features and prof_spec.unsat_ratio and prof_spec.unsat_ratio > 0:
                final_sat = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                final_unsat = [c for c in constraints if c.metadata.get('is_unsat', False)]

                if final_sat and final_unsat:
                    # Target: unsat / (sat + unsat) == unsat_ratio
                    # So: target_unsat_count = round(len(final_sat) * R / (1 - R))
                    R = prof_spec.unsat_ratio
                    desired_unsat = int(round(len(final_sat) * R / (1.0 - R)))
                    desired_unsat = max(desired_unsat, 1)  # keep at least 1

                    if len(final_unsat) > desired_unsat:
                        # Too many UNSAT — trim randomly, but never take a class
                        # below its per-class minimum while another choice exists.
                        removed = len(final_unsat) - desired_unsat
                        _Ltrim = prof_spec.per_class_min or 1
                        _cnt = _Counter(c.context for c in constraints)
                        _pool = list(final_unsat)
                        random.shuffle(_pool)
                        _drop = []
                        for _pass_safe in (True, False):
                            for _u in _pool:
                                if len(_drop) >= removed:
                                    break
                                if any(_u is d for d in _drop):
                                    continue
                                if _pass_safe and _cnt[_u.context] <= _Ltrim:
                                    continue
                                _drop.append(_u)
                                _cnt[_u.context] -= 1
                        trimmed = [u for u in final_unsat if not any(u is d for d in _drop)]
                        constraints = final_sat + trimmed
                        prof_stats['constraint_count'] = len(constraints)
                        prof_stats['sat_count'] = len(final_sat)
                        prof_stats['unsat_count'] = desired_unsat
                        prof_stats['unsat_trimmed_for_ratio'] = removed
                        self.logger.info(
                            f"Trimmed {removed} UNSAT constraints to restore "
                            f"{R*100:.0f}% target ratio "
                            f"({len(final_sat)} SAT + {desired_unsat} UNSAT)"
                        )
                    elif len(final_unsat) < desired_unsat:
                        self.logger.info(
                            f"UNSAT count ({len(final_unsat)}) below target "
                            f"({desired_unsat}) — keeping all confirmed UNSAT"
                        )

            # ============================================================
            # STEP 7.9: Final class-coverage audit (post-refinement)
            # ============================================================
            # VGCR and the consistency/compatibility passes can drop constraints
            # after RepairMinima restored the per-class minima. Re-check coverage on
            # the final set and record any class that met the minimum before
            # refinement but no longer does (reported, never fatal).
            try:
                _L = prof_spec.per_class_min or 1
                _final_counts = _Counter(
                    getattr(c, "context", None) for c in constraints
                )
                _new_deficit = {
                    cls: _final_counts.get(cls, 0)
                    for cls, n0 in pre_refine_counts.items()
                    if cls is not None and n0 >= _L and _final_counts.get(cls, 0) < _L
                }
                prof_stats['post_refine_coverage_deficit'] = (
                    {'min_per_class': _L, 'classes': _new_deficit}
                    if _new_deficit else {}
                )
                if _new_deficit:
                    self.logger.warning(
                        f"Post-refinement coverage deficit (min={_L}): "
                        + ", ".join(f"{k}={v}" for k, v in _new_deficit.items())
                    )
            except Exception as _e:
                self.logger.debug(f"Coverage audit skipped: {_e}")

            # ============================================================
            # STEP 8: Save Outputs
            # ============================================================
            self.logger.step("Saving outputs...")
            self.logger.indent()

            # Rewrite the steered report so its certificate describes the
            # DELIVERED suite. The generation-time fields are kept (they certify
            # the fill), and 'final_suite' is re-measured from the saved text.
            try:
                final_cert = self._final_certificate(constraints, engine, prof_spec, prof_stats)
                if final_cert is not None:
                    report_obj = dict(getattr(engine, "steered_report", None) or {})
                    report_obj["model"] = model_spec.name
                    report_obj["profile"] = prof_spec.name
                    report_obj["semantic_admissibility"] = self._semantic_summary.get(model_spec.name)
                    report_obj["vgcr_enabled"] = bool(
                        getattr(self.suite, 'vgcr', None) and self.suite.vgcr.enable
                    )
                    report_obj["report_scope"] = (
                        "Top-level fields certify the generation phase (pre-verification "
                        "candidates); 'final_suite' is re-measured on the delivered suite."
                    )
                    report_obj["final_suite"] = final_cert
                    with open(output_dir / "report.json", "w") as f:
                        json.dump(report_obj, f, indent=2, default=str)
                    prof_stats['final_suite'] = final_cert
                    _fid = final_cert['complexity_fidelity_sat']
                    _cov = final_cert['class_coverage']
                    self.logger.info(
                        f"Final certificate: {final_cert['delivered']} delivered "
                        f"({final_cert['sat']} SAT + {final_cert['unsat']} UNSAT) of "
                        f"{final_cert['slot_accounting']['requested']} requested"
                    )
                    self.logger.info(
                        f"  Slot tiers {final_cert['slot_tier_distribution']} "
                        f"vs requested {final_cert['requested_tier_distribution']}"
                    )
                    self.logger.info(
                        f"  SAT in-box: {_fid['in_box']}/{_fid['boxed_constraints']} "
                        f"(unlabeled {_fid['unlabeled_constraints']}); "
                        f"UNSAT measured tiers {final_cert['unsat_measured_tier_distribution']}"
                    )
                    self.logger.info(
                        f"  Class coverage: {_cov['covered']}/{_cov['classes']} at min={_cov['min_per_class']}"
                        + (f", below min: {_cov['below_min']}" if _cov['below_min'] else "")
                    )
                    self.logger.success("Rewrote report.json with the final-suite certificate")
            except Exception as _e:
                self.logger.warning(f"Final certificate could not be computed: {_e}")

            # Save traditional formats
            self._save_benchmark(
                output_dir, model_spec, prof_spec, profile,
                constraints, verif_results, prof_stats
            )
            self.logger.success("Saved constraints.json and constraints.ocl")
            
            # Save separate SAT and UNSAT files if advanced features enabled
            if self.enable_research_features:
                sat_constraints = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                unsat_constraints = [c for c in constraints if c.metadata.get('is_unsat', False)]
                
                if sat_constraints:
                    self._save_sat_unsat_files(output_dir, sat_constraints, 'sat', model_spec, prof_spec)
                    self.logger.success(f"Saved constraints_sat.ocl and constraints_sat.json ({len(sat_constraints)} SAT)")
                
                if unsat_constraints:
                    self._save_sat_unsat_files(output_dir, unsat_constraints, 'unsat', model_spec, prof_spec)
                    self.logger.success(f"Saved constraints_unsat.ocl and constraints_unsat.json ({len(unsat_constraints)} UNSAT)")
            
            # Save manifest.jsonl (ML-friendly)
            if self.enable_research_features:
                from pathlib import Path as PathLib
                manifest_path = output_dir / "manifest.jsonl"

                # Build verification_results dict for manifest generator.
                # verif_results covers SAT-intended constraints only; UNSAT-by-mutation
                # constraints are labeled from metadata.
                manifest_verif = None
                if verif_results or any(c.metadata.get('is_unsat') for c in constraints):
                    verif_by_ocl = {}
                    if verif_results:
                        sat_only = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                        for c, r in zip(sat_only, verif_results):
                            verif_by_ocl[c.ocl] = {
                                'result': r.solver_result.upper() if r.solver_result else 'UNKNOWN',
                                'time_ms': int(r.execution_time * 1000),
                                'error': r.errors[0] if r.errors else None,
                            }
                    manifest_verif = {'constraints': []}
                    for c in constraints:
                        if c.ocl in verif_by_ocl:
                            manifest_verif['constraints'].append(verif_by_ocl[c.ocl])
                        elif c.metadata.get('is_unsat'):
                            manifest_verif['constraints'].append({
                                'result': 'UNSAT',
                                'time_ms': None,
                                'error': None,
                            })
                        else:
                            manifest_verif['constraints'].append({
                                'result': 'UNKNOWN',
                                'time_ms': None,
                                'error': None,
                            })

                manifest_generator.generate_manifest(
                    constraints=constraints,
                    model_name=model_spec.name,
                    profile_name=prof_spec.name,
                    output_path=PathLib(manifest_path),
                    verification_results=manifest_verif,
                )
                prof_stats['research_features_applied'].append('manifest_generation')
                self.logger.success(f"Saved manifest.jsonl ({len(constraints)} records)")
            
            self.logger.dedent()
            
            prof_stats['status'] = 'success'
            prof_stats['generation_time_s'] = round(time.monotonic() - profile_t0, 1)
            self.logger.success(
                f"Profile '{prof_spec.name}' complete in "
                f"{_fmt_duration(prof_stats['generation_time_s'])}"
            )

        except Exception as e:
            prof_stats['generation_time_s'] = round(time.monotonic() - profile_t0, 1)
            self.logger.error(
                f"Profile generation failed after "
                f"{_fmt_duration(prof_stats['generation_time_s'])}: {e}"
            )
            import traceback
            self.logger.debug(traceback.format_exc())
            prof_stats['status'] = 'failed'
            prof_stats['error'] = str(e)

        return prof_stats
    
    # ------------------------------------------------------------------
    # Suite-integrity helpers (dedup-with-refill, UNSAT tiers, final certificate)
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_threshold(prof_spec: ProfileSpec) -> float:
        # 'is not None' so an explicit 0.0 (max dedup) is honoured.
        return (prof_spec.similarity_threshold
                if prof_spec.similarity_threshold is not None else 0.85)

    @staticmethod
    def _is_near_duplicate(candidate, suite, threshold: float) -> bool:
        """True when ``candidate`` is structurally > ``threshold`` similar to
        any constraint already in ``suite`` (tree-edit similarity)."""
        for other in suite:
            if other is candidate:
                continue
            try:
                if constraint_similarity.ast_similarity(other, candidate) > threshold:
                    return True
            except Exception:
                continue
        return False

    def _dedup_and_refill(self, constraints: List, engine, threshold: float,
                          metamodel: Metamodel) -> Tuple[List, Dict]:
        """Structural deduplication BEFORE verification, with refill.

        The first occurrence of a near-duplicate group is kept. Every removed
        constraint's slot (its family and complexity box) is sent back through
        the engine's fill loop, and the replacement must itself be distinct
        from everything kept. A slot that cannot be refilled inside its box is
        left unfilled and reported -- it is never padded.
        """
        kept: List = []
        removed: List = []
        for c in constraints:
            (removed if self._is_near_duplicate(c, kept, threshold) else kept).append(c)

        info = {'removed': len(removed), 'refilled': 0, 'unfilled': []}
        if not removed:
            return kept, info

        state = None
        if hasattr(engine, 'generate_refined'):
            try:
                from modules.generation.benchmark.vgcr_refinement import GeneratorState
                state = GeneratorState()
            except Exception:
                state = None

        for c in removed:
            newc = None
            if state is not None:
                try:
                    newc = engine.generate_refined(
                        failed_constraint=c, generator_state=state,
                        accept=lambda nc: not self._is_near_duplicate(nc, kept, threshold),
                    )
                except Exception as e:
                    self.logger.debug(f"Dedup refill failed: {e}")
                    newc = None
            if newc is None:
                info['unfilled'].append({
                    'pattern': c.pattern_id, 'context': c.context,
                    'profile': (c.metadata or {}).get('profile'),
                })
                continue
            newc.metadata['refilled_after'] = 'dedup'
            enriched = metadata_enricher.enrich_constraint_metadata(newc, metamodel=metamodel)
            kept.append(enriched if enriched is not None else newc)
            info['refilled'] += 1
        return kept, info

    @staticmethod
    def _tier_of(gen, ocl: str, cls: str, preferred: Optional[str] = None):
        """Measure a constraint and return ``(tier_label_or_None, vector)``.
        With overlapping boxes the preferred label wins when it fits, else the
        box whose centre is nearest."""
        vec = gen._measure(ocl, cls)
        fits = [b for b in gen.box_by_label.values() if b.contains(vec)]
        if not fits:
            return None, vec
        for b in fits:
            if b.label == preferred:
                return b.label, vec
        return min(fits, key=lambda b: gen._dist(b, vec)).label, vec

    def _relabel_unsat(self, constraints: List, engine) -> Dict:
        """Give every UNSAT mutant its OWN re-measured tier.

        A mutant (e.g. ``P and not P``) is structurally heavier than its parent,
        so inheriting the parent's tier label is wrong. The parent's tier is
        kept as ``slot_profile`` (the slot the mutant occupies for quota
        accounting); ``profile``/``difficulty`` become the measured tier, or
        ``unboxed`` when the mutant lies in no requested box.
        """
        gen = getattr(engine, "_steered_gen", None)
        if gen is None or not getattr(gen, "box_by_label", None):
            return {}
        dist: Dict[str, int] = {}
        for c in constraints:
            if not c.metadata.get('is_unsat', False):
                continue
            parent = c.metadata.get('slot_profile') or c.metadata.get('profile')
            try:
                tier, _ = self._tier_of(gen, c.ocl, c.context, preferred=parent)
            except Exception:
                tier = None
            tier = tier or 'unboxed'
            c.metadata['slot_profile'] = parent
            c.metadata['profile'] = tier
            c.metadata['difficulty'] = tier
            dist[tier] = dist.get(tier, 0) + 1
        return dist

    def _final_certificate(self, constraints: List, engine, prof_spec: ProfileSpec,
                           prof_stats: Dict) -> Optional[Dict]:
        """Certificate computed on the DELIVERED suite (every constraint is
        re-measured from its text), so report.json describes what was saved,
        not the pre-verification candidates."""
        gen = getattr(engine, "_steered_gen", None)
        if gen is None or not getattr(gen, "box_by_label", None):
            return None
        from collections import Counter

        sat = [c for c in constraints if not c.metadata.get('is_unsat', False)]
        unsat = [c for c in constraints if c.metadata.get('is_unsat', False)]

        def fam(c):
            pid = c.metadata.get('original_pattern_id') or c.pattern_id
            try:
                return gen._family(pid)
            except Exception:
                return c.metadata.get('regen_family') or 'unknown'

        in_box = 0
        out_of_box, unlabeled = [], 0
        for c in sat:
            label = str(c.metadata.get('profile') or '').split('+')[0]
            box = gen.box_by_label.get(label)
            if box is None:
                unlabeled += 1
                continue
            vec = gen._measure(c.ocl, c.context)
            if box.contains(vec):
                in_box += 1
            else:
                out_of_box.append({'pattern': c.pattern_id, 'context': c.context,
                                   'profile': label})
        boxed = len(sat) - unlabeled

        # Slot accounting: a mutant occupies its parent's slot.
        def slot_tier(c):
            m = c.metadata
            t = m.get('slot_profile') if m.get('is_unsat', False) else m.get('profile')
            return str(t or 'unlabeled').split('+')[0]

        from modules.generation.benchmark.steered_generation import _largest_remainder
        n = gen.spec.n
        requested_tiers = {
            gen.spec.profiles[i][0].label: q
            for i, q in _largest_remainder(
                {i: pct for i, (_b, pct) in enumerate(gen.spec.profiles)}, n).items()
        }
        requested_fams = dict(_largest_remainder(gen.spec.family_quota, n))

        L = prof_spec.per_class_min or 1
        counts = Counter(c.context for c in constraints)
        all_classes = list(getattr(gen, 'classes', []))
        below = {cls: counts.get(cls, 0) for cls in all_classes if counts.get(cls, 0) < L}

        vg = prof_stats.get('vgcr_stats') or {}
        dd = prof_stats.get('dedup') or {}
        return {
            'delivered': len(constraints),
            'sat': len(sat),
            'unsat': len(unsat),
            'slot_tier_distribution': dict(Counter(slot_tier(c) for c in constraints)),
            'requested_tier_distribution': requested_tiers,
            'sat_tier_distribution': dict(Counter(slot_tier(c) for c in sat)),
            'unsat_measured_tier_distribution': dict(Counter(
                str(c.metadata.get('profile') or 'unboxed') for c in unsat)),
            'family_distribution': dict(Counter(fam(c) for c in constraints)),
            'requested_family_distribution': requested_fams,
            'complexity_fidelity_sat': {
                'boxed_constraints': boxed,
                'in_box': in_box,
                'in_box_rate': round(in_box / boxed, 4) if boxed else None,
                'unlabeled_constraints': unlabeled,
                'out_of_box': out_of_box,
            },
            'class_coverage': {
                'min_per_class': L,
                'classes': len(all_classes),
                'covered': sum(1 for cls in all_classes if counts.get(cls, 0) >= L),
                'below_min': below,
            },
            'slot_accounting': {
                'requested': n,
                'generated': prof_stats.get('initial_sat_count'),
                'generation_gaps': n - (prof_stats.get('initial_sat_count') or 0),
                'dedup_removed': dd.get('removed', 0),
                'dedup_refilled': dd.get('refilled', 0),
                'dedup_unfilled': len(dd.get('unfilled', [])),
                'vgcr_regenerated': vg.get('passed_after_retry', 0),
                'vgcr_unfilled': len(vg.get('unfilled_slots', [])),
                'late_duplicates_removed': prof_stats.get('duplicates_removed', 0),
                'unsat_trimmed_for_ratio': prof_stats.get('unsat_trimmed_for_ratio', 0),
                'delivered': len(constraints),
            },
            'dedup_unfilled_slots': dd.get('unfilled', []),
            'vgcr_unfilled_slots': vg.get('unfilled_slots', []),
            'coverage_repairs': prof_stats.get('coverage_repairs', []),
        }

    def _build_profile_from_spec(self, prof_spec: ProfileSpec, metamodel: Metamodel) -> BenchmarkProfile:
        """Build BenchmarkProfile from ProfileSpec including complexity config."""
        # Get difficulty profile preset
        diff_profile = get_difficulty_profile(prof_spec.complexity_profile)

        # Build quantities config
        quantities = QuantitiesConfig(
            invariants=prof_spec.constraints,
            per_class_min=prof_spec.per_class_min or 1,
            per_class_max=prof_spec.per_class_max or 5
        )

        # Override families if specified
        if prof_spec.families_pct:
            quantities.families_pct = prof_spec.families_pct

        # Build full profile
        from modules.generation.benchmark.bench_config import (
            BenchmarkProfile, CoverageTargets, LibraryConfig,
            RedundancyConfig, ComplexityConfig
        )

        # Build ComplexityConfig from spec + preset defaults
        complexity = ComplexityConfig()

        # TC range: user override > preset > default
        if prof_spec.target_tc_range:
            complexity.min_tc = prof_spec.target_tc_range.get('min', complexity.min_tc)
            complexity.max_tc = prof_spec.target_tc_range.get('max', complexity.max_tc)
        elif 'tc_range' in diff_profile:
            complexity.min_tc = diff_profile['tc_range']['min']
            complexity.max_tc = diff_profile['tc_range']['max']

        # --- Per-dimension complexity configuration (new, takes precedence) ---
        if prof_spec.complexity_dimensions:
            dims = prof_spec.complexity_dimensions

            # Structural dimension
            if 'structural' in dims:
                s = dims['structural']
                complexity.structural_enabled = s.get('enabled', True)
                if not complexity.structural_enabled:
                    complexity.structural_weight = 0.0
                else:
                    complexity.structural_weight = s.get('weight', 1.0)
                # Per-dimension target range
                if 'target_range' in s:
                    complexity.structural_target_min = s['target_range'].get('min')
                    complexity.structural_target_max = s['target_range'].get('max')
                # TNC sub-weights (structural-specific)
                if 'tnc_weights' in s:
                    complexity.tnc_alpha = s['tnc_weights'].get('alpha', 0.4)
                    complexity.tnc_beta = s['tnc_weights'].get('beta', 0.3)
                    complexity.tnc_gamma = s['tnc_weights'].get('gamma', 0.3)

            # Computational dimension
            if 'computational' in dims:
                c = dims['computational']
                complexity.computational_enabled = c.get('enabled', True)
                if not complexity.computational_enabled:
                    complexity.computational_weight = 0.0
                else:
                    complexity.computational_weight = c.get('weight', 1.0)
                # Per-dimension target range
                if 'target_range' in c:
                    complexity.computational_target_min = c['target_range'].get('min')
                    complexity.computational_target_max = c['target_range'].get('max')

        else:
            # Legacy flat keys (backward compatibility)
            if prof_spec.dimension_weights:
                complexity.structural_weight = prof_spec.dimension_weights.get('structural', 1.0)
                complexity.computational_weight = prof_spec.dimension_weights.get('computational', 1.0)
                complexity.dependency_weight = prof_spec.dimension_weights.get('dependency', 1.0)

            if prof_spec.tnc_weights:
                complexity.tnc_alpha = prof_spec.tnc_weights.get('alpha', 0.4)
                complexity.tnc_beta = prof_spec.tnc_weights.get('beta', 0.3)
                complexity.tnc_gamma = prof_spec.tnc_weights.get('gamma', 0.3)

        # --- Operator weights: new full weights > legacy overrides ---
        if prof_spec.operator_weights:
            complexity.operator_weights = prof_spec.operator_weights
        elif prof_spec.operator_weight_overrides:
            complexity.operator_weight_overrides = prof_spec.operator_weight_overrides

        # TC difficulty mix: user override > preset > default
        if prof_spec.tc_difficulty_mix:
            complexity.tc_difficulty_mix = prof_spec.tc_difficulty_mix
        elif 'tc_difficulty_mix' in diff_profile:
            complexity.tc_difficulty_mix = diff_profile['tc_difficulty_mix']

        # Steered mode: per-component complexity profiles + infeasibility policy.
        # Validate at load time so a malformed or over-specified profile fails
        # fast (rejects the engine-reported components oc/wnc/vrc/wnm and checks
        # every range), independent of the generation mode actually used.
        if prof_spec.complexity_profiles:
            from modules.generation.benchmark.steered_generation import validate_complexity_profiles
            validate_complexity_profiles(prof_spec.complexity_profiles)
            complexity.complexity_profiles = prof_spec.complexity_profiles
        if prof_spec.on_infeasible:
            complexity.on_infeasible = prof_spec.on_infeasible

        # Configure the global complexity weights for metadata enrichment
        from modules.generation.benchmark.complexity_calculator import ComplexityWeights, DEFAULT_OPERATOR_WEIGHTS
        from modules.generation.benchmark.metadata_enricher import set_complexity_weights

        # Build effective operator weights: defaults <- overrides <- full weights
        op_weights = dict(DEFAULT_OPERATOR_WEIGHTS)
        op_weights.update(complexity.operator_weight_overrides)
        if complexity.operator_weights:
            op_weights.update(complexity.operator_weights)

        cw = ComplexityWeights(
            operator_weights=op_weights,
            structural_weight=complexity.structural_weight,
            computational_weight=complexity.computational_weight,
            dependency_weight=complexity.dependency_weight,
            tnc_alpha=complexity.tnc_alpha,
            tnc_beta=complexity.tnc_beta,
            tnc_gamma=complexity.tnc_gamma,
        )
        set_complexity_weights(cw)

        profile = BenchmarkProfile(
            quantities=quantities,
            coverage=CoverageTargets(
                difficulty_mix=prof_spec.difficulty_mix or diff_profile['difficulty_mix']
            ),
            library=LibraryConfig(),
            redundancy=RedundancyConfig(
                similarity_threshold=(prof_spec.similarity_threshold
                                      if prof_spec.similarity_threshold is not None else 0.85),
                novelty_boost=prof_spec.novelty_boost if prof_spec.novelty_boost is not None else True
            ),
            complexity=complexity,
        )

        return profile
    
    def _save_benchmark(
        self,
        output_dir: Path,
        model_spec: ModelSpec,
        prof_spec: ProfileSpec,
        profile: BenchmarkProfile,
        constraints: List[OCLConstraint],
        verif_results: List,
        prof_stats: Dict = None
    ):
        """Save benchmark outputs."""
        # Save constraints as OCL text
        ocl_path = output_dir / "constraints.ocl"
        with open(ocl_path, 'w') as f:
            f.write(f"-- Benchmark: {model_spec.name} / {prof_spec.name}\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Seed: {prof_spec.seed}\n")
            f.write(f"-- Complexity: {prof_spec.complexity_profile}\n")
            f.write(f"-- Count: {len(constraints)}\n")
            f.write(f"-- Advanced Features: {'ENABLED' if self.enable_research_features else 'DISABLED'}\n\n")
            
            for i, c in enumerate(constraints, 1):
                f.write(f"-- #{i} {c.pattern_name} in {c.context}\n")
                
                # Add metadata if available
                if self.enable_research_features and c.metadata:
                    if 'difficulty' in c.metadata:
                        f.write(f"-- Difficulty: {c.metadata['difficulty']}\n")
                    if 'mutation_strategy' in c.metadata:
                        f.write(f"-- Type: UNSAT (mutated via {c.metadata['mutation_strategy']})\n")
                    elif 'is_unsat' in c.metadata and c.metadata['is_unsat']:
                        f.write(f"-- Type: UNSAT\n")
                    else:
                        f.write(f"-- Type: SAT\n")
                
                f.write(f"{c.ocl}\n\n")
        
        # Save as JSON with metadata
        json_path = output_dir / "constraints.json"
        data = {
            'metadata': {
                'suite': self.suite.suite_name,
                'model': model_spec.name,
                'profile': prof_spec.name,
                'generated_at': datetime.now().isoformat(),
                'seed': prof_spec.seed,
                'complexity': prof_spec.complexity_profile,
                'framework_version': self.suite.framework_version,
                'git_commit': self.suite.git_commit,
                'constraint_count': len(constraints),
                'research_features_enabled': self.enable_research_features,
                'consistency_verified': prof_stats.get('consistency_verified') if prof_stats else None,
                'scope_per_class': self.suite.verification.scope_per_class
            },
            'constraints': [c.to_dict() for c in constraints]
        }
        
        if verif_results:
            data['verification'] = {
                'enabled': True,
                'results': [
                    {
                        'constraint_id': r.constraint_id,
                        'is_valid': r.is_valid,
                        'solver_result': r.solver_result,
                        'execution_time': r.execution_time
                    }
                    for r in verif_results
                ]
            }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save profile config
        profile_path = output_dir / "profile.json"
        with open(profile_path, 'w') as f:
            json.dump({
                'name': prof_spec.name,
                'seed': prof_spec.seed,
                'constraints': prof_spec.constraints,
                'complexity_profile': prof_spec.complexity_profile,
                'sat_ratio': prof_spec.sat_ratio,
                'unsat_ratio': prof_spec.unsat_ratio,
                'difficulty_mix': prof_spec.difficulty_mix,
                'research_features_enabled': self.enable_research_features
            }, f, indent=2)
    
    def _save_sat_unsat_files(
        self,
        output_dir: Path,
        constraints: List[OCLConstraint],
        constraint_type: str,
        model_spec: ModelSpec,
        prof_spec: ProfileSpec
    ):
        """Save SAT or UNSAT constraints to separate files.
        
        Args:
            output_dir: Output directory
            constraints: List of constraints (all SAT or all UNSAT)
            constraint_type: 'sat' or 'unsat'
            model_spec: Model specification
            prof_spec: Profile specification
        """
        # Save as OCL text
        ocl_path = output_dir / f"constraints_{constraint_type}.ocl"
        with open(ocl_path, 'w') as f:
            f.write(f"-- Benchmark: {model_spec.name} / {prof_spec.name}\n")
            f.write(f"-- Type: {constraint_type.upper()} Constraints Only\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Count: {len(constraints)}\n\n")
            
            for i, c in enumerate(constraints, 1):
                f.write(f"-- #{i} {c.pattern_name} in {c.context}\n")
                
                # Add metadata if available
                if c.metadata:
                    if 'difficulty' in c.metadata:
                        f.write(f"-- Difficulty: {c.metadata['difficulty']}\n")
                    if constraint_type == 'unsat' and 'mutation' in c.metadata:
                        f.write(f"-- Mutation Strategy: {c.metadata['mutation']}\n")
                
                f.write(f"{c.ocl}\n\n")
        
        # Save as JSON
        json_path = output_dir / f"constraints_{constraint_type}.json"
        data = {
            'metadata': {
                'type': constraint_type.upper(),
                'model': model_spec.name,
                'profile': prof_spec.name,
                'generated_at': datetime.now().isoformat(),
                'constraint_count': len(constraints)
            },
            'constraints': [c.to_dict() for c in constraints]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _should_attempt_compatibility_pruning(self, sat_constraint_count: int) -> bool:
        """Limit the greedy recovery pass to sizes that finish in reasonable time."""
        return sat_constraint_count <= self.MAX_GREEDY_COMPATIBILITY_CONSTRAINTS

    @staticmethod
    def _infer_batch_consistency(verif_results: List) -> Optional[bool]:
        """Infer whether a batch verification result proves SAT or UNSAT."""
        solver_results = {r.solver_result for r in verif_results if getattr(r, 'solver_result', None)}
        if 'sat' in solver_results:
            return True
        if 'unsat' in solver_results and solver_results.issubset({'unsat', 'error'}):
            return False
        return None

    def _find_compatible_subset_by_core(self, constraints: List, verifier) -> List:
        """Deterministic conflict removal driven by Z3 unsat cores.

        Repeatedly check the whole set in one tracked call; while it is UNSAT,
        drop the LAST (most recently generated) member of the unsat core and
        re-check. A timeout stops the loop without dropping anything, since
        only a proven core justifies removing a candidate. Needs at most one
        solver call per removed constraint instead of one per constraint."""
        kept = list(constraints)
        for _ in range(len(constraints)):
            res = verifier.check_consistency_with_core(kept)
            if res['solver_result'] != 'unsat':
                break
            core = res['core']
            if not core:
                break  # UNSAT without a usable core: nothing provable to remove
            victim = max(core)
            if getattr(self, 'logger', None):
                self.logger.debug(
                    f"Unsat core {core} -> removing constraint #{victim} "
                    f"({kept[victim].pattern_id}@{kept[victim].context})"
                )
            del kept[victim]
        return kept

    def _find_compatible_subset_batch(self, constraints: List, verifier) -> List:
        """Find maximal compatible subset using batch greedy algorithm.
        
        This is optimized to minimize verification calls by testing batches
        instead of individual constraints.
        
        Args:
            constraints: List of OCLConstraint objects
            verifier: Constraint verifier
            
        Returns:
            List of compatible constraints (subset of input)
        """
        if len(constraints) <= 1 or not getattr(verifier, 'framework_available', False):
            return constraints
        
        # Greedy algorithm: Start with empty set, add constraints one by one
        compatible = []
        
        for idx, constraint in enumerate(constraints, 1):
            test_set = compatible + [constraint]
            
            try:
                # Test if adding this constraint keeps model SAT
                results = verifier.verify_batch(test_set, silent=True)
                is_inconsistent = any(r.solver_result == 'unsat' for r in results)
                
                if not is_inconsistent:
                    # 'sat' keeps the constraint; 'unknown' (timeout) also keeps it,
                    # because only a proven conflict justifies dropping a candidate.
                    compatible.append(constraint)
                # else: skip this constraint (proven conflict with the kept set)
            except Exception:
                pass

            if (idx % 10 == 0 or idx == len(constraints)) and getattr(self, 'logger', None):
                self.logger.debug(
                    f"Compatibility pruning progress: {idx}/{len(constraints)} tested, "
                    f"{len(compatible)} currently compatible"
                )
        
        return compatible
    
