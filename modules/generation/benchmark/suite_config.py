from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
from pathlib import Path


@dataclass
class SemanticSpec:
    """Semantic analysis configuration (LLM-powered, Phi-4 via Ollama)."""
    enable: bool = True
    model: str = "phi4-mini"               # 3.8B — lighter, faster
    # model: str = "phi4"                # 14B — more accurate
    ollama_url: str = "http://localhost:11434"
    use_cache: bool = True


@dataclass
class VerificationSpec:
    """Verification configuration for the suite."""
    enable: bool = True
    objective: str = "both"  # sat | unsat | both
    per_constraint_timeout_ms: int = 8000
    batch_timeout_ms: int = 60000
    check_global_consistency: bool = True
    implication_use_z3: bool = False
    scope_per_class: int = 4  # Bounded scope: max instances per class for Z3


@dataclass
class ProfileSpec:
    """Profile configuration for a single benchmark profile."""
    name: str
    seed: int = 42
    constraints: int = 50
    complexity_profile: str = "standard"  # basic | standard | advanced
    sat_ratio: float = 0.7
    unsat_ratio: float = 0.3
    include_unknown: bool = False

    # Override default profile settings
    families_pct: Optional[Dict[str, int]] = None
    per_class_min: Optional[int] = None
    per_class_max: Optional[int] = None
    similarity_threshold: Optional[float] = None
    novelty_boost: Optional[bool] = None

    # Difficulty mix (legacy 3-bucket system, kept for backward compat)
    difficulty_mix: Optional[Dict[str, float]] = None  # easy/medium/hard

    # --- Complexity-based generation (paper metrics) ---
    # Target TC range: only generate constraints within this range
    target_tc_range: Optional[Dict[str, float]] = None  # {"min": 5.0, "max": 25.0}
    # Dimension weights for TC calculation (legacy flat keys, kept for backward compat)
    dimension_weights: Optional[Dict[str, float]] = None  # {"structural": 1.0, ...}
    # TNC sub-weights (legacy flat keys, kept for backward compat)
    tnc_weights: Optional[Dict[str, float]] = None  # {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}
    # Custom operator weight overrides (legacy, kept for backward compat)
    operator_weight_overrides: Optional[Dict[str, float]] = None
    # TC-based 5-bucket difficulty distribution
    tc_difficulty_mix: Optional[Dict[str, int]] = None  # {"trivial":5,"easy":30,...}

    # --- Per-dimension complexity configuration (new, takes precedence over legacy) ---
    # Each dimension: {enabled: bool, weight: float, target_range: {min, max}, tnc_weights: {alpha, beta, gamma}}
    complexity_dimensions: Optional[Dict[str, Dict]] = None
    # Full operator weights (Table 1 from paper, takes precedence over operator_weight_overrides)
    operator_weights: Optional[Dict[str, float]] = None

    # --- Generation mechanism (per-profile override) ---
    # "construct_select" (default) — over-generate a pool, measure exact
    #     complexity, then stratified-select to fill each tier's quota exactly.
    # "legacy" — older TC-steering path.
    # When None, inherits the suite-level BenchmarkSuite.generation_mode.
    generation_mode: Optional[str] = None

    # --- Steered mode: per-component complexity profiles ---
    # List of {"pct": share, "label": name, "ranges": {component: [lo, hi]}}.
    # Only consumed by generation_mode == "steered".
    complexity_profiles: Optional[List[Dict]] = None
    # Infeasibility policy for steered mode: report | relax | skip.
    on_infeasible: Optional[str] = None

    def __post_init__(self):
        # Fail fast with a clear message on a malformed numeric field (e.g. a YAML
        # typo like 'constraints: 100clear'), instead of a cryptic 'str / int' error
        # surfacing minutes into the run after semantic analysis.
        for fld, val in (('constraints', self.constraints), ('seed', self.seed),
                         ('per_class_min', self.per_class_min),
                         ('per_class_max', self.per_class_max)):
            if val is None:
                continue
            try:
                setattr(self, fld, int(val))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Profile '{self.name}': field '{fld}' must be an integer, "
                    f"got {val!r}. Check the YAML for a typo."
                )
        for fld, val in (('sat_ratio', self.sat_ratio), ('unsat_ratio', self.unsat_ratio)):
            try:
                setattr(self, fld, float(val))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Profile '{self.name}': field '{fld}' must be a number, "
                    f"got {val!r}. Check the YAML for a typo."
                )


@dataclass
class ModelSpec:
    """Model specification with associated profiles."""
    xmi: str
    name: Optional[str] = None  # Auto-inferred from filename if not provided
    profiles: List[ProfileSpec] = field(default_factory=list)
    
    def __post_init__(self):
        if self.name is None:
            self.name = Path(self.xmi).stem


@dataclass
class VGCRSpec:
    """VGCR (Verification-Guided Constraint Refinement) configuration."""
    enable: bool = True
    max_retries: int = 3
    enable_independence_check: bool = True
    independence_threshold: int = 5  # min suite size before q3 activates
    # TC conformance bounds — if None, uses the profile's target_tc_range
    tc_range_override: Optional[Dict[str, float]] = None  # {"min": 3.0, "max": 25.0}


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite specification."""
    suite_name: str
    version: str = "1.0"
    models: List[ModelSpec] = field(default_factory=list)
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    semantic: SemanticSpec = field(default_factory=SemanticSpec)
    vgcr: VGCRSpec = field(default_factory=VGCRSpec)
    output_root: str = "benchmarks/"
    description: Optional[str] = None

    # Generation mechanism for the whole suite (individual profiles may override
    # via ProfileSpec.generation_mode):
    #   "construct_select" (default) — over-generate, measure, stratified-select
    #   "legacy" — older TC-steering path
    generation_mode: str = "construct_select"

    # Reproducibility
    framework_version: str = "2.0"
    git_commit: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BenchmarkSuite':
        """Load suite from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkSuite':
        """Parse suite from dictionary."""
        # Parse verification
        verif_data = data.get('verification', {})
        verification = VerificationSpec(**verif_data)

        # Parse semantic analysis config
        semantic_data = data.get('semantic', {})
        semantic = SemanticSpec(**semantic_data)

        # Parse models
        models = []
        for model_data in data.get('models', []):
            profiles = []
            for prof_data in model_data.get('profiles', []):
                profile = ProfileSpec(**prof_data)
                profiles.append(profile)
            
            model = ModelSpec(
                xmi=model_data['xmi'],
                name=model_data.get('name'),
                profiles=profiles
            )
            models.append(model)
        
        # Parse VGCR config
        vgcr_data = data.get('vgcr', {})
        vgcr = VGCRSpec(**vgcr_data)

        suite = cls(
            suite_name=data['suite_name'],
            version=data.get('version', '1.0'),
            models=models,
            verification=verification,
            semantic=semantic,
            vgcr=vgcr,
            output_root=data.get('output_root', 'benchmarks/'),
            description=data.get('description'),
            generation_mode=data.get('generation_mode', 'construct_select'),
            git_commit=data.get('git_commit')
        )
        
        return suite
    
    def to_yaml(self, yaml_path: str):
        """Save suite to YAML file."""
        data = {
            'suite_name': self.suite_name,
            'version': self.version,
            'description': self.description,
            'models': [
                {
                    'xmi': m.xmi,
                    'name': m.name,
                    'profiles': [
                        {
                            k: v for k, v in {
                                'name': p.name,
                                'seed': p.seed,
                                'constraints': p.constraints,
                                'complexity_profile': p.complexity_profile,
                                'sat_ratio': p.sat_ratio,
                                'unsat_ratio': p.unsat_ratio,
                                'include_unknown': p.include_unknown,
                                'families_pct': p.families_pct,
                                'difficulty_mix': p.difficulty_mix,
                                'target_tc_range': p.target_tc_range,
                                'tc_difficulty_mix': p.tc_difficulty_mix,
                                'complexity_dimensions': p.complexity_dimensions,
                                'operator_weights': p.operator_weights,
                                'per_class_min': p.per_class_min,
                                'per_class_max': p.per_class_max,
                                'similarity_threshold': p.similarity_threshold,
                                'novelty_boost': p.novelty_boost,
                                'generation_mode': p.generation_mode,
                            }.items() if v is not None
                        }
                        for p in m.profiles
                    ]
                }
                for m in self.models
            ],
            'semantic': {
                'enable': self.semantic.enable,
                'model': self.semantic.model,
                'ollama_url': self.semantic.ollama_url,
                'use_cache': self.semantic.use_cache,
            },
            'verification': {
                'enable': self.verification.enable,
                'objective': self.verification.objective,
                'per_constraint_timeout_ms': self.verification.per_constraint_timeout_ms,
                'batch_timeout_ms': self.verification.batch_timeout_ms,
                'scope_per_class': self.verification.scope_per_class
            },
            'output_root': self.output_root,
            'generation_mode': self.generation_mode,
            'framework_version': self.framework_version,
            'git_commit': self.git_commit
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Difficulty profile presets (legacy + complexity-aware)
DIFFICULTY_PROFILES = {
    "easy": {
        "max_depth": 2,
        "max_quantifier_depth": 0,
        "max_hops": 1,
        "allowed_operators": ["=", ">=", "<=", "size", "isEmpty", "notEmpty"],
        "difficulty_mix": {"easy": 1.0, "medium": 0.0, "hard": 0.0},
        # Complexity-aware settings
        "tc_range": {"min": 0.0, "max": 8.0},
        "tc_difficulty_mix": {
            "trivial": 30, "easy": 50, "medium": 20, "hard": 0, "expert": 0
        },
    },
    "standard": {
        "max_depth": 3,
        "max_quantifier_depth": 1,
        "max_hops": 2,
        "allowed_operators": ["forAll", "exists", "select", "collect", "size", "implies"],
        "difficulty_mix": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
        # Complexity-aware settings
        "tc_range": {"min": 3.0, "max": 20.0},
        "tc_difficulty_mix": {
            "trivial": 5, "easy": 30, "medium": 40, "hard": 20, "expert": 5
        },
    },
    "advanced": {
        "max_depth": 5,
        "max_quantifier_depth": 2,
        "max_hops": 4,
        "allowed_operators": None,  # All operators
        "difficulty_mix": {"easy": 0.2, "medium": 0.4, "hard": 0.4},
        # Complexity-aware settings
        "tc_range": {"min": 8.0, "max": 50.0},
        "tc_difficulty_mix": {
            "trivial": 0, "easy": 10, "medium": 30, "hard": 35, "expert": 25
        },
    },
}


def get_difficulty_profile(name: str) -> Dict:
    """Get difficulty profile by name."""
    return DIFFICULTY_PROFILES.get(name, DIFFICULTY_PROFILES["standard"])
