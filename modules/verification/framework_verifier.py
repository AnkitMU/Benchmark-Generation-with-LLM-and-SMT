"""
OCL Constraint Verifier using hybrid-ssr-ocl-full-extended framework
Provides accurate verification using the existing Z3-based verification system.
"""
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

from modules.core.models import OCLConstraint, Metamodel


# Add framework to path
FRAMEWORK_PATH = Path(__file__).parent.parent.parent.parent / "hybrid-ssr-ocl-full-extended"
if FRAMEWORK_PATH.exists():
    sys.path.insert(0, str(FRAMEWORK_PATH / "src"))


@dataclass
class FrameworkVerificationResult:
    """Result from framework verification."""
    constraint_id: str
    is_valid: bool
    is_satisfiable: Optional[bool] = None
    errors: List[str] = None
    warnings: List[str] = None
    execution_time: float = 0.0
    solver_result: Optional[str] = None  # 'sat', 'unsat', 'unknown'
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def status(self) -> str:
        """Get status string."""
        if not self.is_valid:
            return "INVALID"
        if self.is_satisfiable == False:
            return "UNSATISFIABLE"
        if self.warnings:
            return "VALID (warnings)"
        return "VALID"


# Mapping from generator pattern IDs to Z3 encoder pattern names.
# The benchmark generator uses descriptive snake_case IDs, while the Z3 encoder
# uses its own naming convention from the hybrid-ssr-ocl framework.
_PATTERN_ID_TO_Z3 = {
    # Navigation / Association
    'association_exists': 'navigation_chain',
    'association_exists_unsat': 'navigation_chain',
    # Null checks
    'attribute_not_null_simple': 'null_check',
    'attribute_null_check': 'null_check',
    'attribute_null_check_unsat': 'null_check',
    # Collection emptiness / size
    'collection_not_empty_simple': 'size_constraint',
    'collection_not_empty_simple_unsat': 'size_constraint',
    'collection_not_empty_check': 'size_constraint',
    'collection_not_empty_check_unsat': 'size_constraint',
    'collection_empty_check': 'size_constraint',
    'collection_empty_check_unsat': 'size_constraint',
    'collection_has_size': 'size_constraint',
    'collection_size_range': 'size_constraint',
    'collection_min_size': 'size_constraint',
    'collection_min_size_unsat': 'size_constraint',
    'collection_max_size': 'size_constraint',
    'collection_max_size_unsat': 'size_constraint',
    # String operations
    'string_max_length': 'string_operations',
    'string_min_length': 'string_operations',
    'string_min_length_unsat': 'string_operations',
    'string_exact_length': 'string_operations',
    'string_exact_length_unsat': 'string_operations',
    'string_not_empty': 'string_operations',
    'string_starts_with': 'string_pattern',
    'string_contains_substring': 'string_pattern',
    'string_contains_substring_unsat': 'string_pattern',
    # Set operations
    'difference_operation': 'symmetric_difference',
    'union_operation': 'union_intersection',
    'union_operation_unsat': 'union_intersection',
    'includesAll_excludesAll': 'subset_disjointness',
    'includesAll_excludesAll_unsat': 'subset_disjointness',
    # Collection operations
    'collect_operation': 'collect_flatten',
    'collection_asSequence': 'as_set_as_bag',
    'collection_asSet': 'as_set_as_bag',
    'collection_asSet_unsat': 'as_set_as_bag',
    'collection_sum': 'sum_product',
    'collection_at_index': 'collection_navigation',
    'collection_last': 'collection_navigation',
    'sortedBy': 'ordering_ranking',
    'sortedBy_unsat': 'ordering_ranking',
    # Uniqueness
    'all_different': 'pairwise_uniqueness',
    'collection_isUnique_attr': 'uniqueness_constraint',
    'collection_isUnique_attr_unsat': 'uniqueness_constraint',
    # Boolean / guard
    'boolean_guard': 'boolean_guard_implies',
    # Numeric
    'numeric_greater_than_value': 'numeric_comparison',
    'numeric_even': 'arithmetic_expression',
    'numeric_odd': 'arithmetic_expression',
    # Already matching (no mapping needed, but listed for completeness)
    'size_constraint': 'size_constraint',
    'uniqueness_constraint': 'uniqueness_constraint',
    'flatten_operation': 'flatten_operation',
}


class FrameworkConstraintVerifier:
    """
    Verifier that uses hybrid-ssr-ocl-full-extended framework.

    This provides accurate, research-grade verification using:
    - Full OCL parser
    - Z3 SMT solver
    - Association-backed encoding
    - Pattern-based constraint handling
    """
    
    def __init__(self, metamodel: Metamodel, xmi_path: str,
                 scope_per_class: int = 2, timeout_ms: int = 15000):
        """
        Initialize verifier with framework.

        Args:
            metamodel: Metamodel object (for compatibility)
            xmi_path: Path to XMI file (needed by framework)
            scope_per_class: Bounded scope — max instances per class for Z3
            timeout_ms: Z3 solver timeout in milliseconds
        """
        self.metamodel = metamodel
        self.xmi_path = xmi_path
        self.scope_per_class = scope_per_class
        self.timeout_ms = timeout_ms
        self.framework_available = False
        self.unavailable_reason: Optional[str] = None
        self.checker = None
        
        # Try to import framework
        try:
            missing_dependencies = self._get_missing_framework_dependencies()
            if missing_dependencies:
                self.unavailable_reason = (
                    "Missing framework dependencies: "
                    + ", ".join(missing_dependencies)
                )
                print(f"Framework not available: {self.unavailable_reason}")
                print("   Falling back to basic verification")
                return

            # Bypass super_encoder/__init__.py which eagerly imports
            # ComprehensivePatternDetector → torch/sentence-transformers.
            # The Z3-based checker only needs: z3, association_backed_encoder,
            # enhanced_smt_encoder, date_adapter — no ML deps.
            #
            # Strategy: replace super_encoder's __init__ with a stub before
            # importing the checker module, so Python doesn't load the full package.
            import importlib
            import types

            # Create a stub for super_encoder package to prevent __init__.py execution
            stub_pkg_name = "ssr_ocl.super_encoder"
            if stub_pkg_name not in sys.modules:
                stub = types.ModuleType(stub_pkg_name)
                stub.__path__ = [str(FRAMEWORK_PATH / "src" / "ssr_ocl" / "super_encoder")]
                stub.__package__ = stub_pkg_name
                sys.modules[stub_pkg_name] = stub

            # Now import the checker — relative imports resolve against our stub
            from ssr_ocl.super_encoder.generic_global_consistency_checker import GenericGlobalConsistencyChecker

            # Initialize checker (suppress internal prints)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self.checker = GenericGlobalConsistencyChecker(
                    xmi_file=xmi_path,
                    rich_instances=True,  # Enable realistic values
                    timeout_ms=self.timeout_ms,
                    show_raw_values=False
                )
            
            self.framework_available = True
            
        except ImportError as e:
            self.unavailable_reason = str(e)
            print(f"Framework not available: {e}")
            print("   Falling back to basic verification")
            self.framework_available = False
        except Exception as e:
            self.unavailable_reason = str(e)
            print(f"Error initializing framework: {e}")
            print("   Falling back to basic verification")
            self.framework_available = False
    
    def verify(self, constraint: OCLConstraint) -> FrameworkVerificationResult:
        """
        Verify a single constraint.
        
        Args:
            constraint: Constraint to verify
            
        Returns:
            Verification result
        """
        start = time.time()
        
        result = FrameworkVerificationResult(
            constraint_id=f"{constraint.pattern_id}_{constraint.context}",
            is_valid=True
        )
        
        if not self.framework_available:
            # Fallback to basic checking
            reason = self.unavailable_reason or "framework unavailable"
            result.is_satisfiable = None
            result.solver_result = 'unknown'
            result.warnings.append(f"Framework not available - using basic verification ({reason})")
            result.execution_time = time.time() - start
            return result
        
        try:
            # Prepare constraint for framework with mapped pattern ID
            z3_pattern = _PATTERN_ID_TO_Z3.get(constraint.pattern_id, constraint.pattern_id)
            constraint_dict = {
                'name': constraint.pattern_name,
                'pattern': z3_pattern,
                'context': constraint.context,
                'text': constraint.ocl
            }
            
            # Determine scope (how many instances to check)
            scope = self._get_verification_scope()
            
            # Verify using framework
            solver_result, model = self.checker.verify_all_constraints(
                constraints=[constraint_dict],
                scope=scope
            )
            
            result.solver_result = solver_result
            
            if solver_result == 'sat':
                result.is_valid = True
                result.is_satisfiable = True
            elif solver_result == 'unsat':
                result.is_valid = True  # Syntactically valid
                result.is_satisfiable = False
                result.warnings.append("Constraint is unsatisfiable")
            elif solver_result == 'unknown':
                result.is_valid = True
                result.is_satisfiable = None
                result.warnings.append("Verification timeout or unknown")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Framework verification error: {str(e)}")
        
        result.execution_time = time.time() - start
        return result
    
    def verify_batch(self, constraints: List[OCLConstraint], silent: bool = False) -> List[FrameworkVerificationResult]:
        """
        Verify multiple constraints together (checks global consistency).
        
        Args:
            constraints: List of constraints to verify
            silent: If True, suppress all console output
            
        Returns:
            List of verification results
        """
        if not self.framework_available:
            # Fallback - verify individually
            return [self.verify(c) for c in constraints]
        
        start = time.time()
        results = []
        
        try:
            # Prepare all constraints for framework with UNIQUE names and
            # mapped pattern IDs (generator IDs → Z3 encoder pattern names).
            unique_names = []
            constraint_dicts = []
            for i, c in enumerate(constraints):
                unique_name = f"{c.pattern_name}__{i}"
                unique_names.append(unique_name)
                # Map generator pattern ID to Z3 encoder's expected pattern name
                z3_pattern = _PATTERN_ID_TO_Z3.get(c.pattern_id, c.pattern_id)
                constraint_dicts.append({
                    'name': unique_name,
                    'pattern': z3_pattern,
                    'context': c.context,
                    'text': c.ocl
                })

            # Determine scope
            scope = self._get_verification_scope()

            if not silent:
                print(f"\nVerifying {len(constraints)} constraints for global consistency...")

            # Verify all together - suppress output if silent mode
            if silent:
                import sys, io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    solver_result, model = self.checker.verify_all_constraints(
                        constraints=constraint_dicts,
                        scope=scope
                    )
                finally:
                    sys.stdout = old_stdout
            else:
                solver_result, model = self.checker.verify_all_constraints(
                    constraints=constraint_dicts,
                    scope=scope
                )

            # Create results for each constraint using unique name for status lookup.
            # The global solver_result only applies to constraints that were
            # successfully encoded.  Constraints with encoding errors were never
            # added to the solver, so the global result says nothing about them.
            for i, c in enumerate(constraints):
                status = self.checker.constraint_status.get(unique_names[i], 'unknown')

                if status == 'error':
                    # Encoding failed — this constraint was NOT part of the
                    # Z3 check.  Mark as invalid with its own 'error' result.
                    result = FrameworkVerificationResult(
                        constraint_id=f"{c.pattern_id}_{c.context}",
                        is_valid=False,
                        solver_result='error'
                    )
                    result.errors.append("Encoding error — not verified by Z3")
                elif status == 'encoded':
                    result = FrameworkVerificationResult(
                        constraint_id=f"{c.pattern_id}_{c.context}",
                        is_valid=True,
                        solver_result=solver_result
                    )
                    if solver_result == 'sat':
                        result.is_satisfiable = True
                    elif solver_result == 'unsat':
                        result.is_satisfiable = False
                        result.warnings.append("Part of unsatisfiable constraint set")
                    else:
                        result.is_satisfiable = None
                        result.warnings.append("Verification timeout")
                else:
                    # Status 'unknown' — constraint wasn't found in checker
                    # status dict.  Treat as unverified.
                    result = FrameworkVerificationResult(
                        constraint_id=f"{c.pattern_id}_{c.context}",
                        is_valid=True,
                        solver_result='unknown'
                    )
                    result.is_satisfiable = None
                    result.warnings.append("Constraint not found in encoder status")

                results.append(result)
            
        except Exception as e:
            print(f"Batch verification error: {e}")
            # Return error results for all
            for c in constraints:
                result = FrameworkVerificationResult(
                    constraint_id=f"{c.pattern_id}_{c.context}",
                    is_valid=False
                )
                result.errors.append(f"Batch verification error: {str(e)}")
                results.append(result)
        
        total_time = time.time() - start
        
        # Distribute time across constraints
        for r in results:
            r.execution_time = total_time / len(constraints) if constraints else 0
        
        return results
    
    def check_consistency(self, constraints: List[OCLConstraint]) -> Dict:
        """
        Check if constraints are mutually consistent.
        
        Args:
            constraints: List of constraints
            
        Returns:
            Consistency result dict
        """
        if not self.framework_available:
            return {
                'consistent': True,
                'verified': False,
                'message': self.unavailable_reason or 'Framework not available'
            }
        
        try:
            constraint_dicts = [
                {
                    'name': f"{c.pattern_name}__{i}",
                    'pattern': _PATTERN_ID_TO_Z3.get(c.pattern_id, c.pattern_id),
                    'context': c.context,
                    'text': c.ocl
                }
                for i, c in enumerate(constraints)
            ]
            
            scope = self._get_verification_scope()
            
            solver_result, model = self.checker.verify_all_constraints(
                constraints=constraint_dicts,
                scope=scope
            )
            
            return {
                'consistent': solver_result == 'sat',
                'verified': True,
                'solver_result': solver_result,
                'message': f"Global consistency check: {solver_result}"
            }
            
        except Exception as e:
            return {
                'consistent': False,
                'verified': False,
                'error': str(e)
            }
    
    def _get_verification_scope(self) -> Dict:
        """
        Determine verification scope (how many instances of each class).
        
        Returns:
            Scope dict like {'nPerson': 2, 'nCompany': 2, ...}
        """
        scope = {}

        for class_name in self.metamodel.get_class_names():
            scope[f'n{class_name}'] = self.scope_per_class

        return scope

    def _get_missing_framework_dependencies(self) -> List[str]:
        """Return required framework dependencies that are not installed.

        Only checks for dependencies actually needed by the Z3-based verifier.
        sentence-transformers is used by constraint_similarity (Step 5), not here.
        """
        required_modules = {
            'z3': 'z3-solver',
        }

        return [
            package_name
            for module_name, package_name in required_modules.items()
            if find_spec(module_name) is None
        ]
    
    def get_statistics(self, results: List[FrameworkVerificationResult]) -> Dict:
        """
        Get statistics from verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Statistics dict
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        satisfiable = sum(1 for r in results if r.is_satisfiable is True)
        unsatisfiable = sum(1 for r in results if r.is_satisfiable is False)
        unknown = sum(1 for r in results if r.is_satisfiable is None and r.is_valid)
        with_warnings = sum(1 for r in results if r.warnings)
        with_errors = sum(1 for r in results if r.errors)
        
        avg_time = sum(r.execution_time for r in results) / max(1, total)
        
        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'validity_rate': valid / max(1, total),
            'satisfiable': satisfiable,
            'unsatisfiable': unsatisfiable,
            'unknown': unknown,
            'with_warnings': with_warnings,
            'with_errors': with_errors,
            'avg_verification_time': avg_time,
            'framework_used': self.framework_available
        }
