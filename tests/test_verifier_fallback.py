import unittest
from unittest.mock import patch

from modules.core.models import Metamodel, OCLConstraint
from modules.generation.benchmark.suite_controller_enhanced import EnhancedSuiteController
from modules.verification.framework_verifier import FrameworkConstraintVerifier


class DummyUnavailableVerifier:
    framework_available = False

    def verify_batch(self, constraints, silent=False):
        raise AssertionError("verify_batch should not be called when the framework is unavailable")


class VerifierFallbackTests(unittest.TestCase):
    def setUp(self):
        self.metamodel = Metamodel()
        self.constraint = OCLConstraint(
            ocl="context Sensor inv NonEmptyName: self.name <> ''",
            pattern_id="test_pattern",
            pattern_name="NonEmptyName",
            context="Sensor",
            parameters={},
        )

    @patch(
        "modules.verification.framework_verifier.find_spec",
        side_effect=lambda module_name: None if module_name == "z3" else object(),
    )
    def test_basic_fallback_marks_results_unknown(self, _mock_find_spec):
        verifier = FrameworkConstraintVerifier(self.metamodel, "models/iotsensornetwork.xmi")

        self.assertFalse(verifier.framework_available)
        self.assertIn("z3-solver", verifier.unavailable_reason)

        result = verifier.verify(self.constraint)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.solver_result, "unknown")
        self.assertIsNone(result.is_satisfiable)
        self.assertTrue(any("basic verification" in warning for warning in result.warnings))

    def test_compatible_subset_keeps_constraints_without_framework(self):
        controller = EnhancedSuiteController.__new__(EnhancedSuiteController)
        constraints = [
            self.constraint,
            OCLConstraint(
                ocl="context Sensor inv PositiveId: self.id > 0",
                pattern_id="test_pattern_2",
                pattern_name="PositiveId",
                context="Sensor",
                parameters={},
            ),
        ]

        subset = controller._find_compatible_subset_batch(
            constraints,
            DummyUnavailableVerifier(),
        )

        self.assertEqual(subset, constraints)


if __name__ == "__main__":
    unittest.main()
