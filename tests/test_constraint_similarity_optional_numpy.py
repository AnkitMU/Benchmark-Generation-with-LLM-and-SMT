import unittest
from unittest.mock import patch

from modules.core.models import OCLConstraint
from modules.generation.benchmark import constraint_similarity


class ConstraintSimilarityOptionalNumpyTests(unittest.TestCase):
    def setUp(self):
        self.constraint_a = OCLConstraint(
            ocl="context Sensor inv PositiveId: self.id > 0",
            pattern_id="a",
            pattern_name="PositiveId",
            context="Sensor",
            parameters={},
        )
        self.constraint_b = OCLConstraint(
            ocl="context Sensor inv PositiveReading: self.reading >= 0",
            pattern_id="b",
            pattern_name="PositiveReading",
            context="Sensor",
            parameters={},
        )

    def test_ast_similarity_works_without_numpy(self):
        with patch.object(constraint_similarity, "np", None):
            similarity = constraint_similarity.ast_similarity(
                self.constraint_a,
                self.constraint_b,
            )

        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_semantic_embedding_requires_numpy(self):
        with patch.object(constraint_similarity, "np", None):
            with self.assertRaises(ImportError):
                constraint_similarity.compute_embeddings_batch([self.constraint_a.ocl])


if __name__ == "__main__":
    unittest.main()
