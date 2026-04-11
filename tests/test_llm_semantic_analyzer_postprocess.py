import unittest

from modules.semantic.llm_semantic_analyzer import (
    ClassSemanticProfile,
    _classify_attr_domain,
    _postprocess_profiles,
)


class LLMSemanticAnalyzerPostprocessTests(unittest.TestCase):
    def test_domain_classifier_handles_camel_case_without_false_date_match(self):
        self.assertEqual(_classify_attr_domain("updateId"), "identifier")
        self.assertEqual(_classify_attr_domain("scheduledAt"), "date_business")
        self.assertEqual(_classify_attr_domain("memoryMB"), "memory")
        self.assertEqual(_classify_attr_domain("unit"), "unit")
        self.assertEqual(_classify_attr_domain("firmwareVersion"), "version")

    def test_postprocess_removes_count_vs_memory_pair(self):
        profile = ClassSemanticProfile(
            class_name="EdgeNode",
            groups=[],
            comparable_pairs={("cpuCores", "memoryMB")},
            incomparable_pairs=set(),
        )
        classes_data = {
            "EdgeNode": [
                {"name": "cpuCores", "type": "EInt"},
                {"name": "memoryMB", "type": "EInt"},
            ]
        }

        result = _postprocess_profiles({"EdgeNode": profile}, classes_data)["EdgeNode"]

        self.assertNotIn(("cpuCores", "memoryMB"), result.comparable_pairs)
        self.assertIn(("cpuCores", "memoryMB"), result.incomparable_pairs)

    def test_postprocess_removes_label_vs_unit_pair(self):
        profile = ClassSemanticProfile(
            class_name="Metric",
            groups=[],
            comparable_pairs={("name", "unit")},
            incomparable_pairs=set(),
        )
        classes_data = {
            "Metric": [
                {"name": "name", "type": "EString"},
                {"name": "unit", "type": "EString"},
            ]
        }

        result = _postprocess_profiles({"Metric": profile}, classes_data)["Metric"]

        self.assertNotIn(("name", "unit"), result.comparable_pairs)
        self.assertIn(("name", "unit"), result.incomparable_pairs)

    def test_postprocess_removes_identifier_vs_temporal_pair(self):
        profile = ClassSemanticProfile(
            class_name="OTAUpdate",
            groups=[],
            comparable_pairs={("scheduledAt", "updateId")},
            incomparable_pairs=set(),
        )
        classes_data = {
            "OTAUpdate": [
                {"name": "updateId", "type": "EString"},
                {"name": "scheduledAt", "type": "EString"},
            ]
        }

        result = _postprocess_profiles({"OTAUpdate": profile}, classes_data)["OTAUpdate"]

        self.assertNotIn(("scheduledAt", "updateId"), result.comparable_pairs)
        self.assertIn(("scheduledAt", "updateId"), result.incomparable_pairs)

    def test_postprocess_keeps_same_root_qualified_bounds(self):
        profile = ClassSemanticProfile(
            class_name="PricingRule",
            groups=[],
            comparable_pairs=set(),
            incomparable_pairs=set(),
        )
        classes_data = {
            "PricingRule": [
                {"name": "minPrice", "type": "EDouble"},
                {"name": "maxPrice", "type": "EDouble"},
            ]
        }

        result = _postprocess_profiles({"PricingRule": profile}, classes_data)["PricingRule"]

        self.assertIn(("maxPrice", "minPrice"), result.comparable_pairs)


if __name__ == "__main__":
    unittest.main()
