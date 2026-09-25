"""
Automated Test Suite for Model Packaging, Versioning & Registry (v2 compatibility alias).
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System.
Scenario A — Community Pre-Screening (30 non-invasive features, v1.0.0 active).
"""

import os
import copy
import json
import tempfile
import unittest

from src.models.model_registry import (
    load_registry,
    load_registry_v2,
    verify_model_integrity_v2,
    verify_all_models_integrity_v2,
    load_registered_pipeline_v2,
    load_all_registered_pipelines_v2,
    get_registered_thresholds_v2,
    get_feature_schema_v2,
    get_model_metadata_v2,
    compute_sha256,
    DEFAULT_REGISTRY_PATH,
    EXPECTED_MODEL_VERSION,
    EXPECTED_FEATURE_SCHEMA_VERSION,
    LOCKED_APPROVED_THRESHOLDS,
    EXPECTED_30_FEATURES,
    REMOVED_SENSITIVE_FEATURES
)

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

EXPECTED_V1_HASHES = {
    "stunting": "09aa09beb3d79d9b336afe585714ec05c7ae6618e1a0db197858129f9dc95191",
    "underweight": "282e4fb0f2b294fb3bd2e1ea2f0f9c3d8a0b87f386475afc31c9800d78506159",
    "wasting": "a68b5dc29cb98f2806458acefaf934aa44e57f9e8e81038dcd2fce5900287bb2"
}


class TestModelRegistryV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry_v2 = load_registry_v2()

    def test_01_v2_registry_exists(self):
        """1. Verify that registry exists in models/model_registry.json or models/v1/model_registry.json."""
        self.assertTrue(os.path.exists(DEFAULT_REGISTRY_PATH) or os.path.exists("models/v1/model_registry.json"))

    def test_02_v2_registry_json_is_valid(self):
        """2. Verify that registry file parses as valid JSON."""
        with open(DEFAULT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_03_model_version_matches(self):
        """3. Verify model_version is 'nutrisense-scenario-a-v1.0.0'."""
        self.assertEqual(self.registry_v2["model_version"], EXPECTED_MODEL_VERSION)

    def test_04_feature_schema_version_matches(self):
        """4. Verify feature_schema_version is 'scenario-a-30-v1'."""
        self.assertEqual(self.registry_v2["feature_schema_version"], EXPECTED_FEATURE_SCHEMA_VERSION)

    def test_05_scenario_is_a(self):
        """5. Verify scenario is 'A'."""
        self.assertEqual(self.registry_v2["scenario"], "A")

    def test_06_model_family_is_lightgbm(self):
        """6. Verify model_family is 'LightGBM'."""
        self.assertEqual(self.registry_v2["model_family"], "LightGBM")

    def test_07_all_three_targets_exist(self):
        """7. Verify all three target models exist in registry."""
        models = self.registry_v2["models"]
        for target in ["stunting", "underweight", "wasting"]:
            self.assertIn(target, models)

    def test_08_artifact_paths_exist(self):
        """8. Verify all registered artifact files exist on disk in models/v1/."""
        models = self.registry_v2["models"]
        for target, info in models.items():
            path = info["artifact"]
            self.assertTrue(os.path.exists(path), f"Artifact path for {target} not found: {path}")
            self.assertTrue(path.startswith("models/v1/"))

    def test_09_sha256_hashes_match(self):
        """9. Verify actual SHA-256 hashes match registered SHA-256 hashes bit-for-bit."""
        models = self.registry_v2["models"]
        for target, info in models.items():
            path = info["artifact"]
            expected_sha = info["sha256"]
            actual_sha = compute_sha256(path)
            self.assertEqual(actual_sha.lower(), expected_sha.lower())
            self.assertEqual(actual_sha.lower(), EXPECTED_V1_HASHES[target].lower())

    def test_10_thresholds_match_locked_v2_values(self):
        """10. Verify decision thresholds match locked values (0.36, 0.30, 0.17)."""
        thresholds = get_registered_thresholds_v2(self.registry_v2)
        for target, val in LOCKED_APPROVED_THRESHOLDS.items():
            self.assertIn(target, thresholds)
            self.assertAlmostEqual(thresholds[target], val, places=4)

    def test_11_exactly_30_features_exist(self):
        """11. Verify feature schema contains exactly 30 features."""
        features = get_feature_schema_v2(self.registry_v2)
        self.assertEqual(len(features), 30)

    def test_12_feature_names_are_unique(self):
        """12. Verify all feature names are unique."""
        features = get_feature_schema_v2(self.registry_v2)
        self.assertEqual(len(features), len(set(features)))

    def test_13_removed_sensitive_features_strictly_absent(self):
        """13. Verify removed sensitive features are absent."""
        features = get_feature_schema_v2(self.registry_v2)
        for sfeat in REMOVED_SENSITIVE_FEATURES:
            self.assertNotIn(sfeat, features)

    def test_14_missing_artifact_fails_loudly(self):
        """14. Verify missing artifact raises FileNotFoundError."""
        corrupted = copy.deepcopy(self.registry_v2)
        corrupted["models"]["stunting"]["artifact"] = "models/v1/non_existent_model.joblib"
        with self.assertRaises(FileNotFoundError):
            verify_model_integrity_v2("stunting", registry=corrupted)

    def test_15_hash_mismatch_fails_loudly(self):
        """15. Verify hash mismatch raises ValueError."""
        corrupted = copy.deepcopy(self.registry_v2)
        corrupted["models"]["stunting"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with self.assertRaises(ValueError):
            verify_model_integrity_v2("stunting", registry=corrupted)

    def test_16_invalid_threshold_fails_loudly(self):
        """16. Verify invalid threshold raises ValueError during load_registry_v2."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry_v2)
            bad["thresholds"]["stunting"] = 0.99
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry_v2(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_17_missing_target_fails_loudly(self):
        """17. Verify missing target raises ValueError during load_registry_v2."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry_v2)
            del bad["models"]["wasting"]
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry_v2(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_18_duplicate_feature_fails_loudly(self):
        """18. Verify duplicate feature raises ValueError during load_registry_v2."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry_v2)
            bad["features"].append(bad["features"][0])
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry_v2(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_19_sensitive_feature_injected_fails_loudly(self):
        """19. Verify injecting a sensitive feature raises ValueError."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry_v2)
            bad["features"][0] = "caste_category"
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry_v2(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_20_deterministic_model_loading_works(self):
        """20. Verify registered pipelines load deterministically."""
        pipelines = load_all_registered_pipelines_v2(self.registry_v2)
        for target in ["stunting", "underweight", "wasting"]:
            self.assertIn(target, pipelines)
            pipeline = pipelines[target]
            self.assertTrue(hasattr(pipeline, "predict_proba"))

    def test_21_raw_dta_byte_count_exact(self):
        """21. Verify raw DHS DTA file byte count is exactly 441,380,745 bytes."""
        self.assertTrue(os.path.exists(RAW_DTA_PATH))
        actual_bytes = os.path.getsize(RAW_DTA_PATH)
        self.assertEqual(actual_bytes, EXPECTED_RAW_BYTES)


if __name__ == "__main__":
    unittest.main()
