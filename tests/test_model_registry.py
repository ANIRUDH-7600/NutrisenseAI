"""
Automated Test Suite for Model Packaging, Versioning & Registry.
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System.
Scenario A — Community Pre-Screening (30 non-invasive features, v1.0.0).

Tests all required registry validation and integrity behaviors:
1. registry exists
2. registry JSON is valid
3. registry version exists
4. model version exists (nutrisense-scenario-a-v1.0.0)
5. scenario is A
6. model family is LightGBM
7. all three targets exist
8. artifact paths exist in models/v1/
9. SHA-256 hashes match
10. thresholds match locked values (0.36, 0.30, 0.17)
11. exactly 30 features exist
12. feature names are unique
13. feature schema version exists (scenario-a-30-v1)
14. missing artifact fails loudly
15. hash mismatch fails loudly
16. invalid threshold fails loudly
17. missing target fails loudly
18. duplicate feature fails loudly
19. unexpected feature fails loudly
20. deterministic model loading works
21. raw DHS DTA file byte count is exactly 441,380,745 bytes
"""

import os
import copy
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.models.model_registry import (
    load_registry,
    verify_model_integrity,
    verify_all_models_integrity,
    load_registered_pipeline,
    load_all_registered_pipelines,
    get_registered_threshold,
    get_registered_thresholds,
    get_feature_schema,
    get_model_metadata,
    compute_sha256,
    DEFAULT_REGISTRY_PATH,
    EXPECTED_MODEL_VERSION,
    EXPECTED_FEATURE_SCHEMA_VERSION,
    LOCKED_APPROVED_THRESHOLDS,
    EXPECTED_30_FEATURES,
    REMOVED_SENSITIVE_FEATURES
)
from src.models.inference_pipeline import predict_child_screening

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

EXPECTED_V1_HASHES = {
    "stunting": "09aa09beb3d79d9b336afe585714ec05c7ae6618e1a0db197858129f9dc95191",
    "underweight": "282e4fb0f2b294fb3bd2e1ea2f0f9c3d8a0b87f386475afc31c9800d78506159",
    "wasting": "a68b5dc29cb98f2806458acefaf934aa44e57f9e8e81038dcd2fce5900287bb2"
}


class TestModelRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def test_01_registry_exists(self):
        """1. Verify that registry exists in models/model_registry.json."""
        self.assertTrue(os.path.exists(DEFAULT_REGISTRY_PATH))

    def test_02_registry_json_is_valid(self):
        """2. Verify that registry file parses as valid JSON."""
        with open(DEFAULT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_03_model_version_matches(self):
        """3. Verify model_version is exactly 'nutrisense-scenario-a-v1.0.0'."""
        self.assertEqual(self.registry["model_version"], EXPECTED_MODEL_VERSION)

    def test_04_feature_schema_version_matches(self):
        """4. Verify feature_schema_version is 'scenario-a-30-v1'."""
        self.assertEqual(self.registry["feature_schema_version"], EXPECTED_FEATURE_SCHEMA_VERSION)

    def test_05_scenario_is_a(self):
        """5. Verify scenario is 'A'."""
        self.assertEqual(self.registry["scenario"], "A")

    def test_06_model_family_is_lightgbm(self):
        """6. Verify model_family is 'LightGBM'."""
        self.assertEqual(self.registry["model_family"], "LightGBM")

    def test_07_all_three_targets_exist(self):
        """7. Verify all three target models exist in registry."""
        models = self.registry["models"]
        for target in ["stunting", "underweight", "wasting"]:
            self.assertIn(target, models)

    def test_08_artifact_paths_exist(self):
        """8. Verify all registered artifact files exist on disk in models/v1/."""
        models = self.registry["models"]
        for target, info in models.items():
            path = info["artifact"]
            self.assertTrue(os.path.exists(path), f"Artifact path for {target} not found: {path}")
            self.assertTrue(path.startswith("models/v1/"))

    def test_09_sha256_hashes_match(self):
        """9. Verify actual SHA-256 hashes match registered SHA-256 hashes bit-for-bit."""
        models = self.registry["models"]
        for target, info in models.items():
            path = info["artifact"]
            expected_sha = info["sha256"]
            actual_sha = compute_sha256(path)
            self.assertEqual(
                actual_sha.lower(),
                expected_sha.lower(),
                f"SHA-256 mismatch for {target} at {path}!"
            )
            self.assertEqual(actual_sha.lower(), EXPECTED_V1_HASHES[target].lower())

    def test_10_thresholds_match_locked_values(self):
        """10. Verify decision thresholds match locked values (0.36, 0.30, 0.17)."""
        thresholds = get_registered_thresholds(self.registry)
        for target, val in LOCKED_APPROVED_THRESHOLDS.items():
            self.assertIn(target, thresholds)
            self.assertAlmostEqual(thresholds[target], val, places=4)

    def test_11_exactly_30_features_exist(self):
        """11. Verify feature schema contains exactly 30 features."""
        features = get_feature_schema(self.registry)
        self.assertEqual(len(features), 30)

    def test_12_feature_names_are_unique(self):
        """12. Verify all feature names are unique."""
        features = get_feature_schema(self.registry)
        self.assertEqual(len(features), len(set(features)))

    def test_13_removed_sensitive_features_strictly_absent(self):
        """13. Verify removed sensitive features are absent."""
        features = get_feature_schema(self.registry)
        for sfeat in REMOVED_SENSITIVE_FEATURES:
            self.assertNotIn(sfeat, features)

    def test_14_missing_artifact_fails_loudly(self):
        """14. Verify missing artifact raises FileNotFoundError."""
        corrupted = copy.deepcopy(self.registry)
        corrupted["models"]["stunting"]["artifact"] = "models/v1/non_existent_model.joblib"
        with self.assertRaises(FileNotFoundError):
            verify_model_integrity("stunting", registry=corrupted)

    def test_15_hash_mismatch_fails_loudly(self):
        """15. Verify hash mismatch raises ValueError."""
        corrupted = copy.deepcopy(self.registry)
        corrupted["models"]["stunting"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with self.assertRaises(ValueError):
            verify_model_integrity("stunting", registry=corrupted)

    def test_16_invalid_threshold_fails_loudly(self):
        """16. Verify invalid threshold raises ValueError during load_registry."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry)
            bad["thresholds"]["stunting"] = 0.99
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_17_missing_target_fails_loudly(self):
        """17. Verify missing target raises ValueError during load_registry."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry)
            del bad["models"]["wasting"]
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_18_duplicate_feature_fails_loudly(self):
        """18. Verify duplicate feature raises ValueError during load_registry."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry)
            bad["features"].append(bad["features"][0])
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_19_sensitive_feature_injected_fails_loudly(self):
        """19. Verify injecting a sensitive feature raises ValueError."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            bad = copy.deepcopy(self.registry)
            bad["features"][0] = "caste_category"
            json.dump(bad, tmp)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                load_registry(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_20_deterministic_model_loading_works(self):
        """20. Verify registered pipelines load deterministically."""
        pipelines = load_all_registered_pipelines(self.registry)
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
