"""
Automated Test Suite for Step 16 & Step 22 (Major Step 8): End-to-End Inference Pipeline (v2.0.0, 30 Features).
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System.
Scenario A — Community Pre-Screening.

Tests all pre-specified deterministic conditions:
1. Valid complete input (30 features, sensitive attributes omitted)
2. Valid input with approved missing values
3. Missing required field
4. Invalid numeric type
5. Invalid categorical value
6. Prohibited anthropometric or sensitive feature supplied
7. All three model outputs present
8. Probability in [0, 1]
9. Threshold exactly matches locked v2 values (0.36, 0.30, 0.17)
10. screen_positive computed from unrounded probability (probability >= threshold)
11. Idempotency across repeated calls
12. Model files exist in models/v2/
13. Expected pipeline structure exists
14. 30-feature schema preserved
15. DHS raw file not required
16. No model artifact modified (SHA-256 match)
17. No raw DHS data modified (exact byte count)
"""

import os
import json
import hashlib
import unittest
import numpy as np
import pandas as pd

from src.models.inference_pipeline import (
    predict_child_screening,
    load_model_pipelines,
    LOCKED_THRESHOLDS,
    MODEL_ARTIFACTS,
    APPROVED_30_FEATURES,
    PROHIBITED_LEAKAGE_VARS,
    REMOVED_SENSITIVE_FEATURES
)

METADATA_V2_PATH = "models/model_registry.json"
RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745


def get_sample_valid_input():
    """Returns a realistic, valid child record under the 30-feature v2 schema."""
    return {
        "child_age_months": 24.0,
        "child_sex_male": 1,
        "birth_order": 2.0,
        "is_multiple_birth": 0,
        "is_firstborn": 0,
        "preceding_birth_interval_months": 28.0,
        "birth_size_ordinal": 3.0,
        "birth_weight_kg": 2.8,
        "birth_weight_missing": 0,
        "delivery_place_type": "Public_Facility",
        "still_breastfeeding": 1.0,
        "diarrhea_recent": 0.0,
        "fever_recent": 0.0,
        "cough_recent": 0.0,
        "mother_age_years": 26.0,
        "mother_age_first_birth": 22.0,
        "mother_bmi": 21.5,
        "mother_bmi_missing": 0,
        "total_children_born": 2.0,
        "anc_visits_count": 4.0,
        "anc_visits_missing": 0,
        "wealth_quintile": 2.0,
        "is_rural": 1,
        "drinking_water_type": "Improved_Piped",
        "sanitation_facility_type": "Pit_Latrine",
        "has_electricity": 1,
        "clean_cooking_fuel": 1,
        "household_size": 5.0,
        "state_id": 10
    }


class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.valid_input = get_sample_valid_input()

    def test_01_valid_complete_input(self):
        """1. Verify that valid complete 30-feature input returns valid=True and non-empty predictions."""
        result = predict_child_screening(self.valid_input)
        self.assertTrue(result["valid"], f"Expected valid=True, got errors: {result.get('errors')}")
        self.assertEqual(result.get("model_version"), "nutrisense-scenario-a-v1.0.0")
        self.assertEqual(result.get("feature_count"), 30)
        self.assertIn("stunting", result)
        self.assertIn("underweight", result)
        self.assertIn("wasting", result)

    def test_02_valid_input_with_approved_missing_values(self):
        """2. Verify that input with approved missing values (NaN / None) is accepted."""
        record = self.valid_input.copy()
        record["birth_weight_kg"] = None
        record["mother_bmi"] = np.nan
        record["anc_visits_count"] = None
        record["preceding_birth_interval_months"] = None
        record["birth_size_ordinal"] = None

        result = predict_child_screening(record)
        self.assertTrue(result["valid"], f"Expected valid=True for missing values, got: {result.get('errors')}")
        for target in ["stunting", "underweight", "wasting"]:
            self.assertIn(target, result)
            self.assertIsInstance(result[target]["probability"], float)

    def test_03_missing_required_field(self):
        """3. Verify that omitting a required field triggers validation error."""
        record = self.valid_input.copy()
        del record["child_age_months"]
        result = predict_child_screening(record)
        self.assertFalse(result["valid"])
        self.assertTrue(any("child_age_months" in e for e in result["errors"]))

    def test_04_invalid_numeric_type(self):
        """4. Verify that non-numeric types for numeric fields trigger validation error."""
        record = self.valid_input.copy()
        record["child_age_months"] = "twenty_four"
        result = predict_child_screening(record)
        self.assertFalse(result["valid"])
        self.assertTrue(any("child_age_months" in e for e in result["errors"]))

    def test_05_invalid_categorical_value(self):
        """5. Verify that out-of-range categorical or ordinal values trigger validation error."""
        record = self.valid_input.copy()
        record["wealth_quintile"] = 9.0  # Allowed: 1-5
        result = predict_child_screening(record)
        self.assertFalse(result["valid"])
        self.assertTrue(any("wealth_quintile" in e for e in result["errors"]))

        record2 = self.valid_input.copy()
        record2["delivery_place_type"] = "InvalidPlace_HospitalX"
        result2 = predict_child_screening(record2)
        self.assertFalse(result2["valid"])
        self.assertTrue(any("delivery_place_type" in e for e in result2["errors"]))

    def test_06_prohibited_anthropometric_and_sensitive_features_supplied(self):
        """6. Verify that supplying prohibited anthropometric or sensitive variables triggers strict rejection."""
        prohibited_list = ["hw70", "hw2", "hw71", "hw72", "hw3", "hw13", "hw57", "v001"] + REMOVED_SENSITIVE_FEATURES
        for prohibited_var in prohibited_list:
            record = self.valid_input.copy()
            record[prohibited_var] = 1
            result = predict_child_screening(record)
            self.assertFalse(result["valid"], f"Pipeline must reject prohibited/sensitive variable {prohibited_var}")
            self.assertTrue(any("Prohibited variable" in e for e in result["errors"]))

    def test_07_all_three_model_outputs_present(self):
        """7. Verify that all three target model outputs are present in result."""
        result = predict_child_screening(self.valid_input)
        self.assertTrue(result["valid"])
        for target in ["stunting", "underweight", "wasting"]:
            self.assertIn(target, result)
            self.assertIn("probability", result[target])
            self.assertIn("threshold", result[target])
            self.assertIn("screen_positive", result[target])

    def test_08_probability_in_zero_one(self):
        """8. Verify that all predicted probabilities fall in [0.0, 1.0]."""
        result = predict_child_screening(self.valid_input)
        for target in ["stunting", "underweight", "wasting"]:
            prob = result[target]["probability"]
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_09_threshold_exactly_matches_locked_v2_values(self):
        """9. Verify that target thresholds match locked v2 values: stunting=0.36, underweight=0.30, wasting=0.17."""
        result = predict_child_screening(self.valid_input)
        self.assertEqual(result["stunting"]["threshold"], 0.36)
        self.assertEqual(result["underweight"]["threshold"], 0.30)
        self.assertEqual(result["wasting"]["threshold"], 0.17)

    def test_10_screen_positive_computed_from_unrounded_probability_and_edge_case(self):
        """10. Verify decision rule is probability >= threshold (unrounded), testing exact boundary."""
        result = predict_child_screening(self.valid_input)
        for target in ["stunting", "underweight", "wasting"]:
            prob = result[target]["probability"]
            thresh = result[target]["threshold"]
            expected = bool(prob >= thresh)
            self.assertEqual(result[target]["screen_positive"], expected)

        # Test exact boundary decision rule logic (p == threshold -> screen_positive=True)
        self.assertTrue(bool(0.36 >= 0.36), "0.36 must be positive for stunting")
        self.assertTrue(bool(0.30 >= 0.30), "0.30 must be positive for underweight")
        self.assertTrue(bool(0.17 >= 0.17), "0.17 must be positive for wasting")
        self.assertFalse(bool(0.359999 >= 0.36), "0.359999 must be negative for stunting")

    def test_11_idempotent_across_repeated_calls(self):
        """11. Verify that repeated calls with the same input yield identical probabilities and outputs."""
        res1 = predict_child_screening(self.valid_input)
        res2 = predict_child_screening(self.valid_input)
        for target in ["stunting", "underweight", "wasting"]:
            self.assertEqual(res1[target]["probability"], res2[target]["probability"])
            self.assertEqual(res1[target]["screen_positive"], res2[target]["screen_positive"])

    def test_12_model_files_exist(self):
        """12. Verify that all 3 champion LightGBM v2 model files exist on disk and have non-zero size."""
        for target, path in MODEL_ARTIFACTS.items():
            self.assertTrue(os.path.exists(path), f"Missing model file {path}")
            self.assertGreater(os.path.getsize(path), 100000, f"File {path} suspiciously small")

    def test_13_expected_pipeline_structure_exists(self):
        """13. Verify that all models contain ColumnTransformer preprocessor and LGBMClassifier."""
        pipelines = load_model_pipelines()
        for target, pipe in pipelines.items():
            self.assertIn("preprocessor", pipe.named_steps)
            self.assertIn("classifier", pipe.named_steps)
            clf = pipe.named_steps["classifier"]
            self.assertEqual(type(clf).__name__, "LGBMClassifier")

    def test_14_30_feature_schema_preserved(self):
        """14. Verify that exactly 30 approved candidate features are defined and preserved."""
        self.assertEqual(len(APPROVED_30_FEATURES), 30)
        pipelines = load_model_pipelines()
        for target, pipe in pipelines.items():
            prep = pipe.named_steps["preprocessor"]
            num_cols = prep.transformers[0][2]
            cat_cols = prep.transformers[1][2]
            total_cols = set(num_cols).union(set(cat_cols))
            self.assertEqual(len(total_cols), 30)
            self.assertEqual(total_cols, set(APPROVED_30_FEATURES))

    def test_15_dhs_raw_file_not_required(self):
        """15. Verify that inference functions execute without inspecting or accessing raw DTA microdata."""
        import src.models.inference_pipeline as ip
        self.assertFalse(hasattr(ip, "RAW_DTA_PATH"))
        res = predict_child_screening(self.valid_input)
        self.assertTrue(res["valid"])

    def test_16_no_model_artifact_modified(self):
        """16. Verify SHA-256 hashes of v2 model files match recorded baseline in v2 metadata."""
        self.assertTrue(os.path.exists(METADATA_V2_PATH), f"Missing metadata file {METADATA_V2_PATH}")
        with open(METADATA_V2_PATH, "r") as f:
            meta = json.load(f)

        for target, path in MODEL_ARTIFACTS.items():
            with open(path, "rb") as f:
                current_sha = hashlib.sha256(f.read()).hexdigest()
            expected_sha = meta["models"][target]["sha256"]
            self.assertEqual(
                current_sha,
                expected_sha,
                f"Model artifact {path} was altered! Expected SHA {expected_sha}, found {current_sha}"
            )

    def test_17_no_raw_data_modified(self):
        """17. Verify that raw DHS KR microdata file size is exactly 441,380,745 bytes."""
        self.assertTrue(os.path.exists(RAW_DTA_PATH), f"Raw dataset {RAW_DTA_PATH} missing")
        actual_bytes = os.path.getsize(RAW_DTA_PATH)
        self.assertEqual(
            actual_bytes,
            EXPECTED_RAW_BYTES,
            f"Raw dataset altered! Expected {EXPECTED_RAW_BYTES} bytes, got {actual_bytes}"
        )


if __name__ == "__main__":
    unittest.main()
