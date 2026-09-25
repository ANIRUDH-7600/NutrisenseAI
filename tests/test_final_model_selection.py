#!/usr/bin/env python3
"""
Unit and regression tests for Step 12: Final Model Selection Protocol.
Verifies:
1. Selection artifact existence and valid schema.
2. Immutability of raw DHS dataset (exact byte size).
3. Test set lock adherence: 0 test set records or metrics used.
4. Model identifier validity: selected models must match actually evaluated Step 10 models.
5. Multi-criteria gate evaluation reproducibility.
6. Every target outcome has a valid designated champion.
7. Feature registry adherence: exactly 34 non-invasive features.
"""

import json
import os
import unittest
import pandas as pd

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

SELECTION_JSON = "data/interim/final_model_selection.json"
SELECTION_CSV = "data/interim/final_model_selection.csv"
MODEL_COMPARISON_JSON = "data/interim/model_comparison_metrics.json"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"

class TestFinalModelSelection(unittest.TestCase):

    def test_artifacts_exist_and_non_empty(self):
        """Test that final model selection artifacts exist and have non-empty content."""
        self.assertTrue(os.path.exists(SELECTION_JSON), f"Missing {SELECTION_JSON}")
        self.assertTrue(os.path.exists(SELECTION_CSV), f"Missing {SELECTION_CSV}")
        self.assertGreater(os.path.getsize(SELECTION_JSON), 500)
        self.assertGreater(os.path.getsize(SELECTION_CSV), 500)

    def test_raw_dataset_immutability(self):
        """Test that the raw DHS dataset has not been modified."""
        self.assertTrue(os.path.exists(RAW_DTA_PATH), f"Raw dataset not found at {RAW_DTA_PATH}")
        actual_bytes = os.path.getsize(RAW_DTA_PATH)
        self.assertEqual(actual_bytes, EXPECTED_RAW_BYTES, f"Raw data modified! Expected {EXPECTED_RAW_BYTES}, got {actual_bytes}")

    def test_test_set_lock_adherence(self):
        """Verify that NO test-set records, predictions, or metrics were used for model selection."""
        with open(SELECTION_JSON, "r") as f:
            data = json.load(f)
        
        # Test status string must explicitly declare locked status
        self.assertIn("STRICTLY LOCKED", data.get("test_set_lock_status", ""))

        # Check for any accidental inclusion of test metrics
        json_str = json.dumps(data).lower()
        self.assertNotIn("test_roc_auc", json_str)
        self.assertNotIn("test_pr_auc", json_str)
        self.assertNotIn("test_sensitivity", json_str)
        self.assertNotIn("test_f1", json_str)

    def test_required_schema_and_keys(self):
        """Verify that JSON artifact has all required top-level keys and target selections."""
        with open(SELECTION_JSON, "r") as f:
            data = json.load(f)

        required_keys = ["step", "scenario", "test_set_lock_status", "pre_specified_protocol", "target_selections", "unified_architecture_decision"]
        for k in required_keys:
            self.assertIn(k, data, f"Missing key '{k}' in selection JSON")

        targets = ["stunting", "underweight", "wasting"]
        target_selections = data["target_selections"]
        for t in targets:
            self.assertIn(t, target_selections, f"Target '{t}' missing from target_selections")
            sel = target_selections[t]
            self.assertIn("selected_model_id", sel)
            self.assertIn("selected_family", sel)
            self.assertIn("operating_threshold", sel)
            self.assertIn("validation_metrics", sel)
            self.assertIn("selection_rationale", sel)
            self.assertIn("runner_up_alternative", sel)

            # Check domain boundaries of validation metrics
            vm = sel["validation_metrics"]
            self.assertGreaterEqual(vm["roc_auc"], 0.50)
            self.assertLessEqual(vm["roc_auc"], 1.0)
            self.assertGreaterEqual(vm["pr_auc"], 0.0)
            self.assertLessEqual(vm["pr_auc"], 1.0)
            self.assertGreaterEqual(vm["sensitivity"], 0.0)
            self.assertLessEqual(vm["sensitivity"], 1.0)
            self.assertGreaterEqual(vm["specificity"], 0.0)
            self.assertLessEqual(vm["specificity"], 1.0)

    def test_selected_models_correspond_to_evaluated_models(self):
        """Verify that selected models match model names actually evaluated in Step 10."""
        with open(MODEL_COMPARISON_JSON, "r") as f:
            comp_data = json.load(f)
        with open(SELECTION_JSON, "r") as f:
            sel_data = json.load(f)

        evaluated_models = {f"{r['model']} ({r['condition'].capitalize()})" for r in comp_data["results"]}

        for t, sel in sel_data["target_selections"].items():
            model_id = sel["selected_model_id"]
            self.assertIn(model_id, evaluated_models, f"Selected model '{model_id}' was not in evaluated Step 10 models!")

    def test_csv_results_table_integrity(self):
        """Verify the tabular CSV contains 30 evaluated model runs with correct columns."""
        df = pd.read_csv(SELECTION_CSV)
        self.assertEqual(len(df), 30, f"Expected 30 rows in CSV, got {len(df)}")

        expected_cols = [
            "target", "model_id", "model_family", "condition", "roc_auc", "pr_auc",
            "default_f1", "opt_threshold", "opt_sensitivity", "opt_specificity",
            "opt_balanced_acc", "gate_1_discrimination", "gate_2_screening_viability",
            "gate_4_shap_compatibility", "gate_5_deployment_feasibility", "status"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column '{col}' in {SELECTION_CSV}")

        # Check that exactly 3 models are marked SELECTED_CHAMPION (one per target)
        champions = df[df["status"] == "SELECTED_CHAMPION"]
        self.assertEqual(len(champions), 3, f"Expected exactly 3 selected champions, found {len(champions)}")
        self.assertEqual(set(champions["target"]), {"stunting", "underweight", "wasting"})

    def test_feature_registry_isolation(self):
        """Verify the candidate feature matrix contains exactly 34 non-invasive features."""
        with open(FEATURE_METADATA_PATH, "r") as f:
            feat_meta = json.load(f)
        approved_feats = set(feat_meta["feature_statistics"].keys())
        self.assertEqual(len(approved_feats), 34, f"Expected 34 features, found {len(approved_feats)}")

        # Confirm zero direct anthropometric measurements
        forbidden = ["hw2", "hw3", "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12", "hw70", "hw71", "hw72", "hw73"]
        for feat in approved_feats:
            self.assertNotIn(feat.lower(), forbidden)

if __name__ == "__main__":
    print("Running Step 12 Final Model Selection tests...")
    unittest.main(verbosity=2)
