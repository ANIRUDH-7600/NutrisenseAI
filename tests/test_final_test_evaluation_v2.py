#!/usr/bin/env python3
"""
Unit and regression tests for Major Step 5: Final Locked Test Evaluation (30-Feature Scenario-A v2).
Verifies:
1. Test evaluation artifacts exist and are valid.
2. All three targets (stunting, underweight, wasting) are present.
3. Model identifiers are strictly 'LightGBM Unweighted'.
4. Locked thresholds strictly match validation-derived operating cutoffs (0.36, 0.30, 0.17).
5. Thresholds were not re-optimized on test labels.
6. Required metrics exist (ROC-AUC, PR-AUC, Sensitivity, Specificity, PPV, F1, BalAcc, Brier).
7. Confusion matrices are internally consistent.
8. TP + TN + FP + FN equals the evaluated test sample size.
9. Sensitivity is consistent with TP / (TP + FN).
10. Specificity is consistent with TN / (TN + FP).
11. Precision is consistent with TP / (TP + FP).
12. F1 is harmonic mean of precision and recall.
13. Test metrics generated from locked test partition (N = 33,069).
14. No individual test records in output artifacts.
15. Raw DHS DTA file remains unchanged (441,380,745 bytes).
16. Validation-vs-test comparison contains expected targets and metrics.
17. The selected thresholds remain unchanged.
18. Zero model retraining occurred during test evaluation.
19. Figures exist and valid with _v2 suffix.
20. Feature count is explicitly recorded as 30 in metadata.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

FINAL_TEST_JSON_V2 = "data/interim/final_test_metrics_v2.json"
FINAL_TEST_CSV_V2 = "data/interim/final_test_metrics_v2.csv"
VAL_VS_TEST_CSV_V2 = "data/interim/validation_vs_test_metrics_v2.csv"
FIGURES_DIR = "reports/figures"

EXPECTED_THRESHOLDS_V2 = {
    "stunting": 0.36,
    "underweight": 0.30,
    "wasting": 0.17
}


class TestFinalTestEvaluationV2(unittest.TestCase):

    def test_01_artifacts_exist_and_non_empty(self):
        """1. Verify that all test evaluation data artifacts exist and are non-empty."""
        self.assertTrue(os.path.exists(FINAL_TEST_JSON_V2), f"Missing {FINAL_TEST_JSON_V2}")
        self.assertTrue(os.path.exists(FINAL_TEST_CSV_V2), f"Missing {FINAL_TEST_CSV_V2}")
        self.assertTrue(os.path.exists(VAL_VS_TEST_CSV_V2), f"Missing {VAL_VS_TEST_CSV_V2}")
        self.assertGreater(os.path.getsize(FINAL_TEST_JSON_V2), 500)
        self.assertGreater(os.path.getsize(FINAL_TEST_CSV_V2), 200)
        self.assertGreater(os.path.getsize(VAL_VS_TEST_CSV_V2), 200)

    def test_02_all_three_targets_present(self):
        """2. Verify that all three targets are evaluated."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        targets = list(data.get("test_results", {}).keys())
        self.assertEqual(set(targets), {"stunting", "underweight", "wasting"})

    def test_03_selected_model_identifiers_match(self):
        """3. Verify model identifiers are strictly LightGBM Unweighted."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            self.assertEqual(res["model_id"], "LightGBM Unweighted")

    def test_04_selected_thresholds_match_validation(self):
        """4. Verify selected thresholds strictly match validation-derived operating cutoffs (0.36, 0.30, 0.17)."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, expected_th in EXPECTED_THRESHOLDS_V2.items():
            actual_th = data["test_results"][t]["locked_threshold"]
            self.assertEqual(actual_th, expected_th, f"Threshold for {t} altered! Expected {expected_th}, got {actual_th}")

    def test_05_thresholds_not_optimized_on_test(self):
        """5. Verify thresholds were not optimized on test labels."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, expected_th in EXPECTED_THRESHOLDS_V2.items():
            m_th = data["test_results"][t]["metrics"]["threshold"]
            self.assertEqual(m_th, expected_th)

    def test_06_required_metrics_exist(self):
        """6. Verify all required diagnostic metrics exist in final test results."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        required_metrics = [
            "sample_size", "actual_positives", "actual_negatives", "predicted_positives",
            "predicted_negatives", "tp", "fp", "tn", "fn", "unweighted_test_prevalence",
            "threshold", "sensitivity", "specificity", "precision_ppv", "f1_score",
            "balanced_accuracy", "youden_j", "roc_auc", "pr_auc", "brier_score"
        ]
        for t, res in data["test_results"].items():
            for m in required_metrics:
                self.assertIn(m, res["metrics"], f"Missing metric '{m}' for target {t}")

    def test_07_confusion_matrices_internally_consistent(self):
        """7. Verify confusion matrices entries are positive integers."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            m = res["metrics"]
            self.assertGreater(m["tp"], 0)
            self.assertGreater(m["tn"], 0)
            self.assertGreater(m["fp"], 0)
            self.assertGreater(m["fn"], 0)

    def test_08_tp_tn_fp_fn_sum_equals_sample_size(self):
        """8. Verify TP + TN + FP + FN equals the evaluated test sample size."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            m = res["metrics"]
            total = m["tp"] + m["tn"] + m["fp"] + m["fn"]
            self.assertEqual(total, m["sample_size"], f"Sum {total} != sample_size {m['sample_size']} for {t}")

    def test_09_sensitivity_consistent_with_contingency(self):
        """9. Verify Sensitivity == TP / (TP + FN)."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            m = res["metrics"]
            expected_sens = m["tp"] / (m["tp"] + m["fn"])
            self.assertAlmostEqual(m["sensitivity"], expected_sens, places=4)

    def test_10_specificity_consistent_with_contingency(self):
        """10. Verify Specificity == TN / (TN + FP)."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            m = res["metrics"]
            expected_spec = m["tn"] / (m["tn"] + m["fp"])
            self.assertAlmostEqual(m["specificity"], expected_spec, places=4)

    def test_11_precision_consistent_with_contingency(self):
        """11. Verify Precision == TP / (TP + FP)."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            m = res["metrics"]
            expected_ppv = m["tp"] / (m["tp"] + m["fp"])
            self.assertAlmostEqual(m["precision_ppv"], expected_ppv, places=4)

    def test_12_f1_score_consistent(self):
        """12. Verify F1 is the harmonic mean of precision and sensitivity."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        for t, res in data["test_results"].items():
            m = res["metrics"]
            p = m["precision_ppv"]
            r = m["sensitivity"]
            expected_f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            self.assertAlmostEqual(m["f1_score"], expected_f1, places=4)

    def test_13_test_metrics_from_locked_partition(self):
        """13. Verify test metrics are generated from the locked test partition (N = 33,069)."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        self.assertEqual(data["test_partition_total_records"], 33069)
        for t, res in data["test_results"].items():
            self.assertGreater(res["metrics"]["sample_size"], 25000)
            self.assertLessEqual(res["metrics"]["sample_size"], 33069)

    def test_14_no_individual_dhs_records_exported(self):
        """14. Verify no individual microdata records are written to JSON or CSV."""
        df_csv = pd.read_csv(FINAL_TEST_CSV_V2)
        self.assertEqual(len(df_csv), 3, "Test CSV should only contain 3 summary rows!")
        df_val = pd.read_csv(VAL_VS_TEST_CSV_V2)
        self.assertEqual(len(df_val), 3, "Val vs Test CSV should only contain 3 summary rows!")

        with open(FINAL_TEST_JSON_V2, "r") as f:
            content = f.read()
        forbidden_identifiers = ["v001", "v002", "v003", "hh_id", "mother_id"]
        for fid in forbidden_identifiers:
            self.assertNotIn(f'"{fid}"', content)

    def test_15_raw_dhs_data_unchanged(self):
        """15. Verify raw DHS DTA file byte count is exactly 441,380,745 bytes."""
        self.assertTrue(os.path.exists(RAW_DTA_PATH), f"Raw dataset not found at {RAW_DTA_PATH}")
        actual_bytes = os.path.getsize(RAW_DTA_PATH)
        self.assertEqual(actual_bytes, EXPECTED_RAW_BYTES, f"Raw data modified! Expected {EXPECTED_RAW_BYTES}, got {actual_bytes}")

    def test_16_val_vs_test_comparison_integrity(self):
        """16. Verify validation vs test comparison table contains expected targets and deltas."""
        df = pd.read_csv(VAL_VS_TEST_CSV_V2)
        self.assertEqual(set(df["target"]), {"stunting", "underweight", "wasting"})
        for col in ["val_roc_auc", "test_roc_auc", "delta_roc_auc", "val_sensitivity", "test_sensitivity", "delta_sensitivity"]:
            self.assertIn(col, df.columns)

    def test_17_selected_thresholds_unchanged(self):
        """17. Verify selected thresholds remained unchanged throughout test evaluation."""
        df = pd.read_csv(VAL_VS_TEST_CSV_V2)
        for _, row in df.iterrows():
            target = row["target"]
            expected = EXPECTED_THRESHOLDS_V2[target]
            self.assertEqual(row["threshold"], expected)

    def test_18_no_model_retraining_occurred(self):
        """18. Verify no model retraining occurred (check evaluation principle)."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        self.assertIn("zero retraining", data.get("evaluation_principle", "").lower())

    def test_19_figures_exist_and_valid(self):
        """19. Verify all 4 required publication figures exist with _v2 and have valid file sizes."""
        expected_figs = [
            "fig19_test_confusion_matrices_v2.png",
            "fig20_validation_vs_test_metrics_v2.png",
            "fig21_test_roc_curves_v2.png",
            "fig22_test_precision_recall_curves_v2.png"
        ]
        for fig in expected_figs:
            fig_path = os.path.join(FIGURES_DIR, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_path}")
            self.assertGreater(os.path.getsize(fig_path), 5000, f"Figure {fig_path} suspiciously small!")

    def test_20_feature_count_is_30(self):
        """20. Verify feature count is recorded as 30."""
        with open(FINAL_TEST_JSON_V2, "r") as f:
            data = json.load(f)
        self.assertEqual(data.get("feature_count"), 30)
        self.assertEqual(data.get("feature_version"), "v2.0.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
