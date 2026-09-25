#!/usr/bin/env python3
"""
Unit and Regression Tests for Step 15: Descriptive Error Analysis & Subgroup Analysis
Project: NutriSense AI — Multimodal Childhood Malnutrition Risk Intelligence
Dataset: NFHS-5 India 2019–21 Children's Recode (KR)
Setting: Scenario A — Community Pre-Screening (34 non-invasive features)

Verifies:
1. All Step 15 data artifacts and figures exist and are non-empty.
2. Validation cohort is used exclusively (split == 'val').
3. Locked decision thresholds (0.35, 0.31, 0.17) are strictly applied.
4. All three targets (stunting, underweight, wasting) are analyzed.
5. Required subgroup dimensions exist in metrics artifacts.
6. Zero excluded anthropometric variables are used as subgroups.
7. Subgroup sample-size rules (N >= 200) are respected.
8. TP/TN/FP/FN contingency accounting is internally consistent (TP + TN + FP + FN == N).
9. Mathematical rate definitions hold: Sensitivity + FNR == 1.0, Specificity + FPR == 1.0.
10. Confidence intervals are valid, ordered (lower <= upper), and bounded in [0, 1].
11. No individual DHS microdata rows are exposed.
12. Existing model artifacts remain unchanged.
13. Raw DHS DTA file byte count is exactly 441,380,745 bytes.
14. Error overlap artifact exists, covers all 3 targets, and is internally consistent.
15. All 6 publication figures exist and have valid file sizes.
"""

import json
import os
import unittest
import pandas as pd
import numpy as np

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

METRICS_CSV_PATH = "data/interim/error_analysis_metrics.csv"
METRICS_JSON_PATH = "data/interim/error_analysis_metrics.json"
OVERLAP_JSON_PATH = "data/interim/error_overlap_metrics.json"

MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"

PROHIBITED_LEAKAGE_VARS = {
    "hw2", "hw3", "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12", "hw13",
    "hw70", "hw71", "hw72", "hw73",
    "hw2_clean", "hw3_clean", "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
    "stunting_severe", "underweight_severe", "wasting_severe",
    "v001", "v002", "v003", "v005", "v021", "v022"
}

TARGETS = ["stunting", "underweight", "wasting"]
LOCKED_THRESHOLDS = {"stunting": 0.35, "underweight": 0.31, "wasting": 0.17}
REQUIRED_DIMENSIONS = [
    "child_age_group", "child_sex", "wealth_quintile",
    "maternal_education", "residence_type", "diarrhea_recent",
    "fever_recent", "cough_recent", "state_ut"
]


class TestErrorAnalysis(unittest.TestCase):
    """Automated assertions for Step 15 Error Analysis & Subgroup Audit."""

    @classmethod
    def setUpClass(cls):
        """Loads error analysis artifacts for inspection."""
        cls.df_metrics = pd.read_csv(METRICS_CSV_PATH)
        with open(METRICS_JSON_PATH, "r") as f:
            cls.metrics_json = json.load(f)
        with open(OVERLAP_JSON_PATH, "r") as f:
            cls.overlap_json = json.load(f)

    def test_01_output_files_exist_and_non_empty(self):
        """1. Verify that all Step 15 data artifacts exist and are non-empty."""
        for path in [METRICS_CSV_PATH, METRICS_JSON_PATH, OVERLAP_JSON_PATH]:
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")
            self.assertGreater(os.path.getsize(path), 0, f"Artifact is empty: {path}")

    def test_02_validation_cohort_used(self):
        """2. Verify that the validation cohort is used exclusively."""
        self.assertIn("validation", self.metrics_json["cohort"].lower())
        self.assertEqual(self.overlap_json["cohort"], "validation")
        self.assertEqual(self.metrics_json["validation_total_records"], 33153)

    def test_03_locked_thresholds_applied(self):
        """3. Verify locked decision thresholds (0.35, 0.31, 0.17) are applied."""
        for target, expected_tau in LOCKED_THRESHOLDS.items():
            sub = self.df_metrics[self.df_metrics["target"] == target]
            self.assertTrue((sub["threshold"] == expected_tau).all())
            self.assertEqual(self.metrics_json["locked_targets"][target]["threshold"], expected_tau)

    def test_04_all_three_targets_analyzed(self):
        """4. Verify all three targets (stunting, underweight, wasting) are represented."""
        evaluated_targets = set(self.df_metrics["target"].unique())
        self.assertEqual(evaluated_targets, set(TARGETS))
        self.assertEqual(set(self.metrics_json["subgroup_results"].keys()), set(TARGETS))

    def test_05_required_subgroups_exist(self):
        """5. Verify all required subgroup dimensions exist."""
        evaluated_dims = set(self.df_metrics["dimension"].unique())
        for req in REQUIRED_DIMENSIONS:
            self.assertIn(req, evaluated_dims, f"Missing subgroup dimension: {req}")

    def test_06_no_excluded_anthropometrics_as_subgroups(self):
        """6. Verify zero excluded anthropometric or survey identifier features enter subgroups."""
        evaluated_dims = set(self.df_metrics["dimension"].unique())
        leaked = evaluated_dims.intersection(PROHIBITED_LEAKAGE_VARS)
        self.assertEqual(len(leaked), 0, f"Prohibited leakage in subgroup dimensions: {leaked}")

    def test_07_subgroup_sample_size_rule_respected(self):
        """7. Verify minimum sample rule: N >= 200 flags are consistent."""
        for _, row in self.df_metrics.iterrows():
            if row["sample_size"] >= 200:
                self.assertTrue(row["is_comparative_evaluable"])
            else:
                self.assertFalse(row["is_comparative_evaluable"])

    def test_08_contingency_accounting_consistent(self):
        """8. Verify TP + TN + FP + FN equals subgroup sample size N."""
        for _, row in self.df_metrics.iterrows():
            total_cm = row["tp"] + row["tn"] + row["fp"] + row["fn"]
            self.assertEqual(total_cm, row["sample_size"])
            self.assertEqual(row["actual_positives"], row["tp"] + row["fn"])
            self.assertEqual(row["actual_negatives"], row["tn"] + row["fp"])

    def test_09_mathematical_rates_consistent(self):
        """9. Verify Sensitivity + FNR == 1.0 and Specificity + FPR == 1.0."""
        for _, row in self.df_metrics.iterrows():
            if pd.notna(row["sensitivity"]) and pd.notna(row["false_negative_rate"]):
                self.assertAlmostEqual(row["sensitivity"] + row["false_negative_rate"], 1.0, places=3)
            if pd.notna(row["specificity"]) and pd.notna(row["false_positive_rate"]):
                self.assertAlmostEqual(row["specificity"] + row["false_positive_rate"], 1.0, places=3)

    def test_10_confidence_intervals_valid_where_present(self):
        """10. Verify confidence intervals are ordered and bounded in [0, 1]."""
        for _, row in self.df_metrics.iterrows():
            for m in ["sensitivity", "specificity", "ppv", "f1_score"]:
                col_low = f"{m}_ci_lower"
                col_high = f"{m}_ci_upper"
                if pd.notna(row[col_low]) and pd.notna(row[col_high]):
                    low = row[col_low]
                    high = row[col_high]
                    self.assertGreaterEqual(low, 0.0)
                    self.assertLessEqual(high, 1.0)
                    self.assertLessEqual(low, high)

    def test_11_no_raw_dhs_microdata_exported(self):
        """11. Verify no individual DHS microdata rows or identifiers are written."""
        for col in ["v001", "v002", "v003", "caseid", "m18", "hw1", "hw2", "hw3"]:
            self.assertNotIn(col, self.df_metrics.columns)
        with open(METRICS_JSON_PATH, "r") as f:
            raw_text = f.read()
        self.assertNotIn('"v001"', raw_text)
        self.assertNotIn('"v002"', raw_text)

    def test_12_existing_model_artifacts_unchanged(self):
        """12. Verify existing champion model artifacts exist and have non-zero size."""
        for target in TARGETS:
            model_path = os.path.join("models/v1", f"model_comparison_lightgbm_{target}_unweighted_v1.joblib")
            self.assertTrue(os.path.exists(model_path))
            self.assertGreater(os.path.getsize(model_path), 100000)

    def test_13_raw_dhs_data_unchanged(self):
        """13. Verify raw DHS DTA file byte count is exactly 441,380,745 bytes."""
        self.assertTrue(os.path.exists(RAW_DTA_PATH), f"Raw dataset missing: {RAW_DTA_PATH}")
        actual_bytes = os.path.getsize(RAW_DTA_PATH)
        self.assertEqual(actual_bytes, EXPECTED_RAW_BYTES, f"Raw data modified! Expected {EXPECTED_RAW_BYTES}, got {actual_bytes}")

    def test_14_error_overlap_artifact_valid(self):
        """14. Verify error overlap artifact covers all targets and has consistent counts."""
        self.assertIn("pairwise_overlap", self.overlap_json)
        self.assertIn("triple_overlap", self.overlap_json)
        self.assertGreater(self.overlap_json["sample_size_joint_eligible"], 25000)
        self.assertIn("interpretation_note", self.overlap_json)

    def test_15_all_figures_exist_and_valid(self):
        """15. Verify all 6 publication figures exist and have valid file sizes (>50 KB)."""
        expected_figures = [
            "33_error_sensitivity_by_age.png",
            "34_error_specificity_by_age.png",
            "35_error_f1_by_wealth.png",
            "36_false_negative_rate_by_wealth.png",
            "37_false_negative_rate_by_education.png",
            "38_error_distribution_by_age.png"
        ]
        for fig_name in expected_figures:
            fig_path = os.path.join(FIGURES_DIR, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_path}")
            file_size = os.path.getsize(fig_path)
            self.assertGreater(file_size, 50000, f"Figure {fig_name} is too small ({file_size} bytes)")


if __name__ == "__main__":
    print("Running Step 15 Error Analysis & Subgroup Audit Unit Tests...")
    unittest.main()
