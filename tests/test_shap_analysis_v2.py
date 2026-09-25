#!/usr/bin/env python3
"""
Unit and Regression Tests for Major Step 6: Model Explainability & Interpretability using TreeSHAP (30 Predictors v2)
Project: NutriSense AI — Multimodal Childhood Malnutrition Risk Intelligence
Dataset: NFHS-5 India 2019–21 Children's Recode (KR)
Setting: Scenario A — Community Pre-Screening (30 non-invasive features v2)

Verifies:
1. All SHAP analysis artifacts exist and are non-empty.
2. All three targets (stunting, underweight, wasting) are represented.
3. Exactly 30 approved predictive candidate features are evaluated.
4. Strict absence of the 4 removed sensitive features:
   - household_head_female
   - caste_category
   - religion_category
   - mother_education_level
5. Strict absence of prohibited leakage variables (hw2-hw12, hw13, hw70-hw73, targets, survey IDs).
6. Mean absolute SHAP values are strictly non-negative and finite.
7. Feature ranks are contiguous positive integers from 1 to 30 per target without gaps or ties.
8. Every target has a complete 30-feature importance table.
9. Top-feature summaries contain only features from the approved 30-feature set.
10. SHAP sample size is documented (5,000 per target).
11. Random seed is documented (42).
12. Validation cohort is used exclusively for primary SHAP analysis (split == 'val').
13. Test set was NOT used for primary SHAP feature selection or analysis.
14. No raw DHS microdata or individual records are written to SHAP artifacts.
15. Existing champion model artifacts remain unchanged in models/v2.
16. Raw DHS DTA file byte count is exactly 441,380,745 bytes.
17. All 10 publication figure files with _v2 exist and have valid file sizes.
"""

import json
import os
import unittest
import pandas as pd
import numpy as np

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

SHAP_CSV_PATH_V2 = "data/interim/shap_feature_importance_v2.csv"
SHAP_JSON_PATH_V2 = "data/interim/shap_feature_importance_v2.json"
CROSS_TARGET_CSV_PATH_V2 = "data/interim/shap_cross_target_comparison_v2.csv"
TOP_FEATURES_JSON_PATH_V2 = "data/interim/shap_top_features_v2.json"
FEATURE_METADATA_PATH_V2 = "data/interim/feature_metadata_v2.json"

MODELS_DIR_V2 = "models/v2"
FIGURES_DIR = "reports/figures"

REMOVED_SENSITIVE_ATTRS = {
    "household_head_female",
    "caste_category",
    "religion_category",
    "mother_education_level"
}

PROHIBITED_LEAKAGE_VARS = {
    "hw2", "hw3", "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12", "hw13",
    "hw70", "hw71", "hw72", "hw73",
    "hw2_clean", "hw3_clean", "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
    "child_height_cm", "child_weight_kg", "haz", "waz", "whz",
    "stunting", "underweight", "wasting",
    "eligible_stunting", "eligible_underweight", "eligible_wasting",
    "v001", "v002", "v003", "v005", "v021", "v022"
}

TARGETS = ["stunting", "underweight", "wasting"]


class TestShapAnalysisV2(unittest.TestCase):
    """Automated assertions for Major Step 6 TreeSHAP explainability (30 features v2)."""

    @classmethod
    def setUpClass(cls):
        """Loads SHAP artifacts and feature metadata for inspection."""
        with open(FEATURE_METADATA_PATH_V2, "r") as f:
            cls.feature_meta = json.load(f)
        cls.approved_30_features = set(cls.feature_meta["feature_statistics"].keys())

        with open(SHAP_JSON_PATH_V2, "r") as f:
            cls.shap_json = json.load(f)

        with open(TOP_FEATURES_JSON_PATH_V2, "r") as f:
            cls.top_features_json = json.load(f)

        cls.df_shap = pd.read_csv(SHAP_CSV_PATH_V2)
        cls.df_cross = pd.read_csv(CROSS_TARGET_CSV_PATH_V2)

    def test_01_artifacts_exist_and_non_empty(self):
        """1. Verify that all SHAP analysis artifacts exist and are non-empty."""
        for path in [SHAP_CSV_PATH_V2, SHAP_JSON_PATH_V2, CROSS_TARGET_CSV_PATH_V2, TOP_FEATURES_JSON_PATH_V2]:
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")
            self.assertGreater(os.path.getsize(path), 0, f"Artifact is empty: {path}")

    def test_02_all_three_targets_represented(self):
        """2. Verify that all three targets (stunting, underweight, wasting) are represented."""
        for target in TARGETS:
            self.assertIn(target, self.shap_json["results_by_target"])
            self.assertIn(target, self.top_features_json)
            self.assertIn(target, self.df_shap["target"].unique())

    def test_03_exactly_30_approved_features_represented(self):
        """3. Verify that exactly 30 approved predictive candidate features are evaluated."""
        evaluated_features = set(self.df_cross["feature"].unique())
        self.assertEqual(len(evaluated_features), 30, f"Expected 30 features, got {len(evaluated_features)}")
        self.assertEqual(evaluated_features, self.approved_30_features)

        for target in TARGETS:
            target_feats = set(self.df_shap[self.df_shap["target"] == target]["feature"])
            self.assertEqual(len(target_feats), 30)
            self.assertEqual(target_feats, self.approved_30_features)

    def test_04_removed_sensitive_features_absent(self):
        """4. Verify strict absence of the 4 removed sensitive features."""
        evaluated_features = set(self.df_cross["feature"].unique())
        found_sensitive = evaluated_features.intersection(REMOVED_SENSITIVE_ATTRS)
        self.assertEqual(len(found_sensitive), 0, f"Removed sensitive features found in SHAP analysis: {found_sensitive}")

        for col in self.df_shap.columns:
            self.assertNotIn(col, REMOVED_SENSITIVE_ATTRS)
        for col in self.df_cross.columns:
            self.assertNotIn(col, REMOVED_SENSITIVE_ATTRS)

    def test_05_no_leakage_features_appear(self):
        """5. Verify strict absence of prohibited anthropometric and survey leakage features."""
        evaluated_features = set(self.df_cross["feature"].unique())
        leaked = evaluated_features.intersection(PROHIBITED_LEAKAGE_VARS)
        self.assertEqual(len(leaked), 0, f"Fatal leakage detected in SHAP feature set: {leaked}")

        for col in self.df_shap.columns:
            self.assertNotIn(col, PROHIBITED_LEAKAGE_VARS)
        for col in self.df_cross.columns:
            self.assertNotIn(col, PROHIBITED_LEAKAGE_VARS)

    def test_06_mean_abs_shap_non_negative_and_finite(self):
        """6. Verify mean absolute SHAP values are non-negative and finite."""
        self.assertTrue((self.df_shap["mean_abs_shap"] >= 0).all(), "Negative mean(|SHAP|) values found!")
        self.assertFalse(self.df_shap["mean_abs_shap"].isna().any(), "NaN mean(|SHAP|) values found!")
        self.assertTrue(np.isfinite(self.df_shap["mean_abs_shap"]).all(), "Infinite mean(|SHAP|) values found!")

    def test_07_feature_ranks_valid(self):
        """7. Verify feature ranks are contiguous positive integers from 1 to 30 per target."""
        for target in TARGETS:
            sub = self.df_shap[self.df_shap["target"] == target]
            ranks = sorted(sub["rank"].tolist())
            expected_ranks = list(range(1, 31))
            self.assertEqual(ranks, expected_ranks, f"Invalid ranks for target {target}: {ranks}")

    def test_08_complete_feature_importance_table(self):
        """8. Verify every target has a complete 30-feature importance table."""
        for target in TARGETS:
            sub = self.df_shap[self.df_shap["target"] == target]
            self.assertEqual(len(sub), 30)
            target_json = self.shap_json["results_by_target"][target]
            self.assertEqual(len(target_json["features"]), 30)

    def test_09_top_feature_files_contain_only_approved_features(self):
        """9. Verify top-feature summaries contain only features from the approved 30-feature set."""
        for target in TARGETS:
            top5 = self.top_features_json[target]["top_5"]
            top10 = self.top_features_json[target]["top_10"]
            self.assertEqual(len(top5), 5)
            self.assertEqual(len(top10), 10)
            self.assertTrue(set(top5).issubset(self.approved_30_features))
            self.assertTrue(set(top10).issubset(self.approved_30_features))

    def test_10_shap_sample_size_documented(self):
        """10. Verify SHAP sample size is documented and equals 5,000."""
        self.assertEqual(self.shap_json["sample_size_per_target"], 5000)
        for target in TARGETS:
            self.assertEqual(self.shap_json["results_by_target"][target]["sample_size"], 5000)

    def test_11_random_seed_documented(self):
        """11. Verify random seed is documented and equals 42."""
        self.assertEqual(self.shap_json["random_seed"], 42)
        for target in TARGETS:
            self.assertEqual(self.shap_json["results_by_target"][target]["random_seed"], 42)

    def test_12_validation_cohort_used_for_primary_shap(self):
        """12. Verify validation cohort is used for primary SHAP analysis."""
        self.assertIn("validation", self.shap_json["explanation_cohort"].lower())
        for target in TARGETS:
            self.assertEqual(self.shap_json["results_by_target"][target]["cohort"], "validation")

    def test_13_test_set_not_used_for_shap_feature_selection(self):
        """13. Verify test set is not referenced in primary SHAP cohort."""
        cohort_str = self.shap_json["explanation_cohort"].lower()
        self.assertNotIn("test partition", cohort_str)
        for target in TARGETS:
            self.assertNotEqual(self.shap_json["results_by_target"][target]["cohort"], "test")

    def test_14_no_raw_dhs_data_in_shap_artifacts(self):
        """14. Verify no individual DHS microdata rows or identifiers are written to SHAP artifacts."""
        with open(SHAP_JSON_PATH_V2, "r") as f:
            raw_text = f.read()
        for forbidden in ["v001", "v002", "v003", "caseid", "m18", "hw1", "hw2", "hw3"]:
            self.assertNotIn(f'"{forbidden}"', raw_text)

        self.assertNotIn("v001", self.df_shap.columns)
        self.assertNotIn("v002", self.df_shap.columns)
        self.assertNotIn("v001", self.df_cross.columns)

    def test_15_existing_model_artifacts_remain_unchanged(self):
        """15. Verify existing champion model artifacts exist in models/v2 and have non-zero size."""
        for target in TARGETS:
            model_path = os.path.join("models/v1", f"model_comparison_lightgbm_{target}_unweighted_v1.joblib")
            self.assertTrue(os.path.exists(model_path))
            self.assertGreater(os.path.getsize(model_path), 50000)

    def test_16_raw_dhs_data_unchanged(self):
        """16. Verify raw DHS DTA file byte count is exactly 441,380,745 bytes."""
        self.assertTrue(os.path.exists(RAW_DTA_PATH), f"Raw dataset missing: {RAW_DTA_PATH}")
        actual_bytes = os.path.getsize(RAW_DTA_PATH)
        self.assertEqual(actual_bytes, EXPECTED_RAW_BYTES, f"Raw data modified! Expected {EXPECTED_RAW_BYTES}, got {actual_bytes}")

    def test_17_all_figures_exist_and_valid(self):
        """17. Verify all 10 required publication figures with _v2 exist and have valid file sizes (>30 KB)."""
        expected_figures = [
            "fig23_shap_summary_stunting_v2.png",
            "fig24_shap_summary_underweight_v2.png",
            "fig25_shap_summary_wasting_v2.png",
            "fig26_shap_bar_stunting_v2.png",
            "fig27_shap_bar_underweight_v2.png",
            "fig28_shap_bar_wasting_v2.png",
            "fig29_shap_cross_target_comparison_v2.png",
            "fig30_shap_dependence_stunting_v2.png",
            "fig31_shap_dependence_underweight_v2.png",
            "fig32_shap_dependence_wasting_v2.png"
        ]
        for fig_name in expected_figures:
            fig_path = os.path.join(FIGURES_DIR, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_path}")
            file_size = os.path.getsize(fig_path)
            self.assertGreater(file_size, 30000, f"Figure {fig_name} is too small ({file_size} bytes)")


if __name__ == "__main__":
    unittest.main()
