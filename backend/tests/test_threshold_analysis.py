"""
Automated verification test suite for Step 11: Threshold Calibration & Operational Decision Curve Analysis.
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745
INPUT_DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
SPLIT_METADATA_PATH = "data/interim/split_metadata.json"

THRESHOLD_METRICS_JSON = "data/interim/threshold_analysis_metrics.json"
THRESHOLD_METRICS_CSV = "data/interim/threshold_analysis_metrics.csv"
CALIBRATION_METRICS_JSON = "data/interim/calibration_metrics.json"
DCA_METRICS_CSV = "data/interim/decision_curve_metrics.csv"
FIGURES_DIR = "reports/figures"

EXPECTED_TARGETS = ["stunting", "underweight", "wasting"]


def test_required_files_and_artifacts_exist():
    """Asserts that all Step 11 data files, metrics, and generated figures exist."""
    assert os.path.exists(INPUT_DATA_PATH), f"Missing {INPUT_DATA_PATH}"
    assert os.path.exists(FEATURE_METADATA_PATH), f"Missing {FEATURE_METADATA_PATH}"
    assert os.path.exists(SPLIT_METADATA_PATH), f"Missing {SPLIT_METADATA_PATH}"
    assert os.path.exists(THRESHOLD_METRICS_JSON), f"Missing {THRESHOLD_METRICS_JSON}"
    assert os.path.exists(THRESHOLD_METRICS_CSV), f"Missing {THRESHOLD_METRICS_CSV}"
    assert os.path.exists(CALIBRATION_METRICS_JSON), f"Missing {CALIBRATION_METRICS_JSON}"
    assert os.path.exists(DCA_METRICS_CSV), f"Missing {DCA_METRICS_CSV}"

    expected_figures = [
        "fig12_threshold_curves_stunting.png",
        "fig13_threshold_curves_underweight.png",
        "fig14_threshold_curves_wasting.png",
        "fig15_precision_recall_curves.png",
        "fig16_roc_curves.png",
        "fig17_probability_calibration_curves.png",
        "fig18_decision_curve_analysis.png"
    ]
    for fig in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig)
        assert os.path.exists(fig_path), f"Missing figure: {fig_path}"
        assert os.path.getsize(fig_path) > 1000, f"Figure {fig_path} is suspiciously small!"


def test_raw_dataset_immutability():
    """Asserts raw .DTA file has not been altered or overwritten."""
    assert os.path.exists(RAW_DTA_PATH), f"Raw dataset {RAW_DTA_PATH} missing!"
    current_size = os.path.getsize(RAW_DTA_PATH)
    assert current_size == EXPECTED_RAW_BYTES, (
        f"Raw .DTA altered! Expected {EXPECTED_RAW_BYTES} bytes, found {current_size} bytes."
    )


def test_feature_registry_isolation():
    """Asserts exactly 34 approved candidate features are used with zero leakage."""
    with open(FEATURE_METADATA_PATH, "r") as f:
        f_meta = json.load(f)
    approved_feats = set(f_meta["feature_statistics"].keys())
    assert len(approved_feats) == 34, f"Expected 34 features, found {len(approved_feats)}"

    prohibited = {
        "v001", "v002", "v003", "v005", "hh_id", "mother_id", "split",
        "stunting", "underweight", "wasting", "stunting_severe", "underweight_severe", "wasting_severe",
        "eligible_stunting", "eligible_underweight", "eligible_wasting", "eligible_complete_unified",
        "hw70", "hw71", "hw72", "hw73", "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
        "hw2", "hw3", "hw2_clean", "hw3_clean"
    }
    for var in prohibited:
        assert var not in approved_feats, f"CRITICAL LEAKAGE: '{var}' found in candidate feature matrix!"


def test_test_set_lock_principle():
    """Asserts that the test set remains locked and was never accessed for threshold analysis or calibration."""
    with open(THRESHOLD_METRICS_JSON, "r") as f:
        meta = json.load(f)
    assert meta["test_set_status"]["locked"] is True
    assert meta["test_set_status"]["accessed"] is False

    with open(SPLIT_METADATA_PATH, "r") as f_s:
        s_meta = json.load(f_s)

    for target in EXPECTED_TARGETS:
        t_data = meta["threshold_summary"][target]
        expected_val_n = s_meta["target_specific_distributions"][target]["validation"]["valid_count"]
        assert t_data["validation_sample_size"] == expected_val_n


def test_threshold_grid_validity_and_contingency_sums():
    """Asserts that threshold values lie in [0.01, 0.99] and confusion matrix sums match cohort sizes."""
    df_grid = pd.read_csv(THRESHOLD_METRICS_CSV)
    assert (df_grid["threshold"] >= 0.01).all() and (df_grid["threshold"] <= 0.99).all()

    # Non-negative counts
    for col in ["tp", "fp", "fn", "tn"]:
        assert (df_grid[col] >= 0).all()

    # Sum of TP + FP + FN + TN equals total validation cohort
    with open(SPLIT_METADATA_PATH, "r") as f_s:
        s_meta = json.load(f_s)

    for target in EXPECTED_TARGETS:
        sub = df_grid[df_grid["target"] == target]
        expected_val_n = s_meta["target_specific_distributions"][target]["validation"]["valid_count"]
        contingency_sum = sub["tp"] + sub["fp"] + sub["fn"] + sub["tn"]
        assert (contingency_sum == expected_val_n).all(), f"Contingency sum mismatch for {target}"


def test_metric_domain_boundaries():
    """Asserts that diagnostic metrics are bounded within [0, 1]."""
    df_grid = pd.read_csv(THRESHOLD_METRICS_CSV)
    bounded_cols = [
        "accuracy", "precision", "recall_sensitivity", "specificity",
        "f1_score", "balanced_accuracy", "npv", "ppv", "fpr", "fnr"
    ]
    for col in bounded_cols:
        assert (df_grid[col] >= 0.0).all() and (df_grid[col] <= 1.0).all(), f"Values in {col} outside [0, 1]"

    # Youden's J lies in [-1, 1]
    assert (df_grid["youden_j"] >= -1.0).all() and (df_grid["youden_j"] <= 1.0).all()


def test_calibration_metrics_integrity():
    """Asserts that calibration metrics JSON contains valid Brier scores and reliability coordinates."""
    with open(CALIBRATION_METRICS_JSON, "r") as f:
        cal_meta = json.load(f)
    assert "calibration_summary" in cal_meta

    for target in EXPECTED_TARGETS:
        assert target in cal_meta["calibration_summary"]
        t_cal = cal_meta["calibration_summary"][target]
        brier = t_cal["brier_scores"]
        assert 0.0 <= brier["uncalibrated"] <= 0.5
        assert 0.0 <= brier["platt_scaling"] <= 0.5
        assert 0.0 <= brier["isotonic_regression"] <= 0.5

        # Check calibration curve points
        pts = t_cal["calibration_curve_points"]
        for method in ["uncalibrated", "platt_scaling", "isotonic_regression"]:
            preds = pts[method]["prob_pred"]
            trues = pts[method]["prob_true"]
            assert len(preds) > 0 and len(trues) > 0
            assert all(0.0 <= p <= 1.0 for p in preds)
            assert all(0.0 <= t <= 1.0 for t in trues)


def test_decision_curve_analysis_finiteness():
    """Asserts that Decision Curve Analysis Net Benefit values are finite and consistent."""
    df_dca = pd.read_csv(DCA_METRICS_CSV)
    assert len(df_dca) > 0
    assert set(df_dca["target"].unique()) == set(EXPECTED_TARGETS)

    for col in ["net_benefit_model", "net_benefit_all", "net_benefit_none"]:
        assert np.isfinite(df_dca[col]).all(), f"Non-finite values found in DCA column: {col}"

    assert (df_dca["net_benefit_none"] == 0.0).all()


if __name__ == "__main__":
    print("Running Step 11 Threshold Calibration tests...")
    test_required_files_and_artifacts_exist()
    print("[PASS] test_required_files_and_artifacts_exist")
    test_raw_dataset_immutability()
    print("[PASS] test_raw_dataset_immutability")
    test_feature_registry_isolation()
    print("[PASS] test_feature_registry_isolation")
    test_test_set_lock_principle()
    print("[PASS] test_test_set_lock_principle")
    test_threshold_grid_validity_and_contingency_sums()
    print("[PASS] test_threshold_grid_validity_and_contingency_sums")
    test_metric_domain_boundaries()
    print("[PASS] test_metric_domain_boundaries")
    test_calibration_metrics_integrity()
    print("[PASS] test_calibration_metrics_integrity")
    test_decision_curve_analysis_finiteness()
    print("[PASS] test_decision_curve_analysis_finiteness")
    print("\nAll Step 11 verification tests PASSED successfully!")
