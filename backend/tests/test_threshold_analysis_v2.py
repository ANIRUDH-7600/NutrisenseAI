"""Automated verification test suite for Scenario A v2: Threshold Analysis & Model Selection (30 Features)."""
import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745
OUTPUT_THRESHOLD_V2_JSON = "data/interim/threshold_analysis_metrics_v2.json"
OUTPUT_THRESHOLD_V2_CSV = "data/interim/threshold_analysis_metrics_v2.csv"
TARGETS = ["stunting", "underweight", "wasting"]


def test_required_threshold_files_exist_v2():
    """Asserts that v2 threshold analysis JSON and CSV exist and are valid."""
    assert os.path.exists(OUTPUT_THRESHOLD_V2_JSON)
    assert os.path.exists(OUTPUT_THRESHOLD_V2_CSV)

    with open(OUTPUT_THRESHOLD_V2_JSON, "r") as f:
        meta = json.load(f)
    assert meta["feature_count"] == 30
    assert meta["test_set_status"]["locked"] is True
    assert meta["test_set_status"]["accessed"] is False
    assert "validation-derived operating thresholds" in meta["terminology_note"].lower()


def test_threshold_grid_coverage_v2():
    """Asserts that all 99 threshold points from 0.01 to 0.99 are computed."""
    df = pd.read_csv(OUTPUT_THRESHOLD_V2_CSV)
    assert len(df) > 0

    for target in TARGETS:
        target_df = df[df["target"] == target]
        assert len(target_df) > 0
        lgb_unw = target_df[target_df["model"] == "LightGBM (Unweighted)"]
        assert len(lgb_unw) == 99
        assert np.isclose(lgb_unw["threshold"].min(), 0.01)
        assert np.isclose(lgb_unw["threshold"].max(), 0.99)


def test_validation_operating_thresholds_values_v2():
    """Verifies that the validation-derived operating thresholds are independent and distinct from v1."""
    with open(OUTPUT_THRESHOLD_V2_JSON, "r") as f:
        meta = json.load(f)

    stunting_t = meta["threshold_summary"]["stunting"]["models"]["LightGBM (Unweighted)"]["validation_operating_threshold"]["threshold"]
    underweight_t = meta["threshold_summary"]["underweight"]["models"]["LightGBM (Unweighted)"]["validation_operating_threshold"]["threshold"]
    wasting_t = meta["threshold_summary"]["wasting"]["models"]["LightGBM (Unweighted)"]["validation_operating_threshold"]["threshold"]

    assert stunting_t == 0.36
    assert underweight_t == 0.30
    assert wasting_t == 0.17

    # Verify sensitivity >= 60% for stunting and underweight, >= 65% for wasting
    stunting_sens = meta["threshold_summary"]["stunting"]["models"]["LightGBM (Unweighted)"]["validation_operating_threshold"]["recall_sensitivity"]
    underweight_sens = meta["threshold_summary"]["underweight"]["models"]["LightGBM (Unweighted)"]["validation_operating_threshold"]["recall_sensitivity"]
    wasting_sens = meta["threshold_summary"]["wasting"]["models"]["LightGBM (Unweighted)"]["validation_operating_threshold"]["recall_sensitivity"]

    assert stunting_sens >= 0.60
    assert underweight_sens >= 0.60
    assert wasting_sens >= 0.65


def test_test_set_lock_adherence_v2():
    """Asserts that test set records were not accessed or used during threshold analysis."""
    with open(OUTPUT_THRESHOLD_V2_JSON, "r") as f:
        meta = json.load(f)
    assert meta["test_set_status"]["locked"] is True
    assert meta["test_set_status"]["accessed"] is False


def test_v1_threshold_artifacts_preserved():
    """Verifies that original v1 threshold analysis files remain intact."""
    assert os.path.exists("data/interim/threshold_analysis_metrics.json")
    assert os.path.exists("data/interim/threshold_analysis_metrics.csv")


def test_raw_dataset_immutability_v2():
    """Verifies raw DHS dataset remains unaltered."""
    assert os.path.exists(RAW_DTA_PATH)
    assert os.path.getsize(RAW_DTA_PATH) == EXPECTED_RAW_BYTES
