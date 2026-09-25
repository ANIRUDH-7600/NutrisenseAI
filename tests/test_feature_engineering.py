"""Automated verification test suite for Step 7: Feature Engineering."""
import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

OUTPUT_FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

PROHIBITED_LEAKAGE_VARS = [
    "hw70", "hw71", "hw72", "hw73",
    "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
    "hw2", "hw3", "hw2_clean", "hw3_clean",
    "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12"
]

PROHIBITED_TARGETS = [
    "stunting", "underweight", "wasting",
    "stunting_severe", "underweight_severe", "wasting_severe",
    "eligible_stunting", "eligible_underweight", "eligible_wasting", "eligible_complete_unified"
]

PROHIBITED_SURVEY_VARS = [
    "v001", "v002", "v003", "v005", "v021", "v022", "sample_weight"
]


def test_feature_metadata_exists_and_valid():
    """Asserts feature_metadata.json exists, is non-empty, and has valid JSON structure."""
    assert os.path.exists(OUTPUT_FEATURE_METADATA_PATH), f"Missing {OUTPUT_FEATURE_METADATA_PATH}"
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    assert meta["total_records"] == 221263, f"Expected 221,263 records, got {meta['total_records']}"
    assert meta["total_candidate_features_in_matrix"] > 25, "Expected > 25 candidate features"
    assert meta["scenario"] == "Scenario A — Community Pre-Screening"


def test_strict_anthropometric_leakage_exclusion():
    """Verifies that no anthropometric measurement (hw70-hw73, hw2, hw3, hw4-hw12) enters candidate features."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["feature_statistics"].keys())
    
    for var in PROHIBITED_LEAKAGE_VARS:
        assert var not in feature_names, f"CRITICAL LEAKAGE DETECTED: Anthropometric variable '{var}' in feature matrix!"


def test_strict_target_exclusion():
    """Verifies that no target column or eligibility flag enters the candidate feature matrix."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["feature_statistics"].keys())

    for target in PROHIBITED_TARGETS:
        assert target not in feature_names, f"CRITICAL LEAKAGE DETECTED: Target column '{target}' in feature matrix!"


def test_strict_survey_identifier_exclusion():
    """Verifies that survey cluster IDs and sampling weights do not enter the feature matrix."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["feature_statistics"].keys())

    for surv in PROHIBITED_SURVEY_VARS:
        assert surv not in feature_names, f"SURVEY LEAKAGE DETECTED: Survey variable '{surv}' in feature matrix!"


def test_no_duplicate_feature_names():
    """Verifies that all candidate feature names are unique."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    f_list = list(meta["feature_statistics"].keys())
    assert len(f_list) == len(set(f_list)), "Duplicate feature names detected in feature registry!"


def test_raw_dataset_immutability():
    """Verifies that the raw NFHS-5 .DTA file has not been altered."""
    assert os.path.exists(RAW_DTA_PATH), f"Raw dataset {RAW_DTA_PATH} missing!"
    current_size = os.path.getsize(RAW_DTA_PATH)
    assert current_size == EXPECTED_RAW_BYTES, (
        f"Raw .DTA altered! Expected {EXPECTED_RAW_BYTES} bytes, found {current_size} bytes."
    )


def test_categorical_and_numerical_internal_consistency():
    """Verifies that all candidate features are properly categorized and have documented strategies."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    inventory = meta["candidate_feature_inventory"]
    stats = meta["feature_statistics"]

    valid_types = {"numerical", "binary", "ordinal", "nominal"}
    for f_name, f_info in inventory.items():
        assert f_info["feature_type"] in valid_types, f"Invalid feature type for {f_name}: {f_info['feature_type']}"
        assert f_info["domain"] != "", f"Missing domain for {f_name}"
        assert f_info["missingness_strategy"] != "", f"Missing missingness strategy for {f_name}"
        assert f_info["encoding_strategy"] != "", f"Missing encoding strategy for {f_name}"
        assert f_info["scaling_recommendation"] != "", f"Missing scaling recommendation for {f_name}"


def test_candidate_feature_count_exactly_34():
    """Asserts that candidate feature set contains exactly 34 features with zero duplicates."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    assert meta["total_candidate_features_in_matrix"] == 34, (
        f"Expected exactly 34 candidate features, got {meta['total_candidate_features_in_matrix']}"
    )
    assert len(meta["feature_statistics"]) == 34
    assert len(meta["candidate_feature_inventory"]) == 34


def test_excluded_variables_count_exactly_59():
    """Asserts that exactly 59 excluded variables are documented with valid rationale."""
    with open(OUTPUT_FEATURE_METADATA_PATH, "r") as f:
        meta = json.load(f)
    assert meta["total_excluded_variables"] == 59, (
        f"Expected exactly 59 excluded variables, got {meta['total_excluded_variables']}"
    )
    assert len(meta["excluded_variables_registry"]) == 59


def test_reproducible_feature_transformation():
    """Tests the transform_candidate_features pipeline directly on a small batch."""
    from src.features.build_features import transform_candidate_features, validate_scenario_a_feature_matrix
    sample_df = pd.read_csv("data/interim/cleaned_u5_with_targets.csv.gz", nrows=50)
    features_df = transform_candidate_features(sample_df)
    assert len(features_df) == 50
    assert len(features_df.columns) == 34, f"Expected 34 columns, got {len(features_df.columns)}"
    assert validate_scenario_a_feature_matrix(features_df) is True


if __name__ == "__main__":
    print("[*] Running Step 7 Feature Engineering Verification Tests...")
    test_feature_metadata_exists_and_valid()
    print("  [OK] Feature metadata exists, non-empty, and valid JSON structure.")
    test_strict_anthropometric_leakage_exclusion()
    print("  [OK] Strict leakage isolation verified: 0 anthropometric measurements in features.")
    test_strict_target_exclusion()
    print("  [OK] Strict target isolation verified: 0 target columns in features.")
    test_strict_survey_identifier_exclusion()
    print("  [OK] Strict survey isolation verified: 0 cluster IDs or survey weights in features.")
    test_no_duplicate_feature_names()
    print("  [OK] Zero duplicate feature names across all candidate features.")
    test_candidate_feature_count_exactly_34()
    print("  [OK] Candidate feature matrix contains exactly 34 features.")
    test_excluded_variables_count_exactly_59()
    print("  [OK] Excluded variables registry contains exactly 59 documented variables.")
    test_raw_dataset_immutability()
    print("  [OK] Raw NFHS-5 .DTA dataset immutability strictly confirmed (441,380,745 bytes).")
    test_categorical_and_numerical_internal_consistency()
    print("  [OK] Categorical, ordinal, binary, and numerical feature types internally consistent.")
    test_reproducible_feature_transformation()
    print("  [OK] Reproducible feature transformation pipeline verified.")
    print("[SUCCESS] All Step 7 feature engineering tests passed successfully!")
