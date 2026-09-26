"""Automated verification test suite for Scenario A v2: 30-Feature Engineering."""
import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

OUTPUT_FEATURE_METADATA_V2_PATH = "data/interim/feature_metadata_v2.json"
OUTPUT_FEATURE_METADATA_V1_PATH = "data/interim/feature_metadata.json"
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

REMOVED_SENSITIVE_FEATURES = [
    "household_head_female",
    "caste_category",
    "religion_category",
    "mother_education_level"
]

EXPECTED_30_FEATURES = [
    "child_age_months",
    "child_age_group",
    "child_sex_male",
    "birth_order",
    "is_multiple_birth",
    "is_firstborn",
    "preceding_birth_interval_months",
    "birth_size_ordinal",
    "birth_weight_kg",
    "birth_weight_missing",
    "delivery_place_type",
    "still_breastfeeding",
    "diarrhea_recent",
    "fever_recent",
    "cough_recent",
    "mother_age_years",
    "mother_age_first_birth",
    "mother_bmi",
    "mother_bmi_missing",
    "total_children_born",
    "anc_visits_count",
    "anc_visits_missing",
    "wealth_quintile",
    "is_rural",
    "drinking_water_type",
    "sanitation_facility_type",
    "has_electricity",
    "clean_cooking_fuel",
    "household_size",
    "state_id"
]


def test_feature_metadata_v2_exists_and_valid():
    """Asserts feature_metadata_v2.json exists, is non-empty, and has valid JSON structure."""
    assert os.path.exists(OUTPUT_FEATURE_METADATA_V2_PATH), f"Missing {OUTPUT_FEATURE_METADATA_V2_PATH}"
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    assert meta["metadata_version"] == "2.0.0"
    assert meta["schema_version"] == "scenario-a-30-v2"
    assert meta["total_records"] == 221263
    assert meta["total_candidate_features_in_matrix"] == 30
    assert meta["scenario"] == "Scenario A — Community Pre-Screening"


def test_candidate_feature_count_exactly_30():
    """Asserts that candidate feature set contains exactly 30 features matching the expected specification."""
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    assert meta["total_candidate_features_in_matrix"] == 30
    assert len(meta["features"]) == 30
    assert len(meta["feature_statistics"]) == 30
    assert meta["features"] == EXPECTED_30_FEATURES


def test_sensitive_features_strictly_absent_from_v2():
    """Verifies that none of the 4 removed sensitive attributes exist in the v2 feature set."""
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["features"])
    stats_names = set(meta["feature_statistics"].keys())

    for feat in REMOVED_SENSITIVE_FEATURES:
        assert feat not in feature_names, f"Removed sensitive feature '{feat}' found in v2 feature list!"
        assert feat not in stats_names, f"Removed sensitive feature '{feat}' found in v2 feature statistics!"


def test_strict_anthropometric_leakage_exclusion_v2():
    """Verifies that no anthropometric measurement enters candidate features."""
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["features"])
    for var in PROHIBITED_LEAKAGE_VARS:
        assert var not in feature_names, f"CRITICAL LEAKAGE: Anthropometric variable '{var}' in feature matrix!"


def test_strict_target_exclusion_v2():
    """Verifies that no target column or eligibility flag enters the v2 feature matrix."""
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["features"])
    for target in PROHIBITED_TARGETS:
        assert target not in feature_names, f"CRITICAL LEAKAGE: Target column '{target}' in feature matrix!"


def test_strict_survey_identifier_exclusion_v2():
    """Verifies that survey cluster IDs and sampling weights do not enter the v2 feature matrix."""
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    feature_names = set(meta["features"])
    for surv in PROHIBITED_SURVEY_VARS:
        assert surv not in feature_names, f"SURVEY LEAKAGE: Survey variable '{surv}' in feature matrix!"


def test_categorical_and_numerical_counts_v2():
    """Verifies that 5 categorical and 25 numerical/binary/ordinal features are documented in v2."""
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "r") as f:
        meta = json.load(f)
    assert len(meta["categorical_features"]) == 5
    assert len(meta["numerical_features"]) == 25
    assert len(meta["categorical_features"]) + len(meta["numerical_features"]) == 30


def test_reproducible_feature_transformation_v2():
    """Tests the transform_candidate_features_v2 pipeline directly on sample batch."""
    from src.features.build_features import (
        transform_candidate_features_v2,
        validate_scenario_a_v2_feature_matrix
    )
    sample_df = pd.read_csv("data/interim/cleaned_u5_with_targets.csv.gz", nrows=50)
    features_df = transform_candidate_features_v2(sample_df)
    assert len(features_df) == 50
    assert len(features_df.columns) == 30
    assert list(features_df.columns) == EXPECTED_30_FEATURES
    assert validate_scenario_a_v2_feature_matrix(features_df) is True


def test_raw_dataset_immutability_v2():
    """Verifies that the raw NFHS-5 .DTA file has not been altered."""
    assert os.path.exists(RAW_DTA_PATH)
    assert os.path.getsize(RAW_DTA_PATH) == EXPECTED_RAW_BYTES


def test_v1_metadata_preserved():
    """Verifies that the original 34-feature v1 metadata remains intact for reproducibility."""
    assert os.path.exists(OUTPUT_FEATURE_METADATA_V1_PATH)
    with open(OUTPUT_FEATURE_METADATA_V1_PATH, "r") as f:
        meta_v1 = json.load(f)
    assert meta_v1["total_candidate_features_in_matrix"] == 34
    assert meta_v1["metadata_version"] == "1.0.0"


if __name__ == "__main__":
    print("[*] Running Scenario A v2 Feature Engineering Verification Tests...")
    test_feature_metadata_v2_exists_and_valid()
    print("  [OK] v2 Feature metadata exists and valid JSON structure.")
    test_candidate_feature_count_exactly_30()
    print("  [OK] Exactly 30 features verified in specification.")
    test_sensitive_features_strictly_absent_from_v2()
    print("  [OK] All 4 sensitive features strictly absent from v2.")
    test_strict_anthropometric_leakage_exclusion_v2()
    print("  [OK] Anthropometric leakage strictly excluded.")
    test_strict_target_exclusion_v2()
    print("  [OK] Target outcome columns strictly excluded.")
    test_strict_survey_identifier_exclusion_v2()
    print("  [OK] Survey identifiers and weights strictly excluded.")
    test_categorical_and_numerical_counts_v2()
    print("  [OK] Exactly 5 categorical and 25 numerical/binary features verified.")
    test_reproducible_feature_transformation_v2()
    print("  [OK] Reproducible v2 feature transformation pipeline verified.")
    test_raw_dataset_immutability_v2()
    print("  [OK] Raw DHS dataset immutability verified (441,380,745 bytes).")
    test_v1_metadata_preserved()
    print("  [OK] v1 34-feature metadata preserved intact.")
    print("[SUCCESS] All Scenario A v2 feature engineering tests passed successfully!")
