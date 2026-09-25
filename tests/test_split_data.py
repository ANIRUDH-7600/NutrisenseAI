"""
Automated verification test suite for Step 8: Train / Validation / Test Split.
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
OUTPUT_TARGETS_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
SPLIT_METADATA_PATH = "data/interim/split_metadata.json"
SPLIT_MANIFEST_PATH = "data/interim/split_manifest.csv.gz"


def test_split_files_exist_and_metadata_valid():
    """Asserts split_metadata.json and split_manifest.csv.gz exist and metadata is valid JSON."""
    assert os.path.exists(SPLIT_METADATA_PATH), f"Missing {SPLIT_METADATA_PATH}"
    assert os.path.exists(SPLIT_MANIFEST_PATH), f"Missing {SPLIT_MANIFEST_PATH}"
    
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
    assert meta["step"].startswith("Step 8"), f"Unexpected step in metadata: {meta['step']}"
    assert meta["random_seed"] == 42
    assert meta["partition_sizes"]["records"]["total"] == 221263
    assert meta["partition_sizes"]["households"]["total"] == 164339


def test_raw_dataset_immutability():
    """Asserts raw .DTA file has not been altered or overwritten."""
    assert os.path.exists(RAW_DTA_PATH), f"Raw dataset {RAW_DTA_PATH} missing!"
    current_size = os.path.getsize(RAW_DTA_PATH)
    assert current_size == EXPECTED_RAW_BYTES, (
        f"Raw .DTA altered! Expected {EXPECTED_RAW_BYTES} bytes, found {current_size} bytes."
    )


def test_zero_household_leakage():
    """Asserts that no household appears in more than one partition (train ∩ val ∩ test = ∅)."""
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
    
    leakage = meta["leakage_checks"]
    assert leakage["train_val_household_overlap"] == 0, "Train-Val household leakage detected!"
    assert leakage["train_test_household_overlap"] == 0, "Train-Test household leakage detected!"
    assert leakage["val_test_household_overlap"] == 0, "Val-Test household leakage detected!"
    assert leakage["all_leakage_checks_passed"] is True


def test_household_and_record_partition_completeness():
    """Asserts that train + validation + test equals the exact total count without loss."""
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    records = meta["partition_sizes"]["records"]
    assert records["train"] + records["validation"] + records["test"] == records["total"]
    assert records["total"] == 221263

    households = meta["partition_sizes"]["households"]
    assert households["train"] + households["validation"] + households["test"] == households["total"]
    assert households["total"] == 164339


def test_approximate_split_proportions():
    """Asserts household and record split proportions adhere to 70% / 15% / 15% within ±1% tolerance."""
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    records = meta["partition_sizes"]["records"]
    assert 69.0 <= records["train_pct"] <= 71.0, f"Unexpected train record %: {records['train_pct']}"
    assert 14.0 <= records["validation_pct"] <= 16.0, f"Unexpected val record %: {records['validation_pct']}"
    assert 14.0 <= records["test_pct"] <= 16.0, f"Unexpected test record %: {records['test_pct']}"

    households = meta["partition_sizes"]["households"]
    assert 69.5 <= households["train_pct"] <= 70.5, f"Unexpected train hh %: {households['train_pct']}"
    assert 14.5 <= households["validation_pct"] <= 15.5, f"Unexpected val hh %: {households['validation_pct']}"
    assert 14.5 <= households["test_pct"] <= 15.5, f"Unexpected test hh %: {households['test_pct']}"


def test_all_three_targets_have_valid_observations_in_each_partition():
    """Asserts that Stunting, Underweight, and Wasting each have valid positive and negative cases in all partitions."""
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    target_stats = meta["target_specific_distributions"]
    for target in ["stunting", "underweight", "wasting"]:
        assert target in target_stats, f"Missing target: {target}"
        for part in ["train", "validation", "test"]:
            p_data = target_stats[target][part]
            assert p_data["valid_count"] > 25000, f"{target} in {part} has insufficient valid rows: {p_data['valid_count']}"
            assert p_data["positive_count"] > 5000, f"{target} in {part} has insufficient positive cases: {p_data['positive_count']}"
            assert p_data["negative_count"] > 15000, f"{target} in {part} has insufficient negative cases: {p_data['negative_count']}"
            assert p_data["prevalence_pct"] > 10.0, f"{target} in {part} has abnormal prevalence: {p_data['prevalence_pct']}"


def test_target_prevalence_consistency_across_splits():
    """Asserts that prevalence in train, val, and test is reasonably similar (within ±1.0% of cohort prevalence)."""
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    target_stats = meta["target_specific_distributions"]
    for target in ["stunting", "underweight", "wasting"]:
        cohort_prev = target_stats[target]["overall_cohort"]["prevalence_pct"]
        for part in ["train", "validation", "test"]:
            part_prev = target_stats[target][part]["prevalence_pct"]
            diff = abs(cohort_prev - part_prev)
            assert diff <= 1.0, f"{target} prevalence difference in {part} ({diff}%) exceeds 1.0% tolerance"


def test_no_target_or_grouping_identifiers_in_features():
    """Asserts that no target column or grouping/cluster identifier appears in the 34 approved candidate features."""
    with open(FEATURE_METADATA_PATH, "r") as f:
        f_meta = json.load(f)
    approved_feats = set(f_meta["feature_statistics"].keys())
    assert len(approved_feats) == 34, f"Expected 34 candidate features, found {len(approved_feats)}"

    prohibited = {
        "v001", "v002", "v003", "v005", "hh_id", "mother_id", "split",
        "stunting", "underweight", "wasting", "stunting_severe", "underweight_severe", "wasting_severe",
        "eligible_stunting", "eligible_underweight", "eligible_wasting", "eligible_complete_unified",
        "hw70", "hw71", "hw72", "hw73", "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
        "hw2", "hw3", "hw2_clean", "hw3_clean"
    }
    for var in prohibited:
        assert var not in approved_feats, f"CRITICAL LEAKAGE: '{var}' found in candidate feature matrix!"


def test_state_distribution_metadata_complete():
    """Asserts that all 36 Indian states/UTs are represented with max spread across splits < 1.0%."""
    with open(SPLIT_METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    state_dists = meta["state_distributions"]
    assert len(state_dists) >= 36, f"Expected at least 36 states/UTs, found {len(state_dists)}"
    for s_code, s_data in state_dists.items():
        assert "train_pct" in s_data
        assert "val_pct" in s_data
        assert "test_pct" in s_data
        assert "max_spread_pct" in s_data
        assert s_data["max_spread_pct"] < 1.0, f"State {s_code} has excessive variation across splits: {s_data['max_spread_pct']}%"


def test_reproducibility_verification():
    """Asserts that split manifest matches the partition sizes recorded in split_metadata.json."""
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    assert len(manifest) == 221263
    counts = manifest["split"].value_counts().to_dict()
    assert counts["train"] == 155041
    assert counts["val"] == 33153
    assert counts["test"] == 33069


if __name__ == "__main__":
    print("Running Step 8 tests...")
    test_split_files_exist_and_metadata_valid()
    print("[PASS] test_split_files_exist_and_metadata_valid")
    test_raw_dataset_immutability()
    print("[PASS] test_raw_dataset_immutability")
    test_zero_household_leakage()
    print("[PASS] test_zero_household_leakage")
    test_household_and_record_partition_completeness()
    print("[PASS] test_household_and_record_partition_completeness")
    test_approximate_split_proportions()
    print("[PASS] test_approximate_split_proportions")
    test_all_three_targets_have_valid_observations_in_each_partition()
    print("[PASS] test_all_three_targets_have_valid_observations_in_each_partition")
    test_target_prevalence_consistency_across_splits()
    print("[PASS] test_target_prevalence_consistency_across_splits")
    test_no_target_or_grouping_identifiers_in_features()
    print("[PASS] test_no_target_or_grouping_identifiers_in_features")
    test_state_distribution_metadata_complete()
    print("[PASS] test_state_distribution_metadata_complete")
    test_reproducibility_verification()
    print("[PASS] test_reproducibility_verification")
    print("\nAll Step 8 verification tests PASSED successfully!")
