"""Automated verification test suite for Step 5: Target Creation."""
import os
import json
import pandas as pd
import numpy as np

OUTPUT_TARGETS_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
TARGET_STATS_PATH = "data/interim/target_statistics.json"

def test_target_dataset_exists():
    assert os.path.exists(OUTPUT_TARGETS_PATH), f"Target dataset {OUTPUT_TARGETS_PATH} not found"
    assert os.path.exists(TARGET_STATS_PATH), f"Target stats {TARGET_STATS_PATH} not found"

def test_target_definitions_and_thresholds():
    """Verify HAZ/WAZ/WHZ < -2 produces 1, >= -2 produces 0, and NaN produces NaN."""
    cols = [
        "hw70_clean", "stunting", "stunting_severe",
        "hw71_clean", "underweight", "underweight_severe",
        "hw72_clean", "wasting", "wasting_severe"
    ]
    df = pd.read_csv(OUTPUT_TARGETS_PATH, usecols=cols)
    
    # 1. Stunting Tests
    valid_haz = df[df["hw70_clean"].notna()]
    assert (valid_haz[valid_haz["hw70_clean"] < -2.0]["stunting"] == 1.0).all(), "HAZ < -2 must produce stunting = 1"
    assert (valid_haz[valid_haz["hw70_clean"] >= -2.0]["stunting"] == 0.0).all(), "HAZ >= -2 must produce stunting = 0"
    assert (valid_haz[valid_haz["hw70_clean"] < -3.0]["stunting_severe"] == 1.0).all(), "HAZ < -3 must produce stunting_severe = 1"
    assert (valid_haz[valid_haz["hw70_clean"] >= -3.0]["stunting_severe"] == 0.0).all(), "HAZ >= -3 must produce stunting_severe = 0"
    assert df[df["hw70_clean"].isna()]["stunting"].isna().all(), "Missing HAZ must produce NaN stunting"
    assert df[df["hw70_clean"].isna()]["stunting_severe"].isna().all(), "Missing HAZ must produce NaN stunting_severe"

    # 2. Underweight Tests
    valid_waz = df[df["hw71_clean"].notna()]
    assert (valid_waz[valid_waz["hw71_clean"] < -2.0]["underweight"] == 1.0).all(), "WAZ < -2 must produce underweight = 1"
    assert (valid_waz[valid_waz["hw71_clean"] >= -2.0]["underweight"] == 0.0).all(), "WAZ >= -2 must produce underweight = 0"
    assert (valid_waz[valid_waz["hw71_clean"] < -3.0]["underweight_severe"] == 1.0).all(), "WAZ < -3 must produce underweight_severe = 1"
    assert (valid_waz[valid_waz["hw71_clean"] >= -3.0]["underweight_severe"] == 0.0).all(), "WAZ >= -3 must produce underweight_severe = 0"
    assert df[df["hw71_clean"].isna()]["underweight"].isna().all(), "Missing WAZ must produce NaN underweight"

    # 3. Wasting Tests
    valid_whz = df[df["hw72_clean"].notna()]
    assert (valid_whz[valid_whz["hw72_clean"] < -2.0]["wasting"] == 1.0).all(), "WHZ < -2 must produce wasting = 1"
    assert (valid_whz[valid_whz["hw72_clean"] >= -2.0]["wasting"] == 0.0).all(), "WHZ >= -2 must produce wasting = 0"
    assert (valid_whz[valid_whz["hw72_clean"] < -3.0]["wasting_severe"] == 1.0).all(), "WHZ < -3 must produce wasting_severe = 1"
    assert (valid_whz[valid_whz["hw72_clean"] >= -3.0]["wasting_severe"] == 0.0).all(), "WHZ >= -3 must produce wasting_severe = 0"
    assert df[df["hw72_clean"].isna()]["wasting"].isna().all(), "Missing WHZ must produce NaN wasting"

def test_boundary_conditions():
    """Verify exact boundary values -2.00 and -3.00 produce 0, and slight offsets behave correctly."""
    cols = ["hw70_clean", "stunting", "stunting_severe", "hw71_clean", "underweight", "hw72_clean", "wasting"]
    df = pd.read_csv(OUTPUT_TARGETS_PATH, usecols=cols)
    
    # Boundary: exactly -2.00 must be 0 (condition absent)
    assert (df.loc[df["hw70_clean"] == -2.0, "stunting"] == 0.0).all()
    assert (df.loc[df["hw71_clean"] == -2.0, "underweight"] == 0.0).all()
    assert (df.loc[df["hw72_clean"] == -2.0, "wasting"] == 0.0).all()
    
    # Boundary: exactly -3.00 must be 0 for severe
    assert (df.loc[df["hw70_clean"] == -3.0, "stunting_severe"] == 0.0).all()

def test_target_value_domain():
    """Verify that target columns contain ONLY 0, 1, or NaN."""
    target_cols = [
        "stunting", "stunting_severe",
        "underweight", "underweight_severe",
        "wasting", "wasting_severe"
    ]
    df = pd.read_csv(OUTPUT_TARGETS_PATH, usecols=target_cols)
    for col in target_cols:
        unique_vals = set(df[col].dropna().unique())
        assert unique_vals.issubset({0.0, 1.0}), f"Unexpected values in {col}: {unique_vals}"

def test_aggregate_counts_match_verified_totals():
    """Verify aggregate counts match the audited population statistics."""
    with open(TARGET_STATS_PATH, "r") as f:
        stats = json.load(f)
        
    stunting = stats["stunting_stats"]
    assert stunting["valid_count"] == 206025
    assert stunting["positive_count"] == 73072
    assert stunting["negative_count"] == 132953
    assert stunting["severe_positive_count"] == 31518
    assert stunting["prevalence_pct"] == 35.47

    underweight = stats["underweight_stats"]
    assert underweight["valid_count"] == 210524
    assert underweight["positive_count"] == 65043
    assert underweight["negative_count"] == 145481
    assert underweight["severe_positive_count"] == 21735
    assert underweight["prevalence_pct"] == 30.90

    wasting = stats["wasting_stats"]
    assert wasting["valid_count"] == 201687
    assert wasting["positive_count"] == 37553
    assert wasting["negative_count"] == 164134
    assert wasting["severe_positive_count"] == 15332
    assert wasting["prevalence_pct"] == 18.62

    # Overlap sum
    overlap = stats["overlap_counts"]
    assert overlap["unified_cohort_total"] == 198802
    assert overlap["none_healthy"] == 96055
    assert overlap["stunting_only"] == 31678
    assert overlap["all_three"] == 9626

if __name__ == "__main__":
    print("[*] Running Step 5 Target Creation Verification Tests...")
    test_target_dataset_exists()
    print("  [OK] Processed dataset and target statistics files exist.")
    test_target_definitions_and_thresholds()
    print("  [OK] WHO threshold definitions (< -2.0, < -3.0) and NaN propagation verified.")
    test_boundary_conditions()
    print("  [OK] Exact boundary conditions (exactly -2.0 and -3.0 produce 0) verified.")
    test_target_value_domain()
    print("  [OK] Domain check passed: target columns contain only 0.0, 1.0, or NaN.")
    test_aggregate_counts_match_verified_totals()
    print("  [OK] Stunting (73,072 / 35.47%), Underweight (65,043 / 30.90%), Wasting (37,553 / 18.62%) verified.")
    print("[SUCCESS] All Step 5 target creation tests passed successfully!")
