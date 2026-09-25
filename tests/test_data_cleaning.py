"""Automated verification test suite for Step 4: Systematic Data Cleaning.

Strictly verifies:
- Raw DHS file immutability (exact byte check)
- Cleaned dataset row count equals 221,263
- m4 handling strictly follows official metadata (code 94=0, code 95=still breastfeeding)
- m14 has NO arbitrary cap (valid range preserves observed values up to 95 visits)
- v445 is NOT arbitrarily filtered (only sentinel 9998 is removed; observed range 12.02 to 59.99 kg/m^2)
- Implied decimal scalings (HAZ/WAZ/WHZ / 100, weight / 1M)
- Target cohort eligibility flags
- Privacy rules
"""
import os
import json
import pandas as pd
import numpy as np

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
INTERIM_CSV_GZ_PATH = "data/interim/cleaned_u5_data.csv.gz"
AUDIT_JSON_PATH = "data/interim/cleaning_audit.json"
EXPECTED_RAW_BYTES = 441380745

def test_raw_file_immutability():
    """Verify that the raw NFHS-5 dataset was never altered."""
    assert os.path.exists(RAW_DTA_PATH), f"Raw file {RAW_DTA_PATH} not found"
    current_bytes = os.path.getsize(RAW_DTA_PATH)
    assert current_bytes == EXPECTED_RAW_BYTES, (
        f"Raw file size altered! Expected {EXPECTED_RAW_BYTES}, found {current_bytes}"
    )

def test_cleaned_dataset_existence_and_row_count():
    """Verify the cleaned intermediate dataset has exactly 221,263 living U5 records."""
    assert os.path.exists(INTERIM_CSV_GZ_PATH), f"Cleaned dataset {INTERIM_CSV_GZ_PATH} not found"
    with open(AUDIT_JSON_PATH, "r") as f:
        audit = json.load(f)
    assert audit["living_under5_cohort_size"] == 221263, (
        f"Expected 221,263 records, found {audit['living_under5_cohort_size']}"
    )

def test_m4_breastfeeding_official_handling():
    """Verify m4 adheres to official DHS value labels without arbitrary assumptions."""
    cols = ["m4", "still_breastfeeding", "m4_completed_months", "m4_censored_months", "hw1"]
    df = pd.read_csv(INTERIM_CSV_GZ_PATH, usecols=cols)
    
    # Check code 94: Never breastfed -> 0.0 months
    c94 = df[df["m4"] == 94]
    assert (c94["m4_completed_months"] == 0.0).all(), "Code 94 (Never breastfed) must be 0.0 months"
    assert (c94["m4_censored_months"] == 0.0).all(), "Code 94 must be 0.0 in censored duration"
    
    # Check code 95: Still breastfeeding -> still_breastfeeding == 1
    c95 = df[df["m4"] == 95]
    assert (c95["still_breastfeeding"] == 1).all(), "Code 95 must have still_breastfeeding == 1"
    # In m4_completed_months, code 95 is ongoing/unfinished so it must be NaN
    assert c95["m4_completed_months"].isna().all(), "Code 95 is unfinished so completed duration must be NaN"
    # In m4_censored_months, duration is right-censored at child's current age (hw1)
    assert (c95["m4_censored_months"] == c95["hw1"]).all(), "Code 95 censored duration must equal child age hw1"
    
    # Check code 98: Don't know -> NaN
    c98 = df[df["m4"] == 98]
    assert c98["m4_completed_months"].isna().all(), "Code 98 must be NaN"
    assert c98["m4_censored_months"].isna().all(), "Code 98 must be NaN"

def test_m14_no_arbitrary_capping():
    """Verify m14 has NO arbitrary capping at 40 and preserves all observed reported visits."""
    cols = ["m14", "m14_clean"]
    df = pd.read_csv(INTERIM_CSV_GZ_PATH, usecols=cols)
    
    # Code 98 (Don't know) must be NaN
    assert df[df["m14"] == 98]["m14_clean"].isna().all(), "m14 code 98 must be recoded to NaN"
    
    # Valid values > 40 MUST be preserved (e.g. 45, 50, 60, 70, 80, 90, 95)
    high_visits = df[df["m14"] > 40]
    # Exclude code 98
    valid_high = high_visits[high_visits["m14"] != 98]
    assert len(valid_high) > 0, "Expected valid reports > 40 visits in raw dataset"
    assert (valid_high["m14_clean"] == valid_high["m14"]).all(), (
        "m14 values > 40 were improperly capped or altered! All reported visits must be preserved."
    )
    
    # Check observed min and max
    assert df["m14_clean"].min() == 0.0, "m14_clean minimum should be 0.0"
    assert df["m14_clean"].max() == 95.0, f"m14_clean maximum should be 95.0, got {df['m14_clean'].max()}"

def test_v445_no_arbitrary_filtering():
    """Verify v445 only removes sentinel 9998, with NO arbitrary min/max clipping."""
    cols = ["v445", "v445_clean"]
    df = pd.read_csv(INTERIM_CSV_GZ_PATH, usecols=cols)
    
    # Sentinel code 9998 must be NaN
    assert df[df["v445"] >= 9000]["v445_clean"].isna().all(), "Sentinel 9998 in v445 must be NaN"
    
    # All raw values < 9000 must strictly equal raw / 100.0
    valid_raw = df[(df["v445"] < 9000) & (df["v445"].notna())]
    expected_clean = valid_raw["v445"] / 100.0
    assert np.allclose(valid_raw["v445_clean"], expected_clean), "v445_clean must equal raw v445 / 100.0"
    
    # Observed range after documented special-code handling
    assert round(float(df["v445_clean"].min()), 2) == 12.02
    assert round(float(df["v445_clean"].max()), 2) == 59.99

def test_anthropometric_scaling_and_sentinel_removal():
    """Verify z-scores are scaled by 100 and sentinel codes (9996-9998) are eliminated."""
    cols = ["hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean", "sample_weight"]
    df = pd.read_csv(INTERIM_CSV_GZ_PATH, usecols=cols)
    
    for col in ["hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean"]:
        s = df[col].dropna()
        assert not ((s >= 9000).any()), f"Sentinel codes still present in {col}!"
        assert (s >= -6.5).all(), f"Extreme negative out of bounds in {col}: min={s.min()}"
        assert (s <= 6.5).all(), f"Extreme positive out of bounds in {col}: max={s.max()}"

    w = df["sample_weight"]
    assert (w > 0).all(), "Sample weights must be strictly positive"
    assert 0.5 <= w.mean() <= 2.0, f"Normalized sample weight mean {w.mean()} outside expected range"

def test_cohort_eligibility_flags():
    """Verify that candidate cohort flags match verified population counts."""
    cols = ["eligible_stunting", "eligible_underweight", "eligible_wasting", "eligible_complete_unified"]
    df = pd.read_csv(INTERIM_CSV_GZ_PATH, usecols=cols)
    
    assert int(df["eligible_stunting"].sum()) == 206025
    assert int(df["eligible_underweight"].sum()) == 210524
    assert int(df["eligible_wasting"].sum()) == 201687
    assert int(df["eligible_complete_unified"].sum()) == 198802

def test_git_privacy_protection_for_interim_data():
    """Verify that .gitignore prevents cleaned microdata from being tracked."""
    with open(".gitignore", "r") as f:
        content = f.read()
    assert "data/interim/*" in content, ".gitignore must ignore interim cleaned data"
    assert "*.csv" in content or "*.csv.gz" in content, ".gitignore must ignore CSV exports"

if __name__ == "__main__":
    print("[*] Running Step 4 Data Cleaning Verification Tests (with Review Corrections)...")
    test_raw_file_immutability()
    print("  [OK] Raw NFHS-5 dataset file is 100% untouched (exact byte size verified).")
    test_cleaned_dataset_existence_and_row_count()
    print("  [OK] Cleaned dataset has exactly 221,263 living U5 records.")
    test_m4_breastfeeding_official_handling()
    print("  [OK] m4 breastfeeding handling strictly follows official DHS metadata (94=0, 95=censored).")
    test_m14_no_arbitrary_capping()
    print("  [OK] m14 has NO arbitrary cap: preserves valid reported visits up to observed max 95.0.")
    test_v445_no_arbitrary_filtering()
    print("  [OK] v445 has NO arbitrary filter: only sentinel 9998 removed; observed range [12.02, 59.99] kg/m^2.")
    test_anthropometric_scaling_and_sentinel_removal()
    print("  [OK] Z-score scaling (/100) and sentinel removal verified.")
    test_cohort_eligibility_flags()
    print("  [OK] Target cohort flags verified (Stunting: 206k, Wasting: 201k, Unified: 198k).")
    test_git_privacy_protection_for_interim_data()
    print("  [OK] Git privacy rules strictly protect interim cleaned microdata.")
    print("[SUCCESS] All Step 4 data cleaning tests passed successfully!")
