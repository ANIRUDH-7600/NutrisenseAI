"""Empirical Population Definition & Eligibility Evaluator for NutriSense AI."""
import json
import os
import pandas as pd
import numpy as np

DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
SUMMARY_OUTPUT = "data/interim/population_counts.json"

def evaluate_research_population():
    cols = ["b5", "hw1", "b8", "hw13", "hw70", "hw71", "hw72", "hw73", "hw2", "hw3"]
    print(f"Loading {len(cols)} columns from {DTA_PATH} to calculate exact eligibility counts...")
    df = pd.read_stata(DTA_PATH, columns=cols, convert_categoricals=False)
    
    total_raw = len(df)
    print(f"Initial raw dataset records: {total_raw:,}")
    
    # 1. Alive Status (b5)
    alive_mask = (df["b5"] == 1)
    deceased_count = int((df["b5"] == 0).sum())
    alive_count = int(alive_mask.sum())
    print(f"\n1. Alive status (b5):")
    print(f"   Alive (b5 == 1): {alive_count:,}")
    print(f"   Deceased (b5 == 0): {deceased_count:,}")
    
    # 2. Age criteria among alive children
    df_alive = df[alive_mask]
    # Check hw1 (age in months)
    hw1_valid = df_alive["hw1"].notna()
    hw1_0_59 = (df_alive["hw1"] >= 0) & (df_alive["hw1"] <= 59)
    hw1_missing = df_alive["hw1"].isna().sum()
    
    print(f"\n2. Age in months (hw1) among alive children:")
    print(f"   Valid hw1 in [0, 59]: {int(hw1_0_59.sum()):,}")
    print(f"   Missing hw1: {int(hw1_missing):,}")
    print(f"   Min hw1: {df_alive['hw1'].min()}, Max hw1: {df_alive['hw1'].max()}")
    
    # Check b8 (current age in years) for those with missing hw1
    missing_hw1_b8 = df_alive[df_alive["hw1"].isna()]["b8"].value_counts(dropna=False)
    print(f"   b8 distribution for missing hw1:\n{missing_hw1_b8}")
    
    # Filter 1: Living children aged 0-59 months
    # Note: Does b8 confirm that all children in KR are under 5?
    # In DHS KR files, the universe is births in last 5 years.
    # If hw1 is valid, hw1 in 0..59 is living under-five.
    living_u5_mask = alive_mask & hw1_0_59
    living_u5_count = int(living_u5_mask.sum())
    print(f"\nLiving children with valid age 0-59 months: {living_u5_count:,}")

    # 3. Anthropometric Measurement Result (hw13) among living under-5
    df_u5 = df[living_u5_mask]
    hw13_counts = df_u5["hw13"].value_counts(dropna=False)
    print(f"\n3. hw13 (measurement result) distribution among living U5:\n{hw13_counts}")
    
    # 4. Anthropometric Z-scores valid definitions (< 9000 and not NaN)
    # hw70: Stunting
    # hw71: Underweight
    # hw72: Wasting
    # hw73: BMI
    
    def get_z_stats(series):
        not_na = series.notna()
        valid = not_na & (series < 9000)
        c_9996 = int((series == 9996).sum())
        c_9997 = int((series == 9997).sum())
        c_9998 = int((series == 9998).sum())
        c_nan = int(series.isna().sum())
        return {
            "valid_count": int(valid.sum()),
            "flag_9996": c_9996,
            "flag_9997": c_9997,
            "flag_9998": c_9998,
            "missing_nan": c_nan
        }

    stunting_stats = get_z_stats(df_u5["hw70"])
    underweight_stats = get_z_stats(df_u5["hw71"])
    wasting_stats = get_z_stats(df_u5["hw72"])
    bmi_stats = get_z_stats(df_u5["hw73"])
    
    print(f"\n4. Outcome Specific Counts among Living U5 ({len(df_u5):,}):")
    print(f"   Stunting (hw70) valid (<9000): {stunting_stats['valid_count']:,} (flags: {stunting_stats['flag_9998']:,} 9998, {stunting_stats['flag_9997']:,} 9997, {stunting_stats['missing_nan']:,} NaN)")
    print(f"   Underweight (hw71) valid (<9000): {underweight_stats['valid_count']:,} (flags: {underweight_stats['flag_9998']:,} 9998, {underweight_stats['flag_9997']:,} 9997, {underweight_stats['missing_nan']:,} NaN)")
    print(f"   Wasting (hw72) valid (<9000): {wasting_stats['valid_count']:,} (flags: {wasting_stats['flag_9996']:,} 9996, {wasting_stats['flag_9998']:,} 9998, {wasting_stats['missing_nan']:,} NaN)")

    # 5. Overlap between valid outcomes
    valid_stunting = df_u5["hw70"].notna() & (df_u5["hw70"] < 9000)
    valid_underweight = df_u5["hw71"].notna() & (df_u5["hw71"] < 9000)
    valid_wasting = df_u5["hw72"].notna() & (df_u5["hw72"] < 9000)
    
    all_three_valid = valid_stunting & valid_underweight & valid_wasting
    at_least_one_valid = valid_stunting | valid_underweight | valid_wasting
    
    print(f"\n5. Outcome Overlap Counts:")
    print(f"   All three valid (Stunting & Wasting & Underweight): {int(all_three_valid.sum()):,}")
    print(f"   At least one valid outcome: {int(at_least_one_valid.sum()):,}")
    print(f"   Valid Stunting only: {int((valid_stunting & ~valid_wasting & ~valid_underweight).sum()):,}")
    print(f"   Valid Wasting only: {int((~valid_stunting & valid_wasting & ~valid_underweight).sum()):,}")
    print(f"   Valid Underweight only: {int((~valid_stunting & ~valid_wasting & valid_underweight).sum()):,}")
    print(f"   Valid Stunting & Underweight: {int((valid_stunting & valid_underweight).sum()):,}")
    print(f"   Valid Stunting & Wasting: {int((valid_stunting & valid_wasting).sum()):,}")
    print(f"   Valid Wasting & Underweight: {int((valid_wasting & valid_underweight).sum()):,}")

    # Build summary json
    summary = {
        "raw_total_records": total_raw,
        "alive_children": alive_count,
        "deceased_children": deceased_count,
        "living_under_five_0_59_months": living_u5_count,
        "missing_hw1_among_living": int(hw1_missing),
        "stunting_eligible_hw70_valid": stunting_stats["valid_count"],
        "stunting_hw70_flags_and_missing": {
            "flag_9998": stunting_stats["flag_9998"],
            "flag_9997": stunting_stats["flag_9997"],
            "missing_nan": stunting_stats["missing_nan"]
        },
        "underweight_eligible_hw71_valid": underweight_stats["valid_count"],
        "underweight_hw71_flags_and_missing": {
            "flag_9998": underweight_stats["flag_9998"],
            "flag_9997": underweight_stats["flag_9997"],
            "missing_nan": underweight_stats["missing_nan"]
        },
        "wasting_eligible_hw72_valid": wasting_stats["valid_count"],
        "wasting_hw72_flags_and_missing": {
            "flag_9996": wasting_stats["flag_9996"],
            "flag_9998": wasting_stats["flag_9998"],
            "missing_nan": wasting_stats["missing_nan"]
        },
        "complete_anthropometric_cohort_all_three_valid": int(all_three_valid.sum()),
        "union_at_least_one_valid": int(at_least_one_valid.sum())
    }
    
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT), exist_ok=True)
    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUCCESS] Population counts written to {SUMMARY_OUTPUT}")
    return summary

if __name__ == "__main__":
    evaluate_research_population()
