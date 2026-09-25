"""Target Creation Module for NutriSense AI.

Constructs ground-truth clinical targets (stunting, underweight, wasting, and severe forms)
based on WHO Child Growth Standards from cleaned continuous z-scores.
"""
import os
import json
import numpy as np
import pandas as pd

INPUT_CLEANED_PATH = "data/interim/cleaned_u5_data.csv.gz"
OUTPUT_TARGETS_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
TARGET_STATS_PATH = "data/interim/target_statistics.json"

def create_malnutrition_targets():
    print(f"[*] Step 5: Loading cleaned intermediate data from {INPUT_CLEANED_PATH}...")
    df = pd.read_csv(INPUT_CLEANED_PATH)
    total_records = len(df)
    print(f"    Loaded {total_records:,} records.")

    # 1. Primary Binary Malnutrition Targets (< -2.00 SD)
    # Stunting (HAZ < -2.0)
    df["stunting"] = np.where(
        df["hw70_clean"].isna(), np.nan,
        np.where(df["hw70_clean"] < -2.0, 1.0, 0.0)
    )
    # Underweight (WAZ < -2.0)
    df["underweight"] = np.where(
        df["hw71_clean"].isna(), np.nan,
        np.where(df["hw71_clean"] < -2.0, 1.0, 0.0)
    )
    # Wasting (WHZ < -2.0)
    df["wasting"] = np.where(
        df["hw72_clean"].isna(), np.nan,
        np.where(df["hw72_clean"] < -2.0, 1.0, 0.0)
    )

    # 2. Severe Malnutrition Targets (< -3.00 SD)
    # Severe Stunting (HAZ < -3.0)
    df["stunting_severe"] = np.where(
        df["hw70_clean"].isna(), np.nan,
        np.where(df["hw70_clean"] < -3.0, 1.0, 0.0)
    )
    # Severe Underweight (WAZ < -3.0)
    df["underweight_severe"] = np.where(
        df["hw71_clean"].isna(), np.nan,
        np.where(df["hw71_clean"] < -3.0, 1.0, 0.0)
    )
    # Severe Wasting (WHZ < -3.0)
    df["wasting_severe"] = np.where(
        df["hw72_clean"].isna(), np.nan,
        np.where(df["hw72_clean"] < -3.0, 1.0, 0.0)
    )

    # 3. Compute Aggregate Statistics for Target-Specific Cohorts
    def calc_stats(target_col, severe_col):
        s = df[target_col].dropna()
        sev = df[severe_col].dropna()
        valid_cnt = len(s)
        missing_cnt = int(df[target_col].isna().sum())
        pos_cnt = int((s == 1.0).sum())
        neg_cnt = int((s == 0.0).sum())
        pos_pct = round((pos_cnt / valid_cnt) * 100, 2)
        neg_pct = round((neg_cnt / valid_cnt) * 100, 2)
        imbalance_ratio = round(neg_cnt / pos_cnt, 2) if pos_cnt > 0 else None
        
        sev_cnt = int((sev == 1.0).sum())
        sev_pct = round((sev_cnt / len(sev)) * 100, 2)
        sev_imbalance = round((len(sev) - sev_cnt) / sev_cnt, 2) if sev_cnt > 0 else None
        
        return {
            "valid_count": valid_cnt,
            "missing_count": missing_cnt,
            "positive_count": pos_cnt,
            "negative_count": neg_cnt,
            "prevalence_pct": pos_pct,
            "negative_pct": neg_pct,
            "imbalance_ratio": imbalance_ratio,
            "severe_positive_count": sev_cnt,
            "severe_prevalence_pct": sev_pct,
            "severe_imbalance_ratio": sev_imbalance
        }

    stunting_stats = calc_stats("stunting", "stunting_severe")
    underweight_stats = calc_stats("underweight", "underweight_severe")
    wasting_stats = calc_stats("wasting", "wasting_severe")

    # 4. Compute Target Overlap in Unified Complete Cohort (N = 198,802)
    u_mask = df["eligible_complete_unified"] == True
    df_u = df[u_mask].copy()
    
    s = df_u["stunting"].astype(int)
    u = df_u["underweight"].astype(int)
    w = df_u["wasting"].astype(int)
    
    n_unified = len(df_u)
    overlap_stats = {
        "unified_cohort_total": n_unified,
        "none_healthy": int(((s == 0) & (u == 0) & (w == 0)).sum()),
        "stunting_only": int(((s == 1) & (u == 0) & (w == 0)).sum()),
        "underweight_only": int(((s == 0) & (u == 1) & (w == 0)).sum()),
        "wasting_only": int(((s == 0) & (u == 0) & (w == 1)).sum()),
        "stunting_and_underweight_only": int(((s == 1) & (u == 1) & (w == 0)).sum()),
        "stunting_and_wasting_only": int(((s == 1) & (u == 0) & (w == 1)).sum()),
        "underweight_and_wasting_only": int(((s == 0) & (u == 1) & (w == 1)).sum()),
        "all_three": int(((s == 1) & (u == 1) & (w == 1)).sum()),
    }
    
    # Add percentages to overlap stats
    overlap_percentages = {
        k: round((v / n_unified) * 100, 2) for k, v in overlap_stats.items() if k != "unified_cohort_total"
    }

    # 5. Boundary Condition Verification Check
    # Verify that exactly -2.00 is NOT stunted/underweight/wasted
    boundary_audit = {
        "stunting_exact_neg2_is_zero": bool((df.loc[df["hw70_clean"] == -2.0, "stunting"] == 0.0).all()),
        "stunting_exact_neg3_is_zero_in_severe": bool((df.loc[df["hw70_clean"] == -3.0, "stunting_severe"] == 0.0).all()),
        "underweight_exact_neg2_is_zero": bool((df.loc[df["hw71_clean"] == -2.0, "underweight"] == 0.0).all()),
        "wasting_exact_neg2_is_zero": bool((df.loc[df["hw72_clean"] == -2.0, "wasting"] == 0.0).all()),
    }

    # 6. Save Processed Dataset and Statistics
    print(f"[*] Saving target-augmented dataset to {OUTPUT_TARGETS_PATH}...")
    df.to_csv(OUTPUT_TARGETS_PATH, index=False, compression="gzip")
    print(f"    Saved {OUTPUT_TARGETS_PATH} ({os.path.getsize(OUTPUT_TARGETS_PATH):,} bytes).")

    all_target_data = {
        "total_records": total_records,
        "stunting_stats": stunting_stats,
        "underweight_stats": underweight_stats,
        "wasting_stats": wasting_stats,
        "overlap_counts": overlap_stats,
        "overlap_percentages": overlap_percentages,
        "boundary_audit": boundary_audit
    }

    with open(TARGET_STATS_PATH, "w") as f:
        json.dump(all_target_data, f, indent=2)
    print(f"[+] Saved target statistics to {TARGET_STATS_PATH}")
    print("[OK] Target creation completed successfully.")
    return all_target_data

if __name__ == "__main__":
    create_malnutrition_targets()
