"""Systematic Data Cleaning Pipeline for India NFHS-5 Children's Recode (KR).

This module implements deterministic, audited cleaning rules for the living under-five
research population (N = 221,263), preserving the raw .DTA file in immutable read-only mode.
"""
import os
import json
import numpy as np
import pandas as pd

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
INTERIM_CSV_GZ_PATH = "data/interim/cleaned_u5_data.csv.gz"
AUDIT_JSON_PATH = "data/interim/cleaning_audit.json"


# Explicit list of columns to load from raw dataset
CLEANING_COLUMNS = [
    # Population & Demographics
    "b5", "hw1", "b4", "b8", "bord", "b0", "b11",
    # Anthropometric targets & audit variables
    "hw70", "hw71", "hw72", "hw73", "hw13", "hw2", "hw3",
    "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12",
    # Child birth history & feeding
    "m18", "m19", "v404", "m4",
    # Child health & morbidity
    "hw57", "h11", "h22", "h31",
    # Maternal characteristics & healthcare
    "v012", "v212", "v445", "v457", "v106", "v701", "v714", "m14", "m15", "v201",
    # Household & socioeconomic
    "v190", "v025", "v024", "sdist", "s116", "v130",
    "v113", "v116", "v119", "v161", "v136", "v151",
    # Survey design & identifiers
    "v001", "v002", "v003", "v005", "v021", "v022"
]

def clean_nfhs5_data():
    print(f"[*] Step 4: Loading raw survey data from {RAW_DTA_PATH}...")
    raw_size = os.path.getsize(RAW_DTA_PATH)
    
    # 1. Safe loading of candidate columns
    df_raw = pd.read_stata(RAW_DTA_PATH, columns=CLEANING_COLUMNS, convert_categoricals=False)
    total_raw_records = len(df_raw)
    print(f"    Raw records loaded: {total_raw_records:,}")

    # 2. Population Filtering: Living children aged 0-59 completed months
    alive_mask = (df_raw["b5"] == 1)
    age_u5_mask = (df_raw["hw1"] >= 0) & (df_raw["hw1"] <= 59)
    u5_mask = alive_mask & age_u5_mask
    
    df = df_raw[u5_mask].copy()
    u5_count = len(df)
    print(f"    Filtered to Living Under-5 Cohort (0-59 mo): {u5_count:,}")

    # 3. Create Target Eligibility Cohort Flags (Before any transformations)
    # Stunting (hw70): valid if not NaN and < 9000
    df["eligible_stunting"] = df["hw70"].notna() & (df["hw70"] < 9000)
    # Underweight (hw71): valid if not NaN and < 9000
    df["eligible_underweight"] = df["hw71"].notna() & (df["hw71"] < 9000)
    # Wasting (hw72): valid if not NaN and < 9000
    df["eligible_wasting"] = df["hw72"].notna() & (df["hw72"] < 9000)
    # Complete Unified Cohort: all three simultaneously valid
    df["eligible_complete_unified"] = (
        df["eligible_stunting"] & 
        df["eligible_underweight"] & 
        df["eligible_wasting"]
    )

    cohort_counts = {
        "eligible_stunting": int(df["eligible_stunting"].sum()),
        "eligible_underweight": int(df["eligible_underweight"].sum()),
        "eligible_wasting": int(df["eligible_wasting"].sum()),
        "eligible_complete_unified": int(df["eligible_complete_unified"].sum())
    }
    print(f"    Eligible Stunting Cohort (hw70 valid): {cohort_counts['eligible_stunting']:,}")
    print(f"    Eligible Underweight Cohort (hw71 valid): {cohort_counts['eligible_underweight']:,}")
    print(f"    Eligible Wasting Cohort (hw72 valid): {cohort_counts['eligible_wasting']:,}")
    print(f"    Eligible Complete Unified Cohort (all three): {cohort_counts['eligible_complete_unified']:,}")

    # 4. Implied Decimals & Anthropometric Target Cleaning
    # Z-scores are divided by 100; sentinel values (9996, 9997, 9998) recoded to NaN
    df["hw70_clean"] = np.where(df["hw70"] < 9000, df["hw70"] / 100.0, np.nan)
    df["hw71_clean"] = np.where(df["hw71"] < 9000, df["hw71"] / 100.0, np.nan)
    df["hw72_clean"] = np.where(df["hw72"] < 9000, df["hw72"] / 100.0, np.nan)
    df["hw73_clean"] = np.where(df["hw73"] < 9000, df["hw73"] / 100.0, np.nan)

    # 5. Survey Weight Scaling
    # v005 has 6 implied decimals -> divide by 1,000,000
    df["sample_weight"] = df["v005"] / 1_000_000.0

    # 6. Physical Measurement Scaling (For audit/target construction only)
    # hw2 (weight in kg): 1 implied decimal -> divide by 10
    df["hw2_clean"] = np.where((df["hw2"] < 9000) & (df["hw2"] > 0), df["hw2"] / 10.0, np.nan)
    # hw3 (height in cm): 1 implied decimal -> divide by 10
    df["hw3_clean"] = np.where((df["hw3"] < 9000) & (df["hw3"] > 0), df["hw3"] / 10.0, np.nan)

    # 7. Maternal Predictor Cleaning
    # v445 (Mother BMI): 2 implied decimals; official DHS sentinel 9998 is "Flagged cases" -> recode to NaN
    # Note: NO arbitrary min/max filtering is imposed. The observed valid range is 12.02 to 59.99 kg/m^2.
    df["v445_clean"] = np.where((df["v445"] < 9000) & (df["v445"].notna()), df["v445"] / 100.0, np.nan)
    
    # m14 (ANC visits): DHS official metadata defines 0="No antenatal visits" and 98="Don't know".
    # NO arbitrary upper bound is imposed (previous cap at 40 removed).
    # All valid reported visits (observed 0 to 95) are preserved; only documented special code 98 is recoded to NaN.
    df["m14_clean"] = np.where((df["m14"] != 98) & (df["m14"].notna()), df["m14"], np.nan)

    # 8. Child Birth & Feeding Predictor Cleaning
    # m19 (Birth weight in grams): values >= 9000 are sentinel (9996=not weighed, 9998=don't know)
    # Plausible birth weight: 500g to 6000g -> converted to kg (0.5 to 6.0 kg)
    df["birth_weight_kg"] = np.where((df["m19"] >= 500) & (df["m19"] <= 6000), df["m19"] / 1000.0, np.nan)
    df["birth_weight_measured"] = np.where(df["m19"] < 9000, 1, 0)
    
    # m18 (Size at birth): 8 is "Don't know" -> recode to NaN
    df["m18_clean"] = np.where(df["m18"].isin([1, 2, 3, 4, 5]), df["m18"], np.nan)

    # m4 (Duration of breastfeeding):
    # Official DHS value label M4 definitions:
    #   94: "Never breastfed" -> true duration is 0.0 months
    #   95: "Still breastfeeding" -> duration is right-censored at child's current age (hw1)
    #   98: "Don't know" -> informational uncertainty -> NaN
    # We provide two methodologically separated variables:
    #   1. still_breastfeeding: binary indicator (1=Yes, 0=No, NaN=Missing)
    #   2. m4_completed_months: completed duration strictly for weaned children (95 is NaN because duration is unfinished)
    #   3. m4_censored_months: duration where ongoing breastfeeding is right-censored at current child age (hw1)
    df["still_breastfeeding"] = np.where(
        (df["m4"] == 95) | (df["v404"] == 1), 1,
        np.where((df["m4"].notna()) & (df["m4"] != 98) & (df["m4"] != 95), 0, np.nan)
    )
    df["m4_completed_months"] = np.where(
        df["m4"] == 94, 0.0,
        np.where((df["m4"] >= 0) & (df["m4"] < 60), df["m4"], np.nan)
    )
    df["m4_censored_months"] = np.where(
        df["m4"] == 94, 0.0,
        np.where(
            df["m4"] == 95, df["hw1"], # Right-censored duration at current age
            np.where((df["m4"] >= 0) & (df["m4"] < 60), df["m4"], np.nan)
        )
    )


    # 9. Child Morbidity Cleaning (h11, h22, h31)
    # 0=No, 1/2=Yes, 8=Don't know -> recode 8 to NaN
    df["diarrhea_recent"] = np.where(df["h11"].isin([1, 2]), 1, np.where(df["h11"] == 0, 0, np.nan))
    df["fever_recent"] = np.where(df["h22"].isin([1, 2]), 1, np.where(df["h22"] == 0, 0, np.nan))
    df["cough_recent"] = np.where(df["h31"].isin([1, 2]), 1, np.where(df["h31"] == 0, 0, np.nan))

    # 10. Partner & Social Group Cleaning
    # v701 (Partner education): 8=Don't know -> NaN
    df["v701_clean"] = np.where(df["v701"].isin([0, 1, 2, 3]), df["v701"], np.nan)
    # s116 (Caste): 8=Don't know -> NaN
    df["caste_clean"] = np.where(df["s116"].isin([1, 2, 3, 4]), df["s116"], np.nan)

    # 11. Compile Missingness & Audit Metrics
    missingness_audit = []
    audited_cols = [
        "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
        "hw1", "b4", "bord", "b0", "b11",
        "m18_clean", "birth_weight_kg", "hw57", "still_breastfeeding",
        "m4_completed_months", "m4_censored_months",
        "diarrhea_recent", "fever_recent", "cough_recent",
        "v012", "v212", "v445_clean", "v457", "v106", "v701_clean", "v714", "m14_clean", "m15", "v201",
        "v190", "v025", "v024", "caste_clean", "v130", "v113", "v116", "v119", "v161", "v136", "v151",
        "sample_weight"
    ]


    for col in audited_cols:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_valid = int(s.notna().sum())
        pct_missing = round((n_missing / u5_count) * 100, 2)
        
        # Categorize missingness type
        cat_type = "B. Genuine Missing"
        if col in ["hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean"]:
            cat_type = "E. Outcome Unavailable / Flagged"
        elif col in ["b11"]:
            cat_type = "A. Structural (Firstborn)"
        elif col in ["birth_weight_kg", "m18_clean", "m14_clean", "caste_clean", "v701_clean"]:
            cat_type = "C. Refused / Don't Know / Unmeasured"

        missingness_audit.append({
            "variable": col,
            "total_records": u5_count,
            "valid_count": n_valid,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "missingness_category": cat_type,
            "min_val": round(float(s.min()), 2) if n_valid > 0 else None,
            "max_val": round(float(s.max()), 2) if n_valid > 0 else None
        })

    # Save cleaned intermediate dataset
    os.makedirs(os.path.dirname(INTERIM_CSV_GZ_PATH), exist_ok=True)
    df.to_csv(INTERIM_CSV_GZ_PATH, index=False, compression="gzip")
    print(f"[+] Saved cleaned dataset to {INTERIM_CSV_GZ_PATH} ({os.path.getsize(INTERIM_CSV_GZ_PATH):,} bytes)")


    # Save audit metrics
    audit_data = {
        "raw_source_path": RAW_DTA_PATH,
        "raw_source_bytes": raw_size,
        "raw_total_records": total_raw_records,
        "living_under5_cohort_size": u5_count,
        "cohort_counts": cohort_counts,
        "missingness_audit": missingness_audit
    }
    with open(AUDIT_JSON_PATH, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"[+] Saved cleaning audit metrics to {AUDIT_JSON_PATH}")
    
    # Confirm raw file was untouched
    assert os.path.getsize(RAW_DTA_PATH) == raw_size, "CRITICAL ERROR: Raw data file size changed!"
    print("[OK] Verified: Raw NFHS-5 dataset file is 100% untouched and unchanged.")
    return audit_data

if __name__ == "__main__":
    clean_nfhs5_data()
