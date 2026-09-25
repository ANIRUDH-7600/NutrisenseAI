"""Inspect column data types, missingness, and sentinel codes for key NFHS-5 variables."""
import pandas as pd
import numpy as np

DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"

KEY_COLUMNS = [
    # Anthropometric targets & flags
    "hw70", "hw71", "hw72", "hw73", "hw13",
    # Child demographics & birth history
    "hw1", "b4", "b5", "b8", "bord", "m18", "m19", "hw57",
    # Maternal characteristics
    "v012", "v445", "v457", "m14", "m15", "m4", "v404",
    # Household & socioeconomic
    "v001", "v005", "v021", "v022", "v024", "v025", "v106", "v190",
    "v113", "v116", "v119", "v136", "v151", "v152",
    # Child morbidity
    "h11", "h22", "h31"
]

def run_inspection():
    print(f"Loading {len(KEY_COLUMNS)} columns from {DTA_PATH}...")
    df = pd.read_stata(DTA_PATH, columns=KEY_COLUMNS, convert_categoricals=False)
    
    print("\n" + "="*80)
    print(f"DATASET CORE DIMENSIONS: {len(df):,} rows x {len(KEY_COLUMNS)} inspected columns")
    print("="*80)
    
    records = []
    for col in KEY_COLUMNS:
        series = df[col]
        n_nan = int(series.isna().sum())
        pct_nan = (n_nan / len(df)) * 100
        
        # Check for DHS sentinel codes (e.g. 9996, 9997, 9998, 98, 99)
        n_flagged = 0
        if col in ["hw70", "hw71", "hw72", "hw73"]:
            n_flagged = int(series.isin([9996, 9997, 9998]).sum())
        elif col in ["m18"]:
            n_flagged = int(series.isin([8, 9]).sum())
        elif col in ["m19"]:
            n_flagged = int(series.isin([9996, 9998]).sum())
            
        records.append({
            "Variable": col,
            "Dtype": str(series.dtype),
            "NaN Count": n_nan,
            "NaN %": round(pct_nan, 2),
            "Sentinel/Flagged": n_flagged,
            "Min": round(float(series.min()), 2) if np.issubdtype(series.dtype, np.number) and not series.dropna().empty else None,
            "Max": round(float(series.max()), 2) if np.issubdtype(series.dtype, np.number) and not series.dropna().empty else None,
        })
        
    res_df = pd.DataFrame(records)
    print(res_df.to_string(index=False))
    
    print("\n--- ANTHROPOMETRIC SENTINEL BREAKDOWN ---")
    for z_var in ["hw70", "hw71", "hw72"]:
        s = df[z_var]
        print(f"\n{z_var}:")
        print(f"  Valid Range (< 9000) count: {int((s < 9000).sum()):,}")
        print(f"  9996 (Height out of limits): {int((s == 9996).sum()):,}")
        print(f"  9997 (Age out of limits)   : {int((s == 9997).sum()):,}")
        print(f"  9998 (WHO Flagged)         : {int((s == 9998).sum()):,}")
        print(f"  NaN (Not Measured/Missing) : {int(s.isna().sum()):,}")

if __name__ == "__main__":
    run_inspection()
