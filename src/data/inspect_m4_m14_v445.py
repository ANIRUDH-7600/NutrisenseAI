"""Inspect official definitions of m4, m14, v445 from DO and MAP files."""
import re
import pandas as pd

DO_PATH = "IAKR7EDT/IAKR7EFL.DO"
MAP_PATH = "IAKR7EDT/IAKR7EFL.MAP"
DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"

with open(DO_PATH, "r", encoding="latin1") as f:
    do_content = f.read()

target_vars = ["m4", "m14", "v445", "v404"]

for var in target_vars:
    print(f"=== METADATA FOR {var} ===")
    vmatch = re.search(rf'label variable {var}\s+"([^"]+)"', do_content, re.IGNORECASE)
    if vmatch:
        print(f"  Variable Label: {vmatch.group(1)}")
    val_match = re.search(rf'label values {var}\s+([A-Za-z0-9_]+)', do_content, re.IGNORECASE)
    if val_match:
        lbl_name = val_match.group(1)
        print(f"  Value Label: {lbl_name}")
        def_match = re.search(rf'label define {lbl_name}\s+([^;\n]+(?:\n\s+[^;\n]+)*)', do_content, re.IGNORECASE)
        if def_match:
            print(f"  Defined Values:\n    {def_match.group(1).strip()}")
    else:
        print("  No discrete value label attached (continuous variable)")
    print()

# Now inspect MAP file if mentioned
if target_vars:
    with open(MAP_PATH, "r", encoding="latin1") as f:
        map_content = f.read()
    for var in target_vars:
        mmatch = re.search(rf'^{var}\s+.*$', map_content, re.MULTILINE | re.IGNORECASE)
        if mmatch:
            print(f"MAP entry for {var}: {mmatch.group(0)}")

# Check raw distributions in living under-5 cohort
print("\n=== RAW OBSERVED VALUES IN LIVING U5 COHORT ===")
df = pd.read_stata(DTA_PATH, columns=["b5", "hw1", "m4", "m14", "v445", "v404"], convert_categoricals=False)
u5 = df[(df["b5"] == 1) & (df["hw1"] >= 0) & (df["hw1"] <= 59)]

print("\nm4 value distribution (top 20 and special):")
print(u5["m4"].value_counts(dropna=False).head(20))
print("m4 values >= 90:")
print(u5["m4"].value_counts(dropna=False).loc[lambda x: x.index >= 90])

print("\nm14 (ANC visits) value distribution:")
print(u5["m14"].value_counts(dropna=False).sort_index())

print("\nv445 (Mother BMI) distribution summary:")
print(u5["v445"].describe())
print("v445 values >= 9000:")
print(u5["v445"].value_counts().loc[lambda x: x.index >= 9000])
print("v445 values < 9000 min and max:")
valid_v445 = u5.loc[(u5["v445"] < 9000) & (u5["v445"].notna()), "v445"]
print(f"  Raw min: {valid_v445.min()}, Raw max: {valid_v445.max()}")
print(f"  Scaled min: {valid_v445.min() / 100.0:.2f}, Scaled max: {valid_v445.max() / 100.0:.2f}")
