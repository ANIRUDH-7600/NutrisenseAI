"""Verification test for Step 2: Data Dictionary & Metadata Integrity."""
import os
import pandas as pd
from pandas.io.stata import StataReader

CSV_PATH = "data_dictionary.csv"
DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"

def test_data_dictionary_exists():
    assert os.path.exists(CSV_PATH), f"{CSV_PATH} does not exist"
    df = pd.read_csv(CSV_PATH)
    assert len(df) >= 50, f"Expected at least 50 documented variables, got {len(df)}"

def test_variables_exist_in_raw_stata():
    df = pd.read_csv(CSV_PATH)
    with StataReader(DTA_PATH) as reader:
        vlabels = reader.variable_labels()
        
    for var in df["variable_name"]:
        assert var in vlabels, f"Variable {var} documented in dictionary does NOT exist in raw NFHS-5 .DTA"
        # Check that label in dictionary matches the official DHS label
        dict_label = df.loc[df["variable_name"] == var, "official_dhs_label"].values[0]
        assert dict_label == vlabels[var], f"Label mismatch for {var}: dict='{dict_label}', raw='{vlabels[var]}'"

def test_required_roles_and_domains():
    df = pd.read_csv(CSV_PATH)
    roles = set(df["role"].unique())
    expected_roles = {
        "candidate target",
        "candidate feature",
        "potential leakage variable",
        "survey weight",
        "identifier",
        "survey-design variable",
        "potentially derived variable"
    }
    for er in expected_roles:
        assert er in roles, f"Expected role '{er}' not found in data dictionary"

    # Check key targets exist
    targets = df[df["role"] == "candidate target"]["variable_name"].tolist()
    assert "hw70" in targets, "hw70 must be a candidate target"
    assert "hw71" in targets, "hw71 must be a candidate target"
    assert "hw72" in targets, "hw72 must be a candidate target"

def test_total_observation_invariants():
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        total = row["valid_observations_count"] + row["missing_observations_count"]
        assert total == 232920, f"Row sum for {row['variable_name']} is {total}, expected exactly 232,920"

if __name__ == "__main__":
    print("[*] Running Step 2 Data Dictionary Integrity Tests...")
    test_data_dictionary_exists()
    print("  [OK] data_dictionary.csv exists with >= 50 variables.")
    test_variables_exist_in_raw_stata()
    print("  [OK] All variables and official labels strictly verified against raw .DTA.")
    test_required_roles_and_domains()
    print("  [OK] All required roles, domains, and candidate targets verified.")
    test_total_observation_invariants()
    print("  [OK] Observation totals strictly equal 232,920 across all variables.")
    print("[SUCCESS] All Step 2 data dictionary tests passed successfully!")
