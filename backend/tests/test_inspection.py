"""Verification test for Step 1: Dataset Inspection."""
import os
from pandas.io.stata import StataReader

DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"

def test_raw_dataset_exists():
    assert os.path.exists(DTA_PATH), f"Target dataset {DTA_PATH} was not found"

def test_dataset_dimensions():
    with StataReader(DTA_PATH) as reader:
        vlabels = reader.variable_labels()
        nobs = getattr(reader, "_nobs", 0)
        nvar = getattr(reader, "_nvar", 0)
        assert nobs == 232920, f"Expected 232,920 rows, got {nobs}"
        assert nvar == 1644, f"Expected 1,644 variables, got {nvar}"


def test_critical_anthropometric_variables_exist():
    with StataReader(DTA_PATH) as reader:
        vlabels = reader.variable_labels()
        # Anthropometry
        assert "hw70" in vlabels, "hw70 (Height/Age z-score) missing"
        assert "hw71" in vlabels, "hw71 (Weight/Age z-score) missing"
        assert "hw72" in vlabels, "hw72 (Weight/Height z-score) missing"
        assert "hw13" in vlabels, "hw13 (Result of measurement) missing"
        # Demographics & Weights
        assert "b4" in vlabels, "b4 (Child sex) missing"
        assert "b5" in vlabels, "b5 (Child alive status) missing"
        assert "hw1" in vlabels, "hw1 (Child age in months) missing"
        assert "v001" in vlabels, "v001 (Cluster number) missing"
        assert "v005" in vlabels, "v005 (Sample weight) missing"
        assert "v024" in vlabels, "v024 (State) missing"
        assert "v025" in vlabels, "v025 (Residence) missing"

if __name__ == "__main__":
    print("[*] Running Step 1 Dataset Inspection Verification...")
    test_raw_dataset_exists()
    print("  [OK] Raw NFHS-5 dataset file confirmed.")
    test_dataset_dimensions()
    print("  [OK] Dataset dimensions (232,920 rows x 1,644 cols) verified.")
    test_critical_anthropometric_variables_exist()
    print("  [OK] Critical target and predictor variables verified.")
    print("[SUCCESS] All Step 1 inspection tests passed!")
