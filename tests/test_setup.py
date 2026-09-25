"""Verification test for Step 0 project setup and environment integrity."""
import os
import sys

def test_directory_structure():
    """Verify all critical directories exist."""
    required_dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "docs",
        "src/data",
        "src/features",
        "src/models",
        "src/explainability",
        "src/utils",
        "models",
        "reports/figures",
        "tests",
    ]
    for d in required_dirs:
        assert os.path.exists(d), f"Directory {d} does not exist"

def test_gitignore_protects_confidential_data():
    """Verify that .gitignore exists and contains critical rules for DHS/NFHS privacy."""
    assert os.path.exists(".gitignore"), ".gitignore does not exist"
    with open(".gitignore", "r") as f:
        content = f.read()
    
    # Check for crucial DHS protection patterns
    assert "IAKR7EDT" in content, ".gitignore must ignore raw DHS folder IAKR7EDT"
    assert "*.dta" in content or "*.DTA" in content, ".gitignore must ignore Stata microdata"
    assert "data/raw/*" in content, ".gitignore must ignore raw data contents"
    assert "models/*" in content, ".gitignore must ignore trained model weights"

def test_core_imports():
    """Verify core scientific libraries can be imported."""
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost
    import shap
    
    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgboost.__version__ is not None
    assert shap.__version__ is not None

if __name__ == "__main__":
    print("[*] Running Step 0 Verification Tests...")
    test_directory_structure()
    print("  [OK] Directory structure verified.")
    test_gitignore_protects_confidential_data()
    print("  [OK] .gitignore rules verified.")
    test_core_imports()
    print("  [OK] Core imports verified (pandas, numpy, scikit-learn, xgboost, shap).")
    print("[SUCCESS] All Step 0 verification checks passed successfully!")
