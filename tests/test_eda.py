"""Automated verification test suite for Step 6: Exploratory Data Analysis (EDA)."""
import os
import json

OUTPUT_DIR_FIGURES = "reports/figures"
OUTPUT_SUMMARY_JSON = "data/interim/eda_summary.json"

EXPECTED_FIGURES = [
    "fig1_target_prevalence_and_imbalance.png",
    "fig2_missingness_profile.png",
    "fig3_age_dynamics_by_outcome.png",
    "fig4_socioeconomic_maternal_gradients.png",
    "fig5_morbidity_associations_acute_vs_chronic.png",
    "fig6_statistical_association_cramers_v.png",
    "fig7_state_level_stunting_wasting.png"
]

def test_eda_summary_json_exists_and_valid():
    assert os.path.exists(OUTPUT_SUMMARY_JSON), f"Summary JSON {OUTPUT_SUMMARY_JSON} not found"
    with open(OUTPUT_SUMMARY_JSON, "r") as f:
        summary = json.load(f)
    assert summary["dataset_total_records"] == 221263, f"Expected 221,263 records, got {summary['dataset_total_records']}"

def test_target_metrics_in_eda_summary():
    with open(OUTPUT_SUMMARY_JSON, "r") as f:
        summary = json.load(f)
    ts = summary["target_summary"]
    
    # Stunting
    assert ts["stunting"]["valid_n"] == 206025
    assert ts["stunting"]["positive_n"] == 73072
    assert ts["stunting"]["unweighted_prevalence_pct"] == 35.47
    
    # Underweight
    assert ts["underweight"]["valid_n"] == 210524
    assert ts["underweight"]["positive_n"] == 65043
    assert ts["underweight"]["unweighted_prevalence_pct"] == 30.90
    
    # Wasting
    assert ts["wasting"]["valid_n"] == 201687
    assert ts["wasting"]["positive_n"] == 37553
    assert ts["wasting"]["unweighted_prevalence_pct"] == 18.62

def test_all_eda_figures_exist_and_non_empty():
    for fig_name in EXPECTED_FIGURES:
        fig_path = os.path.join(OUTPUT_DIR_FIGURES, fig_name)
        assert os.path.exists(fig_path), f"Expected figure {fig_path} does not exist!"
        size = os.path.getsize(fig_path)
        assert size > 50000, f"Figure {fig_path} file size unexpectedly small ({size} bytes)"

def test_state_level_records_count():
    with open(OUTPUT_SUMMARY_JSON, "r") as f:
        summary = json.load(f)
    states = summary["state_records"]
    assert len(states) >= 35, f"Expected at least 35 states/UTs, found {len(states)}"

def test_cramers_v_features_count():
    with open(OUTPUT_SUMMARY_JSON, "r") as f:
        summary = json.load(f)
    cv = summary["cramers_v_associations"]
    assert len(cv) == 12, f"Expected exactly 12 candidate categorical features in Cramer's V analysis, found {len(cv)}"

if __name__ == "__main__":
    print("[*] Running Step 6 EDA Verification Tests...")
    test_eda_summary_json_exists_and_valid()
    print("  [OK] eda_summary.json exists and valid.")
    test_target_metrics_in_eda_summary()
    print("  [OK] Target metrics strictly verified (Stunting 35.47%, Underweight 30.90%, Wasting 18.62%).")
    test_all_eda_figures_exist_and_non_empty()
    print("  [OK] All 7 publication-quality figures verified on disk with valid file sizes.")
    test_state_level_records_count()
    print("  [OK] State-level geographic records verified across all Indian States/UTs.")
    test_cramers_v_features_count()
    print("  [OK] Exactly 12 candidate categorical features verified in Cramer's V association analysis.")
    print("[SUCCESS] All Step 6 EDA tests passed successfully!")
