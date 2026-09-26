"""Verification test for Step 3: Research Population Definition and Count Integrity."""
import os
import json

SUMMARY_PATH = "data/interim/population_counts.json"

def test_population_summary_exists():
    assert os.path.exists(SUMMARY_PATH), f"Summary file {SUMMARY_PATH} does not exist"

def test_raw_and_alive_counts():
    with open(SUMMARY_PATH, "r") as f:
        counts = json.load(f)
        
    assert counts["raw_total_records"] == 232920, f"Expected 232,920 raw records, got {counts['raw_total_records']}"
    assert counts["alive_children"] == 224218, f"Expected 224,218 alive children, got {counts['alive_children']}"
    assert counts["deceased_children"] == 8702, f"Expected 8,702 deceased children, got {counts['deceased_children']}"
    assert counts["alive_children"] + counts["deceased_children"] == counts["raw_total_records"]

def test_age_eligibility_counts():
    with open(SUMMARY_PATH, "r") as f:
        counts = json.load(f)
        
    assert counts["living_under_five_0_59_months"] == 221263
    assert counts["missing_hw1_among_living"] == 2955
    assert counts["living_under_five_0_59_months"] + counts["missing_hw1_among_living"] == counts["alive_children"]

def test_target_eligible_cohort_counts():
    with open(SUMMARY_PATH, "r") as f:
        counts = json.load(f)
        
    # Check stunting
    assert counts["stunting_eligible_hw70_valid"] == 206025
    # Check underweight
    assert counts["underweight_eligible_hw71_valid"] == 210524
    # Check wasting
    assert counts["wasting_eligible_hw72_valid"] == 201687
    # Check complete cohort (all three valid)
    assert counts["complete_anthropometric_cohort_all_three_valid"] == 198802
    assert counts["union_at_least_one_valid"] == 211646

def test_cohort_logical_bounds():
    with open(SUMMARY_PATH, "r") as f:
        counts = json.load(f)
        
    # The complete cohort must be <= each individual valid count
    complete = counts["complete_anthropometric_cohort_all_three_valid"]
    assert complete <= counts["stunting_eligible_hw70_valid"]
    assert complete <= counts["underweight_eligible_hw71_valid"]
    assert complete <= counts["wasting_eligible_hw72_valid"]
    
    # The union must be >= each individual valid count
    union = counts["union_at_least_one_valid"]
    assert union >= counts["stunting_eligible_hw70_valid"]
    assert union >= counts["underweight_eligible_hw71_valid"]
    assert union >= counts["wasting_eligible_hw72_valid"]

if __name__ == "__main__":
    print("[*] Running Step 3 Research Population Definition Tests...")
    test_population_summary_exists()
    print("  [OK] population_counts.json verified.")
    test_raw_and_alive_counts()
    print("  [OK] Raw (232,920), alive (224,218), and deceased (8,702) counts verified.")
    test_age_eligibility_counts()
    print("  [OK] Living 0-59 months age eligibility (221,263) verified.")
    test_target_eligible_cohort_counts()
    print("  [OK] Stunting (206,025), Underweight (210,524), Wasting (201,687), Complete (198,802) verified.")
    test_cohort_logical_bounds()
    print("  [OK] Mathematical set theory bounds verified.")
    print("[SUCCESS] All Step 3 research population tests passed successfully!")
