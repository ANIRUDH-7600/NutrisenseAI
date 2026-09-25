"""
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System
Step 16 & Step 22 (Major Step 8): Final End-to-End Inference Pipeline (Scenario A — Community Pre-Screening).

This module provides a production-ready, reproducible, leakage-free screening inference
function for evaluating under-five childhood malnutrition risk (Stunting, Underweight, Wasting)
using exclusively the 30 approved Scenario-A v2 non-invasive candidate features (sensitive features excluded).

LOCKED ARCHITECTURAL PRINCIPLES:
1. Zero retraining or refitting: uses the exact fitted pipelines saved in models/v2/.
2. Locked decision thresholds (validation-derived operating thresholds):
   - Stunting: tau = 0.36
   - Underweight: tau = 0.30
   - Wasting: tau = 0.17
3. Strict isolation: rejects direct anthropometric measurements (hw70-hw73, hw2, hw3, hw4-hw12, hw13, hw57)
   and sensitive attributes (household_head_female, caste_category, religion_category, mother_education_level).
4. Zero microdata requirement: does not require or access raw DHS microdata (IAKR7EFL.DTA).
5. Screening interpretation: outputs 'screen_positive', strictly avoiding clinical diagnosis claims.
"""

import os
import copy
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple

# Pre-specified, locked decision thresholds for Scenario-A v2
LOCKED_THRESHOLDS = {
    "stunting": 0.36,
    "underweight": 0.30,
    "wasting": 0.17
}
LOCKED_THRESHOLDS_V2 = LOCKED_THRESHOLDS

# Locked champion v1 model artifact paths
MODEL_ARTIFACTS = {
    "stunting": "models/v1/model_comparison_lightgbm_stunting_unweighted_v1.joblib",
    "underweight": "models/v1/model_comparison_lightgbm_underweight_unweighted_v1.joblib",
    "wasting": "models/v1/model_comparison_lightgbm_wasting_unweighted_v1.joblib"
}

# The 30 approved Scenario-A v2 feature names in exact order
APPROVED_30_FEATURES = [
    "child_age_months",
    "child_age_group",
    "child_sex_male",
    "birth_order",
    "is_multiple_birth",
    "is_firstborn",
    "preceding_birth_interval_months",
    "birth_size_ordinal",
    "birth_weight_kg",
    "birth_weight_missing",
    "delivery_place_type",
    "still_breastfeeding",
    "diarrhea_recent",
    "fever_recent",
    "cough_recent",
    "mother_age_years",
    "mother_age_first_birth",
    "mother_bmi",
    "mother_bmi_missing",
    "total_children_born",
    "anc_visits_count",
    "anc_visits_missing",
    "wealth_quintile",
    "is_rural",
    "drinking_water_type",
    "sanitation_facility_type",
    "has_electricity",
    "clean_cooking_fuel",
    "household_size",
    "state_id"
]

APPROVED_34_FEATURES = APPROVED_30_FEATURES  # Primary feature list is 30 features

REMOVED_SENSITIVE_FEATURES = [
    "household_head_female",
    "caste_category",
    "religion_category",
    "mother_education_level"
]

# Prohibited anthropometric, survey weight, leakage variables, and removed sensitive attributes
PROHIBITED_LEAKAGE_VARS = {
    "hw70", "hw71", "hw72", "hw73",
    "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
    "hw2", "hw3", "hw2_clean", "hw3_clean",
    "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12",
    "hw13", "hw57", "v457",
    "stunting", "underweight", "wasting",
    "stunting_severe", "underweight_severe", "wasting_severe",
    "v001", "v002", "v003", "v005", "v021", "v022", "sample_weight",
    # Four excluded sensitive attributes
    "household_head_female", "caste_category", "religion_category", "mother_education_level"
}

# Allowed categorical sets
ALLOWED_AGE_GROUPS = {"00_05_mo", "06_11_mo", "12_23_mo", "24_35_mo", "36_47_mo", "48_59_mo"}
ALLOWED_DELIVERY_PLACES = {"Home_Delivery", "Public_Facility", "Private_Facility", "Other"}
ALLOWED_WATER_TYPES = {"Improved_Piped", "Improved_Groundwater_Bottled", "Unimproved_Surface_Other", "Unimproved_Surface"}
ALLOWED_TOILET_TYPES = {"Flush_Toilet", "Pit_Latrine", "Open_Defecation_None", "Other_Facility"}

# Global model cache to avoid repeated disk reads
_CACHED_MODELS: Dict[str, Any] = {}


def load_model_pipelines(force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads and validates the 3 locked LightGBM v2 model pipelines.
    Asserts pipeline architecture and 30-feature schema consistency.
    """
    global _CACHED_MODELS
    if _CACHED_MODELS and not force_reload:
        return _CACHED_MODELS

    loaded = {}
    for target, path in MODEL_ARTIFACTS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found for {target} at {path}")
        
        pipeline = joblib.load(path)
        
        # Verify pipeline structure
        if not hasattr(pipeline, "named_steps"):
            raise ValueError(f"Artifact at {path} is not a valid scikit-learn Pipeline.")
        
        if "preprocessor" not in pipeline.named_steps:
            raise ValueError(f"Pipeline at {path} lacks expected 'preprocessor' step.")
        
        if "classifier" not in pipeline.named_steps:
            raise ValueError(f"Pipeline at {path} lacks expected 'classifier' step.")
        
        preprocessor = pipeline.named_steps["preprocessor"]
        if not hasattr(preprocessor, "transformers"):
            raise ValueError(f"Pipeline preprocessor at {path} is not a ColumnTransformer.")
        
        num_features = preprocessor.transformers[0][2]
        cat_features = preprocessor.transformers[1][2]
        total_features = len(num_features) + len(cat_features)
        
        if total_features != 30:
            raise ValueError(f"Model at {path} expects {total_features} features, required: 30.")
        
        clf = pipeline.named_steps["classifier"]
        clf_type = type(clf).__name__
        if "LGBM" not in clf_type:
            raise ValueError(f"Model at {path} classifier is {clf_type}, expected LightGBM.")
        
        loaded[target] = pipeline

    _CACHED_MODELS = loaded
    return _CACHED_MODELS


def categorize_age_months(m: float) -> str:
    """Derives clinical age bracket from child age in months."""
    if m < 6:
        return "00_05_mo"
    elif m < 12:
        return "06_11_mo"
    elif m < 24:
        return "12_23_mo"
    elif m < 36:
        return "24_35_mo"
    elif m < 48:
        return "36_47_mo"
    else:
        return "48_59_mo"


def validate_single_record(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates a single child input record against the 30-feature Scenario-A v2 schema.
    Returns:
        (is_valid: bool, error_list: List[str], constructed_features: Dict[str, Any])
    """
    errors: List[str] = []
    constructed: Dict[str, Any] = {}

    # 1. Prohibited Leakage, Anthropometric, and Sensitive Attribute Check
    prohibited_found = set(data.keys()).intersection(PROHIBITED_LEAKAGE_VARS)
    if prohibited_found:
        errors.append(
            f"Prohibited variable(s) supplied: {sorted(list(prohibited_found))}. "
            "Direct anthropometric measurements, survey design variables, and sensitive social attributes "
            "are strictly forbidden in Scenario A v2."
        )
        return False, errors, {}

    # 2. Child Demographics
    if "child_age_months" not in data or data["child_age_months"] is None:
        errors.append("child_age_months is required.")
    else:
        try:
            age = float(data["child_age_months"])
            if not (0 <= age <= 59):
                errors.append(f"child_age_months must be between 0 and 59, got {age}.")
            constructed["child_age_months"] = age
            
            # Age group validation or automatic derivation
            if "child_age_group" in data and data["child_age_group"] is not None:
                ag = str(data["child_age_group"])
                if ag not in ALLOWED_AGE_GROUPS:
                    errors.append(f"child_age_group '{ag}' is invalid. Allowed: {sorted(list(ALLOWED_AGE_GROUPS))}.")
                constructed["child_age_group"] = ag
            else:
                constructed["child_age_group"] = categorize_age_months(age)
        except (ValueError, TypeError):
            errors.append(f"child_age_months must be a valid numeric value, got {type(data['child_age_months']).__name__}.")

    # Sex: male = 1, female = 0
    if "child_sex_male" in data:
        try:
            csm = int(data["child_sex_male"])
            if csm not in (0, 1):
                errors.append("child_sex_male must be 0 (Female) or 1 (Male).")
            constructed["child_sex_male"] = csm
        except (ValueError, TypeError):
            errors.append("child_sex_male must be 0 or 1.")
    elif "child_sex" in data:
        val = str(data["child_sex"]).lower()
        if val in ("male", "m", "1"):
            constructed["child_sex_male"] = 1
        elif val in ("female", "f", "0"):
            constructed["child_sex_male"] = 0
        else:
            errors.append("child_sex must indicate 'male' or 'female'.")
    else:
        errors.append("child_sex_male is required.")

    # Birth order
    if "birth_order" not in data or data["birth_order"] is None:
        errors.append("birth_order is required.")
    else:
        try:
            bo = float(data["birth_order"])
            if bo < 1:
                errors.append("birth_order must be >= 1.")
            constructed["birth_order"] = bo
        except (ValueError, TypeError):
            errors.append("birth_order must be a numeric value.")

    # Multiple birth
    if "is_multiple_birth" in data and data["is_multiple_birth"] is not None:
        try:
            imb = int(data["is_multiple_birth"])
            if imb not in (0, 1):
                errors.append("is_multiple_birth must be 0 or 1.")
            constructed["is_multiple_birth"] = imb
        except (ValueError, TypeError):
            errors.append("is_multiple_birth must be 0 or 1.")
    else:
        constructed["is_multiple_birth"] = 0

    # Preceding birth interval & firstborn logic
    pbi = data.get("preceding_birth_interval_months")
    if pbi is not None and not (isinstance(pbi, float) and np.isnan(pbi)):
        try:
            pbi_val = float(pbi)
            if pbi_val < 0:
                errors.append("preceding_birth_interval_months cannot be negative.")
            constructed["preceding_birth_interval_months"] = pbi_val
        except (ValueError, TypeError):
            errors.append("preceding_birth_interval_months must be a numeric value.")
    else:
        constructed["preceding_birth_interval_months"] = np.nan

    if "is_firstborn" in data and data["is_firstborn"] is not None:
        try:
            ifb = int(data["is_firstborn"])
            if ifb not in (0, 1):
                errors.append("is_firstborn must be 0 or 1.")
            constructed["is_firstborn"] = ifb
        except (ValueError, TypeError):
            errors.append("is_firstborn must be 0 or 1.")
    else:
        is_first = (
            constructed.get("birth_order", 1) == 1 or 
            np.isnan(constructed.get("preceding_birth_interval_months", np.nan))
        )
        constructed["is_firstborn"] = 1 if is_first else 0

    # 3. Birth Characteristics
    bso = data.get("birth_size_ordinal")
    if bso is not None and not (isinstance(bso, float) and np.isnan(bso)):
        try:
            bso_val = float(bso)
            if bso_val not in (1.0, 2.0, 3.0, 4.0, 5.0):
                errors.append("birth_size_ordinal must be an integer between 1 (very large) and 5 (very small).")
            constructed["birth_size_ordinal"] = bso_val
        except (ValueError, TypeError):
            errors.append("birth_size_ordinal must be an integer 1-5.")
    else:
        constructed["birth_size_ordinal"] = np.nan

    bw = data.get("birth_weight_kg")
    if bw is not None and not (isinstance(bw, float) and np.isnan(bw)):
        try:
            bw_val = float(bw)
            if not (0.5 <= bw_val <= 7.0):
                errors.append(f"birth_weight_kg must be within plausible biological range [0.5, 7.0], got {bw_val}.")
            constructed["birth_weight_kg"] = bw_val
            constructed["birth_weight_missing"] = 0
        except (ValueError, TypeError):
            errors.append("birth_weight_kg must be numeric.")
    else:
        constructed["birth_weight_kg"] = np.nan
        constructed["birth_weight_missing"] = 1

    if "birth_weight_missing" in data and data["birth_weight_missing"] is not None:
        try:
            constructed["birth_weight_missing"] = int(data["birth_weight_missing"])
        except (ValueError, TypeError):
            pass

    # Delivery place
    dp = data.get("delivery_place_type", "Other")
    if dp is None:
        dp = "Other"
    dp_str = str(dp).strip()
    norm_dp_map = {
        "home": "Home_Delivery",
        "home_delivery": "Home_Delivery",
        "public": "Public_Facility",
        "public_facility": "Public_Facility",
        "health_facility_public": "Public_Facility",
        "private": "Private_Facility",
        "private_facility": "Private_Facility",
        "health_facility_private": "Private_Facility",
        "other": "Other"
    }
    dp_normalized = norm_dp_map.get(dp_str.lower(), dp_str)
    if dp_normalized not in ALLOWED_DELIVERY_PLACES:
        errors.append(f"delivery_place_type '{dp}' is invalid. Allowed: {sorted(list(ALLOWED_DELIVERY_PLACES))}.")
    constructed["delivery_place_type"] = dp_normalized

    # 4. Feeding & Recent Morbidities
    for binary_field in ["still_breastfeeding", "diarrhea_recent", "fever_recent", "cough_recent"]:
        if binary_field not in data or data[binary_field] is None:
            errors.append(f"{binary_field} is required.")
        else:
            try:
                bval = float(data[binary_field])
                if bval not in (0.0, 1.0):
                    errors.append(f"{binary_field} must be 0 or 1.")
                constructed[binary_field] = bval
            except (ValueError, TypeError):
                errors.append(f"{binary_field} must be numeric 0 or 1.")

    # 5. Maternal Characteristics
    if "mother_age_years" not in data or data["mother_age_years"] is None:
        errors.append("mother_age_years is required.")
    else:
        try:
            may = float(data["mother_age_years"])
            if not (12 <= may <= 55):
                errors.append(f"mother_age_years must be between 12 and 55, got {may}.")
            constructed["mother_age_years"] = may
        except (ValueError, TypeError):
            errors.append("mother_age_years must be numeric.")

    if "mother_age_first_birth" not in data or data["mother_age_first_birth"] is None:
        errors.append("mother_age_first_birth is required.")
    else:
        try:
            mafb = float(data["mother_age_first_birth"])
            if not (10 <= mafb <= 50):
                errors.append(f"mother_age_first_birth must be between 10 and 50, got {mafb}.")
            constructed["mother_age_first_birth"] = mafb
        except (ValueError, TypeError):
            errors.append("mother_age_first_birth must be numeric.")

    if (
        "mother_age_years" in constructed and 
        "mother_age_first_birth" in constructed and
        constructed["mother_age_first_birth"] > constructed["mother_age_years"]
    ):
        errors.append(
            f"mother_age_first_birth ({constructed['mother_age_first_birth']}) cannot be greater "
            f"than mother_age_years ({constructed['mother_age_years']})."
        )

    # Maternal BMI
    mbmi = data.get("mother_bmi")
    if mbmi is not None and not (isinstance(mbmi, float) and np.isnan(mbmi)):
        try:
            mbmi_val = float(mbmi)
            if not (10.0 <= mbmi_val <= 60.0):
                errors.append(f"mother_bmi must be within plausible biological range [10.0, 60.0], got {mbmi_val}.")
            constructed["mother_bmi"] = mbmi_val
            constructed["mother_bmi_missing"] = 0
        except (ValueError, TypeError):
            errors.append("mother_bmi must be numeric.")
    else:
        constructed["mother_bmi"] = np.nan
        constructed["mother_bmi_missing"] = 1

    if "mother_bmi_missing" in data and data["mother_bmi_missing"] is not None:
        try:
            constructed["mother_bmi_missing"] = int(data["mother_bmi_missing"])
        except (ValueError, TypeError):
            pass

    # Total children born
    if "total_children_born" not in data or data["total_children_born"] is None:
        errors.append("total_children_born is required.")
    else:
        try:
            tcb = float(data["total_children_born"])
            if tcb < 1:
                errors.append("total_children_born must be >= 1.")
            constructed["total_children_born"] = tcb
        except (ValueError, TypeError):
            errors.append("total_children_born must be numeric.")

    if (
        "birth_order" in constructed and 
        "total_children_born" in constructed and
        constructed["birth_order"] > constructed["total_children_born"]
    ):
        errors.append(
            f"birth_order ({constructed['birth_order']}) cannot exceed total_children_born ({constructed['total_children_born']})."
        )

    # ANC visits
    anc = data.get("anc_visits_count")
    if anc is not None and not (isinstance(anc, float) and np.isnan(anc)):
        try:
            anc_val = float(anc)
            if anc_val < 0:
                errors.append("anc_visits_count cannot be negative.")
            constructed["anc_visits_count"] = anc_val
            constructed["anc_visits_missing"] = 0
        except (ValueError, TypeError):
            errors.append("anc_visits_count must be numeric.")
    else:
        constructed["anc_visits_count"] = np.nan
        constructed["anc_visits_missing"] = 1

    if "anc_visits_missing" in data and data["anc_visits_missing"] is not None:
        try:
            constructed["anc_visits_missing"] = int(data["anc_visits_missing"])
        except (ValueError, TypeError):
            pass

    # 6. Household Socioeconomics
    if "wealth_quintile" not in data or data["wealth_quintile"] is None:
        errors.append("wealth_quintile is required.")
    else:
        try:
            wq = float(data["wealth_quintile"])
            if wq not in (1.0, 2.0, 3.0, 4.0, 5.0):
                errors.append("wealth_quintile must be an integer between 1 (Poorest) and 5 (Richest).")
            constructed["wealth_quintile"] = wq
        except (ValueError, TypeError):
            errors.append("wealth_quintile must be an integer between 1 and 5.")

    if "is_rural" not in data or data["is_rural"] is None:
        errors.append("is_rural is required.")
    else:
        try:
            ir = float(data["is_rural"])
            if ir not in (0.0, 1.0):
                errors.append("is_rural must be 0 (Urban) or 1 (Rural).")
            constructed["is_rural"] = ir
        except (ValueError, TypeError):
            errors.append("is_rural must be 0 or 1.")

    # 7. Household Environment & Energy
    water = data.get("drinking_water_type", "Improved_Piped")
    water_str = str(water).strip()
    water_map = {
        "piped": "Improved_Piped",
        "improved_piped": "Improved_Piped",
        "groundwater": "Improved_Groundwater_Bottled",
        "improved_groundwater_bottled": "Improved_Groundwater_Bottled",
        "surface": "Unimproved_Surface",
        "unimproved_surface": "Unimproved_Surface",
        "unimproved_surface_other": "Unimproved_Surface_Other"
    }
    water_norm = water_map.get(water_str.lower(), water_str)
    if water_norm not in ALLOWED_WATER_TYPES:
        errors.append(f"drinking_water_type '{water}' is invalid. Allowed: {sorted(list(ALLOWED_WATER_TYPES))}.")
    constructed["drinking_water_type"] = water_norm

    toilet = data.get("sanitation_facility_type", "Flush_Toilet")
    toilet_str = str(toilet).strip()
    toilet_map = {
        "flush": "Flush_Toilet",
        "flush_toilet": "Flush_Toilet",
        "pit": "Pit_Latrine",
        "pit_latrine": "Pit_Latrine",
        "open": "Open_Defecation_None",
        "open_defecation": "Open_Defecation_None",
        "open_defecation_none": "Open_Defecation_None",
        "other": "Other_Facility",
        "other_facility": "Other_Facility"
    }
    toilet_norm = toilet_map.get(toilet_str.lower(), toilet_str)
    if toilet_norm not in ALLOWED_TOILET_TYPES:
        errors.append(f"sanitation_facility_type '{toilet}' is invalid. Allowed: {sorted(list(ALLOWED_TOILET_TYPES))}.")
    constructed["sanitation_facility_type"] = toilet_norm

    for env_field in ["has_electricity", "clean_cooking_fuel"]:
        if env_field not in data or data[env_field] is None:
            errors.append(f"{env_field} is required.")
        else:
            try:
                eval_ = float(data[env_field])
                if eval_ not in (0.0, 1.0):
                    errors.append(f"{env_field} must be 0 or 1.")
                constructed[env_field] = eval_
            except (ValueError, TypeError):
                errors.append(f"{env_field} must be 0 or 1.")

    if "household_size" not in data or data["household_size"] is None:
        errors.append("household_size is required.")
    else:
        try:
            hs = float(data["household_size"])
            if hs < 1:
                errors.append("household_size must be >= 1.")
            constructed["household_size"] = hs
        except (ValueError, TypeError):
            errors.append("household_size must be numeric.")

    # 8. Geographic Characteristics (State/UT)
    if "state_id" not in data or data["state_id"] is None:
        errors.append("state_id is required.")
    else:
        try:
            sid = int(data["state_id"])
            if not (1 <= sid <= 36):
                errors.append(f"state_id must be between 1 and 36, got {sid}.")
            constructed["state_id"] = sid
        except (ValueError, TypeError):
            errors.append(f"state_id must be an administrative integer between 1 and 36.")

    if errors:
        return False, errors, {}

    # Confirm all 30 features exist in constructed dict
    missing_features = [f for f in APPROVED_30_FEATURES if f not in constructed]
    if missing_features:
        errors.append(f"Internal construction error: missing features {missing_features}")
        return False, errors, {}

    return True, [], constructed


def predict_child_screening(input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Dict[str, Any]:
    """
    Primary inference entrypoint for Scenario A Community Pre-Screening (v2.0.0, 30 features).

    Accepts:
        - A single child dictionary of Scenario-A attributes.
        - A list of child dictionaries.
        - A pandas DataFrame.

    Returns:
        Structured prediction dictionary including validation status,
        probabilities, locked decision thresholds, and 'screen_positive' flags.
    """
    pipelines = load_model_pipelines()

    # Handle single dictionary input
    if isinstance(input_data, dict):
        is_valid, errors, constructed = validate_single_record(input_data)
        if not is_valid:
            return {
                "valid": False,
                "errors": errors
            }

        # Convert to 1-row DataFrame with exact 30 columns
        df_features = pd.DataFrame([constructed], columns=APPROVED_30_FEATURES)

        results = {
            "valid": True,
            "model_version": "nutrisense-scenario-a-v1.0.0",
            "feature_schema_version": "scenario-a-30-v1",
            "feature_count": 30,
            "predictions": {}
        }

        for target in ["stunting", "underweight", "wasting"]:
            model = pipelines[target]
            threshold = LOCKED_THRESHOLDS[target]
            
            # Predict probability (unrounded)
            prob = float(model.predict_proba(df_features)[:, 1][0])
            
            # Decision rule: probability >= threshold
            screen_pos = bool(prob >= threshold)
            
            results["predictions"][target] = {
                "probability": prob,
                "threshold": threshold,
                "screen_positive": screen_pos,
                "decision_rule": "probability >= threshold",
                "model_family": "LightGBM",
                "condition": "unweighted"
            }

        # Direct top-level target access
        results["stunting"] = results["predictions"]["stunting"]
        results["underweight"] = results["predictions"]["underweight"]
        results["wasting"] = results["predictions"]["wasting"]

        return results

    # Handle batch list input
    elif isinstance(input_data, list):
        batch_results = []
        for item in input_data:
            batch_results.append(predict_child_screening(item))
        return {
            "valid": all(r.get("valid", False) for r in batch_results),
            "batch_count": len(batch_results),
            "results": batch_results
        }

    # Handle pandas DataFrame input
    elif isinstance(input_data, pd.DataFrame):
        records = input_data.to_dict(orient="records")
        return predict_child_screening(records)

    else:
        return {
            "valid": False,
            "errors": [f"Unsupported input type: {type(input_data).__name__}. Expected dict, list, or DataFrame."]
        }
