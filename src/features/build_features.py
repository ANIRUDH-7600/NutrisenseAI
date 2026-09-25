"""Feature Engineering Pipeline for NutriSense AI (Scenario A: Community Pre-Screening).

This module defines the candidate predictor set, categorizes features into logical domains,
constructs non-leaking transformed representations (e.g., grouped WASH, missingness flags),
and validates strict isolation between predictors (X), targets (y), and survey metadata.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Scenario A assumes frontline community pre-screening without height/weight scales.
- Excludes hw70-hw73, hw2, hw3, hw4-hw12, and target columns from feature matrix X.
- Does NOT fit dataset-wide imputers, scalers, or encoders (fitted strictly on train split later).
"""
import os
import json
import numpy as np
import pandas as pd

INPUT_TARGETS_DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
OUTPUT_FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
OUTPUT_FEATURE_METADATA_V2_PATH = "data/interim/feature_metadata_v2.json"

REMOVED_SENSITIVE_FEATURES_V2 = [
    "household_head_female",
    "caste_category",
    "religion_category",
    "mother_education_level"
]

SCENARIO_A_V2_FEATURE_NAMES = [
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

# Prohibited anthropometric variables for Scenario A feature matrix
PROHIBITED_LEAKAGE_VARS = {
    "hw70", "hw71", "hw72", "hw73",
    "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
    "hw2", "hw3", "hw2_clean", "hw3_clean",
    "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12"
}

# Prohibited target and audit columns
PROHIBITED_TARGET_VARS = {
    "stunting", "underweight", "wasting",
    "stunting_severe", "underweight_severe", "wasting_severe",
    "eligible_stunting", "eligible_underweight", "eligible_wasting", "eligible_complete_unified"
}

# Prohibited survey design / identifier variables
PROHIBITED_SURVEY_VARS = {
    "v001", "v002", "v003", "v005", "v021", "v022", "sample_weight"
}


def get_feature_inventory():
    """Returns the comprehensive inventory of candidate features, domains, roles, and exclusion rationale."""
    inventory = {
        # Domain A: Child Demographics
        "child_age_months": {
            "source_col": "hw1",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Child's exact age in completed months at interview (0-59 months).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Pass-through as numerical predictor.",
            "scaling_recommendation": "StandardScaler / RobustScaler for linear models; unscaled for tree models."
        },
        "child_age_group": {
            "source_col": "hw1",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "ordinal",
            "description": "Clinical developmental age bracket (0-5, 6-11, 12-23, 24-35, 36-47, 48-59 months).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Ordinal integer encoding or One-Hot encoding for linear baselines.",
            "scaling_recommendation": "None."
        },
        "child_sex": {
            "source_col": "b4",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "binary",
            "description": "Biological sex of child (1=Male, 0=Female).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Binary (1=Male, 0=Female).",
            "scaling_recommendation": "Pass-through."
        },
        "birth_order": {
            "source_col": "bord",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Birth order number of child relative to mother's full birth history.",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Pass-through as numerical count.",
            "scaling_recommendation": "StandardScaler for linear models; unscaled for tree models."
        },
        "is_multiple_birth": {
            "source_col": "b0",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "binary",
            "description": "Indicator if child was born as twin, triplet, or multiple birth (1=Multiple, 0=Single).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Binary indicator (1 if b0 > 0 else 0).",
            "scaling_recommendation": "Pass-through."
        },
        "preceding_birth_interval": {
            "source_col": "b11",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Preceding birth interval in completed months.",
            "missingness_type": "structural_firstborn",
            "missingness_strategy": "Paired with is_firstborn indicator; missing values imputed with median on training split.",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "is_firstborn": {
            "source_col": "b11",
            "domain": "A. Child Demographics",
            "role": "included",
            "feature_type": "binary",
            "description": "Structural indicator for firstborn child with no preceding sibling (1=Firstborn, 0=Has older sibling).",
            "missingness_type": "none",
            "missingness_strategy": "Derived from b11 isna() / bord == 1.",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },

        # Domain B: Birth Characteristics
        "birth_size": {
            "source_col": "m18_clean",
            "domain": "B. Birth Characteristics",
            "role": "included",
            "feature_type": "ordinal",
            "description": "Subjective size of child at birth reported by mother (1=Very large to 5=Very small).",
            "missingness_type": "informative_nonresponse",
            "missingness_strategy": "Imputed with training mode (3=Average) or dedicated missing category.",
            "encoding_strategy": "Ordinal integer encoding (1 to 5).",
            "scaling_recommendation": "Unscaled or MinMax for linear models."
        },
        "birth_weight_kg": {
            "source_col": "birth_weight_kg",
            "domain": "B. Birth Characteristics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Recorded or recalled birth weight in kilograms (0.5 to 6.0 kg).",
            "missingness_type": "unmeasured_home_delivery",
            "missingness_strategy": "Paired with birth_weight_missing indicator; imputed with training median.",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "birth_weight_missing": {
            "source_col": "birth_weight_kg",
            "domain": "B. Birth Characteristics",
            "role": "included",
            "feature_type": "binary",
            "description": "Binary indicator that a valid birth weight value is unavailable after Step 4 cleaning (1=Unavailable, 0=Available).",
            "missingness_type": "none",
            "missingness_strategy": "Derived from birth_weight_kg isna().",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },
        "delivery_place": {
            "source_col": "m15",
            "domain": "B. Birth Characteristics",
            "role": "included",
            "feature_type": "nominal",
            "description": "Grouped place of delivery (Public Facility, Private Facility, Home Delivery, Other).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "One-Hot encoding.",
            "scaling_recommendation": "None."
        },

        # Domain C: Breastfeeding / Feeding
        "still_breastfeeding": {
            "source_col": "still_breastfeeding",
            "domain": "C. Breastfeeding / Feeding",
            "role": "included",
            "feature_type": "binary",
            "description": "Indicator if child is currently being breastfed (1=Yes, 0=No/Weaned).",
            "missingness_type": "nonresponse",
            "missingness_strategy": "Imputed with training mode or explicit missing category.",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },

        # Domain D: Child Health / Recent Morbidity
        "diarrhea_recent": {
            "source_col": "diarrhea_recent",
            "domain": "D. Child Health",
            "role": "included",
            "feature_type": "binary",
            "description": "Child experienced acute diarrhea in the preceding 2 weeks (1=Yes, 0=No).",
            "missingness_type": "minimal_nonresponse",
            "missingness_strategy": "Imputed with 0 (No) on training split (0.11% missing).",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },
        "fever_recent": {
            "source_col": "fever_recent",
            "domain": "D. Child Health",
            "role": "included",
            "feature_type": "binary",
            "description": "Child experienced fever in the preceding 2 weeks (1=Yes, 0=No).",
            "missingness_type": "minimal_nonresponse",
            "missingness_strategy": "Imputed with 0 (No) on training split (0.07% missing).",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },
        "cough_recent": {
            "source_col": "cough_recent",
            "domain": "D. Child Health",
            "role": "included",
            "feature_type": "binary",
            "description": "Child experienced acute cough / respiratory symptoms in preceding 2 weeks (1=Yes, 0=No).",
            "missingness_type": "minimal_nonresponse",
            "missingness_strategy": "Imputed with 0 (No) on training split (0.12% missing).",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },

        # Domain E: Maternal Characteristics & Healthcare
        "mother_age_years": {
            "source_col": "v012",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Mother's current age in completed years at interview (15-49 years).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "mother_age_at_first_birth": {
            "source_col": "v212",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Mother's age at first birth in completed years.",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "mother_education_level": {
            "source_col": "v106",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "ordinal",
            "description": "Mother's highest educational attainment (0=No education, 1=Primary, 2=Secondary, 3=Higher).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Ordinal integer encoding (0 to 3).",
            "scaling_recommendation": "Unscaled or MinMax for linear models."
        },
        "mother_bmi": {
            "source_col": "v445_clean",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Mother's Body Mass Index in kg/m^2 (observed range 12.02 to 59.99).",
            "missingness_type": "nonresponse_flagged",
            "missingness_strategy": "Paired with mother_bmi_missing indicator; imputed with training median.",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler / RobustScaler for linear models."
        },
        "mother_bmi_missing": {
            "source_col": "v445_clean",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "binary",
            "description": "Binary indicator that a valid maternal BMI value is unavailable after Step 4 cleaning (1=Unavailable, 0=Available).",
            "missingness_type": "none",
            "missingness_strategy": "Derived from v445_clean isna().",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },
        "total_children_born": {
            "source_col": "v201",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Total number of children ever born to mother (parity count).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "anc_visits_count": {
            "source_col": "m14_clean",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "numerical",
            "description": "Reported number of antenatal care visits during pregnancy (0 to 95 visits).",
            "missingness_type": "recall_nonresponse",
            "missingness_strategy": "Paired with anc_visits_missing indicator; imputed with training median.",
            "encoding_strategy": "Pass-through numerical.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "anc_visits_missing": {
            "source_col": "m14_clean",
            "domain": "E. Maternal Characteristics",
            "role": "included",
            "feature_type": "binary",
            "description": "Binary indicator that a valid ANC visit count is unavailable after Step 4 cleaning (1=Unavailable, 0=Available).",
            "missingness_type": "none",
            "missingness_strategy": "Derived from m14_clean isna().",
            "encoding_strategy": "Binary indicator.",
            "scaling_recommendation": "Pass-through."
        },

        # Domain F: Household Socioeconomic Characteristics
        "wealth_quintile": {
            "source_col": "v190",
            "domain": "F. Household Socioeconomic",
            "role": "included",
            "feature_type": "ordinal",
            "description": "Household wealth index combined quintile (1=Poorest to 5=Richest). Ordinal encoding represents category order but does not imply equal numerical distances.",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Ordinal integer encoding (1 to 5).",
            "scaling_recommendation": "Unscaled or MinMax for linear models."
        },
        "residence_type": {
            "source_col": "v025",
            "domain": "F. Household Socioeconomic",
            "role": "included",
            "feature_type": "binary",
            "description": "Type of place of residence (1=Rural, 0=Urban).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Binary indicator (1 if v025==2 [Rural] else 0).",
            "scaling_recommendation": "Pass-through."
        },
        "caste_category": {
            "source_col": "caste_clean",
            "domain": "F. Household Socioeconomic",
            "role": "included",
            "feature_type": "nominal",
            "description": "Social group in India (1=Scheduled Caste, 2=Scheduled Tribe, 3=OBC, 4=None/General).",
            "missingness_type": "nonresponse",
            "missingness_strategy": "Imputed with training mode or encoded with dedicated category (5.3% missing).",
            "encoding_strategy": "One-Hot encoding.",
            "scaling_recommendation": "None."
        },
        "religion": {
            "source_col": "v130",
            "domain": "F. Household Socioeconomic",
            "role": "included",
            "feature_type": "nominal",
            "description": "Household religion (Hindu, Muslim, Christian, Sikh, Other).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "One-Hot encoding.",
            "scaling_recommendation": "None."
        },

        # Domain G: Household / Environment Characteristics (WASH & Energy)
        "drinking_water_type": {
            "source_col": "v113",
            "domain": "G. Household Environment",
            "role": "included",
            "feature_type": "nominal",
            "description": "WHO/UNICEF JMP drinking water category (Improved Piped, Improved Groundwater/Bottled, Unimproved/Surface).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "One-Hot encoding.",
            "scaling_recommendation": "None."
        },
        "sanitation_facility_type": {
            "source_col": "v116",
            "domain": "G. Household Environment",
            "role": "included",
            "feature_type": "nominal",
            "description": "WHO/UNICEF JMP toilet facility category (Flush Toilet, Pit Latrine, Open Defecation / None).",
            "missingness_type": "minimal",
            "missingness_strategy": "Imputed with training mode (1 record missing).",
            "encoding_strategy": "One-Hot encoding.",
            "scaling_recommendation": "None."
        },
        "has_electricity": {
            "source_col": "v119",
            "domain": "G. Household Environment",
            "role": "included",
            "feature_type": "binary",
            "description": "Household has electric connection (1=Yes, 0=No).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Binary indicator (1 if v119==1 else 0).",
            "scaling_recommendation": "Pass-through."
        },
        "clean_cooking_fuel": {
            "source_col": "v161",
            "domain": "G. Household Environment",
            "role": "included",
            "feature_type": "binary",
            "description": "Household uses clean cooking fuel [LPG/Natural Gas/Electricity/Biogas] vs Biomass/Solid (1=Clean, 0=Solid/Biomass).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Binary indicator (1 if v161 in [1, 2, 3] else 0).",
            "scaling_recommendation": "Pass-through."
        },
        "household_size": {
            "source_col": "v136",
            "domain": "G. Household Environment",
            "role": "included",
            "feature_type": "numerical",
            "description": "Number of usual household members listed in household roster.",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Pass-through numerical count.",
            "scaling_recommendation": "StandardScaler for linear models."
        },
        "household_head_female": {
            "source_col": "v151",
            "domain": "G. Household Environment",
            "role": "included",
            "feature_type": "binary",
            "description": "Sex of household head (1=Female, 0=Male).",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Binary indicator (1 if v151==2 else 0).",
            "scaling_recommendation": "Pass-through."
        },

        # Domain H: Geographic Characteristics
        "state_id": {
            "source_col": "v024",
            "domain": "H. Geographic Characteristics",
            "role": "included",
            "feature_type": "nominal",
            "description": "State / Union Territory identifier across 36 Indian administrative units. Retained as a candidate predictive feature because state/UT information can be available during community screening. Its contribution and geographic generalization will be evaluated after the train/validation/test strategy is established.",
            "missingness_type": "none",
            "missingness_strategy": "None (0% missing).",
            "encoding_strategy": "Nominal categorical variables will initially be represented using one-hot encoding within a leakage-safe preprocessing pipeline. Alternative encodings may be evaluated after train/validation/test splitting and must be fitted exclusively on training data.",
            "scaling_recommendation": "None."
        }
    }
    return inventory


def get_excluded_variables_registry():
    """Returns the explicit registry of excluded variables and rigorous methodological justifications."""
    excluded = {
        # Prohibited Anthropometric Measurements (Scenario A Leakage)
        "hw70": {"domain": "Child Anthropometry", "reason": "Target leakage: Raw continuous HAZ; mathematical basis for stunting target."},
        "hw71": {"domain": "Child Anthropometry", "reason": "Target leakage: Raw continuous WAZ; mathematical basis for underweight target."},
        "hw72": {"domain": "Child Anthropometry", "reason": "Target leakage: Raw continuous WHZ; mathematical basis for wasting target."},
        "hw73": {"domain": "Child Anthropometry", "reason": "Target leakage: Raw continuous BMI z-score; mathematical derivative of wasting/thinness."},
        "hw70_clean": {"domain": "Child Anthropometry", "reason": "Target leakage: Cleaned HAZ z-score used for ground-truth label generation."},
        "hw71_clean": {"domain": "Child Anthropometry", "reason": "Target leakage: Cleaned WAZ z-score used for ground-truth label generation."},
        "hw72_clean": {"domain": "Child Anthropometry", "reason": "Target leakage: Cleaned WHZ z-score used for ground-truth label generation."},
        "hw73_clean": {"domain": "Child Anthropometry", "reason": "Target leakage: Cleaned BMI-for-age z-score."},
        "hw2": {"domain": "Child Anthropometry", "reason": "Target leakage / Scenario A constraint: Physical child weight measurement (kg) unavailable in scale-free pre-screening."},
        "hw3": {"domain": "Child Anthropometry", "reason": "Target leakage / Scenario A constraint: Physical child length/height measurement (cm) unavailable in stadiometer-free pre-screening."},
        "hw2_clean": {"domain": "Child Anthropometry", "reason": "Target leakage: Scaled child weight in kg."},
        "hw3_clean": {"domain": "Child Anthropometry", "reason": "Target leakage: Scaled child height in cm."},
        "hw4": {"domain": "Child Anthropometry", "reason": "Target leakage: Height/age percentile (direct derivative of hw70)."},
        "hw5": {"domain": "Child Anthropometry", "reason": "Target leakage: NCHS Height/age standard deviation."},
        "hw6": {"domain": "Child Anthropometry", "reason": "Target leakage: Height/age percent of reference median."},
        "hw7": {"domain": "Child Anthropometry", "reason": "Target leakage: Weight/age percentile (direct derivative of hw71)."},
        "hw8": {"domain": "Child Anthropometry", "reason": "Target leakage: NCHS Weight/age standard deviation."},
        "hw9": {"domain": "Child Anthropometry", "reason": "Target leakage: Weight/age percent of reference median."},
        "hw10": {"domain": "Child Anthropometry", "reason": "Target leakage: Weight/height percentile (direct derivative of hw72)."},
        "hw11": {"domain": "Child Anthropometry", "reason": "Target leakage: NCHS Weight/height standard deviation."},
        "hw12": {"domain": "Child Anthropometry", "reason": "Target leakage: Weight/height percent of reference median."},
        "hw13": {"domain": "Child Anthropometry", "reason": "Post-outcome audit variable: Reason anthropometric measurement was not taken."},

        # Target Variables & Eligibility Flags
        "stunting": {"domain": "Clinical Targets", "reason": "Target variable (outcome label y)."},
        "underweight": {"domain": "Clinical Targets", "reason": "Target variable (outcome label y)."},
        "wasting": {"domain": "Clinical Targets", "reason": "Target variable (outcome label y)."},
        "stunting_severe": {"domain": "Clinical Targets", "reason": "Severe target variable (outcome label y)."},
        "underweight_severe": {"domain": "Clinical Targets", "reason": "Severe target variable (outcome label y)."},
        "wasting_severe": {"domain": "Clinical Targets", "reason": "Severe target variable (outcome label y)."},
        "eligible_stunting": {"domain": "Cohort Audit", "reason": "Eligibility cohort selection flag."},
        "eligible_underweight": {"domain": "Cohort Audit", "reason": "Eligibility cohort selection flag."},
        "eligible_wasting": {"domain": "Cohort Audit", "reason": "Eligibility cohort selection flag."},
        "eligible_complete_unified": {"domain": "Cohort Audit", "reason": "Eligibility cohort selection flag."},

        # Invasive Clinical Testing (Unavailable in Community Pre-Screening)
        "hw57": {"domain": "Child Health", "reason": "Invasive clinical test: Child hemoglobin/anemia requires capillary blood finger-prick and HemoCue photometer (not available during routine ASHA community pre-screening; 16.91% missing)."},
        "v457": {"domain": "Maternal Characteristics", "reason": "Invasive clinical test: Maternal hemoglobin/anemia requires capillary blood collection (3.65% missing)."},

        # High Missingness / Subsample Non-Response in KR File
        "v701": {"domain": "Maternal Characteristics", "reason": "High missingness (84.76%): Partner education asked only for married sub-roster in KR file."},
        "v701_clean": {"domain": "Maternal Characteristics", "reason": "High missingness (84.82%): Cleaned partner education."},
        "v714": {"domain": "Maternal Characteristics", "reason": "High missingness (84.74%): Mother employment status asked only for sub-sample of mothers in KR file."},
        "m4_completed_months": {"domain": "Child Feeding", "reason": "Structural non-applicability (67.87%): Completed breastfeeding duration is undefined for 106,538 actively nursing children (code 95). Replaced by still_breastfeeding indicator."},
        "m4_censored_months": {"domain": "Child Feeding", "reason": "Survival analysis variable: Right-censored duration at current age. Retained for survival analysis, excluded from primary tabular classifier."},

        # Redundancy / Coarse Resolution / Constant Filter
        "b5": {"domain": "Child Demographics", "reason": "Constant in cohort: Child alive indicator (constant 1 in living under-5 research population)."},
        "b8": {"domain": "Child Demographics", "reason": "Redundant coarse granularity: Child age in completed single years (hw1 age in months provides superior resolution)."},
        "m19": {"domain": "Birth Characteristics", "reason": "Redundant raw variable: Replaced by cleaned birth_weight_kg (0.5 to 6.0 kg) and birth_weight_missing indicator."},
        "m18": {"domain": "Birth Characteristics", "reason": "Redundant raw variable: Replaced by cleaned m18_clean (special codes removed)."},
        "m14": {"domain": "Maternal Healthcare", "reason": "Redundant raw variable: Replaced by cleaned m14_clean."},
        "v445": {"domain": "Maternal Characteristics", "reason": "Redundant raw variable: Replaced by cleaned v445_clean."},
        "s116": {"domain": "Social & Cultural", "reason": "Redundant raw variable: Replaced by cleaned caste_clean."},
        "h11": {"domain": "Child Morbidity", "reason": "Redundant raw variable: Replaced by binary diarrhea_recent."},
        "h22": {"domain": "Child Morbidity", "reason": "Redundant raw variable: Replaced by binary fever_recent."},
        "h31": {"domain": "Child Morbidity", "reason": "Redundant raw variable: Replaced by binary cough_recent."},
        "v404": {"domain": "Child Feeding", "reason": "Redundant raw variable: Replaced by validated still_breastfeeding indicator."},
        "m4": {"domain": "Child Feeding", "reason": "Redundant raw variable: Replaced by still_breastfeeding indicator."},

        # High Cardinality / Geographic Overfitting
        "sdist": {"domain": "Geographic / Regional", "reason": "High cardinality (707 districts): Risk of severe geographic memorization/overfitting in tree models; retained for subgroup audit."},

        # Survey Design / Sampling / Identifiers
        "v001": {"domain": "Survey Design", "reason": "Survey cluster identifier / PSU; causes arbitrary spatial cluster memorization."},
        "v002": {"domain": "Survey Design", "reason": "Household administrative identifier number within cluster."},
        "v003": {"domain": "Survey Design", "reason": "Respondent line number in household schedule."},
        "v005": {"domain": "Survey Design", "reason": "Sampling probability weight; not an intrinsic child risk predictor. Including it induces model to learn survey selection artifacts."},
        "sample_weight": {"domain": "Survey Design", "reason": "Scaled sampling weight (v005 / 1,000,000); strictly reserved for weighted evaluation."},
        "v021": {"domain": "Survey Design", "reason": "Primary sampling unit (PSU) code for complex survey variance estimation."},
        "v022": {"domain": "Survey Design", "reason": "Sample strata code for complex survey variance estimation."}
    }
    return excluded


def transform_candidate_features(df_input):
    """Deterministic feature transformation without dataset-level statistic fitting.

    Transforms raw/cleaned columns into clean candidate feature matrix X.
    Does NOT fit scalers, imputers, or encoders (which must be fitted strictly on train split).
    """
    df = pd.DataFrame(index=df_input.index)

    # 1. Child Demographics
    df["child_age_months"] = df_input["hw1"].astype(float)
    
    # Clinical age groups: 0-5, 6-11, 12-23, 24-35, 36-47, 48-59 months
    def categorize_age(m):
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
    df["child_age_group"] = df_input["hw1"].apply(categorize_age)
    
    # Sex: 1=Male, 2=Female in DHS -> 1=Male, 0=Female
    df["child_sex_male"] = np.where(df_input["b4"] == 1, 1, 0)
    df["birth_order"] = df_input["bord"].astype(float)
    df["is_multiple_birth"] = np.where(df_input["b0"] > 0, 1, 0)
    
    # Preceding birth interval & firstborn indicator
    df["is_firstborn"] = np.where(df_input["b11"].isna() | (df_input["bord"] == 1), 1, 0)
    df["preceding_birth_interval_months"] = df_input["b11"].astype(float)

    # 2. Birth Characteristics
    df["birth_size_ordinal"] = df_input["m18_clean"].astype(float)
    df["birth_weight_kg"] = df_input["birth_weight_kg"].astype(float)
    df["birth_weight_missing"] = np.where(df_input["birth_weight_kg"].isna(), 1, 0)

    # Place of delivery: 10-14=Home, 20-28=Public Facility, 30-33=Private Facility, Other
    def categorize_delivery(p):
        if pd.isna(p):
            return "Other"
        p = int(p)
        if 10 <= p <= 14 or p == 96:
            return "Home_Delivery"
        elif 20 <= p <= 28:
            return "Public_Facility"
        elif 30 <= p <= 33:
            return "Private_Facility"
        else:
            return "Other"
    df["delivery_place_type"] = df_input["m15"].apply(categorize_delivery)

    # 3. Breastfeeding / Feeding
    df["still_breastfeeding"] = df_input["still_breastfeeding"].astype(float)

    # 4. Child Morbidity
    df["diarrhea_recent"] = df_input["diarrhea_recent"].astype(float)
    df["fever_recent"] = df_input["fever_recent"].astype(float)
    df["cough_recent"] = df_input["cough_recent"].astype(float)

    # 5. Maternal Characteristics
    df["mother_age_years"] = df_input["v012"].astype(float)
    df["mother_age_first_birth"] = df_input["v212"].astype(float)
    df["mother_education_level"] = df_input["v106"].astype(float)
    df["mother_bmi"] = df_input["v445_clean"].astype(float)
    df["mother_bmi_missing"] = np.where(df_input["v445_clean"].isna(), 1, 0)
    df["total_children_born"] = df_input["v201"].astype(float)
    df["anc_visits_count"] = df_input["m14_clean"].astype(float)
    df["anc_visits_missing"] = np.where(df_input["m14_clean"].isna(), 1, 0)

    # 6. Household Socioeconomic
    df["wealth_quintile"] = df_input["v190"].astype(float)
    # Residence: 1=Urban, 2=Rural in DHS -> 1=Rural, 0=Urban
    df["is_rural"] = np.where(df_input["v025"] == 2, 1, 0)
    
    # Social Group / Caste (1=SC, 2=ST, 3=OBC, 4=None/General)
    def categorize_caste(c):
        if pd.isna(c):
            return "Missing_Caste"
        c = int(c)
        if c == 1:
            return "Scheduled_Caste"
        elif c == 2:
            return "Scheduled_Tribe"
        elif c == 3:
            return "OBC"
        elif c == 4:
            return "General_None"
        else:
            return "Other"
    df["caste_category"] = df_input["caste_clean"].apply(categorize_caste)

    # Religion: 1=Hindu, 2=Muslim, 3=Christian, 4=Sikh, 5+=Other
    def categorize_religion(r):
        if pd.isna(r):
            return "Other"
        r = int(r)
        if r == 1:
            return "Hindu"
        elif r == 2:
            return "Muslim"
        elif r == 3:
            return "Christian"
        elif r == 4:
            return "Sikh"
        else:
            return "Other"
    df["religion_category"] = df_input["v130"].apply(categorize_religion)

    # 7. Household Environment / WASH & Energy
    # Drinking Water Source (WHO/UNICEF JMP categories)
    def categorize_water(w):
        if pd.isna(w):
            return "Unimproved_Surface"
        w = int(w)
        if 11 <= w <= 14:
            return "Improved_Piped"
        elif w in [21, 31, 71]:
            return "Improved_Groundwater_Bottled"
        else:
            return "Unimproved_Surface_Other"
    df["drinking_water_type"] = df_input["v113"].apply(categorize_water)

    # Toilet Facility (WHO/UNICEF JMP categories)
    def categorize_toilet(t):
        if pd.isna(t):
            return "Open_Defecation_None"
        t = int(t)
        if t in [11, 12, 14]:
            return "Flush_Toilet"
        elif t in [13, 21, 22, 23, 41, 44]:
            return "Pit_Latrine"
        elif t in [31, 96, 97]:
            return "Open_Defecation_None"
        else:
            return "Other_Facility"
    df["sanitation_facility_type"] = df_input["v116"].apply(categorize_toilet)

    # Electricity
    df["has_electricity"] = np.where(df_input["v119"] == 1, 1, 0)

    # Clean Cooking Fuel: 1=Electricity, 2=LPG/Natural Gas, 3=Biogas vs Biomass/Solid
    df["clean_cooking_fuel"] = np.where(df_input["v161"].isin([1, 2, 3]), 1, 0)
    df["household_size"] = df_input["v136"].astype(float)
    df["household_head_female"] = np.where(df_input["v151"] == 2, 1, 0)

    # 8. Geographic (State)
    df["state_id"] = df_input["v024"].astype(int)

    return df


def validate_scenario_a_feature_matrix(df_features):
    """Enforces strict assertions verifying zero target leakage and zero survey contamination."""
    feature_cols = set(df_features.columns)

    # Check for direct anthropometric leakage
    leaked_anthro = feature_cols.intersection(PROHIBITED_LEAKAGE_VARS)
    assert len(leaked_anthro) == 0, f"FATAL DATA LEAKAGE: Anthropometric variables found in features: {leaked_anthro}"

    # Check for target outcome columns
    leaked_targets = feature_cols.intersection(PROHIBITED_TARGET_VARS)
    assert len(leaked_targets) == 0, f"FATAL DATA LEAKAGE: Target outcome columns found in features: {leaked_targets}"

    # Check for survey design identifiers and weights
    leaked_survey = feature_cols.intersection(PROHIBITED_SURVEY_VARS)
    assert len(leaked_survey) == 0, f"SURVEY DESIGN CONTAMINATION: Survey identifiers/weights found in features: {leaked_survey}"

    # Check for duplicate column names
    assert len(feature_cols) == len(df_features.columns), "Duplicate column names detected in feature matrix!"

    print("[OK] Scenario A Feature Matrix Validation Passed: 0 leakage, 0 target columns, 0 survey identifiers.")
    return True


def build_and_export_feature_metadata():
    """Generates and persists the structured feature metadata registry."""
    print(f"[*] Step 7: Loading target-augmented dataset from {INPUT_TARGETS_DATA_PATH}...")
    df_raw = pd.read_csv(INPUT_TARGETS_DATA_PATH)
    total_records = len(df_raw)
    print(f"    Loaded {total_records:,} records.")

    # Generate transformed features
    df_features = transform_candidate_features(df_raw)
    validate_scenario_a_feature_matrix(df_features)

    inventory = get_feature_inventory()
    excluded_registry = get_excluded_variables_registry()

    # Calculate exact missingness percentages for candidate features
    feature_stats = {}
    for col in df_features.columns:
        n_missing = int(df_features[col].isna().sum())
        n_valid = int(df_features[col].notna().sum())
        pct_missing = round((n_missing / total_records) * 100, 2)
        dtype_str = str(df_features[col].dtype)
        n_unique = int(df_features[col].nunique())
        feature_stats[col] = {
            "valid_count": n_valid,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "data_type": dtype_str,
            "unique_values_count": n_unique
        }

    domain_counts = {}
    for f_name, f_info in inventory.items():
        dom = f_info["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    feature_metadata = {
        "metadata_version": "1.0.0",
        "step": "Step 7 — Feature Engineering",
        "scenario": "Scenario A — Community Pre-Screening",
        "total_records": total_records,
        "total_candidate_features_in_matrix": len(df_features.columns),
        "total_inventory_features": len(inventory),
        "total_excluded_variables": len(excluded_registry),
        "domain_distribution": domain_counts,
        "feature_statistics": feature_stats,
        "candidate_feature_inventory": inventory,
        "excluded_variables_registry": excluded_registry,
        "missing_value_strategy": {
            "core_principle": "No global dropna; principled imputation strictly fitted on training split later.",
            "structural_firstborn": "is_firstborn flag preserves 100% of sample without dropping firstborn children; preceding_birth_interval_months imputed with training median.",
            "unweighed_births": "birth_weight_missing flag preserves 100% of sample without dropping home births; birth_weight_kg imputed with training median.",
            "unmeasured_maternal_bmi": "mother_bmi_missing flag preserves 100% of sample; mother_bmi imputed with training median.",
            "unrecorded_anc": "anc_visits_missing flag preserves 100% of sample; anc_visits_count imputed with training median.",
            "categorical_missingness": "Explicit Missing category for caste_category, mode imputation for binary morbidity symptoms."
        },
        "encoding_strategy": {
            "binary_features": "0/1 integer indicator (child_sex_male, is_rural, has_electricity, clean_cooking_fuel, etc.)",
            "ordinal_features": "Integer rank encoding (wealth_quintile: 1-5, mother_education_level: 0-3, birth_size_ordinal: 1-5)",
            "nominal_features": "Nominal categorical variables will initially be represented using one-hot encoding within a leakage-safe preprocessing pipeline. Alternative encodings may be evaluated after train/validation/test splitting and must be fitted exclusively on training data.",
            "high_cardinality_district": "sdist (707 districts) excluded from baseline features to prevent severe overfitting/memorization."
        },
        "scaling_strategy": {
            "tree_models": "No scaling required for Random Forest, XGBoost, LightGBM.",
            "linear_models": "StandardScaler or RobustScaler fitted strictly on training split for Logistic Regression baseline."
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_FEATURE_METADATA_PATH), exist_ok=True)
    with open(OUTPUT_FEATURE_METADATA_PATH, "w") as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"[+] Feature metadata successfully saved to {OUTPUT_FEATURE_METADATA_PATH}")
    print(f"[+] Candidate features in matrix: {len(df_features.columns)}")
    print(f"[+] Excluded variables documented: {len(excluded_registry)}")
    return feature_metadata


def transform_candidate_features_v2(df_input: pd.DataFrame) -> pd.DataFrame:
    """Deterministic feature transformation for Scenario A v2 (30 predictor features).

    Excludes the four socially sensitive attributes:
    - household_head_female
    - caste_category
    - religion_category
    - mother_education_level

    Returns exactly 30 candidate features in SCENARIO_A_V2_FEATURE_NAMES order.
    Does NOT fit scalers, imputers, or encoders (which must be fitted strictly on train split).
    """
    df_v1 = transform_candidate_features(df_input)
    df_v2 = df_v1[SCENARIO_A_V2_FEATURE_NAMES].copy()
    return df_v2


def validate_scenario_a_v2_feature_matrix(df_features: pd.DataFrame) -> bool:
    """Enforces strict assertions verifying zero target leakage, zero survey contamination,
    and the strict absence of the 4 removed sensitive features for Scenario A v2.
    """
    feature_cols = set(df_features.columns)

    # Check for direct anthropometric leakage
    leaked_anthro = feature_cols.intersection(PROHIBITED_LEAKAGE_VARS)
    assert len(leaked_anthro) == 0, f"FATAL DATA LEAKAGE: Anthropometric variables found in features: {leaked_anthro}"

    # Check for target outcome columns
    leaked_targets = feature_cols.intersection(PROHIBITED_TARGET_VARS)
    assert len(leaked_targets) == 0, f"FATAL DATA LEAKAGE: Target outcome columns found in features: {leaked_targets}"

    # Check for survey design identifiers and weights
    leaked_survey = feature_cols.intersection(PROHIBITED_SURVEY_VARS)
    assert len(leaked_survey) == 0, f"SURVEY DESIGN CONTAMINATION: Survey identifiers/weights found in features: {leaked_survey}"

    # Check for removed sensitive features
    leaked_sensitive = feature_cols.intersection(set(REMOVED_SENSITIVE_FEATURES_V2))
    assert len(leaked_sensitive) == 0, f"SENSITIVE FEATURE CONTAMINATION: Removed sensitive attributes found: {leaked_sensitive}"

    # Verify exactly 30 columns
    assert len(df_features.columns) == 30, f"Feature count mismatch: Expected exactly 30 features, got {len(df_features.columns)}"
    assert list(df_features.columns) == SCENARIO_A_V2_FEATURE_NAMES, "Feature column ordering/naming mismatch for Scenario A v2!"

    print("[OK] Scenario A v2 Feature Matrix Validation Passed: exactly 30 features, 0 leakage, 0 sensitive attributes, 0 survey identifiers.")
    return True


def build_and_export_feature_metadata_v2() -> dict:
    """Generates and persists the structured feature metadata registry for Scenario A v2 (30 features)."""
    print(f"[*] Step 7 v2: Loading target-augmented dataset from {INPUT_TARGETS_DATA_PATH}...")
    df_raw = pd.read_csv(INPUT_TARGETS_DATA_PATH)
    total_records = len(df_raw)
    print(f"    Loaded {total_records:,} records.")

    # Generate transformed v2 features
    df_features = transform_candidate_features_v2(df_raw)
    validate_scenario_a_v2_feature_matrix(df_features)

    inventory_all = get_feature_inventory()
    # Filter inventory for v2 (removing sensitive features: household_head_female, caste_category, religion, mother_education_level)
    inventory_v2 = {
        k: v for k, v in inventory_all.items()
        if k not in REMOVED_SENSITIVE_FEATURES_V2 and k != "religion"
    }

    excluded_registry = get_excluded_variables_registry()
    excluded_registry_v2 = dict(excluded_registry)
    for feat in REMOVED_SENSITIVE_FEATURES_V2:
        excluded_registry_v2[feat] = {
            "domain": "Socially Sensitive Attributes",
            "reason": (
                "The final Scenario-A feature set excludes selected socially sensitive household "
                "and demographic attributes to reduce reliance on sensitive characteristics in "
                "the screening model and simplify community pre-screening input requirements. "
                "Note that removing sensitive predictors does not by itself establish fairness "
                "or absence of bias."
            )
        }

    # Calculate exact missingness percentages for candidate features
    feature_stats = {}
    for col in df_features.columns:
        n_missing = int(df_features[col].isna().sum())
        n_valid = int(df_features[col].notna().sum())
        pct_missing = round((n_missing / total_records) * 100, 2)
        dtype_str = str(df_features[col].dtype)
        n_unique = int(df_features[col].nunique())
        feature_stats[col] = {
            "valid_count": n_valid,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "data_type": dtype_str,
            "unique_values_count": n_unique
        }

    categorical_features_v2 = [
        "child_age_group",
        "delivery_place_type",
        "drinking_water_type",
        "sanitation_facility_type",
        "state_id"
    ]
    numerical_features_v2 = [c for c in df_features.columns if c not in categorical_features_v2]

    domain_counts = {}
    for f_name, f_info in inventory_v2.items():
        dom = f_info["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    feature_metadata_v2 = {
        "metadata_version": "2.0.0",
        "schema_version": "scenario-a-30-v2",
        "step": "Feature Engineering (30-Feature Scenario-A)",
        "scenario": "Scenario A — Community Pre-Screening",
        "rationale_sensitive_exclusion": (
            "The final Scenario-A feature set excludes selected socially sensitive household "
            "and demographic attributes to reduce reliance on sensitive characteristics in "
            "the screening model and simplify community pre-screening input requirements. "
            "Removing sensitive predictors does not by itself establish fairness or absence of bias."
        ),
        "removed_features": REMOVED_SENSITIVE_FEATURES_V2,
        "total_records": total_records,
        "total_candidate_features_in_matrix": len(df_features.columns),
        "features": list(df_features.columns),
        "categorical_features": categorical_features_v2,
        "numerical_features": numerical_features_v2,
        "domain_distribution": domain_counts,
        "feature_statistics": feature_stats,
        "candidate_feature_inventory": inventory_v2,
        "excluded_variables_registry": excluded_registry_v2,
        "missing_value_strategy": {
            "core_principle": "No global dropna; principled imputation strictly fitted on training split later.",
            "structural_firstborn": "is_firstborn flag preserves 100% of sample without dropping firstborn children; preceding_birth_interval_months imputed with training median.",
            "unweighed_births": "birth_weight_missing flag preserves 100% of sample without dropping home births; birth_weight_kg imputed with training median.",
            "unmeasured_maternal_bmi": "mother_bmi_missing flag preserves 100% of sample; mother_bmi imputed with training median.",
            "unrecorded_anc": "anc_visits_missing flag preserves 100% of sample; anc_visits_count imputed with training median.",
            "categorical_missingness": "Mode imputation for binary morbidity symptoms."
        },
        "encoding_strategy": {
            "binary_features": "0/1 integer indicator (child_sex_male, is_rural, has_electricity, clean_cooking_fuel, etc.)",
            "ordinal_features": "Integer rank encoding (wealth_quintile: 1-5, birth_size_ordinal: 1-5)",
            "nominal_features": "One-hot encoding for nominal categories (5 categorical features in v2).",
            "high_cardinality_district": "sdist (707 districts) excluded from baseline features to prevent severe overfitting/memorization."
        },
        "scaling_strategy": {
            "tree_models": "No scaling required for Random Forest, XGBoost, LightGBM, CatBoost.",
            "linear_models": "StandardScaler fitted strictly on training split for Logistic Regression baseline."
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_FEATURE_METADATA_V2_PATH), exist_ok=True)
    with open(OUTPUT_FEATURE_METADATA_V2_PATH, "w") as f:
        json.dump(feature_metadata_v2, f, indent=2)
    print(f"[+] Feature metadata v2 successfully saved to {OUTPUT_FEATURE_METADATA_V2_PATH}")
    print(f"[+] Candidate features in matrix: {len(df_features.columns)}")
    print(f"[+] Excluded variables documented: {len(excluded_registry_v2)}")
    return feature_metadata_v2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline for Scenario A.")
    parser.add_argument("--v2", action="store_true", help="Generate v2 (30-feature) metadata.")
    parser.add_argument("--all", action="store_true", help="Generate both v1 and v2 metadata.")
    args = parser.parse_args()

    if args.v2:
        build_and_export_feature_metadata_v2()
    elif args.all:
        build_and_export_feature_metadata()
        build_and_export_feature_metadata_v2()
    else:
        # Default behavior: run both to ensure v1 is maintained and v2 is produced
        build_and_export_feature_metadata()
        build_and_export_feature_metadata_v2()
