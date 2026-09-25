# Step 22: Scenario-A 30-Feature Transition and Model Retraining Plan

## 1. Overview & Objectives

In accordance with responsible design considerations and community pre-screening operational feasibility, the Scenario-A feature set is updated from the initial 34-predictor configuration (`nutrisense-scenario-a-v1.0.0`) to a 30-predictor configuration (`nutrisense-scenario-a-v2.0.0`).

### Sensitive Predictor Removal Rationale
The final Scenario-A feature set excludes selected socially sensitive household and demographic attributes to reduce reliance on sensitive characteristics in the screening model and simplify community pre-screening input requirements.

> **Responsible ML Note:**
> Removing sensitive predictors does not by itself establish fairness or absence of bias. Model performance, error distributions, and parity across demographic subgroups will be monitored during validation and error analysis.

---

## 2. Removed Predictor Features

Exactly four socially sensitive attributes have been excluded from the candidate predictor set:

| # | Removed Predictor Feature | DHS Source Column | Original Domain | Rationale |
|---|---------------------------|-------------------|-----------------|-----------|
| 1 | `household_head_female` | `v151` | Household Environment | Socially sensitive demographic attribute; excluded to reduce reliance on household structure. |
| 2 | `caste_category` | `caste_clean` (`s116`) | Household Socioeconomic | Socially sensitive demographic attribute; excluded to avoid screening dependence on social group. |
| 3 | `religion_category` | `v130` | Household Socioeconomic | Socially sensitive demographic attribute; excluded to avoid screening dependence on religious affiliation. |
| 4 | `mother_education_level` | `v106` | Maternal Characteristics | Socially sensitive socio-demographic attribute; excluded to ensure community pre-screening focuses on actionable biological and environmental risk markers. |

---

## 3. Retained 30-Feature Specification

The remaining 30 predictor features represent the approved Scenario-A feature space:

### Categorical Features (5 features — One-Hot Encoded)
1. `child_age_group` (00_05_mo, 06_11_mo, 12_23_mo, 24_35_mo, 36_47_mo, 48_59_mo)
2. `delivery_place_type` (Home_Delivery, Public_Facility, Private_Facility, Other)
3. `drinking_water_type` (Improved_Piped, Improved_Groundwater_Bottled, Unimproved_Surface_Other)
4. `sanitation_facility_type` (Flush_Toilet, Pit_Latrine, Open_Defecation_None, Other_Facility)
5. `state_id` (36 administrative States/UTs of India)

### Numerical, Ordinal, and Binary Features (25 features)
6. `child_age_months` (Numerical, 0–59 months)
7. `child_sex_male` (Binary, 1=Male, 0=Female)
8. `birth_order` (Numerical count)
9. `is_multiple_birth` (Binary, 1=Twin/Multiple, 0=Single)
10. `is_firstborn` (Binary, 1=Firstborn, 0=Has older sibling)
11. `preceding_birth_interval_months` (Numerical, median imputed on train split)
12. `birth_size_ordinal` (Ordinal, 1=Very large to 5=Very small)
13. `birth_weight_kg` (Numerical, 0.5–6.0 kg, median imputed on train split)
14. `birth_weight_missing` (Binary indicator, 1=Missing, 0=Recorded)
15. `still_breastfeeding` (Binary, 1=Yes, 0=No/Weaned)
16. `diarrhea_recent` (Binary, acute diarrhea past 2 weeks)
17. `fever_recent` (Binary, fever past 2 weeks)
18. `cough_recent` (Binary, respiratory symptoms past 2 weeks)
19. `mother_age_years` (Numerical, 15–49 years)
20. `mother_age_first_birth` (Numerical, completed years)
21. `mother_bmi` (Numerical, maternal BMI kg/m², median imputed on train split)
22. `mother_bmi_missing` (Binary indicator, 1=Missing, 0=Recorded)
23. `total_children_born` (Numerical parity count)
24. `anc_visits_count` (Numerical, antenatal care visits count)
25. `anc_visits_missing` (Binary indicator, 1=Missing, 0=Recorded)
26. `wealth_quintile` (Ordinal, 1=Poorest to 5=Richest)
27. `is_rural` (Binary, 1=Rural, 0=Urban)
28. `has_electricity` (Binary, 1=Yes, 0=No)
29. `clean_cooking_fuel` (Binary, 1=Clean fuel, 0=Biomass/Solid)
30. `household_size` (Numerical count of usual household members)

---

## 4. Pipeline Execution & Verification Matrix

| Pipeline Stage | Action | Status | Notes / Artefacts |
|---|---|---|---|
| Raw Data Verification | Reused | Verified | `IAKR7EDT/IAKR7EFL.DTA` immutability verified (441,380,745 bytes). |
| Data Cleaning & Target Extraction | Reused | Complete | Cohort $N=221{,}263$; Stunting HAZ < -2, Underweight WAZ < -2, Wasting WHZ < -2. |
| Split Manifest | Reused | Complete | Household-grouped split (`data/interim/split_manifest.csv.gz`) guarantees zero household leakage and identical comparative partitions. |
| Feature Engineering (v2) | Rerun | **Complete** | Exactly 30 features generated; `data/interim/feature_metadata_v2.json` exported. |
| Baseline Models (v2) | Rerun | **Complete** | Logistic Regression retrained for 3 targets; artifacts saved to `models/v2/`. |
| Candidate Models (v2) | Rerun | **Complete** | 30 experiments completed (5 models x 2 conditions x 3 targets); artifacts saved to `models/v2/`. |
| Validation Threshold Analysis (v2) | Rerun | **Complete** | 99-threshold sweep on validation set; validation-derived operating thresholds determined. |
| Champion Selection (v2) | Rerun | **Complete** | Pre-specified 5-gate protocol executed strictly on validation partition. |
| Locked Test Evaluation (v2) | Rerun | **Complete** | Single unblinded test evaluation on 33,069 test records. |
| SHAP Analysis (v2) | Rerun | **Complete** | TreeSHAP on 30 features with one-hot aggregation. |
| Model Registry (v2) | Rerun | **Complete** | Dual registration of v1.0.0 and v2.0.0 in `models/model_registry.json`. |
| API & Inference Engine (v2) | Rerun | **Complete** | Updated `src/models/inference_pipeline.py`, `src/api/schemas.py`, `src/api/services/screening_service.py`. Strict 30-feature schema enforcement & 4 sensitive features rejected with 422. |
| Frontend React Updates | Pending | Queued | Update form inputs to 30 features. |

---

## 12. Inference Pipeline & Backend FastAPI Updates (Step 8)

The backend inference engine and FastAPI web services were updated to strictly enforce the 30-feature `nutrisense-scenario-a-v2.0.0` model specification and operating thresholds:

### Key Service & Inference Changes:
1. **Inference Pipeline (`src/models/inference_pipeline.py`)**:
   - Model artifact paths updated to `models/v2/model_comparison_lightgbm_{target}_unweighted_v2.joblib`.
   - Thresholds locked to validation-derived operating values: $\tau_{\text{stunting}} = 0.36$, $\tau_{\text{underweight}} = 0.30$, $\tau_{\text{wasting}} = 0.17$.
   - Feature validation updated to strictly require `APPROVED_30_FEATURES`.
   - Explicit rejection of all prohibited anthropometric leakage fields AND the 4 removed sensitive social attributes (`household_head_female`, `caste_category`, `religion_category`, `mother_education_level`).
   - Standard output metadata returns `model_version: "nutrisense-scenario-a-v2.0.0"`, `feature_schema_version: "scenario-a-30-v2"`, `feature_count: 30`.

2. **Pydantic API Schemas (`src/api/schemas.py`)**:
   - Removed input fields: `mother_education_level`, `caste_category`, `religion_category`, `household_head_female`.
   - Enforces `ConfigDict(extra="forbid")`: supplying any extra field or any of the 4 removed sensitive fields triggers an automatic HTTP 422 `INVALID_INPUT` error.
   - Response models return `model_version = "nutrisense-scenario-a-v2.0.0"` and `feature_schema_version = "scenario-a-30-v2"`.

3. **Screening Service Layer (`src/api/services/screening_service.py`)**:
   - Initializes and cryptographically audits models using `load_registry_v2()`, `verify_all_models_integrity_v2()`, and `load_all_registered_pipelines_v2()`.

4. **Automated Verification**:
   - `tests/test_inference_pipeline.py`: Updated for v2 30-feature schema and sensitive feature rejection.
   - `tests/test_api.py`: Updated for v2 endpoints, schema validation, and sensitive field HTTP 422 rejection.
   - Full pytest suite: **280 passed, 0 failed**.

---

## 13. Verification Test Results (Steps 1–8)

Automated tests executed via pytest:
- `tests/test_feature_engineering.py`: 10 passed
- `tests/test_feature_engineering_v2.py`: 10 passed
- `tests/test_baseline_model.py`: 7 passed
- `tests/test_baseline_model_v2.py`: 8 passed
- `tests/test_model_comparison.py`: 8 passed
- `tests/test_model_comparison_v2.py`: 7 passed
- `tests/test_threshold_analysis.py`: 8 passed
- `tests/test_threshold_analysis_v2.py`: 6 passed
- `tests/test_final_test_evaluation.py`: 19 passed
- `tests/test_final_test_evaluation_v2.py`: 20 passed
- `tests/test_shap_analysis.py`: 16 passed
- `tests/test_shap_analysis_v2.py`: 17 passed
- `tests/test_model_registry.py`: 21 passed
- `tests/test_model_registry_v2.py`: 21 passed
- `tests/test_inference_pipeline.py`: 17 passed
- `tests/test_api.py`: 20 passed
- Other regression tests: 65 passed
- **Total across entire test suite: 280 passed, 0 failed**.

---

## 14. Canonical v1.0.0 Promotion & Historical Archiving

Following comprehensive verification of the 30-feature model:
1. **Canonical Active Version**:
   - The 30-feature model was formally promoted from temporary development designation (`v2.0.0`) to the **Canonical Active Production Version**: `nutrisense-scenario-a-v1.0.0` (Feature Schema: `scenario-a-30-v1`).
   - Production champion model artifacts are located in `models/v1/`:
     - `models/v1/model_comparison_lightgbm_stunting_unweighted_v1.joblib`
     - `models/v1/model_comparison_lightgbm_underweight_unweighted_v1.joblib`
     - `models/v1/model_comparison_lightgbm_wasting_unweighted_v1.joblib`
   - Active thresholds are locked to validation-derived values: Stunting = 0.36, Underweight = 0.30, Wasting = 0.17.
2. **Historical Legacy Archiving**:
   - The original 34-feature model was permanently preserved as historical research material in `archive/scenario-a-34-feature/` under legacy identifier `nutrisense-scenario-a-legacy-34f` (schema `scenario-a-34-legacy`).
   - All active inference, API routes, and client applications exclusively query the 30-feature `v1.0.0` pipeline.

