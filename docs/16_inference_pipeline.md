# Step 16 — Final End-to-End Inference Pipeline

## 1. Purpose

The purpose of Step 16 is to package the pre-specified, champion machine learning models developed and evaluated across Steps 0–15 into a production-ready, reproducible, and leakage-safe inference pipeline (`predict_child_screening`).

In the **Scenario A — Community Pre-Screening** setting, frontline community health workers (e.g., ASHAs or Anganwadi workers in India) require a rapid, non-invasive risk intelligence tool to identify children at elevated risk of undernutrition without requiring physical anthropometric equipment (scales or stadiometers).

> [!IMPORTANT]
> The inference pipeline reproduces the transformations learned during model training by loading the existing fitted pipelines. It does not retrain or refit preprocessing components.

---

## 2. Architecture

The inference pipeline follows a decoupled, modular architecture designed for local or edge deployment:

```
                      [User / Application Input]
                                  │
                                  ▼
                   [Input Validation Layer]
           ├── Prohibited Variables Gate (hw70-hw73, hw2, hw3, hw13, hw57)
           ├── Type & Range Verification (child age, birth order, maternal age)
           └── Categorical & Cardinality Mapping (WASH, caste, religion, state)
                                  │
                                  ▼
                [34 Scenario-A Feature Assembler]
           ├── Automatic Age Bracket Categorization
           ├── Missing Indicator Consistency (birth weight, maternal BMI, ANC)
           └── 1-row Feature DataFrame Construction
                                  │
                                  ▼
             [Fitted Preprocessing & Estimator Pipelines]
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
 [Stunting Pipeline]    [Underweight Pipeline]     [Wasting Pipeline]
 (ColumnTransformer +     (ColumnTransformer +      (ColumnTransformer +
   LGBMClassifier)          LGBMClassifier)           LGBMClassifier)
         │                        │                        │
         ▼                        ▼                        ▼
  P(Stunting >= 0.35)    P(Underweight >= 0.31)    P(Wasting >= 0.17)
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                    [Structured Decision Output]
       {
         "valid": True,
         "stunting": {"probability": ..., "threshold": 0.35, "screen_positive": ...},
         "underweight": {"probability": ..., "threshold": 0.31, "screen_positive": ...},
         "wasting": {"probability": ..., "threshold": 0.17, "screen_positive": ...}
       }
```

---

## 3. Input Schema

The pipeline entrypoint `predict_child_screening` accepts:
1. A Python `dict` representing an individual child record.
2. A `list` of `dict` objects for batch processing.
3. A `pandas.DataFrame`.

---

## 4. The 34 Approved Scenario-A Features

The inference pipeline evaluates the exact 34 non-leaking predictors approved in Step 7:

| # | Feature Name | Source | Domain | Type | Allowed Values / Ranges |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | `child_age_months` | `hw1` | A. Demographics | Numeric | $0$ to $59$ completed months |
| 2 | `child_age_group` | Derived | A. Demographics | Categorical | `00_05_mo`, `06_11_mo`, `12_23_mo`, `24_35_mo`, `36_47_mo`, `48_59_mo` |
| 3 | `child_sex_male` | `b4` | A. Demographics | Binary | $1 = \text{Male}$, $0 = \text{Female}$ |
| 4 | `birth_order` | `bord` | A. Demographics | Numeric | Integer $\ge 1$ |
| 5 | `is_multiple_birth` | `b0` | A. Demographics | Binary | $1 = \text{Twin/Triplet}$, $0 = \text{Single}$ |
| 6 | `is_firstborn` | Derived | A. Demographics | Binary | $1 = \text{Firstborn child}$, $0 = \text{Later born}$ |
| 7 | `preceding_birth_interval_months` | `b11` | A. Demographics | Numeric | $\ge 0$ months (or NaN if firstborn) |
| 8 | `birth_size_ordinal` | `m18` | B. Birth Profile | Ordinal | $1 = \text{Very Large}$ to $5 = \text{Very Small}$ (or NaN) |
| 9 | `birth_weight_kg` | Derived | B. Birth Profile | Numeric | $0.5$ to $7.0$ kg (or NaN) |
| 10 | `birth_weight_missing` | Derived | B. Birth Profile | Binary | $1 = \text{Missing}$, $0 = \text{Recorded}$ |
| 11 | `delivery_place_type` | `m15` | B. Birth Profile | Categorical | `Home_Delivery`, `Public_Facility`, `Private_Facility`, `Other` |
| 12 | `still_breastfeeding` | `m4` | C. Feeding | Binary | $1 = \text{Yes}$, $0 = \text{No}$ |
| 13 | `diarrhea_recent` | `h11` | D. Health | Binary | $1 = \text{Yes (past 2 weeks)}$, $0 = \text{No}$ |
| 14 | `fever_recent` | `h22` | D. Health | Binary | $1 = \text{Yes (past 2 weeks)}$, $0 = \text{No}$ |
| 15 | `cough_recent` | `h31` | D. Health | Binary | $1 = \text{Yes (past 2 weeks)}$, $0 = \text{No}$ |
| 16 | `mother_age_years` | `v012` | E. Maternal | Numeric | $12$ to $55$ years |
| 17 | `mother_age_first_birth` | `v212` | E. Maternal | Numeric | $10$ to $50$ years |
| 18 | `mother_education_level` | `v106` | E. Maternal | Ordinal | $0 = \text{None}, 1 = \text{Primary}, 2 = \text{Secondary}, 3 = \text{Higher}$ |
| 19 | `mother_bmi` | `v445` | E. Maternal | Numeric | $10.0$ to $60.0$ $\text{kg/m}^2$ (or NaN) |
| 20 | `mother_bmi_missing` | Derived | E. Maternal | Binary | $1 = \text{Missing}$, $0 = \text{Recorded}$ |
| 21 | `total_children_born` | `v201` | E. Maternal | Numeric | Integer $\ge 1$ |
| 22 | `anc_visits_count` | `m14` | E. Maternal | Numeric | $\ge 0$ visits (or NaN) |
| 23 | `anc_visits_missing` | Derived | E. Maternal | Binary | $1 = \text{Missing}$, $0 = \text{Recorded}$ |
| 24 | `wealth_quintile` | `v190` | F. Socioeconomic | Ordinal | $1 = \text{Poorest}, 2 = \text{Poorer}, 3 = \text{Middle}, 4 = \text{Richer}, 5 = \text{Richest}$ |
| 25 | `is_rural` | `v025` | F. Socioeconomic | Binary | $1 = \text{Rural}, 0 = \text{Urban}$ |
| 26 | `caste_category` | `s116` | F. Socioeconomic | Categorical | `Scheduled_Caste`, `Scheduled_Tribe`, `OBC`, `General_None`, `Other`, `Missing_Caste` |
| 27 | `religion_category` | `v130` | F. Socioeconomic | Categorical | `Hindu`, `Muslim`, `Christian`, `Sikh`, `Other` |
| 28 | `drinking_water_type` | `v113` | G. Environment | Categorical | `Improved_Piped`, `Improved_Groundwater_Bottled`, `Unimproved_Surface_Other`, `Unimproved_Surface` |
| 29 | `sanitation_facility_type` | `v116` | G. Environment | Categorical | `Flush_Toilet`, `Pit_Latrine`, `Open_Defecation_None`, `Other_Facility` |
| 30 | `has_electricity` | `v119` | G. Environment | Binary | $1 = \text{Yes}, 0 = \text{No}$ |
| 31 | `clean_cooking_fuel` | `v161` | G. Environment | Binary | $1 = \text{Yes [LPG/Electricity/Biogas]}, 0 = \text{Solid Biomass/Kerosene}$ |
| 32 | `household_size` | `v136` | G. Environment | Numeric | Integer $\ge 1$ |
| 33 | `household_head_female` | `v151` | G. Environment | Binary | $1 = \text{Female}, 0 = \text{Male}$ |
| 34 | `state_id` | `v024` | H. Geographic | Nominal | Integer administrative code $1$ to $36$ |

---

## 5. Input Validation

The validation layer (`validate_single_record`) strictly checks:
1. **Target and Anthropometric Leakage**: Explicitly rejects `hw70`, `hw71`, `hw72`, `hw73`, `hw2`, `hw3`, `hw4`–`hw12`, `hw13`, `hw57`, and survey cluster weights (`v001`, `v005`, etc.).
2. **Type Safety**: Enforces proper numeric conversions; rejects strings in numeric fields.
3. **Biological & Plausible Range Boundaries**:
   - `child_age_months`: $0 \le \text{age} \le 59$.
   - `mother_age_years`: $12 \le \text{age} \le 55$.
   - `birth_weight_kg`: $0.5 \le \text{wt} \le 7.0$ (when present).
   - `mother_bmi`: $10.0 \le \text{BMI} \le 60.0$ (when present).
4. **Logical Consistency**:
   - `mother_age_first_birth` cannot exceed `mother_age_years`.
   - `birth_order` cannot exceed `total_children_born`.

---

## 6. Missing-Value Handling

The pipeline replicates the missing-data handling strategy established in Step 7 and Step 10:
- **Missing Value Indicators**: Continuous variables subject to non-random non-response (`birth_weight_kg`, `mother_bmi`, `anc_visits_count`) are automatically paired with their corresponding indicator flags (`birth_weight_missing`, `mother_bmi_missing`, `anc_visits_missing`).
- **In-Pipeline Imputation**: The loaded `ColumnTransformer` handles numerical imputation via the training-fitted `SimpleImputer(strategy='median')` and categorical imputation via `SimpleImputer(strategy='most_frequent')`. No global `dropna` is applied, and no external imputers are refitted.

---

## 7. Model Loading

Models are loaded from disk via `joblib.load()` and cached in memory:
- **Stunting**: `models/model_comparison_lightgbm_stunting_unweighted.joblib`
- **Underweight**: `models/model_comparison_lightgbm_underweight_unweighted.joblib`
- **Wasting**: `models/model_comparison_lightgbm_wasting_unweighted.joblib`

On first invocation, `load_model_pipelines()` asserts:
1. Existence of `named_steps['preprocessor']` (`ColumnTransformer`).
2. Existence of `named_steps['classifier']` (`LGBMClassifier`).
3. Correct feature partition: 27 numeric and 7 categorical features.

---

## 8. Preprocessing

The preprocessing stage executes the fitted transformations inside the scikit-learn pipeline:
- **Numerical Features (27)**: Median imputation followed by standard scaling (`StandardScaler`).
- **Categorical Features (7)**: Most frequent imputation followed by one-hot encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
- **Nominal State Encoding**: `state_id` is processed as a categorical column and one-hot encoded, ensuring nominal administrative treatment without imposing artificial ordinal rank.

---

## 9. Probability Generation

For each target outcome, class posterior probabilities are computed via:

$$\hat{P}(Y = 1 \mid X) = \text{classifier.predict\_proba}(X)[:, 1]$$

Probabilities are preserved as unrounded double-precision floating-point numbers during threshold evaluation.

---

## 10. Threshold Application

Screening classification applies the validation-derived decision thresholds selected in Step 11/Step 12:

$$\text{screen\_positive} = \mathbb{I}(\hat{P}(Y=1 \mid X) \ge \tau)$$

| Target Outcome | Locked Decision Threshold ($\tau$) | Decision Boundary Rule |
| :--- | :---: | :--- |
| **Stunting** | **0.35** | $\hat{P} \ge 0.35 \implies \text{screen\_positive} = \text{True}$ |
| **Underweight** | **0.31** | $\hat{P} \ge 0.31 \implies \text{screen\_positive} = \text{True}$ |
| **Wasting** | **0.17** | $\hat{P} \ge 0.17 \implies \text{screen\_positive} = \text{True}$ |

> [!NOTE]
> The threshold boundary is strictly inclusive ($\ge$). A predicted probability of exactly 0.35 is classified as `screen_positive = True` for stunting.

---

## 11. Output Schema

The inference function returns a structured dictionary:

```json
{
  "valid": true,
  "predictions": {
    "stunting": {
      "probability": 0.235118,
      "threshold": 0.35,
      "screen_positive": false,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    },
    "underweight": {
      "probability": 0.170036,
      "threshold": 0.31,
      "screen_positive": false,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    },
    "wasting": {
      "probability": 0.128001,
      "threshold": 0.17,
      "screen_positive": false,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    }
  },
  "stunting": { ... },
  "underweight": { ... },
  "wasting": { ... }
}
```

If validation fails, the output structure is:

```json
{
  "valid": false,
  "errors": [
    "child_age_months is required."
  ]
}
```

---

## 12. Privacy and Data Governance

In strict compliance with DHS agreements and patient privacy principles:
- **Zero Microdata Bundling**: The inference module operates completely independently of `IAKR7EFL.DTA`.
- **Zero Identifier Retention**: No child names, household phone numbers, GPS coordinates, or DHS cluster identifiers (`v001`) are accepted, logged, or stored by the pipeline.
- **Minimal Sufficient Input**: The inference pipeline requires only the 34 non-invasive demographic, health, and household covariates.

---

## 13. Model Artifact Integrity

The three serialized model files in `models/` are cryptographically verified via SHA-256 hashes recorded in `data/interim/model_artifact_metadata.json`:

| Target | Model Artifact File | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Stunting** | `model_comparison_lightgbm_stunting_unweighted.joblib` | 372,191 | `3a1a9c0be5d8eb35483857812da2d77441d85ec3d233952c8a650b3e03f5b736` |
| **Underweight** | `model_comparison_lightgbm_underweight_unweighted.joblib` | 371,983 | `2fc16e1a56d0b036398dd6d428ee7231cc09cfdec34c3dcb3cfd104caf2a3eef` |
| **Wasting** | `model_comparison_lightgbm_wasting_unweighted.joblib` | 361,999 | `7900278ce83da2aafacccd8d77a924a82cf0d7b2fdfab20aa37a1ac6bdd7e919` |

---

## 14. Reproducibility

The inference pipeline is fully deterministic:
- Given an identical input dictionary, repeated evaluations return identical probability values and screening classifications.
- Model hyperparameters (e.g., `random_state = 42`) and preprocessing encoders are fixed within the loaded joblib artifacts.

---

## 15. Scientific & Clinical Limitations

1. **Pre-Screening, Not Diagnosis**: The pipeline is designed for risk triage and secondary referral. It does not provide clinical diagnosis. A `screen_positive` flag indicates that a child exceeds the pre-specified risk threshold and warrants direct anthropometric assessment.
2. **Threshold Context**: Thresholds were optimized on the NFHS-5 validation cohort ($N_{val} = 33{,}153$) to maximize F1 and Youden balance. Different community screening contexts with varying clinical resource constraints may require alternative threshold calibration.
3. **Subgroup Heterogeneity**: As demonstrated in Step 15, classifier sensitivity varies across age brackets and socioeconomic quintiles. In higher wealth strata, the models exhibit higher false-negative rates among positive cases.
4. **External Validation Required**: Field deployment requires prospective clinical validation in local primary healthcare centers prior to routine adoption.
5. **Non-Causal Interpretation**: Model feature weights and predictions reflect statistical correlations and do not establish causal etiologies for childhood undernutrition.

---

## 16. Example Valid Input

```python
sample_child = {
    "child_age_months": 18.0,
    "child_sex_male": 1,
    "birth_order": 2.0,
    "is_multiple_birth": 0,
    "preceding_birth_interval_months": 24.0,
    "birth_size_ordinal": 3.0,
    "birth_weight_kg": 2.6,
    "delivery_place_type": "Public_Facility",
    "still_breastfeeding": 1.0,
    "diarrhea_recent": 0.0,
    "fever_recent": 0.0,
    "cough_recent": 0.0,
    "mother_age_years": 24.0,
    "mother_age_first_birth": 21.0,
    "mother_education_level": 2.0,
    "mother_bmi": 19.8,
    "total_children_born": 2.0,
    "anc_visits_count": 4.0,
    "wealth_quintile": 2.0,
    "is_rural": 1,
    "caste_category": "OBC",
    "religion_category": "Hindu",
    "drinking_water_type": "Improved_Piped",
    "sanitation_facility_type": "Pit_Latrine",
    "has_electricity": 1,
    "clean_cooking_fuel": 1,
    "household_size": 5.0,
    "household_head_female": 0,
    "state_id": 10
}
```

---

## 17. Example Output

```python
from src.models.inference_pipeline import predict_child_screening

result = predict_child_screening(sample_child)
print(result)
```

Output:
```json
{
  "valid": true,
  "predictions": {
    "stunting": {
      "probability": 0.312845,
      "threshold": 0.35,
      "screen_positive": false,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    },
    "underweight": {
      "probability": 0.284192,
      "threshold": 0.31,
      "screen_positive": false,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    },
    "wasting": {
      "probability": 0.185204,
      "threshold": 0.17,
      "screen_positive": true,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    }
  },
  "stunting": { ... },
  "underweight": { ... },
  "wasting": { ... }
}
```

---

## 18. Invalid-Input Examples and Responses

### Example A: Prohibited Anthropometric Feature Supplied
```python
bad_child_1 = sample_child.copy()
bad_child_1["hw70"] = -2.4  # Prohibited continuous HAZ
```
*Response*:
```json
{
  "valid": false,
  "errors": [
    "Prohibited variable(s) supplied: ['hw70']. Direct anthropometric measurements and survey design variables are strictly forbidden in Scenario A."
  ]
}
```

### Example B: Missing Required Field
```python
bad_child_2 = sample_child.copy()
del bad_child_2["child_age_months"]
```
*Response*:
```json
{
  "valid": false,
  "errors": [
    "child_age_months is required."
  ]
}
```

### Example C: Impossible Combination
```python
bad_child_3 = sample_child.copy()
bad_child_3["birth_order"] = 5.0
bad_child_3["total_children_born"] = 2.0
```
*Response*:
```json
{
  "valid": false,
  "errors": [
    "birth_order (5.0) cannot exceed total_children_born (2.0)."
  ]
}
```

---

## 19. Tests

Automated testing is maintained in [`tests/test_inference_pipeline.py`](file:///d:/finalyearproj/Nutrisense-Ai/tests/test_inference_pipeline.py) covering 17 test cases:
1. Valid complete input
2. Valid input with approved missing values
3. Missing required field
4. Invalid numeric type
5. Invalid categorical value
6. Prohibited anthropometric feature supplied
7. All three model outputs present
8. Probability in $[0, 1]$
9. Threshold exact match (stunting = 0.35, underweight = 0.31, wasting = 0.17)
10. Unrounded threshold evaluation and boundary condition ($p \ge \tau$)
11. Deterministic idempotency across repeated calls
12. Model files exist on disk
13. Expected pipeline architecture exists
14. 34-feature schema preserved
15. DHS raw file not required
16. SHA-256 model artifact immutability
17. Raw DHS data immutability

---

## 20. Exact Commands to Run

### Execute Unit Test Suite:
```bash
python -m unittest tests/test_inference_pipeline.py -v
```

### Execute Full Pipeline Regression Suite (Steps 0–16):
```bash
python -c "
import unittest
loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
assert result.wasSuccessful(), f'Failed with {len(result.failures)} failures'
print('ALL 17 TEST SUITES PASSED!')
"
```
