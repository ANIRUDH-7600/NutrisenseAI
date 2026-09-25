# Step 17 — Model Packaging, Versioning & Registry

## 1. Purpose

The purpose of Step 17 is to establish a formal, machine-readable, and cryptographically verified Model Registry for the NutriSense AI childhood undernutrition screening pipeline.

As machine learning systems transition from research development to production backend services, explicit versioning and strict artifact integrity are required to guarantee that only the exact, approved model artifacts and decision thresholds are deployed.

> [!IMPORTANT]
> The model registry does not retrain, recalibrate, or modify the approved models. It provides deterministic identification and integrity verification of the already-approved artifacts.

---

## 2. Why Model Versioning Is Needed

1. **Elimination of Silent Drifts**: Without strict cryptographic binding, model files can be inadvertently overwritten, corrupted, or replaced with different training iterations.
2. **Deterministic Reproducibility**: Future backend APIs and edge applications must load verified model weights and decision thresholds without relying on heuristics such as file timestamps or directory wildcards.
3. **Auditability and Governance**: Public health intelligence tools require complete provenance linking research validation milestones directly to the deployed serialized models.
4. **Clinical Safety**: Enforcing locked decision thresholds ($\tau = 0.35, 0.31, 0.17$) directly in the registry prevents arbitrary threshold shifts that could alter screening sensitivity and specificity.

---

## 3. Model Registry Architecture

The registry system separates metadata specification from the verification engine:

```
                    [models/model_registry.json]
                                 │
                                 ▼
                   [src/models/model_registry.py]
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
  [Schema & Version]    [Cryptographic Audit]   [Deterministic Load]
  - model_version        - SHA-256 calculation   - joblib.load()
  - feature_schema       - Bit-for-bit check     - Pipeline validation
  - 34 Scenario-A cols   - Fail-loudly abort     - Threshold binding
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
              [Application / Backend Inference Layer]
```

---

## 4. Model Version

- **Identifier**: `nutrisense-scenario-a-v1.0.0`
- **Scope**: Represents the complete tri-target screening model configuration developed and evaluated across Steps 0–16.
- **Specification**: This version denotes the validated research model configuration; it does not claim regulatory medical device clearance.

---

## 5. Feature Schema Version

- **Identifier**: `scenario-a-34-v1`
- **Scope**: Exactly 34 approved Scenario-A non-invasive candidate features (demographic, household asset, maternal, feeding, and recent morbidity indicators).
- **Integrity**: Enforces zero direct anthropometric measurements (`hw70`–`hw73`, `hw2`, `hw3`, `hw4`–`hw12`, `hw13`, `hw57`) and strict feature order preservation.

---

## 6. Three Model Artifacts

| Target Outcome | Serialized Model Artifact | Family | Training Condition | Size (Bytes) |
| :--- | :--- | :---: | :---: | :---: |
| **Stunting** | `models/model_comparison_lightgbm_stunting_unweighted.joblib` | LightGBM | Unweighted | 372,191 |
| **Underweight** | `models/model_comparison_lightgbm_underweight_unweighted.joblib` | LightGBM | Unweighted | 371,983 |
| **Wasting** | `models/model_comparison_lightgbm_wasting_unweighted.joblib` | LightGBM | Unweighted | 361,999 |

---

## 7. Threshold Configuration

Decision thresholds are pre-specified, validation-derived, and locked:

```json
{
  "stunting": 0.35,
  "underweight": 0.31,
  "wasting": 0.17
}
```

The decision boundary applies an inclusive inequality on unrounded posterior probabilities:

$$\text{screen\_positive} = \mathbb{I}(\hat{P}(Y=1 \mid X) \ge \tau)$$

---

## 8. SHA-256 Integrity Verification

Every model artifact is cryptographically hashed using SHA-256. The registry verification engine recomputes these digests at runtime and matches them against the registered baseline:

| Target | Model Artifact | Expected SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Stunting** | `model_comparison_lightgbm_stunting_unweighted.joblib` | `3a1a9c0be5d8eb35483857812da2d77441d85ec3d233952c8a650b3e03f5b736` | **VERIFIED** |
| **Underweight** | `model_comparison_lightgbm_underweight_unweighted.joblib` | `2fc16e1a56d0b036398dd6d428ee7231cc09cfdec34c3dcb3cfd104caf2a3eef` | **VERIFIED** |
| **Wasting** | `model_comparison_lightgbm_wasting_unweighted.joblib` | `7900278ce83da2aafacccd8d77a924a82cf0d7b2fdfab20aa37a1ac6bdd7e919` | **VERIFIED** |

---

## 9. Artifact Loading

Loading is deterministic and explicit through `src/models/model_registry.py`:
- `load_registry()`: Parses and validates the schema.
- `verify_model_integrity(target)`: Computes and asserts SHA-256 match.
- `load_registered_pipeline(target)`: Loads verified joblib pipeline.
- `load_all_registered_pipelines()`: Returns dictionary of all 3 verified pipelines.

---

## 10. Failure Behavior (Fail Loudly)

The registry rejects silent degradation or automated file repair:
- **Missing File**: Raises `FileNotFoundError`.
- **Hash Mismatch**: Raises `ValueError` detailing expected vs. actual digests.
- **Threshold Tampering**: Raises `ValueError` if thresholds deviate from locked values ($0.35, 0.31, 0.17$) or fall outside $(0, 1)$.
- **Feature Schema Drift**: Raises `ValueError` if features are added, removed, or duplicated.

---

## 11. Security and Privacy Considerations

- **No Microdata Contamination**: The registry contains only model configuration metadata; zero DHS microdata (`IAKR7EFL.DTA`) or individual predictions are stored.
- **Supply Chain Integrity**: SHA-256 verification prevents unauthorized model tampering or accidental regression.
- **Repository Safety**: Model artifacts and registry files are tracked in accordance with project governance; DHS microdata remains strictly excluded via `.gitignore`.

---

## 12. Reproducibility

The model registry is fully deterministic:
- Given the registered artifacts, any deployment environment running compatible Python/scikit-learn/LightGBM dependencies will load identical pipelines and reproduce exact predictions.

---

## 13. Inference Compatibility

Step 17 includes automated integration verification with Step 16:
- Evaluating identical child input data through pipelines loaded via `model_registry.py` produces bit-for-bit identical probabilities and screening decisions as Step 16's `predict_child_screening`.

---

## 14. Scientific and Operational Limitations

1. **Research Artifact Baseline**: The registered models reflect the community pre-screening setting (Scenario A) evaluated on the NFHS-5 dataset.
2. **Threshold Fixedness**: Thresholds in the registry are optimized for community pre-screening sensitivity/specificity trade-offs and should not be modified without explicit clinical re-validation.
3. **Subgroup Performance Awareness**: As documented in Step 15, false-negative rates among positive cases vary across socioeconomic and age strata; screening output must be interpreted within this operational context.

---

## 15. Exact Commands to Run

### Run Registry Test Suite:
```bash
python -m unittest tests/test_model_registry.py -v
```

### Run Full 18-Test Regression Suite (Steps 0–17):
```bash
python -c "
import unittest
loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
assert result.wasSuccessful(), f'Failed with {len(result.failures)} failures'
print(f'ALL 18 TEST SUITES ({result.testsRun} TESTS) PASSED!')
"
```

---

## 16. Test Results

- **Registry Unit Tests**: 21/21 tests passed (`tests/test_model_registry.py`).
- **Inference Pipeline Tests**: 17/17 tests passed (`tests/test_inference_pipeline.py`).
- **Full Regression Suite**: All 18 test suites (95 tests) passed with zero errors.
- **Raw Data Immutability**: `IAKR7EDT/IAKR7EFL.DTA` size verified at exactly 441,380,745 bytes.
- **Model Hash Verification**: All 3 champion models match their registered SHA-256 signatures.
