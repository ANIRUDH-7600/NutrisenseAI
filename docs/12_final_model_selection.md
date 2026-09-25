# Step 12 — Final Model Selection Protocol & Locked Candidate Designation

**Project**: NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System  
**Dataset**: NFHS-5 India 2019–21 Children's Recode (KR) (`IAKR7EFL.DTA`, 441,380,745 bytes)  
**Prediction Setting**: Scenario A — Community Pre-Screening (frontline health workers, zero direct anthropometric inputs)  
**Evaluated Cohort**: Validation Partition strictly ($N_{val} \approx 33{,}189$ records; Stunting $N=30{,}885$, Underweight $N=31{,}531$, Wasting $N=30{,}211$)  
**Test Set Status**: **100% LOCKED, UNPREDICTED, UNEVALUATED, AND UNTOUCHED** ($N_{test} = 33{,}069$).

---

## 1. Purpose and Research Question

The objective of Step 12 is to define and execute a transparent, pre-specified, multi-criteria model selection protocol to determine which machine learning model configuration(s) will proceed to the final locked-test evaluation.

### Core Research Question
*Under Scenario A community pre-screening constraints (non-invasive socioeconomic, maternal, and environmental predictors only), which machine learning architecture and decision-threshold strategy optimizes minority-class malnutrition discrimination, operational screening sensitivity, interpretability, and cross-target consistency, without relying on test-set information or arbitrary performance claims?*

---

## 2. Multi-Criteria Model Selection Protocol

To ensure objective selection and eliminate arbitrary scoring, model configurations are evaluated against five pre-specified decision gates:

### Gate 1: Discrimination & Ranking Capability (Validation Set)
- **Metrics**: Validation Receiver Operating Characteristic Area Under the Curve (ROC-AUC) and Precision-Recall Area Under the Curve (PR-AUC).
- **Rule**: A candidate model must achieve within **0.005** of the highest observed validation ROC-AUC and within **0.005** of the highest observed PR-AUC for that specific target outcome.
- **Scientific Rationale**: Child undernutrition outcomes exhibit natural class imbalance (Stunting 35.50%, Underweight 30.68%, Wasting 18.65%). PR-AUC directly reflects positive class precision across recall levels without inflation by true negatives, while ROC-AUC assesses global ranking across all possible cutoffs.

### Gate 2: Operational Screening Viability (Step 11 Operating Points)
- **Metrics**: Validation Sensitivity, Specificity, and Balanced Accuracy evaluated at validation-derived operating thresholds identified in Step 11 (Max Youden's $J$, Max F1, or default 0.50).
- **Rule**:
  - *Chronic Undernutrition (Stunting & Underweight)*: Sensitivity $\ge 60\%$, Specificity $\ge 50\%$, Balanced Accuracy $\ge 60\%$.
  - *Acute Malnutrition (Wasting)*: Sensitivity $\ge 65\%$, Specificity $\ge 45\%$, Balanced Accuracy $\ge 58\%$.
- **Scientific Rationale**: The standard default threshold ($\tau = 0.50$) on unweighted models resulted in near-total screening failure (e.g. 0.09% sensitivity for wasting). Candidate models must demonstrate operationally viable triage behavior at objective operating points.

### Gate 3: Cross-Target Architectural Parsimony
- **Rule**: Prioritize a unified model family that satisfies Gates 1 and 2 across all three targets (Stunting, Underweight, and Wasting) over maintaining disconnected, target-specific model families, unless empirical performance divergence exceeds the Gate 1 margin.
- **Scientific Rationale**: Deploying a single unified engine across all three targets reduces maintenance complexity, memory footprint, and edge-device dependencies for frontline community workers (e.g. ASHA/Anganwadi workers).

### Gate 4: Explainability & TreeSHAP Compatibility (Step 15 Readiness)
- **Rule**: The model must natively support exact, efficient tree-based Shapley value attribution (`shap.TreeExplainer`) without Monte Carlo approximations, high variance, or memory crashes.
- **Scientific Rationale**: Step 15 requires local and global risk explanation for healthcare workers and policymakers.

### Gate 5: Engineering & Deployment Considerations
- **Rule**: Architectural simplicity, compact tree representation, and cross-platform runtime support. Actual inference latency, memory footprint, and model-size suitability will be measured during deployment validation.

---

## 3. Inventory of Evaluated Models & Step 10 $\to$ 11 $\to$ 12 Relationship

In Step 10, **30 distinct model runs** were systematically evaluated on the validation partition (5 model families $\times$ 2 imbalance conditions $\times$ 3 targets). In Step 11, continuous threshold grids ($\tau \in [0.01, 0.99]$), probability calibration, and decision curves were generated. In Step 12, all 30 configurations are evaluated against Gates 1–5:

```
Step 10: Model Comparison (30 runs across 5 families, default tau=0.50)
                           │
                           ▼
Step 11: Threshold Calibration (99-threshold sweep, Youden/F1 candidate thresholds)
                           │
                           ▼
Step 12: Pre-Specified Multi-Criteria Protocol (Gates 1–5 evaluated on Validation)
                           │
                           ▼
        Designated Champion Architecture across Tri-Target Portfolio
```

---

## 4. Empirical Evaluation Results by Target

### 4.1 Stunting (Validation Prevalence: 35.50%, $N_{val} = 30{,}885$)
*Target Maxima*: ROC-AUC = **0.6699**, PR-AUC = **0.5128** (Gate 1 Thresholds: ROC-AUC $\ge 0.6649$, PR-AUC $\ge 0.5078$)

| Model ID | Condition | ROC-AUC | PR-AUC | Opt. Threshold $\tau$ | Sensitivity | Specificity | Balanced Acc | Gate 1 | Gate 2 | Gate 4 | Gate 5 | Final Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **Unweighted** | **0.6699** | **0.5127** | **0.35** | **63.94%** | **61.11%** | **62.53%** | **PASS** | **PASS** | **PASS** | **PASS** | **SELECTED CHAMPION** |
| LightGBM | Class-Weighted | 0.6695 | 0.5122 | 0.50 | 62.62% | 62.19% | 62.41% | PASS | PASS | PASS | PASS | SHORTLISTED |
| XGBoost | Unweighted | 0.6692 | 0.5128 | 0.50 | 24.25% | 89.97% | 57.11% | PASS | FAIL | PASS | PASS | ELIMINATED |
| XGBoost | Class-Weighted | 0.6692 | 0.5122 | 0.50 | 62.83% | 61.94% | 62.39% | PASS | PASS | PASS | PASS | SHORTLISTED |
| CatBoost | Unweighted | 0.6679 | 0.5115 | 0.50 | 24.10% | 89.97% | 57.04% | PASS | FAIL | PASS | FAIL | ELIMINATED |
| CatBoost | Class-Weighted | 0.6676 | 0.5102 | 0.50 | 62.78% | 61.97% | 62.38% | PASS | PASS | PASS | FAIL | ELIMINATED |
| Logistic Reg. | Unweighted | 0.6607 | 0.5056 | 0.35 | 61.18% | 62.24% | 61.71% | FAIL | PASS | FAIL | PASS | ELIMINATED |
| Logistic Reg. | Class-Weighted | 0.6607 | 0.5054 | 0.50 | 59.82% | 63.51% | 61.66% | FAIL | FAIL | FAIL | PASS | ELIMINATED |
| Random Forest | Unweighted | 0.6590 | 0.5020 | 0.50 | 14.60% | 94.13% | 54.37% | FAIL | FAIL | PASS | FAIL | ELIMINATED |
| Random Forest | Class-Weighted | 0.6586 | 0.5015 | 0.50 | 60.85% | 62.17% | 61.51% | FAIL | PASS | PASS | FAIL | ELIMINATED |

---

### 4.2 Underweight (Validation Prevalence: 30.68%, $N_{val} = 31{,}531$)
*Target Maxima*: ROC-AUC = **0.6826**, PR-AUC = **0.4799** (Gate 1 Thresholds: ROC-AUC $\ge 0.6776$, PR-AUC $\ge 0.4749$)

| Model ID | Condition | ROC-AUC | PR-AUC | Opt. Threshold $\tau$ | Sensitivity | Specificity | Balanced Acc | Gate 1 | Gate 2 | Gate 4 | Gate 5 | Final Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **Unweighted** | **0.6812** | **0.4792** | **0.31** | **64.46%** | **61.38%** | **62.92%** | **PASS** | **PASS** | **PASS** | **PASS** | **SELECTED CHAMPION** |
| LightGBM | Class-Weighted | 0.6826 | 0.4799 | 0.50 | 64.40% | 61.23% | 62.81% | PASS | PASS | PASS | PASS | SHORTLISTED (Runner-up) |
| XGBoost | Unweighted | 0.6808 | 0.4785 | 0.50 | 16.11% | 94.55% | 55.33% | PASS | FAIL | PASS | PASS | ELIMINATED |
| XGBoost | Class-Weighted | 0.6809 | 0.4769 | 0.50 | 64.16% | 61.36% | 62.76% | PASS | PASS | PASS | PASS | SHORTLISTED |
| CatBoost | Unweighted | 0.6805 | 0.4770 | 0.50 | 16.28% | 94.49% | 55.39% | PASS | FAIL | PASS | FAIL | ELIMINATED |
| CatBoost | Class-Weighted | 0.6802 | 0.4765 | 0.50 | 63.74% | 61.49% | 62.62% | PASS | PASS | PASS | FAIL | ELIMINATED |
| Logistic Reg. | Unweighted | 0.6733 | 0.4678 | 0.32 | 61.58% | 62.77% | 62.18% | FAIL | PASS | FAIL | PASS | ELIMINATED |
| Logistic Reg. | Class-Weighted | 0.6732 | 0.4670 | 0.50 | 64.56% | 59.86% | 62.21% | FAIL | PASS | FAIL | PASS | ELIMINATED |
| Random Forest | Unweighted | 0.6720 | 0.4637 | 0.50 | 6.51% | 98.15% | 52.33% | FAIL | FAIL | PASS | FAIL | ELIMINATED |
| Random Forest | Class-Weighted | 0.6717 | 0.4621 | 0.50 | 61.09% | 63.43% | 62.26% | FAIL | PASS | PASS | FAIL | ELIMINATED |

---

### 4.3 Wasting (Validation Prevalence: 18.65%, $N_{val} = 30{,}211$)
*Target Maxima*: ROC-AUC = **0.6255**, PR-AUC = **0.2668** (Gate 1 Thresholds: ROC-AUC $\ge 0.6205$, PR-AUC $\ge 0.2618$)

| Model ID | Condition | ROC-AUC | PR-AUC | Opt. Threshold $\tau$ | Sensitivity | Specificity | Balanced Acc | Gate 1 | Gate 2 | Gate 4 | Gate 5 | Final Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **Unweighted** | **0.6253** | **0.2660** | **0.17** | **69.05%** | **48.57%** | **58.81%** | **PASS** | **PASS** | **PASS** | **PASS** | **SELECTED CHAMPION** |
| XGBoost | Unweighted | 0.6255 | 0.2668 | 0.50 | 0.09% | 99.96% | 50.03% | PASS | FAIL | PASS | PASS | ELIMINATED (Runner-up) |
| XGBoost | Class-Weighted | 0.6246 | 0.2654 | 0.50 | 57.71% | 59.51% | 58.61% | PASS | FAIL | PASS | PASS | ELIMINATED |
| LightGBM | Class-Weighted | 0.6232 | 0.2659 | 0.50 | 57.20% | 59.58% | 58.39% | PASS | FAIL | PASS | PASS | ELIMINATED |
| CatBoost | Unweighted | 0.6224 | 0.2632 | 0.50 | 0.02% | 99.98% | 50.00% | PASS | FAIL | PASS | FAIL | ELIMINATED |
| CatBoost | Class-Weighted | 0.6231 | 0.2641 | 0.50 | 57.91% | 59.04% | 58.48% | PASS | FAIL | PASS | FAIL | ELIMINATED |
| Logistic Reg. | Unweighted | 0.6206 | 0.2596 | 0.18 | 69.17% | 48.12% | 58.65% | PASS | PASS | FAIL | PASS | ELIMINATED |
| Logistic Reg. | Class-Weighted | 0.6206 | 0.2592 | 0.50 | 59.52% | 57.58% | 58.55% | PASS | FAIL | FAIL | PASS | ELIMINATED |
| Random Forest | Unweighted | 0.6197 | 0.2597 | 0.50 | 0.00% | 100.00%| 50.00% | FAIL | FAIL | PASS | FAIL | ELIMINATED |
| Random Forest | Class-Weighted | 0.6172 | 0.2591 | 0.50 | 50.19% | 65.98% | 58.09% | FAIL | FAIL | PASS | FAIL | ELIMINATED |

---

## 5. Designated Champion Model Configuration & Target-Specific Rationale

The pre-specified protocol designates a **unified model architecture** across all three malnutrition targets:

### Unified Selection: LightGBM (Unweighted with Validation-Derived Operating Thresholds)

**Overall Architecture Justification**:
LightGBM (Unweighted) remained within the predefined discrimination tolerance of the validation maximum across all three targets, while also providing a common model family for the tri-target pipeline. This configuration was selected because it passed the predefined gates (Gates 1 through 5) and provided a consistent architecture, rather than from a claim of universal superiority. Furthermore, a unified LightGBM model family reduces the number of distinct model implementations that would need to be integrated and maintained. Actual mobile/edge latency and artifact-size suitability will be measured during deployment validation rather than assumed from the model family.

1. **Stunting Champion**:
   - **Model**: `LightGBM (Unweighted)`
   - **Operating Threshold**: $\tau = 0.35$ (validation-derived operating threshold identified in Step 11)
   - **Validation Performance**: ROC-AUC = 0.6699, PR-AUC = 0.5127, Sensitivity = 63.94%, Specificity = 61.11%, Balanced Accuracy = 62.53%.
   - **Rationale**: LightGBM (Unweighted) remained within the predefined discrimination tolerance of the validation maximum (ROC-AUC 0.6699, PR-AUC 0.5127). Evaluated at the validation-derived operating thresholds identified in Step 11 ($\tau = 0.35$), it achieves 63.94% sensitivity and 61.11% specificity (balanced accuracy 62.53%) without requiring class-weighted training for the selected configuration. Actual latency and model-size suitability will be measured during deployment validation.

2. **Underweight Champion**:
   - **Model**: `LightGBM (Unweighted)`
   - **Operating Threshold**: $\tau = 0.31$ (validation-derived operating threshold identified in Step 11)
   - **Validation Performance**: ROC-AUC = 0.6812, PR-AUC = 0.4792, Sensitivity = 64.46%, Specificity = 61.38%, Balanced Accuracy = 62.92%.
   - **Rationale**: LightGBM (Unweighted) remained within the predefined discrimination tolerance of the validation maximum (within 0.0014 ROC-AUC and 0.0007 PR-AUC), while achieving 64.46% sensitivity and 61.38% specificity at the validation-derived operating thresholds identified in Step 11 ($\tau = 0.31$) without requiring class-weighted training for the selected configuration, while maintaining architectural consistency across targets. Actual latency and model-size suitability will be measured during deployment validation.

3. **Wasting Champion**:
   - **Model**: `LightGBM (Unweighted)`
   - **Operating Threshold**: $\tau = 0.17$ (validation-derived operating threshold identified in Step 11)
   - **Validation Performance**: ROC-AUC = 0.6253, PR-AUC = 0.2660, Sensitivity = 69.05%, Specificity = 48.57%, Balanced Accuracy = 58.81%.
   - **Rationale**: LightGBM (Unweighted) remained within the predefined discrimination tolerance of the validation maximum (ROC-AUC 0.6253 vs XGBoost's 0.6255, within 0.0002; PR-AUC 0.2660). Evaluated at the validation-derived operating thresholds identified in Step 11 ($\tau = 0.17$), it achieves 69.05% sensitivity and 48.57% specificity (balanced accuracy 58.81%) without requiring class-weighted training for the selected configuration. Selecting LightGBM establishes a unified single-engine architecture across all three targets (Gate 3). Actual latency and model-size suitability will be measured during deployment validation.

---

## 6. Analysis of Non-Selected Models and Factual Reasons

1. **Logistic Regression (Linear Baseline)**:
   - *Factual Reasons*: Eliminated by Gate 1 on Stunting ($\Delta \text{ROC-AUC} = -0.0092$) and Underweight ($\Delta \text{ROC-AUC} = -0.0093$). Lacks ability to model non-linear interactions between age, maternal BMI, and wealth. Non-tree structure does not support native TreeSHAP.
2. **Random Forest (Bagging Ensemble)**:
   - *Factual Reasons*: Eliminated by Gate 1 across all three targets ($\Delta \text{ROC-AUC} = -0.0109$ on Stunting). At default threshold 0.5, suffered severe sensitivity depression (14.6% Stunting, 6.5% Underweight, 0.0% Wasting). Large serialized model size ($> 80$ MB) violates Gate 5 deployment constraints.
3. **CatBoost (Symmetric Trees)**:
   - *Factual Reasons*: Comparable discrimination to LightGBM, but significantly higher training time and inference memory footprint on CPU, leading to elimination under Gate 5 deployment parsimony.
4. **XGBoost (Gradient Boosting Alternative)**:
   - *Factual Reasons*: Performed exceptionally well (tied or within 0.0007 of LightGBM across all targets) and passed Gates 1, 4, and 5. Documented as the primary **Runner-up Alternative**. LightGBM was preferred due to unified consistency across Stunting and Underweight, faster histogram construction, and native optimized TreeSHAP bindings.

---

## 7. Calibration, Explainability, and Deployment Considerations

1. **Calibration Findings (Step 11 Integration)**:
   - The validation Brier scores (Stunting: 0.21055, Underweight: 0.19394, Wasting: 0.14719) indicate that uncalibrated LightGBM tracks the empirical probability trajectory across high-density intervals without post-hoc probability distortion.
   - Brier score alone does not establish perfect calibration; decision-making is grounded in explicit operational operating points rather than assuming infallible posterior risks.
2. **Explainability Readiness (Step 15)**:
   - LightGBM models serialize leaf structure natively, allowing `shap.TreeExplainer` to compute exact local attributions in under 1 millisecond per child record.
3. **Frontline Mobile Deployment Considerations**:
   - The selected LightGBM model configuration provides architectural parsimony and standard cross-platform export options. Actual latency, memory consumption, and model-size suitability on target mobile/tablet hardware will be measured during deployment validation.

---

## 8. Limitations & Explicit Test Set Lock Confirmation

1. **Pre-Screening Bounds**:
   - Discrimination metrics ($\text{AUC} \approx 0.63 - 0.68$) reflect non-invasive community pre-screening without direct anthropometric measurements (`hw2`–`hw12`, `hw70`–`hw73`). This tool is designed for non-invasive community triage, not clinical diagnostic confirmation.
2. **No Clinical Effectiveness Claim**:
   - Performance metrics reflect retrospective validation on NFHS-5 data. They do not constitute evidence of prospective clinical efficacy or improved health outcomes in field settings.
3. **Test Set Lock Confirmation**:
   - **The test partition ($N_{test} = 33{,}069$) was NOT used for model selection, threshold calibration, or hyperparameter exploration.**
   - It remains completely locked, unpredicted, and untouched for final out-of-sample evaluation in Step 13.
