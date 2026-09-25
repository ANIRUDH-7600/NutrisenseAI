# Step 13 — Final Locked Test Evaluation

**Project**: NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System  
**Dataset**: NFHS-5 India 2019–21 Children's Recode (KR) (`IAKR7EFL.DTA`, 441,380,745 bytes)  
**Prediction Setting**: Scenario A — Community Pre-Screening (frontline health workers, zero direct anthropometric inputs)  
**Evaluated Cohort**: Locked Out-of-Sample Test Partition ($N_{test} = 33{,}069$ records)  
**Evaluation Protocol**: Strictly out-of-sample evaluation using pre-specified model configurations and thresholds selected in Step 12; zero retraining; zero threshold re-tuning; zero post-hoc calibration fitting.  

---

## 1. Executive Summary & Purpose of Locked Test Evaluation

Step 13 constitutes the definitive out-of-sample empirical evaluation of the NutriSense AI Scenario A triage models. Prior to this step, the test partition ($N_{test} = 33{,}069$ children) remained completely untouched, sealed, and unpredicted across Steps 0 through 12. Pre-specified model configurations and thresholds selected in Step 12 (LightGBM Unweighted at $\tau = 0.35$ for Stunting, $\tau = 0.31$ for Underweight, and $\tau = 0.17$ for Wasting) were finalized exclusively on validation partition evidence.

The purpose of this locked test evaluation is to provide an unbiased assessment of model generalization capability on previously unseen clusters and households across India:
- **Stunting** ($\tau = 0.35$): Test ROC-AUC of **0.6671** [95% CI: 0.6606–0.6729], PR-AUC of **0.5170** [95% CI: 0.5064–0.5266], Sensitivity of **63.42%** [95% CI: 62.46%–64.31%], and Specificity of **61.18%** [95% CI: 60.54%–61.85%].
- **Underweight** ($\tau = 0.31$): Test ROC-AUC of **0.6848** [95% CI: 0.6784–0.6915], PR-AUC of **0.4865** [95% CI: 0.4763–0.4972], Sensitivity of **64.60%** [95% CI: 63.56%–65.57%], and Specificity of **61.79%** [95% CI: 61.16%–62.44%].
- **Wasting** ($\tau = 0.17$): Test ROC-AUC of **0.6349** [95% CI: 0.6274–0.6428], PR-AUC of **0.2752** [95% CI: 0.2654–0.2851], Sensitivity of **69.82%** [95% CI: 68.64%–70.97%], and Specificity of **48.42%** [95% CI: 47.77%–49.04%].

Divergence between validation and test discrimination metrics was bounded within $\Delta \text{ROC-AUC} \in [-0.0028, +0.0096]$ and $\Delta \text{Sensitivity} \in [-0.52\%, +0.77\%]$, confirming that the household-grouped split effectively prevented optimistic data leakage.

---

## 2. Pre-Specified Evaluation Protocol & Methodological Governance

To maintain methodological integrity, the evaluation followed these strict governance rules:

1. **Frozen Pipelines**: Preprocessing scalers, imputers, encoders, and LightGBM tree structures were loaded directly from `models/lgbm_unweighted_{target}.joblib`. No parameter was re-estimated, tuned, or updated on the test data.
2. **Locked Operating Thresholds**: Operating thresholds ($\tau$) established in Steps 11 and 12 were applied directly to continuous predicted probabilities. No search for optimal Youden's $J$, F1, or cost-minimizing threshold was conducted on the test set.
3. **Absence of Calibration Fitting**: No Platt scaling, isotonic regression, or temperature scaling was fitted on test labels.
4. **Scenario A Feature Boundary**: Exactly the 34 approved candidate features established in Step 7 were ingested. Direct anthropometric variables (`hw2`–`hw12`, `hw70`–`hw73`, z-scores, percentiles) remained strictly excluded from predictor matrices.
5. **Data Protection**: All results are reported as aggregate performance metrics and confusion counts. Microdata identifiers are never disclosed or modified.

---

## 3. Test Partition Characteristics

The test set was partitioned during Step 8 using household-level grouping on `(v001, v002)` with a fixed random seed (`42`), ensuring that all children residing in the same household cluster into exactly one partition.

| Partition Attribute | Stunting Cohort | Underweight Cohort | Wasting Cohort |
| :--- | :---: | :---: | :---: |
| **Total Test Records Assigned** | 33,069 | 33,069 | 33,069 |
| **Eligible Records (Non-Missing Target)** | 30,772 | 31,433 | 30,148 |
| **Missing/Excluded Cases (DHS Flags)** | 2,297 (6.95%) | 1,636 (4.95%) | 2,921 (8.83%) |
| **Actual Positive Cases ($y=1$)** | 10,961 | 9,706 | 5,627 |
| **Actual Negative Cases ($y=0$)** | 19,811 | 21,727 | 24,521 |
| **Unweighted Test Prevalence** | 35.62% | 30.88% | 18.66% |
| **Unweighted Validation Prevalence** | 35.50% | 30.68% | 18.65% |
| **Prevalence Shift ($\Delta$)** | +0.12% | +0.20% | +0.01% |
| **Household Leakage with Train/Val** | **0 (Zero)** | **0 (Zero)** | **0 (Zero)** |

The stability in prevalence across validation and test partitions confirms the integrity of the randomized household-grouped sampling scheme.

---

## 4. Primary Discrimination Performance

Global discrimination was evaluated using ROC-AUC and PR-AUC. Non-parametric bootstrap resampling ($B = 1{,}000$ iterations) was executed on the locked test partition to compute empirical 95% confidence intervals.

| Target | Model | Test ROC-AUC [95% CI] | Validation ROC-AUC | Test PR-AUC [95% CI] | Validation PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stunting** | LightGBM Unweighted | **0.6671** [0.6606, 0.6729] | 0.6699 | **0.5170** [0.5064, 0.5266] | 0.5127 |
| **Underweight** | LightGBM Unweighted | **0.6848** [0.6784, 0.6915] | 0.6812 | **0.4865** [0.4763, 0.4972] | 0.4792 |
| **Wasting** | LightGBM Unweighted | **0.6349** [0.6274, 0.6428] | 0.6253 | **0.2752** [0.2654, 0.2851] | 0.2660 |

ROC curves and Precision-Recall curves are illustrated in `reports/figures/fig21_test_roc_curves.png` and `reports/figures/fig22_test_precision_recall_curves.png`.

---

## 5. Threshold-Specific Operating Performance

Operating metrics evaluated at the locked validation-derived decision thresholds ($\tau$) on the test set:

| Target | Locked $\tau$ | Sensitivity (%) [95% CI] | Specificity (%) [95% CI] | PPV (%) [95% CI] | NPV (%) | Balanced Acc (%) [95% CI] | F1 Score [95% CI] | Youden's $J$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stunting** | 0.35 | **63.42** [62.46, 64.31] | **61.18** [60.54, 61.85] | **47.48** [46.67, 48.25] | 75.15 | **62.30** [61.73, 62.82] | **0.5431** [0.5355, 0.5499] | 0.2461 |
| **Underweight** | 0.31 | **64.60** [63.56, 65.57] | **61.79** [61.16, 62.44] | **43.03** [42.26, 43.85] | 79.62 | **63.19** [62.59, 63.74] | **0.5165** [0.5086, 0.5241] | 0.2639 |
| **Wasting** | 0.17 | **69.82** [68.64, 70.97] | **48.42** [47.77, 49.04] | **23.70** [23.02, 24.36] | 87.49 | **59.12** [58.41, 59.77] | **0.3539** [0.3457, 0.3619] | 0.1824 |

Key observations:
1. **Screening Sensitivity Target**: Chronic forms (Stunting and Underweight) exceeded the pre-specified 60% operational sensitivity benchmark at 63.42% and 64.60%, respectively. Acute wasting exceeded the 65% benchmark at 69.82%.
2. **False Negative Minimization**: High NPV across targets (75.15% to 87.49%) indicates that a negative screening result strongly reduces the post-test probability of unflagged undernutrition.

---

## 6. Detailed Confusion Matrix Analysis

Operating triage counts across the test partition ($N_{test}$ eligible cases) are presented below:

```
STUNTING (tau = 0.35, N = 30,772)
                       Actual Stunted (y=1)    Actual Not Stunted (y=0)    Total Predicted
Flagged High-Risk:             6,952 (TP)               7,690 (FP)             14,642 (47.6%)
Flagged Low-Risk:              4,009 (FN)              12,121 (TN)             16,130 (52.4%)
Total Actual:                 10,961 (35.6%)           19,811 (64.4%)          30,772 (100.0%)

UNDERWEIGHT (tau = 0.31, N = 31,433)
                       Actual Underweight (y=1) Actual Not Underweight (y=0) Total Predicted
Flagged High-Risk:             6,270 (TP)               8,302 (FP)             14,572 (46.4%)
Flagged Low-Risk:              3,436 (FN)              13,425 (TN)             16,861 (53.6%)
Total Actual:                  9,706 (30.9%)           21,727 (69.1%)          31,433 (100.0%)

WASTING (tau = 0.17, N = 30,148)
                       Actual Wasted (y=1)     Actual Not Wasted (y=0)     Total Predicted
Flagged High-Risk:             3,929 (TP)              12,648 (FP)             16,577 (55.0%)
Flagged Low-Risk:              1,698 (FN)              11,873 (TN)             13,571 (45.0%)
Total Actual:                  5,627 (18.7%)           24,521 (81.3%)          30,148 (100.0%)
```

Figure 19 (`reports/figures/fig19_test_confusion_matrices.png`) provides visual confusion matrix heatmaps alongside rate breakdowns.

### Frontline Operational Interpretation:
- In community pre-screening, high false-positive counts (e.g. 12,648 for wasting) are expected when prioritizing sensitivity in low-prevalence regimes (18.66%). 
- A "false positive" in this context represents a child flagged for confirmatory anthropometric assessment (MUAC tape, digital weighing scales, stadiometer measurement) who is subsequently determined to be within normal anthropometric thresholds. The cost of a false positive is frontline screening time (approx. 5–10 minutes for physical measurement).
- In contrast, a "false negative" represents a malnourished child missed during triage, remaining without dietary supplementation or clinical intervention. The selected operating points prioritize capturing ~63%–70% of affected children.

---

## 7. Calibration & Probabilistic Fidelity

Probabilistic calibration was evaluated out-of-sample using the Brier score:

| Target | Unweighted Test Prevalence | Test Brier Score | Baseline Null Model Brier Score [$p \cdot (1-p)$] | Brier Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Stunting** | 0.3562 | 0.2111 | 0.2293 | -0.0182 (-7.9%) |
| **Underweight** | 0.3088 | 0.1938 | 0.2134 | -0.0196 (-9.2%) |
| **Wasting** | 0.1866 | 0.1466 | 0.1518 | -0.0052 (-3.4%) |

The Brier score for all three models improves upon the non-informative base rate predictor ($p$). Consistent with Step 11 findings, while raw unweighted probability outputs exhibit minor compression toward the mean in extreme deciles, they maintain monotonic ordering, validating post-hoc threshold adjustment over probability-altering class weights.

---

## 8. Comparative Generalization Analysis: Validation vs. Test

To evaluate potential overfitting, test-set metrics were compared directly with validation-set metrics across identical model checkpoints and thresholds:

| Target | Metric | Validation Value | Test Value | Absolute Shift ($\Delta$) | Stability Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stunting** | ROC-AUC | 0.6699 | 0.6671 | -0.0028 | Within predefined $|\Delta| < 0.005$ reference |
| | PR-AUC | 0.5127 | 0.5170 | +0.0043 | Within predefined $|\Delta| < 0.005$ reference |
| | Sensitivity | 63.94% | 63.42% | -0.52% | Percentage difference ($-0.52\%$) |
| | Specificity | 61.11% | 61.18% | +0.07% | Percentage difference ($+0.07\%$) |
| | Balanced Acc | 62.53% | 62.30% | -0.23% | Percentage difference ($-0.23\%$) |
| | F1 Score | 0.5451 | 0.5431 | -0.0020 | Within predefined $|\Delta| < 0.005$ reference |
| **Underweight** | ROC-AUC | 0.6812 | 0.6848 | +0.0036 | Within predefined $|\Delta| < 0.005$ reference |
| | PR-AUC | 0.4792 | 0.4865 | +0.0073 | Exceeds predefined $|\Delta| < 0.005$ reference |
| | Sensitivity | 64.46% | 64.60% | +0.14% | Percentage difference ($+0.14\%$) |
| | Specificity | 61.38% | 61.79% | +0.41% | Percentage difference ($+0.41\%$) |
| | Balanced Acc | 62.92% | 63.19% | +0.27% | Percentage difference ($+0.27\%$) |
| | F1 Score | 0.5121 | 0.5165 | +0.0044 | Within predefined $|\Delta| < 0.005$ reference |
| **Wasting** | ROC-AUC | 0.6253 | 0.6349 | +0.0096 | Exceeds predefined $|\Delta| < 0.005$ reference |
| | PR-AUC | 0.2660 | 0.2752 | +0.0092 | Exceeds predefined $|\Delta| < 0.005$ reference |
| | Sensitivity | 69.05% | 69.82% | +0.77% | Percentage difference ($+0.77\%$) |
| | Specificity | 48.57% | 48.42% | -0.15% | Percentage difference ($-0.15\%$) |
| | Balanced Acc | 58.81% | 59.12% | +0.31% | Percentage difference ($+0.31\%$) |
| | F1 Score | 0.3511 | 0.3539 | +0.0028 | Within predefined $|\Delta| < 0.005$ reference |

Figure 20 (`reports/figures/fig20_validation_vs_test_metrics.png`) illustrates this comparison.

### Analysis of Generalization Integrity:
- Across discrimination metrics:
  - Stunting ROC-AUC ($-0.0028$), Stunting PR-AUC ($+0.0043$), and Underweight ROC-AUC ($+0.0036$) remain **Within predefined $|\Delta| < 0.005$ reference**.
  - Underweight PR-AUC ($+0.0073$), Wasting ROC-AUC ($+0.0096$), and Wasting PR-AUC ($+0.0092$) **Exceeds predefined $|\Delta| < 0.005$ reference**.
- Operating sensitivity and specificity shifts remained within $\pm 0.8\%$ across all three targets ($\Delta \text{Sensitivity} \in [-0.52\%, +0.77\%]$, $\Delta \text{Specificity} \in [-0.15\%, +0.41\%]$).
- This consistency validates the household-level grouping strategy established in Step 8, confirming that the models learned generalized relationships rather than cluster-specific artifacts.

---

## 9. Clinical & Public Health Screening Contextualization

In resource-constrained settings across India, frontline Anganwadi and ASHA workers are tasked with monitoring child nutrition across large catchment populations. Conducting physical anthropometric assessments on every child during routine visits can be challenging due to broken or uncalibrated scales, unavailable stadiometers, and time constraints.

Scenario A models operate as a **first-line digital triage mechanism**:
1. **Community Pre-Screening (Non-Invasive)**: Community workers collect demographic, maternal, dietary, and household indicators through standard questionnaire responses.
2. **Risk Scoring & Stratification**: The model evaluates the risk score against the target threshold ($\tau$).
3. **Targeted Anthropometric Assessment**: Children flagged as high risk (46%–55% of the screened population) are prioritized for immediate physical anthropometric verification, clinical evaluation, and referral to Nutrition Rehabilitation Centres (NRCs) or Supplementary Nutrition Programmes (SNP).
4. **Routine Surveillance**: Children categorized as low risk continue under routine community observation.

This approach focuses physical anthropometry resources on children with higher predicted risk.

---

## 10. Model vs. Anthropometric Status Discrepancies

A non-invasive model relying on socio-demographic, maternal, and environmental predictors will inevitably exhibit divergence from objective anthropometric measurements:

### Sources of Divergence:
1. **Biological Latency**: Socioeconomic and maternal risk factors (e.g. maternal short stature, severe poverty, poor sanitation) indicate chronic environmental vulnerability. However, a child living under adverse conditions may not yet have crossed the $-2.0$ SD anthropometric deficit threshold. These cases appear as "false positives" in cross-sectional survey data, yet they often represent children at high ongoing risk.
2. **Acute Episodic Shocks**: Acute wasting (WHZ $< -2.0$) can be triggered rapidly by diarrheal episodes, acute infections, or sudden household food insecurity in children who otherwise reside in moderate-wealth households. Without recent infection or dietary biomarkers, non-invasive demographic features capture only a portion of acute wasting variance (explaining the lower test ROC-AUC of 0.6349).
3. **Genetic vs. Nutritional Stunting**: A child may be constitutionally short due to familial genetics rather than nutritional stunting, leading to non-nutritional false positives.
4. **DHS Measurement Noise**: Field measurements in large-scale demographic surveys carry documented measurement variances (e.g. child movement during length measurement, uncooperative infants).

---

## 11. Multi-Target Performance Synthesis

Across the three undernutrition manifestations, a consistent hierarchy of predictive performance emerges:

$$\text{Underweight } (\text{ROC-AUC } 0.6848) > \text{Stunting } (\text{ROC-AUC } 0.6671) > \text{Wasting } (\text{ROC-AUC } 0.6349)$$

### Underlying Drivers:
- **Underweight (Weight-for-Age)** integrates signals from both linear growth deficits (stunting) and soft tissue/muscle mass deficits (wasting). It exhibits the strongest composite correlation with household wealth, maternal education, sanitation, and child age.
- **Stunting (Height-for-Age)** reflects chronic, cumulative deprivation accumulated over conception and the first 1,000 days. It associates strongly with maternal height, household wealth, and maternal birth history.
- **Wasting (Weight-for-Height)** reflects acute, rapid tissue loss. Cross-sectional demographic features provide moderate signal for acute wasting because short-term shocks (recent fevers, acute waterborne pathogens) are only partially captured by survey-level indicators.

---

## 12. Comparison with Base Literature

The selected literature baseline is **Islam et al. (2024)** (*PLOS ONE*), which applied machine learning algorithms to BDHS children's data in Bangladesh.

| Methodological Dimension | Islam et al. (2024) [Base Literature] | NutriSense AI (Current Study) |
| :--- | :--- | :--- |
| **Target Dataset** | Bangladesh DHS 2017–18 ($N \approx 7{,}700$) | India NFHS-5 2019–21 ($N = 221{,}263$) |
| **Splitting Protocol** | Random individual split (household clustering unaddressed) | Cluster-safe household-grouped split (`(v001, v002)`) |
| **Evaluation Discipline** | Cross-validation / unpartitioned metrics reported | Strictly locked out-of-sample test partition ($N = 33{,}069$) |
| **Operating Thresholds** | Unspecified / default 0.50 | Pre-specified thresholds from validation grid ($\tau \in \{0.35, 0.31, 0.17\}$) selected in Step 12 |
| **Class Weighting / Resampling** | Evaluated SMOTE resampling on small sample | Evaluated cost-sensitive weighting vs. post-hoc threshold adjustment |
| **Direct Anthropometrics** | Discussed with mixed predictor boundaries | Strictly excluded under Scenario A (zero direct anthropometrics) |
| **Bootstrap Validation** | Not reported | 1,000-iteration non-parametric bootstrap 95% CIs |

While literature studies utilizing random splits or including physical proxy measurements occasionally report higher unvalidated accuracies, NutriSense AI adopts strict household-level split isolation and non-invasive pre-screening constraints to reflect practical deployment settings.

---

## 13. Diagnostic & Operational Trade-offs

The choice of operating decision thresholds involves operational trade-offs:

1. **Threshold Selection Trade-off**:
   - Raising $\tau$ increases specificity and PPV, reducing the number of children referred for physical measurement, but increases false negatives (missed cases).
   - Lowering $\tau$ increases sensitivity, capturing more affected children, but increases the frontline screening workload (false positives).
2. **Community Screening Context**:
   - In rural and peri-urban India, an undetected malnourished child risks developmental delay, cognitive impairment, or mortality from opportunistic infections.
   - The workload cost of measuring a child (MUAC/weight) is modest compared to the clinical risk of missing severe undernutrition.
   - Consequently, the pre-specified model configurations and thresholds selected in Step 12 ($\tau = 0.35, 0.31, 0.17$) prioritize sensitivity ($\sim 63\% - 70\%$) while maintaining specificity ($\sim 48\% - 62\%$).

---

## 14. Threats to Validity & Study Limitations

The test findings should be interpreted in light of the following methodological limitations:

1. **Cross-Sectional Design**: The NFHS-5 dataset is cross-sectional. Observed associations between socioeconomic/maternal factors and malnutrition outcomes are predictive rather than causal.
2. **Self-Reported Covariates**: Features such as maternal recall of birth size, dietary diversity, and immunization history are subject to recall bias.
3. **Temporal Invariance**: The data represents the 2019–2021 survey period. Post-pandemic economic shifts, localized climate shocks, and recent food security changes are not reflected in model weights.
4. **Pre-Screening Boundary**: The model is an early pre-screening tool designed to prioritize physical anthropometric assessment. It is not a diagnostic device and cannot substitute for calibrated clinical measurement.

---

## 15. Algorithmic Fairness & Subgroup Generalization Notes

Observational subgroup analysis indicates consistent operational behavior:
- **Child Sex**: Sensitivity and specificity remained balanced between male and female children across all three targets within $\pm 1.5\%$.
- **Urban vs. Rural**: Performance showed higher screening referral rates in rural areas, reflecting the higher underlying prevalence of malnutrition in rural clusters rather than model bias.
- **Geographic Representation**: Household grouping ensured that test children were drawn from all surveyed states and union territories, supporting pan-Indian geographical generalizability.

---

## 16. Reproducibility & Audit Trail

| Audit Attribute | Value / Specification |
| :--- | :--- |
| **Execution Script** | `src/models/final_test_evaluation.py` |
| **Test Verification Script** | `tests/test_final_test_evaluation.py` (19 automated assertions passed) |
| **Source Data File** | `data/raw/IAKR7EDT/IAKR7EFL.DTA` (SHA256 verified) |
| **Test Output Manifest** | `data/interim/final_test_metrics.json` |
| **Comparative CSV** | `data/interim/validation_vs_test_metrics.csv` |
| **Generated Figures** | `fig19_test_confusion_matrices.png`, `fig20_validation_vs_test_metrics.png`, `fig21_test_roc_curves.png`, `fig22_test_precision_recall_curves.png` |
| **Random State / Seed** | Fixed `seed = 42` (splits, bootstrap iterations) |
| **Environment Specifications** | Python 3.13.5, `scikit-learn==1.6.1`, `lightgbm==4.7.0`, `joblib==1.4.2` |

---

## 17. Conclusion & Transition to Post-Evaluation Analysis

Step 13 completes the locked test evaluation for the NutriSense AI Scenario A pre-screening pipeline. The LightGBM Unweighted models demonstrated stable discrimination and operational sensitivity across the locked test partition ($N_{test} = 33{,}069$).

With test evaluation finalized and all models, thresholds, and splits frozen:
- **Step 14** will proceed to model interpretability and explainability analysis using TreeSHAP (`shap.TreeExplainer`). This will extract global feature attributions, local individual child risk waterfall plots, and subgroup interaction effects to facilitate interpretation by healthcare workers and policy analysts.
