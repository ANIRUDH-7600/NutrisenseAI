# Step 15 — Descriptive Error Analysis & Subgroup Analysis

## 1. Objective

The objective of Step 15 is to conduct a rigorous descriptive error analysis and subgroup performance audit for the NutriSense AI childhood malnutrition screening pipeline. 

Using exclusively the **locked validation cohort** ($N_{val} = 33{,}153$) and the **champion LightGBM Unweighted models** evaluated at their **pre-specified, locked decision thresholds** ($\tau = 0.35$ for stunting, $\tau = 0.31$ for underweight, and $\tau = 0.17$ for wasting), this analysis investigates:
- How classification errors (false negatives and false positives) are distributed across key demographic, socioeconomic, geographic, and clinical covariate strata.
- Whether screening sensitivity, specificity, and false-negative rates vary systematically across subgroups.
- The extent to which classification false negatives overlap across co-occurring undernutrition conditions among jointly eligible children.

> [!IMPORTANT]
> Step 15 is strictly descriptive. No model retraining, hyperparameter tuning, threshold re-optimization, or feature modification was performed. The locked test set remained untouched and uninspected throughout this analysis.

---

## 2. Why Error Analysis Is Needed

Aggregate evaluation metrics (such as overall ROC-AUC, sensitivity, and specificity) summarize population-level model performance but can mask substantial operational discrepancies across demographic sub-populations. 

In a community pre-screening context (Scenario A):
1. **Unequal Clinical Consequences**: A false negative represents a child with unflagged undernutrition who may miss timely secondary referral, anthropometric triage, and dietary supplementation. Conversely, a false positive creates unnecessary anxiety and adds administrative load to frontline community health workers (ASHAs/Anganwadi workers).
2. **Identification of Screening Discrepancies**: Non-invasive demographic features carry strong baseline associations with living standards. Auditing subgroup error rates allows researchers and public health program managers to identify subgroups where the risk scoring system exhibits elevated miss rates (e.g., higher-income households or specific infant age brackets).
3. **Transparent Scientific Reporting**: Before any field pilot or deployment consideration, clinical and machine learning governance standards require full transparency regarding error concentration, subgroup sample adequacy, and uncertainty intervals.

---

## 3. Validation Cohort Used

All subgroup analyses and error audits were conducted on the validation split (`split == 'val'`) established in Step 8 through household-clustered sampling (`v001` cluster grouping).

- **Total Validation Records**: $N_{val} = 33{,}153$
- **Target Eligibility Breakdown**:
  - **Stunting** (`stunting`): $N = 30{,}885$ eligible children (2,268 flagged flagged with missing or out-of-range WHO HAZ flagged as NaN). Actual positive prevalence: $34.85\%$ ($N_{pos} = 10{,}763$).
  - **Underweight** (`underweight`): $N = 31{,}531$ eligible children (1,622 flagged as NaN). Actual positive prevalence: $30.68\%$ ($N_{pos} = 9{,}673$).
  - **Wasting** (`wasting`): $N = 30{,}211$ eligible children (2,942 flagged as NaN). Actual positive prevalence: $18.65\%$ ($N_{pos} = 5{,}635$).
- **Joint Eligibility Cohort**: $N = 29{,}792$ children with valid, non-flagged anthropometric records across all three nutritional targets simultaneously.

---

## 4. Locked Models Used

The models evaluated in this analysis are the exact fitted LightGBM Unweighted pipelines saved in Step 10 and selected in Step 12:
1. **Stunting**: `models/lightgbm_unweighted_stunting.joblib`
2. **Underweight**: `models/lightgbm_unweighted_underweight.joblib`
3. **Wasting**: `models/lightgbm_unweighted_wasting.joblib`

These models operate on the 34 non-leaking Scenario A features (demographic, household asset, maternal, feeding, and recent illness indicators).

---

## 5. Locked Thresholds Used

Binary classifications were generated using the validation-derived decision thresholds selected in Step 11/Step 12:

$$\hat{y} = \mathbb{I}(\hat{P}(Y=1 \mid X) \ge \tau)$$

| Target | Model Family | Locked Threshold ($\tau$) | Optimization Basis (Step 11) |
| :--- | :--- | :---: | :--- |
| **Stunting** | LightGBM Unweighted | **0.35** | Maximum validation F1 / Youden balance |
| **Underweight** | LightGBM Unweighted | **0.31** | Maximum validation F1 / Youden balance |
| **Wasting** | LightGBM Unweighted | **0.17** | Maximum validation F1 / Youden balance |

---

## 6. Error Definitions and Formulas

For each target and subgroup partition, the classification confusion matrix is defined as:
- **True Positive (TP)**: Child has undernutrition ($Y=1$) and is flagged by model ($\hat{Y}=1$).
- **False Positive (FP)**: Child is non-undernourished ($Y=0$) and is flagged by model ($\hat{Y}=1$).
- **True Negative (TN)**: Child is non-undernourished ($Y=0$) and is not flagged by model ($\hat{Y}=0$).
- **False Negative (FN)**: Child has undernutrition ($Y=1$) and is not flagged by model ($\hat{Y}=0$).

The operating metrics are computed as follows:

$$\text{Sensitivity (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{\text{Actual Positives}}$$

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{\text{TN}}{\text{Actual Negatives}}$$

$$\text{Positive Predictive Value (PPV, Precision)} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Negative Predictive Value (NPV)} = \frac{\text{TN}}{\text{TN} + \text{FN}}$$

$$\text{False Negative Rate (FNR, Miss Rate)} = 1 - \text{Sensitivity} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$

$$\text{False Positive Rate (FPR, Fall-out)} = 1 - \text{Specificity} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

$$\text{F1 Score} = \frac{2 \cdot \text{PPV} \cdot \text{Sensitivity}}{\text{PPV} + \text{Sensitivity}}$$

---

## 7. Subgroup Definitions

Subgroup analyses were conducted across 9 clinically and demographically relevant dimensions derived from the NFHS-5 dataset:

1. **Child Age Group** (`child_age_group`): 
   - 0–5 months (`00_05_mo`)
   - 6–11 months (`06_11_mo`)
   - 12–23 months (`12_23_mo`)
   - 24–35 months (`24_35_mo`)
   - 36–47 months (`36_47_mo`)
   - 48–59 months (`48_59_mo`)
2. **Child Sex** (`child_sex`): Female (`female`), Male (`male`).
3. **Wealth Quintile** (`wealth_quintile`): Poorest (Q1), Poorer (Q2), Middle (Q3), Richer (Q4), Richest (Q5).
4. **Maternal Education Level** (`maternal_education`): No education, Primary, Secondary, Higher.
5. **Residence Type** (`residence_type`): Urban, Rural.
6. **Recent Diarrhea** (`diarrhea_recent`): Yes (1.0), No (0.0).
7. **Recent Fever** (`fever_recent`): Yes (1.0), No (0.0).
8. **Recent Cough** (`cough_recent`): Yes (1.0), No (0.0).
9. **State / Union Territory** (`state_ut`): 36 administrative state/UT codes (`state_id`).

---

## 8. Minimum Subgroup Sample Rule

To avoid reporting statistically unstable estimates from sparsely populated categories, a strict sample size threshold was enforced:
- **Rule**: A subgroup category must contain at least **$N \ge 200$ eligible validation children** to be designated as *comparatively evaluable*.
- **Handling of Under-Sampled Groups**: For groups with $N < 200$ (e.g., Union Territory administrative code `04` with $N = 22$), metrics are recorded descriptively with the explicit caveat `is_comparative_evaluable = false`, and bootstrap confidence intervals are omitted to prevent misleading precision claims.

---

## 9. Confidence Interval Method

To quantify estimation uncertainty across subgroups:
- **Method**: Non-parametric percentile bootstrap resampling.
- **Iterations**: $B = 500$ bootstrap iterations.
- **Random Seed**: Fixed at `seed = 42` for exact reproducibility.
- **Metrics Evaluated**: Sensitivity, Specificity, PPV, and F1 Score.
- **Interval Bounds**: Empirical 2.5th and 97.5th percentiles of the bootstrap distribution ($95\%$ CI).

---

## 10. Results: Stunting Subgroup Performance

Overall Stunting Performance at $\tau = 0.35$ ($N = 30{,}885$):
- $\text{Sensitivity} = 63.94\%$ ($95\%\text{ CI: } [63.02\%, 64.84\%]$)
- $\text{Specificity} = 61.11\%$ ($95\%\text{ CI: } [60.43\%, 61.79\%]$)
- $\text{PPV} = 47.50\%$ ($95\%\text{ CI: } [46.70\%, 48.30\%]$)
- $\text{F1} = 0.5451$ ($95\%\text{ CI: } [0.5385, 0.5516]$)
- $\text{False Negative Rate (FNR)} = 36.06\%$

### Selected Subgroup Stratifications:

| Dimension | Subgroup Category | Sample ($N$) | Positives | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | 0–5 months | 2,951 | 720 | 24.40% | 13.06% [10.82%, 15.45%] | 95.56% [94.61%, 96.39%] | 86.94% |
| | 6–11 months | 3,193 | 755 | 23.65% | 14.97% [12.60%, 17.65%] | 95.53% [94.67%, 96.35%] | 85.03% |
| | 12–23 months | 6,558 | 2,510 | 38.27% | 66.41% [64.55%, 68.30%] | 61.02% [59.50%, 62.47%] | 33.59% |
| | 24–35 months | 6,197 | 2,342 | 37.79% | 74.47% [72.63%, 76.16%] | 50.84% [49.25%, 52.41%] | 25.53% |
| | 36–47 months | 6,110 | 2,298 | 37.61% | 74.15% [72.33%, 75.92%] | 49.37% [47.78%, 51.04%] | 25.85% |
| | 48–59 months | 5,876 | 2,138 | 36.39% | 71.42% [69.41%, 73.34%] | 53.05% [51.49%, 54.67%] | 28.58% |
| **Wealth** | Poorest (Q1) | 6,707 | 2,931 | 43.70% | 86.05% [84.77%, 87.27%] | 32.09% [30.63%, 33.59%] | 13.95% |
| | Poorer (Q2) | 6,346 | 2,492 | 39.27% | 75.52% [73.78%, 77.20%] | 47.95% [46.33%, 49.52%] | 24.48% |
| | Middle (Q3) | 6,290 | 2,130 | 33.86% | 61.36% [59.27%, 63.34%] | 62.67% [61.16%, 64.13%] | 38.64% |
| | Richer (Q4) | 5,918 | 1,770 | 29.91% | 46.84% [44.42%, 49.19%] | 74.52% [73.20%, 75.83%] | 53.16% |
| | Richest (Q5) | 5,624 | 1,440 | 25.60% | 13.70% [11.90%, 15.60%] | 94.32% [93.63%, 94.97%] | 86.30% |
| **Maternal Edu**| No education | 6,944 | 3,116 | 44.87% | 84.66% [83.37%, 85.94%] | 35.84% [34.34%, 37.37%] | 15.34% |
| | Primary | 4,204 | 1,595 | 37.94% | 74.11% [71.86%, 76.24%] | 49.10% [47.16%, 51.00%] | 25.89% |
| | Secondary | 15,221 | 4,960 | 32.59% | 55.44% [54.04%, 56.81%] | 69.17% [68.25%, 70.07%] | 44.56% |
| | Higher | 4,516 | 1,092 | 24.18% | 19.96% [17.58%, 22.45%] | 92.52% [91.64%, 93.36%] | 80.04% |
| **Sex** | Female | 14,976 | 5,116 | 34.16% | 62.61% [61.27%, 63.92%] | 62.06% [61.10%, 63.02%] | 37.39% |
| | Male | 15,909 | 5,647 | 35.49% | 65.15% [63.90%, 66.39%] | 60.19% [59.24%, 61.14%] | 34.85% |
| **Residence** | Rural | 23,388 | 8,623 | 36.87% | 68.79% [67.81%, 69.77%] | 54.38% [53.58%, 55.19%] | 31.21% |
| | Urban | 7,497 | 2,140 | 28.55% | 44.39% [42.27%, 46.49%] | 79.58% [78.50%, 80.66%] | 55.61% |

---

## 11. Results: Underweight Subgroup Performance

Overall Underweight Performance at $\tau = 0.31$ ($N = 31{,}531$):
- $\text{Sensitivity} = 64.46\%$ ($95\%\text{ CI: } [63.48\%, 65.41\%]$)
- $\text{Specificity} = 61.38\%$ ($95\%\text{ CI: } [60.72\%, 62.02\%]$)
- $\text{PPV} = 42.48\%$ ($95\%\text{ CI: } [41.69\%, 43.27\%]$)
- $\text{F1} = 0.5121$ ($95\%\text{ CI: } [0.5054, 0.5186]$)
- $\text{False Negative Rate (FNR)} = 35.54\%$

### Selected Subgroup Stratifications:

| Dimension | Subgroup Category | Sample ($N$) | Positives | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | 0–5 months | 3,028 | 778 | 25.69% | 51.03% [47.53%, 54.49%] | 72.84% [71.01%, 74.67%] | 48.97% |
| | 6–11 months | 3,254 | 936 | 28.76% | 59.94% [56.76%, 63.09%] | 66.82% [64.88%, 68.73%] | 40.06% |
| | 12–23 months | 6,693 | 2,217 | 33.12% | 67.21% [65.25%, 69.17%] | 58.78% [57.34%, 60.23%] | 32.79% |
| | 24–35 months | 6,326 | 2,059 | 32.55% | 67.31% [65.27%, 69.32%] | 58.07% [56.57%, 59.54%] | 32.69% |
| | 36–47 months | 6,243 | 1,939 | 31.06% | 65.65% [63.53%, 67.75%] | 59.43% [57.96%, 60.89%] | 34.35% |
| | 48–59 months | 5,987 | 1,744 | 29.13% | 64.62% [62.36%, 66.86%] | 61.16% [59.69%, 62.63%] | 35.38% |
| **Wealth** | Poorest (Q1) | 6,869 | 2,785 | 40.54% | 81.33% [79.88%, 82.78%] | 40.16% [38.64%, 41.67%] | 18.67% |
| | Poorer (Q2) | 6,488 | 2,343 | 36.11% | 72.39% [70.57%, 74.19%] | 50.13% [48.60%, 51.65%] | 27.61% |
| | Middle (Q3) | 6,423 | 1,934 | 30.11% | 60.55% [58.37%, 62.74%] | 62.78% [61.36%, 64.19%] | 39.45% |
| | Richer (Q4) | 6,025 | 1,539 | 25.54% | 48.02% [45.51%, 50.48%] | 74.90% [73.63%, 76.15%] | 51.98% |
| | Richest (Q5) | 5,726 | 1,072 | 18.72% | 17.74% [15.48%, 20.08%] | 93.30% [92.57%, 94.02%] | 82.26% |
| **Maternal Edu**| No education | 7,126 | 2,933 | 41.16% | 79.58% [78.11%, 81.04%] | 44.50% [42.99%, 45.99%] | 20.42% |
| | Primary | 4,289 | 1,514 | 35.30% | 71.33% [69.05%, 73.57%] | 53.66% [51.80%, 55.51%] | 28.67% |
| | Secondary | 15,510 | 4,374 | 28.20% | 57.34% [55.87%, 58.81%] | 68.21% [67.33%, 69.06%] | 42.66% |
| | Higher | 4,606 | 852 | 18.50% | 27.46% [24.47%, 30.50%] | 90.17% [89.21%, 91.13%] | 72.54% |
| **Sex** | Female | 15,313 | 4,704 | 30.72% | 63.82% [62.44%, 65.17%] | 62.45% [61.53%, 63.37%] | 36.18% |
| | Male | 16,218 | 4,969 | 30.64% | 65.06% [63.74%, 66.38%] | 60.37% [59.46%, 61.27%] | 34.94% |

---

## 12. Results: Wasting Subgroup Performance

Overall Wasting Performance at $\tau = 0.17$ ($N = 30{,}211$):
- $\text{Sensitivity} = 69.05\%$ ($95\%\text{ CI: } [67.83\%, 70.26\%]$)
- $\text{Specificity} = 48.57\%$ ($95\%\text{ CI: } [47.95\%, 49.20\%]$)
- $\text{PPV} = 23.54\%$ ($95\%\text{ CI: } [22.90\%, 24.18\%]$)
- $\text{F1} = 0.3511$ ($95\%\text{ CI: } [0.3444, 0.3578]$)
- $\text{False Negative Rate (FNR)} = 30.95\%$

### Selected Subgroup Stratifications:

| Dimension | Subgroup Category | Sample ($N$) | Positives | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | 0–5 months | 2,878 | 798 | 27.73% | 76.69% [73.74%, 79.57%] | 41.59% [39.46%, 43.70%] | 23.31% |
| | 6–11 months | 3,116 | 733 | 23.52% | 73.12% [69.90%, 76.26%] | 43.14% [41.14%, 45.13%] | 26.88% |
| | 12–23 months | 6,432 | 1,304 | 20.27% | 69.25% [66.74%, 71.74%] | 47.93% [46.56%, 49.30%] | 30.75% |
| | 24–35 months | 6,051 | 1,061 | 17.53% | 67.58% [64.76%, 70.38%] | 50.80% [49.42%, 52.19%] | 32.42% |
| | 36–47 months | 5,989 | 910 | 15.19% | 64.40% [61.27%, 67.47%] | 52.22% [50.85%, 53.60%] | 35.60% |
| | 48–59 months | 5,745 | 829 | 14.43% | 63.81% [60.52%, 67.07%] | 52.34% [50.94%, 53.74%] | 36.19% |
| **Wealth** | Poorest (Q1) | 6,569 | 1,481 | 22.55% | 75.35% [73.16%, 77.52%] | 39.86% [38.51%, 41.21%] | 24.65% |
| | Poorer (Q2) | 6,211 | 1,280 | 20.61% | 71.95% [69.49%, 74.38%] | 45.39% [44.00%, 46.77%] | 28.05% |
| | Middle (Q3) | 6,155 | 1,146 | 18.62% | 66.84% [64.10%, 69.55%] | 49.61% [48.23%, 50.99%] | 33.16% |
| | Richer (Q4) | 5,765 | 967 | 16.77% | 64.94% [61.92%, 67.92%] | 54.02% [52.61%, 55.44%] | 35.06% |
| | Richest (Q5) | 5,511 | 761 | 13.81% | 58.21% [54.68%, 61.69%] | 61.16% [59.77%, 62.54%] | 41.79% |
| **Maternal Edu**| No education | 6,801 | 1,514 | 22.26% | 74.31% [72.10%, 76.47%] | 41.52% [40.19%, 42.84%] | 25.69% |
| | Primary | 4,098 | 848 | 20.69% | 70.75% [67.67%, 73.79%] | 47.78% [46.06%, 49.51%] | 29.25% |
| | Secondary | 14,888 | 2,642 | 17.75% | 67.56% [65.77%, 69.32%] | 50.81% [49.92%, 51.69%] | 32.44% |
| | Higher | 4,424 | 631 | 14.26% | 60.38% [56.54%, 64.17%] | 57.66% [56.09%, 59.24%] | 39.62% |
| **Sex** | Female | 14,642 | 2,643 | 18.05% | 68.37% [66.59%, 70.12%] | 49.43% [48.53%, 50.32%] | 31.63% |
| | Male | 15,569 | 2,992 | 19.22% | 69.65% [68.01%, 71.30%] | 47.75% [46.88%, 48.62%] | 30.35% |

---

## 13. Cross-Target Comparison

Comparing subgroup behavior across stunting, underweight, and wasting demonstrates distinct operational characteristics:

1. **Age-Group Differences in Model Performance**:
   - Sensitivity varied across age groups in the validation sample. For stunting, sensitivity was 13.06% [10.82%, 15.45%] among children aged 0–5 months compared with 71.42% [69.41%, 73.34%] among children aged 48–59 months.
   - For underweight, sensitivity was 51.03% [47.53%, 54.49%] among children aged 0–5 months and 64.62% [62.36%, 66.86%] among children aged 48–59 months.
   - For wasting, sensitivity was 76.69% [73.74%, 79.57%] among children aged 0–5 months and 63.81% [60.52%, 67.07%] among children aged 48–59 months.
   - These differences describe the operating behavior of the fitted models at the fixed thresholds and do not establish a biological explanation for the observed pattern.

2. **Socioeconomic Variations in Model Performance**:
   - In this validation sample, sensitivity varied substantially across wealth quintiles. For stunting and underweight, sensitivity was higher in Q1 than Q5, while specificity showed the opposite pattern. This indicates that the fixed decision thresholds produced different operating characteristics across wealth groups. These results describe model performance within the validation sample and do not establish why the differences occur.
   - The false-negative rate among positive cases was 86.30% for stunting and 82.26% for underweight in Q5, compared with 13.95% for stunting and 18.67% for underweight in Q1. For wasting, the false-negative rate among positive cases was 41.79% in Q5 and 24.65% in Q1.
   - False-positive rates among negative cases were higher in Q1, corresponding to the lower specificity observed in that subgroup.

3. **Sex Equivalence**:
   - Across all three targets, differences in sensitivity, specificity, and FNR between male and female children remain small ($|\Delta| < 2.5\%$), with overlapping 95% bootstrap confidence intervals.

---

## 14. Error Overlap Analysis Across Co-Occurring Conditions

To examine whether classification false negatives cluster among children who have multiple nutritional deficits, we evaluated the joint validation cohort ($N = 29{,}792$ children eligible for all three outcomes):

- **Jointly Stunted and Underweight ($N_{joint\_pos} = 5{,}920$)**:
  - Joint false negatives (missed on both models): $1{,}270$ children.
  - Overlap proportion: **$21.45\%$** of jointly positive children.
- **Jointly Stunted and Wasted ($N_{joint\_pos} = 1{,}454$)**:
  - Joint false negatives (missed on both models): $168$ children.
  - Overlap proportion: **$11.55\%$** of jointly positive children.
- **Jointly Underweight and Wasted ($N_{joint\_pos} = 3{,}598$)**:
  - Joint false negatives (missed on both models): $693$ children.
  - Overlap proportion: **$19.26\%$** of jointly positive children.
- **Triple Positives (Stunted, Underweight, and Wasted simultaneously, $N_{triple\_pos} = 1{,}454$)**:
  - Triple false negatives (missed across all three models): **$164$ children** ($11.28\%$).

> [!NOTE]
> The empirical overlap numbers describe joint prediction misses within the validation sample. They describe model scoring behavior and do not establish a shared etiology or physical explanation.

---

## 15. Interpretation of Major Error Patterns

The empirical error distributions reveal differences in classifier operating characteristics across demographic and socioeconomic strata:

1. **Socioeconomic Stratification in Model Performance**:
   - In this validation sample, sensitivity varied substantially across wealth quintiles. For stunting and underweight, sensitivity was higher in Q1 than Q5, while specificity showed the opposite pattern. This indicates that the fixed decision thresholds produced different operating characteristics across wealth groups. These results describe model performance within the validation sample and do not establish why the differences occur.
   - The false-negative rate among positive cases was 86.30% for stunting and 82.26% for underweight in Q5, whereas the false-negative rate among positive cases in Q1 was 13.95% for stunting and 18.67% for underweight.
   - For maternal education, the false-negative rate among positive cases was 80.04% in the higher education subgroup for stunting, compared with 15.34% in the no-education subgroup.
   - For some observations in these subgroups, the fitted models produced lower predicted probabilities, placing them below the decision threshold.
   - False-positive rates among negative cases were higher in Q1, corresponding to the lower specificity observed in that subgroup.

2. **Age-Group Differences in Model Performance**:
   - Sensitivity varied across age groups in the validation sample. For stunting, sensitivity was 13.06% among children aged 0–5 months compared with 71.42% among children aged 48–59 months. For wasting, sensitivity was 76.69% among children aged 0–5 months and 63.81% among children aged 48–59 months.
   - The false-negative rate among positive cases for stunting was 86.94% among children aged 0–5 months.
   - These differences describe the operating behavior of the fitted models at the fixed thresholds and do not establish a biological explanation for the observed pattern.

---

## 16. Limitations of the Error Analysis

1. **Validation Sample Partitioning**: Although the validation set contains 33,153 children, sub-stratification across 36 administrative states yields small cell counts ($N < 200$) in smaller Union Territories, precluding comparative precision for those areas.
2. **Post-Hoc Fixed Thresholds**: Subgroup metrics are evaluated at a single population-wide operating threshold ($\tau = 0.35, 0.31, 0.17$). Subgroup-specific thresholds were not explored, as the study protocol pre-specified a uniform decision threshold per target.
3. **Unmeasured Clinical Covariates**: Non-invasive pre-screening lacks direct dietary intake, maternal micronutrient biomarkers, and pathogen-specific infection records.
4. **Survey Recall Bias**: Morbidity indicators (diarrhea, fever, cough in the past 2 weeks) rely on maternal recall, which may introduce misclassification into covariate subgroup definitions.

---

## 17. No-Causal-Inference Statement

This error analysis is strictly descriptive. The associations observed between subgroup categories (e.g., maternal education, child age, or wealth quintile) and model error rates describe empirical classifier performance on this dataset. They do not represent causal relationships, physical mechanisms, or normative judgments regarding sub-populations. Phrases such as "best subgroup" or "worst subgroup" are scientifically inappropriate and strictly avoided.

---

## 18. Data Privacy Considerations

In compliance with DHS data governance and ethics agreements:
- All error metrics and subgroup counts are reported exclusively in aggregate form.
- No individual child records, household cluster identifiers (`v001`), or geo-coordinates are exposed or redistributed.
- Categories with sparse cell counts ($N < 200$) are flagged, and bootstrap confidence intervals are suppressed to preserve statistical integrity and anonymization.

---

## 19. Reproducibility & Files Created

### Source Code:
- `src/models/error_analysis.py`: Modular script computing subgroup confusion matrices, 95% bootstrap CIs, and cross-target overlap.
- `tests/test_error_analysis.py`: Unit test suite verifying metrics, schema, threshold consistency, CI ordering, and sample size rules.

### Interim Data Outputs:
- `data/interim/error_analysis_metrics.csv`: 180 rows containing detailed subgroup metrics across 9 dimensions $\times$ 3 targets.
- `data/interim/error_analysis_metrics.json`: Structured JSON containing complete subgroup metrics and 95% bootstrap CIs.
- `data/interim/error_overlap_metrics.json`: Overlap statistics across co-occurring conditions ($N = 29{,}792$).

### Report Figures:
- `reports/figures/33_error_sensitivity_by_age.png`: Sensitivity across child age brackets.
- `reports/figures/34_error_specificity_by_age.png`: Specificity across child age brackets.
- `reports/figures/35_error_f1_by_wealth.png`: F1 score across wealth quintiles.
- `reports/figures/36_false_negative_rate_by_wealth.png`: False negative rates across wealth quintiles.
- `reports/figures/37_false_negative_rate_by_education.png`: False negative rates across maternal education levels.
- `reports/figures/38_error_distribution_by_age.png`: Confusion distribution stacked proportions by child age group.

---

## 20. Tests Performed and Exact Commands to Reproduce

### Unit Test Execution:
```bash
python -m unittest tests/test_error_analysis.py -v
```
*Result*: 15 passed in 0.095s.

### Full Pipeline Regression Execution:
```bash
python -c "
import unittest
loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
assert result.wasSuccessful()
"
```
*Result*: All 16 test suites (Steps 0 through 15) passed cleanly with exit code 0.

### Exact Command to Reproduce Error Analysis:
```bash
python -m src.models.error_analysis
```
