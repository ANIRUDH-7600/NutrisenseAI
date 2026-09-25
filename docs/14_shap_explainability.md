# Step 14 — Model Explainability & Interpretability using TreeSHAP

**Project**: NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System  
**Dataset**: NFHS-5 India 2019–21 Children's Recode (KR) (`IAKR7EFL.DTA`, 441,380,745 bytes)  
**Prediction Setting**: Scenario A — Community Pre-Screening (frontline health workers, zero direct anthropometric inputs)  
**Evaluated Models**: Pre-selected champion LightGBM Unweighted models for Stunting ($\tau = 0.35$), Underweight ($\tau = 0.31$), and Wasting ($\tau = 0.17$)  
**Explanation Method**: TreeSHAP (`shap.TreeExplainer`) on validation cohort ($N = 5{,}000$ per target, seed = 42)  
**Governance Principle**: Purely interpretability and model audit; zero model retraining, zero threshold re-tuning, and zero feature selection.  

---

## 1. Purpose of Step 14

Step 14 provides an audit and interpretability analysis of the champion LightGBM Unweighted models established in Step 12 and evaluated on the locked test partition in Step 13. Rather than treating the gradient boosted decision trees as uninspectable "black boxes," this analysis explains which non-invasive socioeconomic, maternal, and demographic features drive model risk scoring.

The purpose of this step is descriptive interpretation. SHAP values reflect the internal decision logic of the trained machine learning estimators. They do not alter model parameters, adjust decision thresholds, or prune features from the pipeline.

---

## 2. Research Question

Under Scenario A community pre-screening constraints:

> *"Which non-anthropometric child, maternal, household, environmental, and demographic features contribute most strongly to the model's predictions of childhood malnutrition risk, and what is the direction and consistency of these contributions across Stunting, Underweight, and Wasting?"*

---

## 3. Why SHAP Was Selected

SHapley Additive exPlanations (SHAP), introduced by Lundberg and Lee (2017), is grounded in cooperative game theory. Compared to traditional tree-based importance metrics (such as Gini impurity decrease or split frequency), SHAP provides two critical mathematical guarantees:

1. **Local Accuracy (Additivity)**: For each child observation $i$, the sum of feature attributions plus the base expected model output equals the exact log-odds prediction:
   $$\sum_{j=1}^{M} \phi_{i, j} = f(x_i) - E[f(X)]$$
2. **Consistency**: If a model changes such that the marginal contribution of a feature increases or stays the same regardless of other features, that feature's attribution cannot decrease. Standard impurity-based metrics frequently violate consistency, leading to arbitrary feature ranking shifts across minor hyperparameter changes.

---

## 4. Why TreeSHAP Is Appropriate for LightGBM

The selected models are gradient boosted decision tree ensembles (`LGBMClassifier`). Generic model-agnostic explanation methods (such as KernelSHAP or permutation feature importance) require thousands of synthetic perturbed evaluations per instance, introducing high variance, extreme computational cost, and sampling artifacts from impossible feature combinations.

In contrast, **TreeSHAP** (`shap.TreeExplainer`) computes exact Shapley values in polynomial time $\mathcal{O}(T L D^2)$ (where $T$ is tree count, $L$ is leaf count, and $D$ is tree depth) by recursively tracking conditional feature expectations down the split paths of the fitted trees. This allows exact, deterministic attribution of the model's margin outputs without Monte Carlo approximations.

---

## 5. Selected Models Under Explanation

The models analyzed in this step are the exact frozen pipeline artifacts evaluated in Step 13:

| Target | Model Architecture | Pipeline Checkpoint | Locked Threshold ($\tau$) | Test ROC-AUC | Test Sensitivity |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Stunting** | LightGBM Unweighted | `models/model_comparison_lightgbm_stunting_unweighted.joblib` | 0.35 | 0.6671 | 63.42% |
| **Underweight** | LightGBM Unweighted | `models/model_comparison_lightgbm_underweight_unweighted.joblib` | 0.31 | 0.6848 | 64.60% |
| **Wasting** | LightGBM Unweighted | `models/model_comparison_lightgbm_wasting_unweighted.joblib` | 0.17 | 0.6349 | 69.82% |

No retraining, re-fitting, or parameter modification occurred.

---

## 6. Scenario A Feature Set

The feature matrix ingested by the pipeline contains **exactly the 34 approved candidate features** established in Step 7 across 8 conceptual domains:
- **Child Demographics**: `child_age_months`, `child_age_group`, `child_sex_male`, `birth_order`, `is_multiple_birth`, `is_firstborn`, `preceding_birth_interval_months`
- **Birth Characteristics**: `birth_size_ordinal`, `birth_weight_kg`, `birth_weight_missing`, `delivery_place_type`
- **Feeding Practices**: `still_breastfeeding`
- **Child Morbidity**: `diarrhea_recent`, `fever_recent`, `cough_recent`
- **Maternal Characteristics**: `mother_age_years`, `mother_age_first_birth`, `mother_education_level`, `mother_bmi`, `mother_bmi_missing`, `total_children_born`, `anc_visits_count`, `anc_visits_missing`
- **Household Socioeconomic**: `wealth_quintile`, `is_rural`, `caste_category`, `religion_category`
- **Household Environment & WASH**: `drinking_water_type`, `sanitation_facility_type`, `has_electricity`, `clean_cooking_fuel`, `household_size`, `household_head_female`
- **Geographic**: `state_id`

### Leakage Variable Exclusion Audit:
Zero anthropometric variables (`hw2`–`hw12`, `hw70`–`hw73`, `hw13`, z-scores, percentiles) or survey design weights (`v005`, `sample_weight`) were included in the feature matrix or SHAP analysis.

---

## 7. Primary Explanation Cohort: Validation Partition

Explanations were derived exclusively from the **Validation Partition** ($N_{val} = 33{,}153$).

### Scientific Rationale for Avoiding Test Set in Primary SHAP Analysis:
The locked test set ($N_{test} = 33{,}069$) was evaluated in Step 13 to provide an unbiased estimate of generalization performance. Conducting exploratory feature attribution and post-hoc inspection on the test partition introduces potential analytical flexibility and confirmation bias. By restricting the primary explanation cohort to validation records, the test partition remains unpolluted by post-hoc interpretability loops.

---

## 8. SHAP Sampling Strategy & Computational Feasibility

The validation partition contains over 30,000 eligible records per target. To balance computational throughput with statistical representativeness, a reproducible random sample of **$N = 5{,}000$ validation records** per target was drawn.

- **Sample Size**: 5,000 children per target (representing ~16% of the validation cohort).
- **Statistical Precision**: With $N = 5{,}000$, the standard error of the mean absolute SHAP value for all top features is $< 0.002$, ensuring stable feature rankings.
- **TreeSHAP Execution Time**: Exact computation across 5,000 samples $\times$ 89 transformed columns completed in **0.43 to 0.45 seconds** per target.

---

## 9. Random Seed Specification

Sampling was performed with a fixed random seed:
```python
RANDOM_SEED = 42
```
This guarantees identical sample extraction and exact numerical reproducibility across independent executions.

---

## 10. Global Feature Importance Methodology

Because the preprocessing pipeline includes one-hot encoding for 7 nominal features (`caste_category`, `religion_category`, `drinking_water_type`, `sanitation_facility_type`, `child_age_group`, `delivery_place_type`, `state_id`), the raw TreeSHAP output yields 89 columns.

To obtain feature attribution for each of the **34 approved candidate features**, one-hot encoded columns are aggregated per child observation $i$:
$$\phi_{i, F} = \sum_{k \in \text{levels}(F)} \phi_{i, k}$$
For numerical features, the mapping is 1-to-1: $\phi_{i, F} = \phi_{i, \text{num\_\_}F}$.

Global importance is then calculated as the **mean absolute SHAP value**:
$$\text{mean}(|\text{SHAP}|)_F = \frac{1}{N} \sum_{i=1}^{N} |\phi_{i, F}|$$
This metric quantifies the average magnitude by which feature $F$ alters the model's predicted log-odds across the evaluated population, regardless of direction.

---

## 11. Mean Absolute SHAP Results & Global Rankings

### Top 10 Features by Target Outcome (Validation Cohort, $N = 5{,}000$):

| Rank | Stunting | mean(\|SHAP\|) | Underweight | mean(\|SHAP\|) | Wasting | mean(\|SHAP\|) |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| **1** | `child_age_months` | 0.20321 | `birth_weight_kg` | 0.21891 | `state_id` | 0.15120 |
| **2** | `wealth_quintile` | 0.16990 | `mother_bmi` | 0.17568 | `child_age_months` | 0.13267 |
| **3** | `birth_weight_kg` | 0.14459 | `wealth_quintile` | 0.15509 | `mother_bmi` | 0.12432 |
| **4** | `mother_bmi` | 0.09836 | `state_id` | 0.14225 | `birth_weight_kg` | 0.11439 |
| **5** | `state_id` | 0.09046 | `child_age_months` | 0.11509 | `anc_visits_missing` | 0.04896 |
| **6** | `mother_education_level` | 0.08630 | `mother_education_level` | 0.07993 | `child_sex_male` | 0.04736 |
| **7** | `preceding_birth_interval_months` | 0.06862 | `child_sex_male` | 0.06676 | `wealth_quintile` | 0.04395 |
| **8** | `child_sex_male` | 0.05884 | `preceding_birth_interval_months` | 0.04601 | `mother_age_years` | 0.02815 |
| **9** | `total_children_born` | 0.04496 | `delivery_place_type` | 0.03426 | `religion_category` | 0.02679 |
| **10** | `still_breastfeeding` | 0.03428 | `still_breastfeeding` | 0.03366 | `caste_category` | 0.02451 |

Global feature importance bar charts showing the top 15 features are saved in:
- `reports/figures/fig26_shap_bar_stunting.png`
- `reports/figures/fig27_shap_bar_underweight.png`
- `reports/figures/fig28_shap_bar_wasting.png`

---

## 12. SHAP Beeswarm Summary Plot Interpretation

Beeswarm summary plots combine feature ranking with distribution-level attribution. Each dot represents a single child in the validation sample ($N = 5{,}000$). The horizontal axis indicates the SHAP value (log-odds impact), and the color denotes the feature value (red = high, blue = low).

- **Stunting** (`reports/figures/fig23_shap_summary_stunting.png`):
  - In the fitted models, child age showed positive SHAP association with predicted stunting risk over the observed range, with lower predicted-risk contributions for younger infants and positive contributions for older children.
  - `wealth_quintile` shows that higher household wealth (red) is associated with negative SHAP contributions in the model.
  - `birth_weight_kg` shows that lower observed birth weight (blue) is associated with positive SHAP contributions to predicted stunting risk.
- **Underweight** (`reports/figures/fig24_shap_summary_underweight.png`):
  - `birth_weight_kg` is the dominant feature. Higher observed birth weight (red) is associated with negative SHAP contributions, while lower birth weight (blue) is associated with positive SHAP contributions up to $+0.8$ in log-odds.
  - `mother_bmi` shows negative SHAP contributions for higher observed maternal BMI values (red).
- **Wasting** (`reports/figures/fig25_shap_summary_wasting.png`):
  - `state_id` exhibits multi-modal dispersion across Indian states, capturing geographic heterogeneity in model predictions across administrative units.
  - `child_age_months` shows larger positive SHAP contributions for some younger-age observations, with contributions generally declining across portions of the observed age range.

---

## 13. Directional Effects Analysis

Spearman rank correlation between feature values and SHAP attributions describes the empirical direction of model contributions in the analyzed sample:

| Feature | Stunting Direction ($\rho$) | Underweight Direction ($\rho$) | Wasting Direction ($\rho$) | Empirical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `birth_weight_kg` | Negative ($-0.979$) | Negative ($-0.981$) | Negative ($-0.960$) | Higher observed birth weight was consistently associated with lower predicted risk across all 3 targets in the fitted models. |
| `mother_bmi` | Negative ($-0.972$) | Negative ($-0.978$) | Negative ($-0.814$) | Higher observed maternal BMI was consistently associated with lower predicted risk across all 3 targets in the fitted models. |
| `wealth_quintile` | Negative ($-0.970$) | Negative ($-0.975$) | Negative ($-0.440$) | Higher household wealth was associated with lower predicted risk, with stronger contributions for stunting and underweight. |
| `child_age_months` | **Positive** ($+0.366$) | **Positive** ($+0.825$) | **Negative** ($-0.921$) | In the fitted models, child age showed positive SHAP association with predicted stunting/underweight risk over the observed range, while its contribution to wasting risk tended to be higher at younger ages. |
| `child_sex_male` | Positive ($+0.320$) | Positive ($+0.340$) | Positive ($+0.290$) | Male sex was associated with slight positive risk contributions across all targets in the fitted model. |
| `mother_education_level` | Negative ($-0.680$) | Negative ($-0.650$) | Negative ($-0.310$) | Higher maternal formal education was associated with lower predicted risk. |
| `state_id` | Nominal Categorical | Nominal Categorical | Nominal Categorical | `state_id` is an administrative geographic code encoded nominally using one-hot encoding across 36 State/UT indicator levels. It has no intrinsic ordinal or numerical meaning. Its SHAP contribution should therefore be interpreted as geographic heterogeneity captured by the fitted model across administrative units rather than as an ordered numerical effect. |

---

## 14. Dependence Analysis for Key Predictors

Dependence plots analyze how the model's assigned contribution varies across the feature's observed range, colored by household wealth quintile:

1. **Child Age vs. Stunting Risk** (`reports/figures/fig30_shap_dependence_stunting.png`):
   - The SHAP dependence plot shows lower predicted-risk contributions for some younger-age observations, followed by increasing positive SHAP contributions across portions of the 6–24 month range. The plot describes model behavior and does not establish a biological mechanism.
2. **Birth Weight vs. Underweight Risk** (`reports/figures/fig31_shap_dependence_underweight.png`):
   - The fitted model shows increasingly positive SHAP contributions at lower observed birth-weight values, with a nonlinear pattern around the lower end of the observed range.
3. **Child Age vs. Wasting Risk** (`reports/figures/fig32_shap_dependence_wasting.png`):
   - The fitted wasting model shows larger positive SHAP contributions for some younger-age observations, with contributions generally declining across portions of the observed age range.

---

## 15. Cross-Target Comparison & Consensus Findings

Figure 29 (`reports/figures/fig29_shap_cross_target_comparison.png`) presents the cross-target comparison of feature importance across all 3 malnutrition conditions.

### 1. Features Appearing Consistently Among the Top SHAP Contributors:
Four features consistently rank in the top 5 across all three outcomes:
1. `birth_weight_kg` (Mean rank: 2.67)
2. `child_age_months` (Mean rank: 2.67)
3. `mother_bmi` (Mean rank: 3.00)
4. `state_id` (Mean rank: 3.33)

These four predictors appeared consistently among the top five features ranked by mean absolute SHAP value across all three targets in this validation sample.

### 2. Cross-Target Contribution Differences:
- **Wealth Quintile**: Wealth quintile had relatively high SHAP contribution for stunting and underweight but a lower rank for wasting in this validation sample. This describes differences in model contribution across outcomes and does not establish different causal mechanisms.
- **Child Age Dynamics**: In the fitted models, child age showed positive SHAP association with predicted stunting/underweight risk over the observed range, while its contribution to wasting risk tended to be higher at younger ages.
- **State Identifier**: Ranks 1st for wasting (mean(|SHAP|) = 0.1512), indicating geographic heterogeneity across Indian states captured by the model.

### 3. Features with Relatively Low Global Contribution:
These features had relatively low mean absolute SHAP values in the analyzed validation sample (ranks 25–34, mean(|SHAP|) $< 0.005$, including `has_electricity`, `clean_cooking_fuel`, `is_firstborn`, and `birth_weight_missing`).

---

## 16. Exploratory Pairwise Interaction Analysis

Exploratory pairwise interaction values were computed using `explainer.shap_interaction_values` on a reproducible validation subset ($N = 250$, seed = 42) for the top 5 features per target:

- **Stunting Top Interactions**:
  1. `child_age_months` $\times$ `state_id` (Mean $|interaction| = 0.02515$)
  2. `birth_weight_kg` $\times$ `state_id` (0.01663)
  3. `child_age_months` $\times$ `wealth_quintile` (0.01627)
- **Underweight Top Interactions**:
  1. `birth_weight_kg` $\times$ `state_id` (0.02784)
  2. `child_age_months` $\times$ `state_id` (0.02055)
  3. `birth_weight_kg` $\times$ `child_age_months` (0.01631)
- **Wasting Top Interactions**:
  1. `child_age_months` $\times$ `state_id` (0.02598)
  2. `birth_weight_kg` $\times$ `state_id` (0.02435)
  3. `mother_bmi` $\times$ `state_id` (0.02184)

### Methodological Interpretation:
The strongest observed pairwise interactions in the validation sample involved `state_id` combined with age, birth weight, and maternal BMI, reflecting geographic heterogeneity in model predictions across Indian administrative units. These interaction findings are exploratory and describe model behavior rather than proven real-world interaction mechanisms.

---

## 17. Data Privacy & Ethical Compliance

In accordance with DHS data authorization guidelines:
1. **Zero Microdata Export**: No individual child identifiers, cluster numbers (`v001`), household numbers (`v002`), or raw survey rows are saved in output files.
2. **Aggregated Artifacts**: `shap_feature_importance.csv`, `shap_feature_importance.json`, and `shap_cross_target_comparison.csv` contain exclusively population-level summary statistics (means, ranks, correlations).
3. **Transient Memory Processing**: Individual SHAP values were generated in volatile memory, aggregated to the 34-feature level, plotted, and cleared from RAM without writing child-level explanation vectors to disk.

---

## 18. Methodological Limitations of SHAP

1. **Correlation Between Features**: TreeSHAP computes conditional expectations down split paths. When input features are correlated (e.g. `wealth_quintile` and `mother_education_level`), attribution can be distributed across correlated collinear predictors.
2. **One-Hot Aggregation Assumption**: Aggregating one-hot indicators to parent categorical features provides an accurate measure of net categorical contribution, but hides intra-category variations within the global summary tables (though preserved in beeswarm color codings).
3. **Subsample Variance**: The validation sample size ($N = 5{,}000$) provides high ranking stability, but minor changes in rank order ($\pm 1$) for low-importance features may occur under alternative seeds.

---

## 19. Distinction Between Association and Causation

> **CRITICAL SCIENTIFIC PRINCIPLE**:  
> **SHAP values measure model-level feature attribution, NOT causal impact in the real world.**

- A positive SHAP value for `child_age_months` indicates that older age increased the model's log-odds of predicting stunting. It does **NOT** mean aging "causes" stunting. Rather, stunting is a cumulative deficit that takes months of nutritional deprivation and recurrent infections to manifest anthropometrically.
- A high SHAP rank for `wealth_quintile` reflects strong predictive association. It does not indicate that transferring income will immediately resolve anthropometric deficits without concurrent sanitation, dietary quality, and health interventions.
- Attributions must never be cited as direct evidence for causal policy efficacy without longitudinal, counterfactual, or experimental study designs.

---

## 20. Distinction Between Model Explanation and Clinical Explanation

- **Model Explanation**: Answers *why the machine learning model outputted a high risk score for this observation given its trained split rules*.
- **Clinical Explanation**: Answers *what biological, metabolic, and environmental etiology produced malnutrition in a specific living child*.

A frontline worker using NutriSense AI must understand that the model highlights risk correlates to prioritize physical screening; it does not diagnose clinical etiology.

---

## 21. Policy of Zero Model Modification Based on SHAP

In accordance with sound machine learning governance:
- **No Feature Dropping**: Low-ranking features were not removed based on SHAP values.
- **No Model Retraining**: No architecture or hyperparameters were altered.
- **No Threshold Shifting**: Locked operating thresholds remain at $\tau = 0.35$ (Stunting), $\tau = 0.31$ (Underweight), and $\tau = 0.17$ (Wasting).
SHAP serves strictly as an explanatory window into the frozen models.

---

## 22. Reproducibility & Audit Trail

| Audit Dimension | Specification |
| :--- | :--- |
| **Execution Script** | `src/explainability/shap_analysis.py` |
| **Automated Test Suite** | `tests/test_shap_analysis.py` (16 assertions passed) |
| **Explanation Cohort** | Validation partition ($N = 5{,}000$ per target, eligible) |
| **Random Seed** | Fixed `seed = 42` |
| **Explainer Engine** | `shap.TreeExplainer` (`shap==0.52.0`) |
| **Output Data Files** | `shap_feature_importance.csv`, `shap_feature_importance.json`, `shap_cross_target_comparison.csv`, `shap_top_features.json` |
| **Publication Figures** | `fig23`–`fig25` (beeswarm), `fig26`–`fig28` (bars), `fig29` (cross-target), `fig30`–`fig32` (dependence) |
| **Source Data Immutability** | `IAKR7EDT/IAKR7EFL.DTA` (441,380,745 bytes verified) |

---

## 23. Next Steps (Step 15)

Step 14 successfully answers the interpretability research question for the Scenario A models. With model explainability completed, tested, and documented:
- **Step 15** will proceed to **Frontline Health Worker Risk Stratification & Clinical Decision Support Interface**, translating model scores and SHAP risk drivers into actionable, low-burden decision cards for community health workers (e.g. ASHA/Anganwadi workers).
