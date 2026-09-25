# Step 8 — Train / Validation / Test Split Documentation

**NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System**  
*Scenario A — Community Pre-Screening Setting*  
*Dataset: NFHS-5 India (2019–21) Children's Recode (KR)*

---

## 1. Objective

The objective of Step 8 is to establish and validate an experimental split strategy that is strictly leakage-safe for the NutriSense AI project. In clinical, demographic, and public health predictive modeling, splitting child observations across partitions without considering intra-household clustering leads to overly optimistic performance estimates due to information leakage across related records. 

Step 8 achieves:
1. Identifying and validating the household cluster unit (`v001`, `v002`) as the primary grouping entity.
2. Grouping child records by household so that all children residing in the same household are strictly assigned to the exact same partition.
3. Partitioning the eligible cohort ($N = 221{,}263$) into **70% Training**, **15% Validation**, and **15% Test** partitions.
4. Stratifying households using a composite multi-label target profile across the three clinical outcomes: **Stunting**, **Underweight**, and **Wasting**.
5. Preserving target-specific outcome cohorts while holding out complete test sets.
6. Establishing the **Test-Set Lock Principle** and enforcing zero preprocessing snooping.

---

## 2. Why Data Splitting is Required

In machine learning pipelines for public health and clinical pre-screening, evaluating a model on the same data used to fit its parameters leads to empirical over-optimism (overfitting). A rigorous three-way split is necessary:
- **Training Set (70%)**: Used solely to learn model parameters (decision trees, linear coefficients, neural weights) and fit data transformers (imputers, encoders, scalers).
- **Validation Set (15%)**: Used to compare competing algorithms, select optimal hyperparameters, choose decision thresholds for sensitivity/specificity operating points, and calibrate probabilities without contaminating the final evaluation.
- **Test Set (15%)**: Held out in strict isolation. It serves as an uncompromised benchmark to estimate future out-of-sample generalization performance on unseen households.

---

## 3. Potential Household Leakage

In the NFHS-5 Children's Recode (KR) dataset, individual rows represent children born in the preceding 5 years to surveyed mothers. Multiple children frequently reside in the same household as siblings, twins, or cousins.

### The Leakage Mechanism:
If row-level random splitting were applied:
1. Child A and Child B from the same household could be assigned to Train and Test partitions respectively.
2. Both children share near-identical predictor values:
   - Maternal characteristics (maternal education `v106`, maternal age at birth, maternal BMI `v445`, maternal stature).
   - Household economic and environmental indicators (wealth index `v190`, sanitation facility `v116`, drinking water source `v113`, cooking fuel `v161`, electricity `v119`).
   - Geographical and cultural practices (state `v024`, place of residence `v025`, caste/tribe group `s116`).
   - Unobserved genetic and biological traits.
3. A decision tree or gradient boosting model could memorize specific combinations of household features in the training partition and falsely achieve near-perfect predictions on sibling records in the test partition.
4. This produces artificially inflated AUC and accuracy metrics that collapse when deployed in real-world community settings where new, un-encountered households are screened.

---

## 4. Split Unit Determination

To determine the appropriate split unit, we analyzed the hierarchical structure of NFHS-5:
- `v001`: Cluster number (Primary Sampling Unit / Census Enumeration Block).
- `v002`: Household number within cluster.
- `v003`: Respondent line number (mother's position in household schedule).
- `bidx`: Birth index of child for the mother.

**Household Identifier**: The tuple `(v001, v002)`, formulated as `hh_id = str(v001) + "_" + str(v002)`, uniquely identifies each domestic living unit.
**Maternal Identifier**: The tuple `(v001, v002, v003)`, formulated as `mother_id = hh_id + "_" + str(v003)`, identifies individual mothers.

### Mother-Level Nesting Check:
We verified whether mothers can span multiple households:
- Analysis confirmed that **100% of mothers** belong to exactly one household schedule (`max_households_per_mother = 1`).
- Therefore, grouping by household `(v001, v002)` guarantees that **no mother's children are ever split across partitions**. Household-level grouped splitting prevents both household-level and mother-level information leakage.

---

## 5. Household Grouping Verification & Statistics

From the living under-five cohort ($N = 221{,}263$), quantitative clustering metrics were extracted:

| Metric | Value | Proportion |
| :--- | :---: | :---: |
| **Total Child Observations** | 221,263 | 100.00% |
| **Total Unique Households** | 164,339 | 100.00% |
| **Single-Child Households** | 115,922 | 70.54% of households |
| **Multi-Child Households** | 48,417 | **29.46% of households** |
| **Children in Multi-Child Households** | 105,341 | **47.61% of all children** |
| **Maximum Children in Single Household** | 9 | — |
| **Total Unique Mothers** | 172,421 | 100.00% |
| **Max Children per Mother** | 5 | — |

### Household Size Distribution (Children per Household):
- **1 child**: 115,922 households (70.54%)
- **2 children**: 41,330 households (25.15%)
- **3 children**: 5,948 households (3.62%)
- **4 children**: 922 households (0.56%)
- **5 children**: 169 households (0.10%)
- **6 children**: 35 households (0.02%)
- **7 children**: 11 households (0.007%)
- **8 children**: 1 household (0.0006%)
- **9 children**: 1 household (0.0006%)

> [!IMPORTANT]
> **Key Finding**: Nearly half of all children (**47.61%**) reside in households with more than one eligible child. This empirically demonstrates that row-level random splitting would compromise almost half of the dataset, making grouped household splitting methodologically essential.

---

## 6. Train / Validation / Test Proportions

The splitting pipeline strictly targets a **70% / 15% / 15%** partition at the household grouping level.

| Partition | Households ($N$) | Household Share | Child Records ($N$) | Child Record Share |
| :--- | :---: | :---: | :---: | :---: |
| **Training** | 115,037 | **70.00%** | 155,041 | **70.07%** |
| **Validation** | 24,651 | **15.00%** | 33,153 | **14.98%** |
| **Test** | 24,651 | **15.00%** | 33,069 | **14.95%** |
| **Total** | **164,339** | **100.00%** | **221,263** | **100.00%** |

*Note*: Because households vary slightly in child count (1 to 9), mapping exact 70.00% / 15.00% / 15.00% household partitions to child records yields 70.07% / 14.98% / 14.95% child-level proportions, perfectly within the $\pm 0.1\%$ operational tolerance.

---

## 7. Random Seed & Reproducibility

- **Random Seed**: `SEED = 42`.
- The split is implemented via `sklearn.model_selection.train_test_split` with explicit deterministic seeding across both stages:
  - Stage 1: Partition 164,339 households into 70% Train ($N = 115{,}037$) and 30% Holdout ($N = 49{,}302$) with `random_state=42`.
  - Stage 2: Partition 49,302 holdout households into 50% Validation ($N = 24{,}651$) and 50% Test ($N = 24{,}651$) with `random_state=42`.
- Exact reproducibility was verified programmatically: running the pipeline independently produced identical household assignments (`train_hh == train_hh2`, `val_hh == val_hh2`, `test_hh == test_hh2`) across 100% of records.

---

## 8. Stratification Methodology (Multi-Label Target Patterns)

Because NutriSense AI predicts three distinct malnutrition targets (**Stunting**, **Underweight**, **Wasting**), naive single-target stratification would distort the distributions of the remaining two targets.

### Formulation:
1. For each household, we compute the maximum status across eligible children for each target:
   $$\text{hh\_s} = \max(\text{stunting}), \quad \text{hh\_u} = \max(\text{underweight}), \quad \text{hh\_w} = \max(\text{wasting})$$
2. An outcome string is constructed: `s_u_w`, where `s, u, w` $\in \{0, 1, \text{'M'}\}$, representing positive (1), negative (0), or missing/unmeasured (M).
3. This creates up to 25 theoretical pattern combinations (e.g., `0_0_0` healthy, `1_1_0` stunting + underweight, `1_1_1` composite deficit, `M_M_M` unmeasured).
4. **Rare Stratum Binning**: Any stratum with fewer than 30 households is mapped to a composite `'rare_mix'` stratum. This guarantees that all strata have sufficient samples to be divided into 70/15/15 partitions without singleton errors or zero-division anomalies.
5. In practice, 24 robust strata were formed, ensuring optimal representation across all joint malnutrition configurations.

---

## 9. Target-Specific Cohort Handling

As established in Step 3 and Step 5, anthropometric measurements in NFHS-5 have target-specific eligibility:
- Stunting cohort: $N = 206{,}025$ valid observations.
- Underweight cohort: $N = 210{,}524$ valid observations.
- Wasting cohort: $N = 201{,}687$ valid observations.

The grouped household split was executed across the **entire living under-five cohort ($N = 221{,}263$)**. When training or evaluating a target-specific model:
- The model filters to records where `eligible_{target} == True` within the respective partition (`train`, `val`, or `test`).
- This preserves maximum sample size for each target while guaranteeing that test households remain completely held out across all three targets.

---

## 10. Household Leakage Checks

Automated set-intersection assertions were executed on the household partitions:
$$\text{Train} \cap \text{Validation} = \emptyset$$
$$\text{Train} \cap \text{Test} = \emptyset$$
$$\text{Validation} \cap \text{Test} = \emptyset$$

### Audit Results:
- `intersection(train_households, validation_households)`: **0**
- `intersection(train_households, test_households)`: **0**
- `intersection(validation_households, test_households)`: **0**
- Complete partition sum: $115{,}037 + 24{,}651 + 24{,}651 = 164{,}339$ (100.00%).

Additionally, candidate feature matrix verification confirmed that no cluster or household identifier (`v001`, `v002`, `v003`, `v005`, `hh_id`, `mother_id`) or split assignment variable (`split`) enters the candidate feature registry.

---

## 11. Target Distribution Checks Across Partitions

The prevalence and class balance of all three targets were checked across partitions and compared against the full cohort:

### 1. Stunting (`hw70_clean < -2.0`)
- **Overall Cohort**: Valid $N = 206{,}025$, Pos = 73,072, Neg = 132,953, **Prevalence = 35.47%**, Ratio = 0.5496
- **Training Set**: Valid $N = 144{,}368$, Pos = 51,148, Neg = 93,220, **Prevalence = 35.43%**, Ratio = 0.5487
- **Validation Set**: Valid $N = 30{,}885$, Pos = 10,963, Neg = 19,922, **Prevalence = 35.50%**, Ratio = 0.5503
- **Test Set**: Valid $N = 30{,}772$, Pos = 10,961, Neg = 19,811, **Prevalence = 35.62%**, Ratio = 0.5533
- *Max Prevalence Difference Across Splits*: **0.19%**

### 2. Underweight (`hw71_clean < -2.0`)
- **Overall Cohort**: Valid $N = 210{,}524$, Pos = 65,043, Neg = 145,481, **Prevalence = 30.90%**, Ratio = 0.4471
- **Training Set**: Valid $N = 147{,}560$, Pos = 45,664, Neg = 101,896, **Prevalence = 30.95%**, Ratio = 0.4481
- **Validation Set**: Valid $N = 31{,}531$, Pos = 9,673, Neg = 21,858, **Prevalence = 30.68%**, Ratio = 0.4425
- **Test Set**: Valid $N = 31{,}433$, Pos = 9,706, Neg = 21,727, **Prevalence = 30.88%**, Ratio = 0.4467
- *Max Prevalence Difference Across Splits*: **0.27%**

### 3. Wasting (`hw72_clean < -2.0`)
- **Overall Cohort**: Valid $N = 201{,}687$, Pos = 37,553, Neg = 164,134, **Prevalence = 18.62%**, Ratio = 0.2288
- **Training Set**: Valid $N = 141{,}328$, Pos = 26,291, Neg = 115,037, **Prevalence = 18.60%**, Ratio = 0.2285
- **Validation Set**: Valid $N = 30{,}211$, Pos = 5,635, Neg = 24,576, **Prevalence = 18.65%**, Ratio = 0.2293
- **Test Set**: Valid $N = 30{,}148$, Pos = 5,627, Neg = 24,521, **Prevalence = 18.66%**, Ratio = 0.2295
- *Max Prevalence Difference Across Splits*: **0.06%**

> [!NOTE]
> All target prevalence differences between Train, Validation, and Test are below **0.27%**, indicating that the multi-target household stratification strategy preserved representative target distributions without distortion.

---

## 12. State Distribution Checks Across Partitions

Because `state_id` (`v024`) is an approved candidate predictor representing regional and administrative variance, state proportions were audited across partitions.

- Total administrative units: **36 states and Union Territories**.
- State distributions across Train, Validation, and Test partitions were recorded in `split_metadata.json`.
- The maximum variation (`max_spread_pct = max(train, val, test) - min(train, val, test)`) across any state was **0.43%** (State 20: 4.28% in train, 4.13% in val, 4.56% in test).
- Highly populated states were naturally balanced:
  - Uttar Pradesh (`v024 = 9`): 15.00% Train, 15.35% Val, 15.28% Test (spread: 0.35%).
  - Bihar (`v024 = 10`): 8.92% Train, 8.70% Val, 8.59% Test (spread: 0.33%).
  - Madhya Pradesh (`v024 = 23`): 8.08% Train, 7.96% Val, 8.07% Test (spread: 0.12%).
- No state shows severe representation imbalance.

---

## 13. Reproducibility Verification

1. The random seed (`42`) is stored in `split_metadata.json`.
2. A dedicated test (`test_reproducibility_verification` in `tests/test_split_data.py`) asserts that rebuilding the split yields 100% identical household assignments.
3. A lightweight partition manifest (`data/interim/split_manifest.csv.gz`) records the split assignment for every child index, enabling deterministic reloading across future training and evaluation stages.

---

## 14. Test-Set Lock Principle

To preserve scientific validity, NutriSense AI enforces the **Test-Set Lock Principle**:
1. **Zero Access During Development**: The test set ($N = 33{,}069$) must remain completely locked and untouched during model exploration, feature selection, pipeline architecture experiments, and hyperparameter tuning.
2. **No Threshold Tuning on Test**: Decision thresholds (e.g., probability cutoffs for high-sensitivity screening) must be calibrated exclusively on the Validation partition ($N = 33{,}153$).
3. **Single Final Benchmark**: The test set will be evaluated exactly once per final candidate model architecture to record the unbiased out-of-sample performance metrics.

---

## 15. Preprocessing Leakage Prevention Rule

A strict constraint enforced in Step 8 is **Zero Preprocessing Snooping**:
- **No transformers fitted in Step 8**: Imputers (median/mode), scalers (robust/standard), encoders (one-hot/ordinal), and feature selectors must NOT be fitted in Step 8.
- In subsequent steps, all learned transformations must be:
  $$\text{transformer.fit}(X_{\text{train}})$$
  $$X_{\text{train\_trans}} = \text{transformer.transform}(X_{\text{train}})$$
  $$X_{\text{val\_trans}} = \text{transformer.transform}(X_{\text{val}})$$
  $$X_{\text{test\_trans}} = \text{transformer.transform}(X_{\text{test}})$$
- Fitting any imputer or scaler on the complete cohort before splitting leaks statistical properties (means, variances, frequencies) into the test partition.

---

## 16. Base-Paper Comparison

### Reference Study: Islam et al. (2024), PLOS ONE
*"Prediction of undernutrition and identification of its influencing predictors among under-five children in Bangladesh using explainable machine learning algorithms"*

| Dimension | Islam et al. (2024) | NutriSense AI (Step 8) |
| :--- | :--- | :--- |
| **Survey Data** | BDHS 2017–18 (Bangladesh) | NFHS-5 2019–21 (India) |
| **Sample Size** | ~8,759 under-five children | **221,263 under-five children** |
| **Splitting Strategy** | Standard 80/20 train/test row split (unclear grouping) | **70 / 15 / 15 Grouped Household Split** |
| **Household Clustering** | Not explicitly grouped by household | **Explicitly grouped by `(v001, v002)`** |
| **Leakage Controls** | Not documented for intra-household sibling pairs | **Strict zero-overlap assertions ($0$ overlapping households)** |
| **Stratification** | Single target or default random | **Multi-target composite stratification across 3 outcomes** |
| **Validation Set** | Cross-validation on train or split | **Dedicated 15% ($N=33{,}153$) validation set for tuning/thresholds** |

*Methodological Note*: We do not claim our pipeline is superior to Islam et al. in performance, but our design addresses a structural feature of DHS microdata: in NFHS-5, 47.61% of children share a household with another eligible child. Explicitly grouping by household provides stronger protection against intra-household leakage.

---

## 17. Methodological Limitations

1. **Not a Geographic Generalization Test**:
   - The grouped household split evaluates generalization to *unseen households within surveyed clusters/regions*.
   - It does **not** evaluate spatial or out-of-region generalization (e.g., leave-one-state-out or leave-one-district-out). Geographic generalization must be assessed in separate spatial transferability experiments.
2. **Cluster-Level Correlations**:
   - Households within the same primary sampling unit (`v001` cluster/village) may share environmental risk factors (local groundwater contamination, local healthcare infrastructure). Cluster-level grouped splitting would require grouping by village, which alters the primary prediction unit for pre-screening.
3. **Survey Weight Omission in Splitting**:
   - Splitting was performed using unweighted household counts. Survey weights (`v005`) are reserved for descriptive epidemiological estimations and not applied to distort machine learning validation sets.

---

## 18. Viva / Defense Interview Q&A

**Q1: Why did you use a grouped split instead of standard random splitting?**  
*Answer*: In NFHS-5, 48,417 households (29.46%) contain multiple eligible under-five children, representing 47.61% of all children in the analytical cohort. Randomly splitting by child would place siblings in both training and test sets. Since siblings share maternal education, household wealth, sanitation, diet, genetics, and water sources, the model could achieve inflated test scores by memorizing household features. Grouping by household `(v001, v002)` ensures clean separation.

**Q2: How does household grouping affect maternal-level leakage?**  
*Answer*: In DHS methodology, respondent mothers are enumerated within their household schedule (`v002`). Our empirical audit verified that 100% of mothers are nested within exactly one household (`max_households_per_mother = 1`). Grouping by household therefore automatically groups by mother.

**Q3: Why a 70 / 15 / 15 split instead of an 80 / 20 split?**  
*Answer*: A three-way split provides a dedicated validation set (15%, $N = 33{,}153$) to tune hyperparameters, compare model families (LightGBM, XGBoost, CatBoost, Logistic Regression), and select operational screening thresholds (e.g., fixing sensitivity $\ge 80\%$) without touching the held-out test set ($N = 33{,}069$).

**Q4: How did you handle stratification for three simultaneous targets?**  
*Answer*: We constructed a household-level composite outcome profile combining the maximum status for Stunting, Underweight, and Wasting (e.g., `1_1_0`, `0_0_1`). Households with rare combinations (< 30 occurrences) were pooled into a `'rare_mix'` stratum. This yielded 24 robust strata, maintaining target prevalence across all three partitions within $\pm 0.27\%$.

**Q5: Why are children with missing anthropometric measurements retained in the split?**  
*Answer*: NFHS-5 has target-specific missingness ($N = 206{,}025$ for stunting, $N = 210{,}524$ for underweight, $N = 201{,}687$ for wasting). Retaining all 221,263 living children in the partition manifest ensures that models trained for a specific target utilize all valid observations for that target, while keeping holdout households consistent.

**Q6: Did you apply SMOTE, imputation, or scaling before splitting?**  
*Answer*: No. Applying any learned transformation (imputation, scaling, one-hot encoding, or resampling) before splitting constitutes data leakage. All transformers must be fitted strictly on `X_train`.

**Q7: How did you verify that zero households leaked into the test set?**  
*Answer*: We evaluated set intersections: `intersection(train_hh, val_hh) == ∅`, `intersection(train_hh, test_hh) == ∅`, and `intersection(val_hh, test_hh) == ∅`. All three returned 0 overlapping households.

**Q8: Does this split prove that the model will generalize to other states or districts?**  
*Answer*: No. Grouped household splitting tests generalization to unseen households in surveyed clusters. Evaluating geographical generalization across unseen districts or states requires a spatial cross-validation strategy (e.g., Leave-One-State-Out), which is a separate experimental design.

**Q9: Why are household IDs (`v001`, `v002`) excluded from the feature matrix?**  
*Answer*: Grouping variables serve solely to partition the data. Including cluster or household IDs in $X$ would cause models to memorize geographical locations or household keys rather than learning generalizable risk factors.

**Q10: What is the Test-Set Lock Principle?**  
*Answer*: The test set is sequestered from the moment of creation. It is never used for feature selection, hyperparameter tuning, model architecture selection, or threshold calibration. It is evaluated only once on finalized candidate models to provide an honest estimate of real-world generalization.

---

## 19. Exact Commands

```powershell
# 1. Execute Step 8 split pipeline
python src/data/split_data.py

# 2. Run Step 8 verification tests
python tests/test_split_data.py

# 3. Run complete regression test suite (Steps 0 to 8)
python tests/test_setup.py
python tests/test_inspection.py
python tests/test_data_dictionary.py
python tests/test_research_population.py
python tests/test_data_cleaning.py
python tests/test_target_creation.py
python tests/test_eda.py
python tests/test_feature_engineering.py
python tests/test_split_data.py
```

---

## 20. Files Created / Modified

| File | Type | Description |
| :--- | :---: | :--- |
| [`src/data/split_data.py`](file:///d:/finalyearproj/Nutrisense-Ai/src/data/split_data.py) | **Created** | Script executing 70/15/15 grouped household split, stratification, and zero-leakage assertions. |
| [`data/interim/split_metadata.json`](file:///d:/finalyearproj/Nutrisense-Ai/data/interim/split_metadata.json) | **Created** | Comprehensive metadata recording clustering stats, partition sizes, target/state distributions, and leakage audits. |
| [`data/interim/split_manifest.csv.gz`](file:///d:/finalyearproj/Nutrisense-Ai/data/interim/split_manifest.csv.gz) | **Created** | Reproducible mapping connecting each child record to its split partition (`train`, `val`, `test`). |
| [`data/interim/cleaned_u5_with_targets.csv.gz`](file:///d:/finalyearproj/Nutrisense-Ai/data/interim/cleaned_u5_with_targets.csv.gz) | **Updated** | Added `'split'` column to interim dataset for downstream model access. |
| [`tests/test_split_data.py`](file:///d:/finalyearproj/Nutrisense-Ai/tests/test_split_data.py) | **Created** | Automated test suite verifying 10 assertions for Step 8. |
| [`docs/08_train_validation_test_split.md`](file:///d:/finalyearproj/Nutrisense-Ai/docs/08_train_validation_test_split.md) | **Created** | Exhaustive technical documentation covering all 21 prompt requirements. |

---

## 21. Next Step (Step 9 Preview)

**Step 9 — Preprocessing Pipeline Design & Imputation Baseline**:
- Build a scikit-learn `ColumnTransformer` / `Pipeline` fitted **exclusively on the training partition**.
- Implement domain-informed missingness handlers:
  - Missing indicator creation for informatively missing predictors (e.g., birthweight `m19`, antenatal visits `m14`).
  - Median/mode imputers for continuous/categorical variables fitted strictly on `X_train`.
  - One-hot encoding for nominal categories with unobserved category handling.
- Verify zero information leakage from validation and test sets during transformation.
- Prepare baseline pipeline for model exploration.
