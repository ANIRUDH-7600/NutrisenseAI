# Step 5 — Ground-Truth Target Creation

## Objective
The objective of Step 5 is to systematically construct and validate the ground-truth clinical outcome targets for **NutriSense AI** based on internationally accepted World Health Organization (WHO) 2006 Child Growth Standards, using cleaned continuous standard deviation z-scores (`hw70_clean`, `hw71_clean`, and `hw72_clean`).

We define and validate three distinct childhood undernutrition conditions—**Stunting**, **Underweight**, and **Wasting**—along with their corresponding severe clinical manifestations. We establish strict boundary-condition rules, enforce exact missing-value propagation (missing z-score produces missing target `NaN`, never $0$), analyze epidemiological target overlap in the unified complete cohort ($N = 198{,}802$), and compute exact class imbalance statistics.

In strict adherence to research integrity, **no machine learning models were trained, no feature selection was performed, no SMOTE or class balancing was applied, and no data splitting was executed yet.**

---

## 1. Clinical Outcome Formulation: Three Distinct Targets

A fundamental principle of NutriSense AI is that **we do NOT create a generic, monolithic "malnutrition" target.** 

From an epidemiological and public health perspective, stunting, wasting, and underweight represent physiologically distinct conditions that correlate with different risk factors reported in the literature and require distinct public health intervention strategies:
- **Stunting (Height-for-Age deficit, HAZ)**: Reflects chronic linear growth retardation associated with long-term nutritional inadequacy and recurrent health challenges over developmental periods.
- **Wasting (Weight-for-Height deficit, WHZ)**: Reflects acute deficit in body mass relative to height, frequently associated in the literature with recent illness (such as acute diarrhea or fever) or sudden nutritional disruptions.
- **Underweight (Weight-for-Age deficit, WAZ)**: A composite indicator influenced by both linear growth status (stunting) and acute body mass deficit (wasting).

Merging these three distinct conditions into a single binary label would obscure distinct physiological patterns, confuse feature attribution in explainable AI (e.g., recent acute illness strongly associates with wasting in empirical studies but shows weaker association with chronic stunting), and prevent frontline health workers from triaging specific interventions (e.g., immediate supplementary feeding for acute wasting vs long-term dietary diversity for chronic stunting).

---

## 2. WHO 2006 Growth-Standard Thresholds

Based on the official WHO Child Growth Standards and DHS Program statistical guidelines, undernutrition cutoffs are defined using standard deviation units ($Z$) from the international reference median:

| Target Name | Cleaned Source Variable | WHO Growth Index | Documented Formula | Clinical Classification |
| :--- | :--- | :--- | :---: | :--- |
| **`stunting`** | `hw70_clean` | Height-for-Age (HAZ) | $\text{HAZ} < -2.00\text{ SD}$ | Moderate-or-severe chronic stunting |
| **`stunting_severe`** | `hw70_clean` | Height-for-Age (HAZ) | $\text{HAZ} < -3.00\text{ SD}$ | Severe chronic stunting |
| **`underweight`** | `hw71_clean` | Weight-for-Age (WAZ) | $\text{WAZ} < -2.00\text{ SD}$ | Moderate-or-severe composite underweight |
| **`underweight_severe`** | `hw71_clean` | Weight-for-Age (WAZ) | $\text{WAZ} < -3.00\text{ SD}$ | Severe composite underweight |
| **`wasting`** | `hw72_clean` | Weight-for-Height (WHZ) | $\text{WHZ} < -2.00\text{ SD}$ | Moderate-or-severe acute wasting |
| **`wasting_severe`** | `hw72_clean` | Weight-for-Height (WHZ) | $\text{WHZ} < -3.00\text{ SD}$ | Severe acute wasting |

---

## 3. Boundary-Condition Audit & Missing-Value Governance

### Strict Inequality vs. Boundary Values
In official WHO and DHS definitions, undernutrition is explicitly defined as **falling below** $-2\text{ SD}$ and $-3\text{ SD}$. 
Therefore:
$$\text{Condition} = \begin{cases} 1.0 & \text{if } Z < -2.00\text{ SD} \\ 0.0 & \text{if } Z \ge -2.00\text{ SD} \\ \text{NaN} & \text{if } Z \text{ is missing}\end{cases}$$

- **Boundary Test at Exactly $-2.00$**: A child whose z-score is exactly $-2.00$ is at the boundary of the reference standard and is classified as **condition absent ($0.0$)**. In our dataset, exactly 522 children have $\text{HAZ} = -2.00$, 585 children have $\text{WAZ} = -2.00$, and 311 children have $\text{WHZ} = -2.00$. All are correctly classified as $0.0$.
- **Boundary Test at Exactly $-3.00$**: A child whose z-score is exactly $-3.00$ is classified as **non-severe ($0.0$)** in the severe indicators. In our dataset, 303 children have $\text{HAZ} = -3.00$, 262 children have $\text{WAZ} = -3.00$, and 136 children have $\text{WHZ} = -3.00$. All are correctly classified as $0.0$ for the severe indicators.
- **Strict NaN Propagation**: If a child was unmeasured, refused, absent, or flagged by the WHO standard as biologically implausible, their target is strictly assigned `NaN`. **Zero cases of missing outcomes were converted to 0.**

---

## 4. Target Distributions & Analytical Sample Prevalence

Every number reported below was computed directly from the cleaned NFHS-5 dataset ($N = 221{,}263$ living under-five children):

### A. Target-Specific Cohort Distributions (Unweighted Sample Prevalence)

| Clinical Target | Total Records | Valid Observations | Missing Observations | Positive Count ($1$) | Negative Count ($0$) | Unweighted Sample Prevalence (%) | Negative (%) | Imbalance Ratio (Neg : Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`stunting`** | 221,263 | **206,025** | 15,238 (6.89%) | **73,072** | 132,953 | **35.47%** | 64.53% | **1 : 1.82** |
| **`underweight`** | 221,263 | **210,524** | 10,739 (4.85%) | **65,043** | 145,481 | **30.90%** | 69.10% | **1 : 2.24** |
| **`wasting`** | 221,263 | **201,687** | 19,576 (8.85%) | **37,553** | 164,134 | **18.62%** | 81.38% | **1 : 4.37** |

### B. Severe Target Distributions (Unweighted Sample Prevalence)

| Severe Target | Valid Observations | Severe Positive Count ($1$) | Severe Negative Count ($0$) | Severe Prevalence (%) | Severe Imbalance Ratio (Neg : Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`stunting_severe`** | 206,025 | **31,518** | 174,507 | **15.30%** | **1 : 5.54** |
| **`underweight_severe`** | 210,524 | **21,735** | 188,789 | **10.32%** | **1 : 8.69** |
| **`wasting_severe`** | 201,687 | **15,332** | 186,355 | **7.60%** | **1 : 12.15** |

> [!NOTE]
> **Survey-Weighting Context**: The prevalence percentages reported above represent the **unweighted analytical sample proportions** in our cleaned under-five cohort. Official survey reports published by the International Institute for Population Sciences (IIPS) and MoHFW incorporate complex survey design weights (`v005`), cluster stratification, and non-response adjustments (which yield weighted national figures of 35.5% for stunting, 32.1% for underweight, and 19.3% for wasting). We preserve the raw sampling weights (`sample_weight`) to investigate survey-weighted statistics separately in downstream analysis.

---

## 5. Target Overlap Analysis (Complete Unified Cohort, $N = 198{,}802$)

In the complete unified cohort, all three anthropometric indicators are simultaneously valid and unflagged:

```
Total Complete Unified Cohort: N = 198,802
-------------------------------------------------------------
None / Healthy (All 3 Normal):                     96,055 (48.32%)
At least one anthropometric failure (Deficit >= 1): 102,747 (51.68%)
```

### Granular Anthropometric Overlap Breakdown

| Overlap Category | Stunted? | Underweight? | Wasted? | Child Count | Percentage (%) | Epidemiological Context in Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **None (Healthy)** | No | No | No | **96,055** | **48.32%** | Child meets reference standards across all three growth indices. |
| **Stunting Only** | **Yes** | No | No | **31,678** | **15.93%** | Linear growth retardation with mass-for-height and mass-for-age $\ge -2\text{ SD}$. |
| **Wasting Only** | No | No | **Yes** | **12,768** | **6.42%** | Low mass-for-height with height-for-age and mass-for-age $\ge -2\text{ SD}$. |
| **Underweight Only** | No | **Yes** | No | **4,169** | **2.10%** | Low mass relative to age without meeting stunting or wasting criteria. |
| **Stunting + Underweight** | **Yes** | **Yes** | No | **30,043** | **15.11%** | Chronic growth deficit concurrent with low mass relative to age. |
| **Underweight + Wasting** | No | **Yes** | **Yes** | **14,463** | **7.28%** | Low mass-for-height concurrent with low mass relative to age. |
| **Stunting + Wasting Only** | **Yes** | No | **Yes** | **0** | **0.00%** | **Dataset Observation**: No children in the NFHS-5 complete unified cohort were observed with stunting and wasting without simultaneously meeting the underweight criterion. |
| **All Three (S + U + W)** | **Yes** | **Yes** | **Yes** | **9,626** | **4.84%** | Children meeting all three anthropometric criteria represent simultaneous anthropometric deficits and form an important subgroup for descriptive analysis. |

*Note on Terminology*: The proportion of children experiencing at least one deficit (51.68%) is conceptually related to Peter Svedberg's (2000) Composite Index of Anthropometric Failure (CIAF). However, formal CIAF aggregation and subgroup classification have not yet been adopted in this project.

---

## 6. Target-Specific Cohorts vs. Unified Cohort Summary

| Cohort Formulation | Sample Size ($N$) | Stunting Prevalence | Underweight Prevalence | Wasting Prevalence | Research Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Complete Unified Cohort** | **198,802** | 35.89% (71,347) | 29.33% (58,301) | 18.54% (36,857) | **Primary Benchmark Cohort**: Strict head-to-head comparison across all models on identical children; direct replication of Islam et al. (2024). |
| **Stunting Cohort** | **206,025** | 35.47% (73,072) | — | — | Standalone stunting model; retains 7,223 additional valid height measurements. |
| **Underweight Cohort** | **210,524** | — | 30.90% (65,043) | — | Standalone underweight model; retains 11,722 additional valid weight measurements. |
| **Wasting Cohort** | **201,687** | — | — | 18.62% (37,553) | Standalone wasting model; retains 2,885 additional valid wasting measurements. |

---

## 7. Class Imbalance Assessment

Machine learning models require careful monitoring of class imbalance ratios:
- **Stunting (1 : 1.82)**: Mild imbalance (35.47% positive). Tree models and logistic regression typically train stably under this ratio without synthetic oversampling.
- **Underweight (1 : 2.24)**: Moderate imbalance (30.90% positive).
- **Wasting (1 : 4.37)**: Substantial imbalance (18.62% positive). Class weighting or threshold tuning will be evaluated in Step 11 to address potential false-negative rates in acute wasting screening.
- **Severe Targets (1 : 5.54 to 1 : 12.15)**: Marked imbalance (7.60% for severe wasting). Requires evaluation using PR-AUC and cost-sensitive loss functions.

---

## 8. Target Leakage Protection (Scenario A Contract)

The newly created target columns:
- `stunting`, `stunting_severe`
- `underweight`, `underweight_severe`
- `wasting`, `wasting_severe`

are **ground-truth labels ($y$)**.

> [!IMPORTANT]
> In our **Scenario A (Community Pre-Screening)** contract:
> 1. Target labels ($y$) must **NEVER** enter the input feature matrix ($X$).
> 2. The source anthropometric variables (`hw70_clean`, `hw71_clean`, `hw72_clean`, `hw73_clean`) and physical measurements (`hw2_clean`, `hw3_clean`, `hw4`–`hw12`) **remain strictly isolated from $X$**.
> 3. An automated assertion will be enforced during feature matrix assembly in Step 7 to guarantee zero target leakage.

---

## 9. Base Paper Comparison (Islam et al., 2024, PLOS ONE)

| Dimension | Base Research Paper (Islam et al., 2024) | NutriSense AI (Our Project) | Comparison & Alignment |
| :--- | :--- | :--- | :--- |
| **Target Formulation** | Evaluated stunting, wasting, and underweight as three separate classification tasks. | Evaluated stunting, wasting, and underweight as three separate classification tasks. | **Exact Methodological Match**. |
| **Cutoff Thresholds** | $\text{HAZ} < -2\text{ SD}$, $\text{WAZ} < -2\text{ SD}$, $\text{WHZ} < -2\text{ SD}$. | $\text{HAZ} < -2.00\text{ SD}$, $\text{WAZ} < -2.00\text{ SD}$, $\text{WHZ} < -2.00\text{ SD}$. | **Exact Match** based on WHO 2006 Child Growth Standards. |
| **Severe Targets** | Did not model severe malnutrition ($Z < -3\text{ SD}$). | Implemented severe targets (`stunting_severe`, `underweight_severe`, `wasting_severe`). | **Our Extension**: Adds targeted risk intelligence for severe clinical deficits. |
| **Sample Size** | $N = 7{,}859$ (Bangladesh BDHS 2017–18) | $N = 198{,}802$ (Unified) to $206{,}025$ (Target-Specific) | **>25x Larger Cohort** in India NFHS-5. |
| **Unweighted Prevalence** | Stunting: 30.8% (2,423)<br>Underweight: 22.1% (1,737)<br>Wasting: 8.4% (660) | Stunting: 35.47% (73,072)<br>Underweight: 30.90% (65,043)<br>Wasting: 18.62% (37,553) | Higher prevalence in our Indian analytical sample across all three indicators, reflecting distinct demographic cohorts. |

---

## 10. Research Decisions Made vs. Intentionally Deferred

### Decisions Made in Step 5:
1. Created three clinically distinct ground-truth target variables (`stunting`, `underweight`, `wasting`) and three severe targets (`stunting_severe`, `underweight_severe`, `wasting_severe`).
2. Enforced strict WHO growth-standard thresholds ($Z < -2.00$ and $Z < -3.00$).
3. Verified boundary values: exactly $-2.00$ and $-3.00$ are classified as condition absent ($0.0$).
4. Enforced strict `NaN` propagation for missing z-scores.
5. Saved the target-augmented intermediate dataset to `data/interim/cleaned_u5_with_targets.csv.gz` (gitignored).

### Decisions Intentionally Deferred to Step 6+:
1. **Exploratory Data Analysis**: Univariate and spatial correlation analysis is deferred to **Step 6 (EDA)**.
2. **Feature Selection & Missing Value Imputation**: Deferred to **Step 7 (Feature Engineering)**.
3. **Train / Test Splitting**: Deferred to **Step 8 (Data Splitting)**.
4. **Handling Class Imbalance**: Applying SMOTE or class weights is strictly forbidden before splitting and is deferred to **Step 11 (Class Imbalance)**.

---

## 11. Viva / Interview Questions & Answers

### Q1: Why did you model stunting, underweight, and wasting as three separate binary targets rather than creating a single composite malnutrition target?
**Sample Answer**:
"In public health and clinical pediatrics, malnutrition is not a single uniform entity. Stunting (HAZ < -2 SD) is a chronic condition reflecting cumulative linear growth retardation associated with long-term socioeconomic and environmental factors. Wasting (WHZ < -2 SD) is an acute condition reflecting deficit in body mass relative to height, frequently associated in the literature with recent illness such as acute diarrhea. Underweight (WAZ < -2 SD) is a composite index. Modeling them as separate targets maintains clinical specificity, avoids confounding distinct physiological conditions, and allows future explainability tools (like SHAP) to isolate predictors specific to chronic vs acute nutritional deficits."

### Q2: In your target creation logic, is a child with a HAZ of exactly -2.00 considered stunted or normal?
**Sample Answer**:
"A child with a HAZ of exactly -2.00 SD is classified as non-stunted (target = 0). According to official WHO 2006 Child Growth Standards and DHS guidelines, undernutrition is mathematically defined as strictly *below* -2 standard deviations from the international reference median ($Z < -2.00$). The value -2.00 represents the boundary threshold. In our dataset, exactly 522 children have a HAZ of exactly -2.00, and our automated boundary tests verify that all 522 are assigned target = 0."

### Q3: What did your target overlap analysis show regarding stunting and wasting without underweight?
**Sample Answer**:
"In our complete unified cohort of 198,802 children, no children were observed with stunting and wasting who were not also underweight. This is an empirical dataset observation reflecting how severe combined deficits in height-for-age and weight-for-height correspond with low weight-for-age in this population. Furthermore, 48.32% of children had no anthropometric deficit across all three indices, while 51.68% exhibited at least one anthropometric failure, and 4.84% (9,626 children) met the criteria for all three conditions simultaneously."

### Q4: Does your machine learning model claim to prove what causes child malnutrition?
**Sample Answer**:
"No. Predictive machine learning models identify correlational patterns and statistical associations between predictors and target outcomes; they do not establish causal relationships. While factors such as maternal education, wealth index, and sanitation are strong predictive features documented in the epidemiological literature, claiming that a feature causes undernutrition requires formal counterfactual causal inference frameworks or prospective randomized trials, which is outside the scope of this predictive screening system."

---

## Commands Used
```bash
# Run target creation pipeline
python src/features/create_targets.py

# Run Step 5 verification test suite
python tests/test_target_creation.py
```

---

## Next Step
**Step 6 — Exploratory Data Analysis (EDA)**: Conduct comprehensive exploratory data analysis on feature distributions, demographic patterns, geographic variations across Indian states, bivariate associations with malnutrition targets, and publication-quality visualizations saved under `reports/figures/`.
