# Step 6 — Exploratory Data Analysis (EDA)

## Objective
The objective of Step 6 is to perform a rigorous, non-causal exploratory data analysis of the cleaned NFHS-5 under-five analytical cohort ($N = 221{,}263$) and the three ground-truth clinical targets (**Stunting**, **Underweight**, and **Wasting**). 

We investigate demographic, maternal, socioeconomic, environmental, and morbidity distributions, evaluate missingness patterns, analyze statistical associations using appropriate non-parametric metrics (Cramer's V across 12 candidate categorical variables), examine unweighted sample-level geographic heterogeneity across 36 Indian States and Union Territories, report unweighted analytical cohort prevalence alongside descriptive prevalence using DHS sample weights, and generate publication-quality figures saved under `reports/figures/`.

In strict adherence to scientific integrity:
- **No causal claims are made** (we do not claim any variable "causes" malnutrition).
- **No machine learning models were trained**.
- **No feature selection was conducted based on test-set information**.
- **No data splitting was performed**.
- **No SMOTE or resampling was applied**.
- **No imputation was executed**.

---

## 1. Dataset & Analytical Cohort Overview

- **Source File**: `data/interim/cleaned_u5_with_targets.csv.gz`
- **Total Eligible Cohort**: **221,263** living children aged 0–59 completed months.
- **Total Columns**: 72 columns (candidate predictors, cleaned continuous z-scores, audit variables, survey design parameters, and binary clinical targets).
- **Target Availability & Target-Specific Cohorts**:
  - **Stunting (`stunting`)**: 206,025 valid observations (93.11% of living cohort; 15,238 unmeasured or flagged).
  - **Underweight (`underweight`)**: 210,524 valid observations (95.15% of living cohort; 10,739 unmeasured or flagged).
  - **Wasting (`wasting`)**: 201,687 valid observations (91.15% of living cohort; 19,576 unmeasured or flagged).
  - **Complete Unified Cohort**: **198,802** children with all three outcomes simultaneously valid (89.85% of living cohort).

---

## 2. Target Distributions: Unweighted Analytical Cohort Prevalence vs. Descriptive Estimates Using DHS Sample Weights

We report both **unweighted analytical cohort prevalence** (used for model loss optimization and sample evaluation) and **descriptive estimates using DHS sample weights** (`sample_weight = v005 / 1,000,000`).

### Methodological Distinction
To maintain rigorous epidemiological standards, we clearly distinguish three categories of estimates:
1. **Unweighted Analytical Cohort Prevalence**: The raw empirical proportion of positive cases observed within our cleaned, eligible analytical under-five cohort ($N = 221{,}263$). This empirical distribution directly reflects the dataset on which candidate screening models will be developed.
2. **Descriptive Prevalence Using DHS Sample Weights**: Descriptive prevalence computed by weighting individual observations by normalized DHS sampling weights (`v005 / 1,000,000`). This descriptive estimate accounts for unequal probability of selection across survey strata within this analytical cohort.
3. **Official Published NFHS-5 National Estimates**: Official population estimates published by IIPS and MoHFW in national reports, which incorporate full complex survey design parameters—including multi-stage stratified cluster sampling, primary sampling unit (PSU) clustering, strata definitions, finite population corrections, and official post-stratification.

> [!IMPORTANT]
> Applying `v005 / 1,000,000` alone does NOT fully reproduce the complete NFHS-5 complex survey estimation methodology. Descriptive prevalence using DHS sample weights must NOT be claimed as or conflated with official published NFHS-5 national estimates. Furthermore, weighted and unweighted descriptive estimates can differ across outcomes because sample weights adjust for strata with disproportionate sampling fractions; calculating descriptive estimates using DHS sample weights is useful for descriptive context alongside unweighted cohort distributions.

| Clinical Target | Valid Sample ($N$) | Positive Count ($1$) | Negative Count ($0$) | Unweighted Analytical Cohort Prevalence (%) | Descriptive Prevalence Using DHS Sample Weights (%) | Class Imbalance Ratio (Neg : Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stunting (`stunting`)** | 206,025 | 73,072 | 132,953 | **35.47%** | **35.47%** | **1 : 1.82** |
| **Underweight (`underweight`)** | 210,524 | 65,043 | 145,481 | **30.90%** | **32.08%** | **1 : 2.24** |
| **Wasting (`wasting`)** | 201,687 | 37,553 | 164,134 | **18.62%** | **19.22%** | **1 : 4.37** |
| **Severe Stunting (`stunting_severe`)** | 206,025 | 31,518 | 174,507 | **15.30%** | **15.09%** | **1 : 5.54** |
| **Severe Underweight (`underweight_severe`)** | 210,524 | 21,735 | 188,789 | **10.32%** | **10.62%** | **1 : 8.69** |
| **Severe Wasting / SAM (`wasting_severe`)** | 201,687 | 15,332 | 186,355 | **7.60%** | **7.66%** | **1 : 12.15** |

*Descriptive Observation*: Weighted and unweighted descriptive estimates differ slightly across outcomes (e.g., stunting is 35.47% unweighted vs 35.47% weighted; underweight is 30.90% unweighted vs 32.08% weighted; wasting is 18.62% unweighted vs 19.22% weighted). This difference is expected as sampling weights re-balance the relative contribution of sampling strata with differing sampling fractions.

---

## 3. Predictor Missingness Architecture

In Step 6, missingness patterns were audited across candidate predictors without performing imputation:
- **Structural Non-Applicability**: Preceding birth interval (`b11`) has 38.47% missingness ($N = 85{,}130$), reflecting firstborn children (`bord == 1`) who logically have no preceding sibling. Similarly, weaned breastfeeding duration (`m4_completed_months`) has 67.78% non-applicability because 106,538 infants are actively still breastfeeding (`still_breastfeeding == 1`) and 43,050 were not in the detailed feeding recall roster.
- **Invasive Test Consent**: Child anemia (`hw57`) has 16.91% missingness ($N = 37{,}408$) and maternal anemia (`v457`) has 3.60% missingness ($N = 7{,}970$) due to refusal or absence during capillary blood collection.
- **Institutional Scale Availability**: Child birth weight (`birth_weight_kg`) has 9.41% missingness ($N = 20{,}824$) due to unmeasured home deliveries or forgotten birth weights.
- **Zero Missingness**: Core demographic and socioeconomic predictors (`hw1` age, `b4` sex, `bord` birth order, `v012` maternal age, `v106` maternal education, `v190` wealth quintile, `v025` residence, `v024` state) have **0.00% missingness** across all 221,263 children.

---

## 4. Key Exploratory Findings Across Analytical Domains

### A. Age Dynamics (Figure 3)
We examined the prevalence of each malnutrition condition across 6 developmental age brackets:
- **Acute Wasting**: Highest in early infancy (**26.00% in 0–5 months** and **21.96% in 6–11 months**), followed by a steady decline to **15.94% in 48–59 months**.
- **Chronic Stunting**: Relatively lower in early infancy (**24.47% in 0–5 months**). Stunting prevalence was highest in the 12–23-month age group in this analytical cohort (**39.70%**), remaining elevated at **39.17% in 36–47 months**. (From an epidemiological background perspective in published pediatric literature, the 12–23 month window coincides with the introduction of complementary feeding and heightened environmental pathogen exposure, though this descriptive EDA does not infer that complementary feeding caused the observed stunting pattern.)
- **Underweight**: Shows an intermediate pattern, rising from 27.89% in early infancy to plateau around 32.5% between 24 and 59 months.
- *Analytical Implication*: Age shows nonlinear empirical patterns across the outcomes. Step 7 will evaluate suitable representations of age and allow candidate models to capture potential nonlinear relationships.

### B. Socioeconomic & Maternal Gradients (Figure 4)
- **Household Wealth Quintile (`v190`)**:
  - Stunting prevalence is lower in higher wealth quintiles, decreasing from **44.95% in the poorest quintile** to **22.60% in the richest quintile** (a 22.35 percentage point difference).
  - Underweight prevalence is **40.56% (poorest)** versus **18.78% (richest)**.
  - Wasting prevalence is **21.72% (poorest)** versus **14.80% (richest)**.
- **Maternal Education (`v106`)**:
  - Stunting prevalence is **44.81% among children of mothers with no formal education**, compared to **23.51% among children of mothers with higher education** (a 21.30 percentage point difference).
  - Underweight prevalence is **40.31% (no education)** versus **19.69% (higher education)**.
- **Maternal Body Mass Index (`v445_clean`)**:
  - Children of mothers categorized as underweight ($\text{BMI} < 18.5\text{ kg/m}^2$) exhibit **43.30% stunting**, **42.57% underweight**, and **23.61% wasting**.
  - Children of overweight/obese mothers exhibit lower stunting ($20.91\%$) and underweight ($13.14\%$).
- **Place of Residence (`v025`)**:
  - Rural residence associates with higher stunting (**37.44% rural** vs **29.74% urban**) and underweight (**32.61% rural** vs **25.92% urban**).

### C. Morbidity Patterns: Acute vs. Chronic Deficits (Figure 5)
- **Recent Acute Diarrhea (`diarrhea_recent`)**:
  - Wasting prevalence among children who experienced diarrhea in the preceding 2 weeks is **23.54%**, compared to **18.25% among children without recent diarrhea** (a relative difference of 29.0%).
  - In contrast, stunting prevalence shows minimal divergence (**36.87% with diarrhea** vs **35.37% without diarrhea**, a relative difference of 4.2%).
- **Recent Fever (`fever_recent`)**:
  - Wasting prevalence is **22.02% among children with recent fever**, compared to **18.14% among those without fever**.
- *Analytical Implication*: Recent acute morbidity recall features demonstrate an associated pattern with acute wasting in bivariate comparisons, whereas their bivariate divergence with chronic stunting is comparatively small.

---

## 5. Statistical Association Matrix: Cramer's V Across 12 Categorical Variables (Figure 6)

Because many candidate predictors in DHS are categorical or binned, we computed **Cramer's V** to evaluate bivariate association effect sizes across all **12 candidate categorical features** and the three binary malnutrition targets:

$$V = \sqrt{\frac{\chi^2 / N}{\min(r - 1, k - 1)}}$$

| Candidate Predictor Feature | Variable Name | Type / Levels | Association with Stunting ($V$) | Association with Underweight ($V$) | Association with Wasting ($V$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Household Wealth Quintile** | `wealth_label` | Ordinal (5 levels) | **0.1637** | **0.1642** | 0.0600 |
| **Maternal Education Level** | `edu_label` | Ordinal (4 levels) | **0.1459** | **0.1421** | 0.0447 |
| **Maternal BMI Category** | `maternal_bmi_cat`| Categorical (4 levels) | **0.1053** | **0.1544** | **0.0851** |
| **Size at Birth** | `m18_clean` | Ordinal (5 levels) | 0.0434 | 0.0639 | 0.0318 |
| **Place of Residence** | `residence_label` | Binary (Urban/Rural) | 0.0618 | 0.0569 | 0.0146 |
| **Household Electricity** | `v119` | Binary (Yes/No) | 0.0517 | 0.0429 | 0.0158 |
| **Still Breastfeeding** | `still_breastfeeding` | Binary (Yes/No) | 0.0273 | 0.0302 | 0.0394 |
| **Child Sex** | `b4` | Binary (Male/Female) | 0.0249 | 0.0185 | 0.0207 |
| **Twin / Multiple Birth** | `b0` | Binary (Single/Multiple) | 0.0140 | 0.0196 | 0.0048 |
| **Recent Diarrhea (2 wks)** | `diarrhea_recent` | Binary (Yes/No) | 0.0055 | 0.0171 | 0.0184 |
| **Recent Fever (2 wks)** | `fever_recent` | Binary (Yes/No) | 0.0057 | 0.0158 | 0.0137 |
| **Recent Cough / ARI (2 wks)**| `cough_recent` | Binary (Yes/No) | 0.0000 | 0.0017 | 0.0000 |

*Key Statistical Observation*:
- **Stunting and Underweight**: Household wealth and maternal education show the largest Cramer's V values among the categorical variables examined for stunting and underweight (with $V \approx 0.16$ for wealth and $V \approx 0.14$ for maternal education; maternal BMI category also shows $V \approx 0.15$ for underweight and $0.11$ for stunting). We do not automatically describe these values as "strong" without a specific interpretation framework, but note them as the largest bivariate associations observed among the tested categorical variables.
- **Wasting**: The examined static household variables show smaller bivariate Cramer's V values for wasting than for stunting and underweight (all static household variables have $V \le 0.060$, with maternal BMI category at $0.085$). We do not state that this proves wasting is dynamically episodic, but observe that the examined static household variables show smaller bivariate Cramer's V values for wasting than for stunting and underweight in this dataset.

---

## 6. Geographic Heterogeneity Across Indian States / UTs (Figure 7)

State/UT results are reported as **unweighted analytical-cohort prevalence** (or **sample prevalence**) across 36 Indian States and Union Territories ($N = 221{,}263$):

- **Stunting Variation (Sample Prevalence)**: Unweighted analytical-cohort prevalence ranges from **18.78% in Puducherry** and **22.39% in Kerala** to **41.77% in Uttar Pradesh**, **42.92% in Bihar**, and **46.73% in Meghalaya**.
- **Wasting Variation (Sample Prevalence)**: Unweighted analytical-cohort prevalence ranges from **8.39% in Mizoram** and **11.23% in Manipur** to **21.84% in Maharashtra**, **22.46% in Gujarat**, and **23.11% in Dadra & Nagar Haveli**.

### Methodological Limitations & Confidentiality
1. **Sample Prevalence Only**: These figures represent **unweighted analytical-cohort prevalence** within our cleaned under-five dataset, NOT official state-level NFHS-5 estimates. States are not labeled as "high-burden" or "low-burden" based solely on this analysis.
2. **Not Official NFHS-5 State Estimates**: Official state estimates published by IIPS/MoHFW incorporate complex survey weights, primary sampling unit clustering, and survey strata adjustments.
3. **Data Confidentiality Maintained**: In strict adherence to DHS data privacy and research ethics, no cluster-level identifiers, primary sampling unit (PSU) coordinates, or GPS geographic data are exposed or analyzed.

---

## 7. Generated Visualizations in `reports/figures/`

All 7 figures have been generated and saved with high resolution (300 DPI) in [reports/figures/](file:///d:/finalyearproj/Nutrisense-Ai/reports/figures):

1. **`fig1_target_prevalence_and_imbalance.png`** (139 KB): Grouped bar chart depicting moderate-or-severe ($< -2\text{ SD}$) versus severe ($< -3\text{ SD}$) prevalence across Stunting (35.5% vs 15.3%), Underweight (30.9% vs 10.3%), and Wasting (18.6% vs 7.6%).
2. **`fig2_missingness_profile.png`** (204 KB): Horizontal bar chart displaying predictor missingness percentages and differentiating structural non-applicability (firstborn spacing `b11`: 38.5%) from unmeasured clinical consent (`hw57`: 16.9%).
3. **`fig3_age_dynamics_by_outcome.png`** (253 KB): Longitudinal age-profile curves illustrating how acute wasting prevalence is highest in early infancy (0–5 months: 26.0%) while stunting prevalence was highest in the 12–23-month age group (39.7%) in this analytical cohort.
4. **`fig4_socioeconomic_maternal_gradients.png`** (350 KB): 2x2 multi-panel figure visualizing the monotonic prevalence declines across household wealth quintiles, maternal education categories, maternal BMI brackets, and place of residence.
5. **`fig5_morbidity_associations_acute_vs_chronic.png`** (135 KB): Paired bar chart demonstrating that recent 2-week diarrhea and fever episodes correlate with elevated acute wasting rates in bivariate comparison while showing negligible divergence for chronic stunting.
6. **`fig6_statistical_association_cramers_v.png`** (332 KB): Heatmap visualizing Cramer's V association effect sizes across all 12 candidate categorical features and the three malnutrition targets.
7. **`fig7_state_level_stunting_wasting.png`** (174 KB): Horizontal bar chart displaying unweighted analytical-cohort prevalence across selected Indian States and Union Territories with contrasting sample prevalence.

---

## 8. What These EDA Observations Mean for Feature Engineering (Step 7)

1. **Age Representations**: Age shows nonlinear empirical patterns across the outcomes. Step 7 will evaluate suitable representations of age and allow candidate models to capture potential nonlinear relationships.
2. **Missingness Indicator Features**: Missingness in birth weight (`birth_weight_kg`) and preceding birth interval (`b11`) is non-random and carries informative signal (e.g., firstborn children vs unweighed home deliveries). We will evaluate creating explicit binary missingness flags (`birth_weight_missing`, `is_firstborn`) during feature engineering.
3. **Target-Specific Association Patterns**: The differing association patterns will be considered when designing and evaluating feature sets for the three target outcomes.
4. **Regional Variation**: Given the variance across Indian states in this cohort (18% to 46% unweighted sample stunting), state-level indicators (`v024`) and place of residence (`v025`) should be encoded appropriately to allow models to account for subnational variation.

---

## 9. What Was Deliberately NOT Done in Step 6

To maintain research integrity:
- **No Causal Inferences**: We do not claim that poverty "causes" stunting or that diarrhea "causes" wasting; we report observed statistical associations and prevalence differences.
- **No Data Splitting**: Train/test splitting must occur in Step 8 before any model training or preprocessing transformations.
- **No Imputation**: Missing values were analyzed but zero values were imputed.
- **No SMOTE or Resampling**: Class imbalance was measured (1:1.82 to 1:4.37) but not resampled.
- **No Feature Selection**: All candidate predictors remain under consideration.

---

## 10. Base Paper Comparison (Islam et al., 2024, PLOS ONE)

| Dimension | Base Research Paper (Islam et al., 2024) | NutriSense AI (Our Project) | Comparison & Methodological Notes |
| :--- | :--- | :--- | :--- |
| **EDA Sample Size** | $N = 7{,}859$ children (BDHS 2017–18) | **$N = 221{,}263$** living U5 ($N = 198{,}802$ unified) | **>25x Larger Sample Size** in India NFHS-5. |
| **Statistical Associations** | Used Chi-square tests of independence without detailed effect-size heatmaps. | Computed **Cramer's V effect sizes across 12 candidate categorical features** alongside bivariate comparisons. | Our EDA provides normalized effect sizes ($V$) across all three separate target outcomes. |
| **Age Profiling** | Grouped age into 5 coarse bins. | Examined granular monthly trajectories (0–5, 6–11, 12–23, 24–35, 36–47, 48–59 mo). | Captures early infancy acute wasting patterns ($26.0\%$ at 0–5 mo). |
| **Geographic Diversity** | 8 administrative divisions of Bangladesh. | **36 States and Union Territories of India**. | Captures subnational regional heterogeneity across India. |
| **Morbidity Focus** | Included diarrhea and fever in general descriptive tables. | Evaluated acute illness divergence between acute wasting vs chronic stunting. | Clarified that acute symptoms show stronger empirical association with wasting than stunting in bivariate comparisons. |

---

## 11. Limitations of Step 6 EDA
- **Unweighted Analytical Cohort Prevalence vs. Official Population Estimates**: Bivariate crosstabs and state distributions represent unweighted analytical cohort prevalence within the cleaned under-five dataset. While descriptive estimates using DHS sample weights are also reported for context, applying `v005 / 1,000,000` alone does not reproduce complex survey design estimation (PSU clustering, stratification, finite population correction) and should not be cited as official published census/NFHS-5 figures.
- **Confounding & Multicollinearity**: Bivariate associations (e.g., between wealth and stunting) do not control for confounding variables like maternal education, sanitation, or geographic location. Multivariate modeling in Step 9+ will be required to assess independent feature contributions.
- **Self-Reported Morbidity**: Diarrhea, fever, and cough rely on 2-week maternal recall, which is subject to recall bias and subjective symptom perception.

---

## 12. Viva / Interview Questions & Answers

### Q1: Why did you use Cramer's V instead of Pearson correlation for categorical variables in your EDA?
**Sample Answer**:
"Pearson correlation assumes continuous, normally distributed variables with linear relationships, making it mathematically invalid for nominal or multi-category survey variables such as maternal education, wealth quintiles, or child sex. Cramer's V is a non-parametric effect-size measure derived from the Chi-square statistic ($\chi^2$), bounded strictly between 0 (no association) and 1 (perfect association). It provides a standardized metric to compare the relative strength of association between categorical demographic and socioeconomic predictors and binary malnutrition outcomes without violating distributional assumptions."

### Q2: Why does acute wasting peak in early infancy (0–5 months) while chronic stunting peaks in the 12–23 month group in this analytical cohort?
**Sample Answer**:
"From an epidemiological background perspective in published pediatric literature, acute wasting represents immediate mass deficit relative to length, which in early infancy often associates with intrauterine growth restriction, low birth weight, sub-optimal exclusive breastfeeding, or acute neonatal infections. In contrast, stunting represents cumulative linear growth retardation that accumulates over time. In our analytical cohort, stunting prevalence was highest in the 12–23-month age group (39.70%), which in broader literature corresponds to the complementary feeding window where children face increased environmental exposure and weaning transitions. However, our EDA observes this descriptive empirical pattern and does not claim that complementary feeding causes stunting."

### Q3: Did your EDA find that poverty causes child malnutrition?
**Sample Answer**:
"No. Observational survey data analyzed via cross-tabulations and association statistics cannot prove causality. Our EDA demonstrated a pronounced statistical association—children in the poorest wealth quintile had higher stunting prevalence (44.95%) compared to the richest quintile (22.60%), with a Cramer's V of 0.1637. However, wealth index is a proxy for multiple correlated structural factors, including sanitation, dietary diversity, maternal education, and healthcare access. Inferring true causality would require counterfactual modeling or prospective randomized trials, which is outside the scope of observational screening systems."

---

## Commands Used
```bash
# Run comprehensive EDA and generate figures
python src/data/eda.py

# Run Step 6 automated verification test suite
python tests/test_eda.py
```

---

## Files Created / Modified
- `src/data/eda.py`: EDA calculation and visualization script with refined terminology and labels.
- `reports/figures/fig1_target_prevalence_and_imbalance.png` to `fig7_state_level_stunting_wasting.png`: 7 generated figures.
- `data/interim/eda_summary.json`: Comprehensive structured summary metrics.
- `tests/test_eda.py`: Automated verification test suite.
- `docs/06_eda.md`: Updated comprehensive EDA documentation.

---

## Next Step
**Step 7 — Feature Engineering**: Systematically assemble candidate predictors, encode categorical variables, evaluate suitable representations of age and missingness indicator flags, enforce strict data leakage isolation between labels ($y$) and features ($X$), and document all included and rejected variables.
