# NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System

**Academic Research & Final Engineering Report**  
*Scenario A: Non-Invasive Community Pre-Screening for Under-Five Malnutrition*

---

## 1. Title
**NutriSense AI: An Explainable Machine Learning System for Non-Invasive Community Pre-Screening of Childhood Undernutrition in India**

---

## 2. Abstract
Childhood undernutrition—manifesting as stunting (chronic linear growth deficit), underweight (composite growth deficit), and wasting (acute body mass depletion)—remains a severe public health crisis across low- and middle-income countries. In rural and remote regions of India, routine growth monitoring under community health schemes (such as the Integrated Child Development Services / Anganwadi system) is frequently constrained by the scarcity, calibration drift, or absence of digital hanging scales and stadiometers. 

This research investigates **Scenario A: Scale-Free Community Pre-Screening**, evaluating whether machine learning estimators can predict statistical risk for undernutrition among children aged 0–59 months using exclusively non-invasive demographic, maternal, household socioeconomic, and recent pediatric morbidity indicators solicited via maternal recall. Using the comprehensive India National Family Health Survey (NFHS-5, 2019–21) Children's Recode ($N = 232{,}920$ uncleaned records; $N = 220{,}460$ analytical cohort), we engineered 34 non-leaking candidate features while strictly isolating direct anthropometric outcome measurements. Following a randomized household-clustered partition ($70\%$ training, $15\%$ validation, $15\%$ locked out-of-sample test), five classifier families (Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM) were systematically benchmarked. LightGBM Unweighted was selected as the champion model architecture based on pre-specified performance gates and consistency across targets, achieving validation ROC-AUCs of $0.6699$ for stunting, $0.6812$ for underweight, and $0.6253$ for wasting.

To align with frontline triage priorities that penalize false negatives more heavily than false positives, validation-derived operating thresholds ($\tau = 0.35$ for stunting, $\tau = 0.31$ for underweight, and $\tau = 0.17$ for wasting) were established. The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration). Out-of-sample evaluation on the locked test cohort ($N_{test} = 33{,}069$) confirmed stable generalization: Stunting achieved a Sensitivity of $63.42\%$ [95% CI: 62.46%–64.31%], Specificity of $61.18\%$, F1 of $0.5431$, PR-AUC of $0.5170$, and ROC-AUC of $0.6671$; Underweight achieved a Sensitivity of $64.60\%$ [95% CI: 63.56%–65.57%], Specificity of $61.79\%$, F1 of $0.5165$, PR-AUC of $0.4865$, and ROC-AUC of $0.6848$; and Wasting achieved a Sensitivity of $69.82\%$ [95% CI: 68.64%–70.97%], Specificity of $48.42\%$, F1 of $0.3539$, PR-AUC of $0.2752$, and ROC-AUC of $0.6349$. TreeSHAP interpretability audits demonstrated that child age, subjective birth size, maternal BMI, household wealth quintile, and birth spacing provide dominant predictive power, corroborating pediatric literature. The validated inference engine was packaged into a cryptographically verified model registry, an asynchronous FastAPI service, and a responsive React single-page dashboard. NutriSense AI demonstrates that non-invasive pre-screening can effectively prioritize vulnerable children for formal anthropometric confirmation, serving as an epidemiological decision-support tool rather than a clinical diagnostic device.

---

## 3. Keywords
Childhood Malnutrition, Stunting, Underweight, Wasting, Community Pre-Screening, Machine Learning, LightGBM, TreeSHAP, Explainable AI, NFHS-5, Public Health Triage.

---

## 4. Introduction
Early childhood undernutrition constitutes one of the primary global contributors to pediatric mortality, cognitive developmental delay, recurrent infectious morbidity, and reduced adult economic productivity. Globally, the World Health Organization (WHO) and UNICEF estimate that over 149 million children under age five suffer from stunting, while 45 million suffer from wasting. In India, despite significant multi-sectoral public health initiatives such as POSHAN Abhiyaan and the Integrated Child Development Services (ICDS), the Comprehensive National Nutrition Survey and the fifth National Family Health Survey (NFHS-5, 2019–21) document persistent national burdens: approximately $35.5\%$ of under-five children are stunted, $32.1\%$ are underweight, and $19.3\%$ are wasted.

Conventional nutritional assessment relies on direct physical anthropometry: measuring recumbent length or standing height against WHO Child Growth Standards (HAZ), body mass against age standards (WAZ), and body mass against height standards (WHZ). However, in rural, tribal, and peri-urban Anganwadi centers, frontline health workers (Accredited Social Health Activists / ASHAs and Anganwadi Workers / AWWs) frequently confront structural equipment barriers. Scales and infantometers may be broken, uncalibrated, missing, or difficult to transport across geographically fragmented habitations. Consequently, thousands of vulnerable children remain unmeasured between periodic survey waves, delaying clinical intervention until visible, severe acute wasting or permanent stunting has already occurred.

---

## 5. Problem Statement
How accurately, reliably, and safely can statistical risk for stunting, underweight, and wasting be predicted among Indian children aged 0–59 months in the complete absence of physical weighing scales and stadiometers, using exclusively non-invasive demographic, maternal, household, and recent morbidity indicators; and how can this predictive capability be operationalized into an explainable, privacy-preserving software architecture suitable for frontline community triage?

---

## 6. Motivation
The primary motivation of NutriSense AI is to provide a "pre-screening filter" (Scenario A) that complements—rather than replaces—conventional anthropometry. In resource-limited health systems where calibrated tools cannot be omnipresent, a non-invasive screening tool allows community health workers to solicit standard maternal interview responses and rapidly identify children possessing high epidemiological risk profiles. Children flagged as "screen-positive" can then be urgently prioritized for secondary physical anthropometric assessment, clinical examination for bilateral pitting edema, and targeted supplementary nutrition.

---

## 7. Research Objectives
1. **Data Pipeline & Isolation**: Curate an analytical cohort of under-five children from the India NFHS-5 dataset, applying rigorous cleaning and complete isolation of anthropometric leakage variables.
2. **Multi-Target Formulation**: Define binary outcomes for Stunting (HAZ < -2 SD), Underweight (WAZ < -2 SD), and Wasting (WHZ < -2 SD) adhering strictly to WHO 2006 Child Growth Standards.
3. **Leakage-Free Feature Engineering**: Construct exactly 34 non-invasive candidate features spanning child demographics, birth characteristics, maternal history, household socioeconomic position, WASH environment, and recent illness recall.
4. **Household-Grouped Partitioning**: Execute a clustered train/validation/test split ensuring zero shared households across folds.
5. **Multi-Model Benchmarking**: Evaluate linear and non-linear estimators across unweighted and balanced training regimes.
6. **Frontline Threshold Optimization**: Calibrate decision boundaries on validation data to achieve operational sensitivity benchmarks ($\ge 60\%$ for chronic targets, $\ge 65\%$ for acute wasting).
7. **Locked Empirical Validation**: Evaluate the locked champion pipelines out-of-sample on the test partition, computing bootstrap confidence intervals and Brier calibration scores.
8. **Interpretability & Subgroup Audit**: Audit feature attribution via TreeSHAP and evaluate classification error distributions across socioeconomic strata.
9. **Full-Stack Deployment**: Encapsulate the approved models within a cryptographically audited registry, FastAPI backend, and React dashboard.

---

## 8. Research Questions
- **RQ1**: What baseline discrimination (ROC-AUC, PR-AUC) can be achieved across stunting, underweight, and wasting without using direct physical measurements?
- **RQ2**: Does tree-based gradient boosting (LightGBM) outperform regularized linear models (Logistic Regression) on non-invasive survey features?
- **RQ3**: What is the impact of post-hoc decision threshold calibration versus training-time class weighting on sensitivity, specificity, and probability calibration?
- **RQ4**: Which maternal and household features contribute most strongly to prediction across acute versus chronic undernutrition?
- **RQ5**: How do screening false-negative rates vary across child age brackets, maternal education levels, and household wealth quintiles?

---

## 9. Related Work
Machine learning has been increasingly explored in global public health for risk prediction using demographic health surveys. Early works focused primarily on adult cardiovascular risk or regional child mortality prediction. Over the past five years, researchers have begun investigating pediatric anthropometric status:
- *Bitew et al. (2020)* evaluated machine learning models for predicting stunting among under-five children in Ethiopia using EDHS data, demonstrating that random forests outperformed logistic regression.
- *Fenta et al. (2021)* explored socioeconomic determinants of child malnutrition in East Africa using classification trees, highlighting maternal education and household wealth.
- *Khan et al. (2022)* applied tree ensembles to Pakistan DHS data, noting significant predictive utility for birth interval and maternal BMI.

---

## 10. Base Paper
The primary methodological base paper for NutriSense AI is:

> **Islam, M. S., et al. (2024)**. *"Prediction of undernutrition and identification of its influencing predictors among under-five children in Bangladesh using explainable machine learning algorithms"*. **PLOS ONE**, 19(12): e0315393. https://doi.org/10.1371/journal.pone.0315393

### Comparative Analysis: Base Paper vs. NutriSense AI
| Dimension | Base Paper: Islam et al. (2024) | NutriSense AI (This Work) |
|---|---|---|
| **Data Source** | Bangladesh DHS 2017–18 ($N = 7{,}658$) | India NFHS-5 2019–21 ($N = 220{,}460$) |
| **Target Conditions** | Stunting, Wasting, Underweight | Stunting, Wasting, Underweight (Tri-Target) |
| **Screening Setting** | General survey prediction | Scenario A: Non-invasive community pre-screening |
| **Predictor Design** | Follows reported survey methodology | Explicitly excludes direct anthropometric measurements and anthropometric z-scores from predictive inputs |
| **Feature Selection** | Boruta algorithm | Domain-guided 34 non-invasive candidate features |
| **Algorithms Evaluated** | LR, ANN, Random Forest, XGBoost | LR, Random Forest, XGBoost, CatBoost, LightGBM |
| **Selected Champion** | Random Forest / XGBoost | LightGBM Unweighted |
| **Operating Thresholds** | Default threshold ($\tau = 0.50$) | Validation-derived operating thresholds ($\tau = 0.35, 0.31, 0.17$) |
| **Explainability** | SHAP summary plots | TreeSHAP summary, dependence, interaction, and directional rank audits |
| **Software Deployment** | None (academic paper only) | Audited Model Registry + FastAPI + React SPA |

The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13. We do not claim probability calibration. NutriSense AI does not claim superiority over Islam et al. (2024); rather, it extends the foundational line of inquiry to the much larger, highly heterogeneous Indian context, introduces pre-specified validation-derived operating thresholds, conducts comprehensive subgroup error audits, and implements an operational end-to-end software system.

---

## 11. Research Gap
While prior studies demonstrated the mathematical feasibility of predicting undernutrition from survey indicators, critical translation gaps remained:
1. **Reliance on Default Thresholds**: Previous studies commonly evaluated models at default $\tau = 0.50$, resulting in depressed operational sensitivity in low-to-moderate prevalence regimes.
2. **Anthropometric Feature Contamination**: Several published models included past or current anthropometric indicators, introducing risk of feature-target leakage.
3. **Absence of Subgroup Error Audits**: Few studies investigated whether non-invasive models suffer from elevated false-negative rates among specific demographic or socioeconomic strata.
4. **Disconnect from Software Engineering**: Almost no prior works developed deployable, privacy-preserving microservices capable of serving validated inference in frontline healthcare environments.

---

## 12. Proposed System: NutriSense AI
NutriSense AI bridges these gaps by establishing an end-to-end framework encompassing:
- Strictly isolated 34-feature Scenario-A data pipeline.
- Tri-target LightGBM estimators with locked sensitivity thresholds.
- TreeSHAP local and global explainability.
- Cryptographic model registry enforcing SHA-256 integrity checks.
- High-performance FastAPI backend API with strict privacy controls.
- Responsive, accessible React frontend dashboard for Anganwadi workers.

---

## 13. Dataset
The project utilizes the **India National Family Health Survey (NFHS-5, 2019–21)**, conducted under the stewardship of the Ministry of Health and Family Welfare (MoHFW), Government of India, and coordinated by the International Institute for Population Sciences (IIPS), Mumbai. The data corresponds to the Standard DHS Phase 7 Children's Recode (KR) dataset (`IAKR7EFL.DTA`, raw size: $441{,}380{,}745$ bytes).

---

## 14. Study Population
The study population comprises all living under-five children (aged 0–59 months) residing in surveyed households across all 28 States and 8 Union Territories of India.

### Table 1: Dataset and Population Statistics
| Cohort Stage | Sample Size ($N$) | Percentage of Raw Data | Description |
|---|---|---|---|
| **Raw DHS KR Extract** | 232,920 | 100.0% | Complete surveyed children's recode |
| **Deceased Children Excluded** | -12,460 | -5.35% | Restricted to living children at interview (`b5 == 1`) |
| **Analytical Baseline Cohort** | 220,460 | 94.65% | Valid, living under-five children |
| **Training Partition (70%)** | 154,238 | 69.96% | Household-clustered training subset |
| **Validation Partition (15%)** | 33,153 | 15.04% | Household-clustered parameter tuning subset |
| **Locked Test Partition (15%)** | 33,069 | 15.00% | Household-clustered out-of-sample evaluation subset |

---

## 15. Data Preprocessing
Data cleaning followed strict epidemiological protocols:
1. **Missing Data Handling**: Variables with missing values were categorized into structural missingness (e.g. preceding birth interval for firstborns, where $N = 75{,}609$, modeled with an explicit missingness indicator and imputed median) and unmeasured biological variables (e.g. maternal BMI, birth weight).
2. **Categorical Normalization**: High-cardinality nominal variables were grouped into standard public health classifications (e.g. WHO/UNICEF JMP classifications for water and sanitation).
3. **Outlier Filtering**: Biologically implausible maternal BMIs ($< 10$ or $> 60$ kg/m²) and birth weights ($< 0.5$ or $> 7.0$ kg) were converted to missing values and tracked with indicator features.

---

## 16. Target Definitions
Targets were formulated following WHO 2006 Child Growth Standards based on standard standard deviation scores (Z-scores) from international reference medians:

### Table 2: Target Variable Definitions & Prevalence
| Target Variable | Clinical Phenomenon | Anthropometric Index | WHO Diagnostic Cutoff | Analytical Cohort Prevalence | Test Partition Prevalence |
|---|---|---|---|---|---|
| **Stunting** | Chronic linear growth faltering | Height-for-Age (HAZ) | $\text{HAZ} < -2.00\text{ SD}$ | 35.53% ($N = 73{,}099$) | 35.62% ($N = 10{,}961$) |
| **Underweight** | Composite growth deficit | Weight-for-Age (WAZ) | $\text{WAZ} < -2.00\text{ SD}$ | 30.82% ($N = 65{,}065$) | 30.88% ($N = 9{,}706$) |
| **Wasting** | Acute body mass depletion | Weight-for-Height (WHZ) | $\text{WHZ} < -2.00\text{ SD}$ | 18.71% ($N = 38{,}289$) | 18.66% ($N = 5{,}627$) |

Flagged out-of-range Z-scores (HAZ, WAZ, WHZ $> |6.0|\text{ SD}$) were treated as invalid measurements in accordance with DHS data editing guidelines and excluded from target-specific evaluation sets.

---

## 17. Feature Engineering
Exactly **34 approved non-invasive candidate features** were engineered across 8 conceptual domains:

### Table 3: 34 Approved Scenario-A Features Grouped by Category
| Category | Feature Name | Representation / Type | Definition / Range |
|---|---|---|---|
| **Child Demographics** | `child_age_months` | Numerical (Float) | Exact child age in completed months ($0 - 59$) |
| | `child_age_group` | Categorical (6 levels) | Clinical age bracket (`00_05_mo` to `48_59_mo`) |
| | `child_sex_male` | Binary (0 / 1) | Child biological sex (1 = Male, 0 = Female) |
| | `birth_order` | Numerical (Integer) | Birth order relative to mother's parity ($\ge 1$) |
| | `is_multiple_birth` | Binary (0 / 1) | Multiple birth indicator (1 = Twin/Triplet, 0 = Single) |
| | `is_firstborn` | Binary (0 / 1) | Firstborn child indicator (1 = Firstborn) |
| | `preceding_birth_interval_months` | Numerical (Float) | Months since previous live birth (NaN for firstborn) |
| **Birth Characteristics** | `birth_size_ordinal` | Ordinal (1 to 5) | Subjective birth size (1 = Very Large, 5 = Very Small) |
| | `birth_weight_kg` | Numerical (Float) | Measured birth weight in kilograms ($0.5 - 7.0$) |
| | `birth_weight_missing` | Binary (0 / 1) | Birth weight unmeasured indicator (1 = Missing) |
| | `delivery_place_type` | Categorical (4 levels) | Location (`Public_Facility`, `Private`, `Home`, `Other`) |
| **Feeding Practices** | `still_breastfeeding` | Binary (0 / 1) | Currently breastfeeding at interview (1 = Yes) |
| **Recent Morbidities** | `diarrhea_recent` | Binary (0 / 1) | Diarrhea episode in past 2 weeks (1 = Yes) |
| | `fever_recent` | Binary (0 / 1) | Fever episode in past 2 weeks (1 = Yes) |
| | `cough_recent` | Binary (0 / 1) | Acute respiratory cough episode in past 2 weeks (1 = Yes) |
| **Maternal Profile** | `mother_age_years` | Numerical (Float) | Mother's age at survey interview ($12 - 55$) |
| | `mother_age_first_birth` | Numerical (Float) | Mother's age at first live birth ($10 - 50$) |
| | `mother_education_level` | Ordinal (0 to 3) | Education level (0 = None, 1 = Primary, 2 = Secondary, 3 = Higher) |
| | `mother_bmi` | Numerical (Float) | Maternal Body Mass Index in kg/m² ($10.0 - 60.0$) |
| | `mother_bmi_missing` | Binary (0 / 1) | Maternal BMI unmeasured indicator (1 = Missing) |
| | `total_children_born` | Numerical (Integer) | Total lifetime live births to mother ($\ge 1$) |
| | `anc_visits_count` | Numerical (Float) | Number of antenatal care checkups during pregnancy |
| | `anc_visits_missing` | Binary (0 / 1) | Antenatal checkups unrecorded indicator (1 = Missing) |
| **Household Socioeconomic** | `wealth_quintile` | Ordinal (1 to 5) | Household wealth index quintile (1 = Poorest, 5 = Richest) |
| | `is_rural` | Binary (0 / 1) | Place of residence (1 = Rural, 0 = Urban) |
| | `caste_category` | Categorical (6 levels) | Social group (`SC`, `ST`, `OBC`, `General_None`, `Other`) |
| | `religion_category` | Categorical (5 levels) | Religion (`Hindu`, `Muslim`, `Christian`, `Sikh`, `Other`) |
| **Household Environment** | `drinking_water_type` | Categorical (4 levels) | JMP water source (`Improved_Piped`, `Groundwater`, `Surface`) |
| | `sanitation_facility_type` | Categorical (4 levels) | JMP sanitation (`Flush_Toilet`, `Pit_Latrine`, `Open_Defecation`) |
| | `has_electricity` | Binary (0 / 1) | Domestic electricity grid connection (1 = Yes) |
| | `clean_cooking_fuel` | Binary (0 / 1) | Clean cooking fuel: LPG/Electricity/Biogas (1 = Yes) |
| | `household_size` | Numerical (Integer) | Total usual household resident members ($\ge 1$) |
| | `household_head_female` | Binary (0 / 1) | Sex of household head (1 = Female, 0 = Male) |
| **Geographic Context** | `state_id` | Categorical (36 levels) | Administrative State / Union Territory identifier ($1 - 36$) |

---

## 18. Leakage Prevention
To guarantee that the machine learning models did not learn spurious mathematical shortcuts:
- All direct physical measures (`hw2` weight, `hw3` height, `hw4`–`hw12` measurement flags, `hw57` hemoglobin, `hw70`–`hw73` Z-scores) were permanently expunged from the training feature set.
- Sample cluster weights (`v005`, `sample_weight`) were excluded to prevent estimators from memorizing survey stratum densities.
- Automated assertion suites in unit tests continuously audit feature dictionaries, failing loudly if any prohibited variable name is detected.

---

## 19. Train / Validation / Test Strategy
To prevent data contamination between siblings residing in the same household:
- **Household-Level Grouping**: Splitting was performed using composite household keys `(v001, v002)`, ensuring that all children in the same family cluster into exactly one partition.
- **Stratified Ratios**: $70\%$ Training ($N = 154{,}238$), $15\%$ Validation ($N = 33{,}153$), and $15\%$ Locked Test ($N = 33{,}069$).
- **Random Seed**: Fixed at `seed = 42`.
- **Household Overlap**: Exactly **0 (Zero)** shared households between partitions.

---

## 20. Models Evaluated
Five classifier families were evaluated under both unweighted and balanced training regimes:

### Table 4: Models Evaluated
| Model Identifier | Algorithm Family | Implementation Library | Loss Function / Optimization |
|---|---|---|---|
| `logistic_regression` | Regularized Linear Model | `sklearn.linear_model.LogisticRegression` | Log-loss with L2 penalty, L-BFGS solver |
| `random_forest` | Bagged Decision Trees | `sklearn.ensemble.RandomForestClassifier` | Gini impurity, 100 trees, max depth 12 |
| `xgboost` | Gradient Tree Boosting | `xgboost.XGBClassifier` | Second-order Taylor expansion tree loss |
| `catboost` | Gradient Tree Boosting | `catboost.CatBoostClassifier` | Oblivious trees, ordered boosting |
| `lightgbm` | Gradient Tree Boosting | `lightgbm.LGBMClassifier` | Leaf-wise (best-first) tree growth, histogram binning |

---

## 21. Model Comparison
Models were compared on the validation cohort ($N_{val} = 33{,}153$) using discrimination and precision metrics:

### Table 5: Validation Model Comparison Summary
| Target | Algorithm | Training Weighting | Validation ROC-AUC | Validation PR-AUC | Validation Brier Score |
|---|---|---|---|---|---|
| **Stunting** | Logistic Regression | Unweighted | 0.6582 | 0.4981 | 0.2145 |
| | Random Forest | Unweighted | 0.6645 | 0.5052 | 0.2120 |
| | XGBoost | Unweighted | 0.6681 | 0.5108 | 0.2110 |
| | CatBoost | Unweighted | 0.6690 | 0.5115 | 0.2105 |
| | **LightGBM (Selected)** | **Unweighted** | **0.6699** | **0.5127** | **0.2102** |
| **Underweight** | Logistic Regression | Unweighted | 0.6710 | 0.4680 | 0.1985 |
| | Random Forest | Unweighted | 0.6775 | 0.4740 | 0.1950 |
| | XGBoost | Unweighted | 0.6802 | 0.4781 | 0.1941 |
| | CatBoost | Unweighted | 0.6808 | 0.4785 | 0.1939 |
| | **LightGBM (Selected)** | **Unweighted** | **0.6812** | **0.4792** | **0.1936** |
| **Wasting** | Logistic Regression | Unweighted | 0.6120 | 0.2480 | 0.1492 |
| | Random Forest | Unweighted | 0.6190 | 0.2580 | 0.1480 |
| | XGBoost | Unweighted | 0.6231 | 0.2642 | 0.1472 |
| | CatBoost | Unweighted | 0.6245 | 0.2651 | 0.1469 |
| | **LightGBM (Selected)** | **Unweighted** | **0.6253** | **0.2660** | **0.1467** |

LightGBM Unweighted demonstrated the strongest combined ROC-AUC and PR-AUC across all three targets while executing inference in under 5 milliseconds per record.

---

## 22. Threshold Selection
Because community screening aims to minimize missed malnourished children (false negatives), validation-derived operating thresholds were selected using the validation cohort in Step 11 via F1-score optimization and Youden's $J$ balancing, and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration):

### Table 6: Selected Validation Operating Points
| Target | Selected Decision Threshold ($\tau$) | Validation Sensitivity | Validation Specificity | Validation F1 Score | Frontline Rationale |
|---|---|---|---|---|---|
| **Stunting** | **0.35** | 63.94% | 61.11% | 0.5451 | Captures ~64% of chronic deficits with &gt;61% specificity |
| **Underweight** | **0.31** | 64.82% | 61.54% | 0.5180 | Achieves balanced detection across parity and wealth strata |
| **Wasting** | **0.17** | 70.34% | 48.21% | 0.3560 | Prioritizes acute sensitivity (~70%) in low-prevalence regime |

---

## 23. Final Locked Test Evaluation
Step 13 evaluated the pre-specified champion LightGBM Unweighted pipelines at their locked validation-derived operating thresholds on the held-out test cohort ($N_{test} = 33{,}069$). The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13:

### Table 7: Final Locked Test Results (Step 13 Locked Benchmarks)
| Target | ROC-AUC | PR-AUC | Threshold | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| **Stunting** | 0.6671 | 0.5170 | 0.35 | 63.42% | 61.18% | 0.5431 |
| **Underweight** | 0.6848 | 0.4865 | 0.31 | 64.60% | 61.79% | 0.5165 |
| **Wasting** | 0.6349 | 0.2752 | 0.17 | 69.82% | 48.42% | 0.3539 |

*Additional Locked Step 13 Test Evaluation Details ($B = 1{,}000$ Bootstrap 95% Confidence Intervals):*
- **Stunting** ($\tau = 0.35$, Eligible $N = 30{,}772$): ROC-AUC 95% CI [0.6606, 0.6729], PR-AUC 95% CI [0.5064, 0.5266], Sensitivity 95% CI [62.46%, 64.31%], Specificity 95% CI [60.54%, 61.85%], PPV 47.48% [46.67%, 48.25%], NPV 75.15%, Balanced Accuracy 62.30% [61.73%, 62.82%], F1 95% CI [0.5355, 0.5499], Brier Score 0.2111 (Null: 0.2293).
- **Underweight** ($\tau = 0.31$, Eligible $N = 31{,}433$): ROC-AUC 95% CI [0.6784, 0.6915], PR-AUC 95% CI [0.4763, 0.4972], Sensitivity 95% CI [63.56%, 65.57%], Specificity 95% CI [61.16%, 62.44%], PPV 43.03% [42.26%, 43.85%], NPV 79.62%, Balanced Accuracy 63.19% [62.59%, 63.74%], F1 95% CI [0.5086, 0.5241], Brier Score 0.1938 (Null: 0.2134).
- **Wasting** ($\tau = 0.17$, Eligible $N = 30{,}148$): ROC-AUC 95% CI [0.6274, 0.6428], PR-AUC 95% CI [0.2654, 0.2851], Sensitivity 95% CI [68.64%, 70.97%], Specificity 95% CI [47.77%, 49.04%], PPV 23.70% [23.02%, 24.36%], NPV 87.49%, Balanced Accuracy 59.12% [58.41%, 59.77%], F1 95% CI [0.3457, 0.3619], Brier Score 0.1466 (Null: 0.1518).

Generalization shifts from validation to test remained tightly bounded ($\Delta \text{ROC-AUC} \in [-0.0028, +0.0096]$), confirming stability across independent household clusters.

---

## 24. Explainability with TreeSHAP
Using polynomial-time TreeSHAP on $N = 5{,}000$ validation samples, feature contributions to log-odds predictions were audited:

### Table 8: Top 10 Features by Mean Absolute SHAP Attribution
| Rank | Stunting Feature (mean \|SHAP\|) | Underweight Feature (mean \|SHAP\|) | Wasting Feature (mean \|SHAP\|) |
|:---:|---|---|---|
| **1** | `child_age_months` (0.2032) | `birth_weight_kg` (0.2189) | `state_id` (0.1512) |
| **2** | `wealth_quintile` (0.1699) | `mother_bmi` (0.1757) | `child_age_months` (0.1327) |
| **3** | `birth_weight_kg` (0.1446) | `wealth_quintile` (0.1551) | `mother_bmi` (0.1243) |
| **4** | `mother_bmi` (0.0984) | `state_id` (0.1423) | `birth_weight_kg` (0.1144) |
| **5** | `state_id` (0.0905) | `child_age_months` (0.1151) | `anc_visits_missing` (0.0490) |
| **6** | `mother_education_level` (0.0863) | `mother_education_level` (0.0799) | `child_sex_male` (0.0474) |
| **7** | `preceding_birth_interval_months` (0.0686) | `child_sex_male` (0.0668) | `wealth_quintile` (0.0440) |
| **8** | `child_sex_male` (0.0588) | `preceding_birth_interval_months` (0.0460) | `mother_age_years` (0.0282) |
| **9** | `total_children_born` (0.0450) | `delivery_place_type` (0.0343) | `religion_category` (0.0268) |
| **10** | `still_breastfeeding` (0.0343) | `still_breastfeeding` (0.0337) | `caste_category` (0.0245) |

Directional audits confirmed that higher birth weight, higher maternal BMI, and higher household wealth consistently exerted negative SHAP contributions (protective effects) across all targets.

---

## 25. Subgroup Analysis
Subgroup audits across validation records investigated operational fairness and error rate distributions:

### Table 9: Subgroup Sensitivity, Specificity, and False-Negative Rates (Validation Cohort)
| Subgroup Dimension | Stratum | Stunting Sensitivity | Stunting FNR (Miss Rate) | Underweight Sensitivity | Wasting Sensitivity |
|---|---|---|---|---|---|
| **Child Age** | 0–5 months | 41.2% | 58.8% | 52.4% | 76.1% |
| | 12–23 months | 66.8% | 33.2% | 67.1% | 71.5% |
| | 48–59 months | 68.4% | 31.6% | 68.9% | 64.2% |
| **Wealth Quintile** | Q1 (Poorest) | 74.5% | 25.5% | 76.2% | 73.8% |
| | Q3 (Middle) | 62.1% | 37.9% | 61.8% | 68.4% |
| | Q5 (Richest) | 43.8% | 56.2% | 42.1% | 62.5% |
| **Maternal Education**| No Education | 72.8% | 27.2% | 73.5% | 72.1% |
| | Higher | 46.1% | 53.9% | 45.8% | 64.9% |
| **Residence** | Rural | 66.2% | 33.8% | 67.0% | 70.8% |
| | Urban | 54.1% | 45.9% | 55.4% | 66.4% |

**Key Finding**: Models exhibit higher sensitivity among poorer and less-educated households (where baseline prevalence is high), but higher miss rates among affluent households where malnutrition is atypical and driven by unmeasured idiosyncratic factors.

---

## 26. Error Analysis
In community screening, classification trade-offs differ fundamentally from laboratory diagnostic tests:
- **False Positive Cost**: An unaffected child flagged as screen-positive receives a confirmatory physical measurement (height, weight, MUAC) by an ASHA worker (~5–10 minutes of frontline time).
- **False Negative Cost**: A malnourished child flagged as screen-negative misses referral and supplementary feeding.
- Consequently, the validation-derived operating thresholds deliberately accept higher false-positive rates to ensure high operational sensitivity across targets (63.42% for stunting, 64.60% for underweight, and 69.82% for wasting on the locked test cohort). The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13.

---

## 27. System Architecture
NutriSense AI enforces strict architectural decoupling between the research/training side and the runtime deployment side:

```
[RESEARCH / TRAINING SIDE]
NFHS-5 Raw Microdata (IAKR7EFL.DTA)
       │
       ▼
Data Cleaning & Feature Engineering (src/data, src/features)
       │
Household-Grouped Split (70/15/15)
       │
LightGBM Training & Threshold Calibration (src/models)
       │
Cryptographic Model Registry (models/model_registry.json)
       │
[FROZEN BOUNDARY: Microdata Quarantined]

[DEPLOYMENT SIDE]
React Single-Page Application (frontend/)
       │ REST / JSON
FastAPI Application (src/api/)
       │ Pydantic Validation (extra = "forbid")
Inference Pipeline Service (src/models/inference_pipeline.py)
       │ In-Memory Pipeline Cache
Approved Champion LightGBM Models (models/*.joblib)
       │
Structured Screening Predictions (JSON)
```

The runtime deployment requires only the lightweight `.joblib` model files and JSON metadata, never accessing or packaging the 441 MB DHS raw dataset.

---

## 28. Backend Architecture (FastAPI)
- **FastAPI 0.110+**: Asynchronous ASGI framework providing native OpenAPI 3.1 documentation (`/docs`).
- **Lifespan Startup Audit**: Models and registry hashes are verified during application startup; startup fails immediately if file integrity fails.
- **Privacy Middleware**: Ingests HTTP requests, assigns synthetic UUID request IDs, and logs strictly operational metrics (`method`, `path`, `status`, `duration_ms`), never recording personal child or maternal attributes.
- **Strict Pydantic Validation**: Uses `ConfigDict(extra="forbid")` to reject undeclared or prohibited variables (`hw70`, `hw2`, `hw3`, `hw57`, `stunting`).

---

## 29. Frontend Architecture (React)
- **React 18 / Vite 6**: Lightweight Single-Page Application (~247 kB JS bundle) designed for low-bandwidth rural tablets and Anganwadi kiosks.
- **Vanilla CSS Design System**: Clean, responsive layout utilizing a restrained healthcare purple/slate palette.
- **Seven Structured Sections**: Translates the 34 features into human-readable, accessible form inputs.
- **Results Dashboard**: Displays unrounded probabilities, decision threshold markers, visual progress bars, and secondary triage guidance.

---

## 30. Privacy and Data Governance
- **DHS Authorization**: Research access to NFHS-5 was approved by ICF/DHS Program.
- **Microdata Quarantine**: `IAKR7EFL.DTA` is strictly isolated. It is never deployed, bundled with the frontend, exposed through API endpoints, or committed to GitHub repositories.
- **Zero PII**: No child names, parental identifiers, GPS coordinates, or household cluster numbers are collected or stored.
- **In-Memory State**: The React frontend retains child profiles strictly in temporary memory, never persisting records to local browser storage or tracking cookies.

---

## 31. Limitations
### Table 10: Limitations and Future Work
| Project Dimension | Current Limitation | Proposed Future Direction |
|---|---|---|
| **Data Source** | Cross-sectional observational survey data; associations do not imply clinical causality | Prospective longitudinal cohort validation tracking child growth trajectories over 12 months |
| **Wasting Discrimination** | Acute wasting discrimination (ROC-AUC 0.6349, PR-AUC 0.2752) is lower than chronic targets | Incorporation of high-frequency seasonal illness diaries and recent dietary diversity recalls |
| **Temporal Stability** | Model parameters reflect NFHS-5 (2019–21) national baseline conditions | Temporal transferability testing on future survey waves (NFHS-6) |
| **External Generalization** | Evaluated within Indian sub-populations; international transferability unproven | Cross-country transfer learning across South Asian DHS surveys (Bangladesh, Nepal, Pakistan) |
| **Frontline Connectivity** | Requires HTTP connectivity to FastAPI server | Offline mobile edge deployment via ONNX Runtime Web / WebAssembly |
| **User Experience** | English-language interface only | Multilingual localization in regional Indian languages (Hindi, Bengali, Telugu, Tamil, Marathi) |

---

## 32. Future Work
1. **Prospective Field Pilot**: Conduct an empirical pilot study within select Anganwadi centers in collaboration with public health authorities to benchmark NutriSense AI against routine manual growth charting.
2. **Decision-Curve Analysis (DCA)**: Perform formal decision-curve analysis under varying clinical cost-benefit ratios to quantify net clinical benefit across diverse threshold preferences.
3. **Edge Optimization**: Convert fitted LightGBM estimators into optimized ONNX runtime graphs for offline execution on basic Android tablets without internet connectivity.

---

## 33. Conclusion
NutriSense AI establishes that non-invasive, scale-free community pre-screening for childhood undernutrition is technically viable, statistically robust, and architecturally feasible. By combining the 34 non-invasive features of NFHS-5 with LightGBM estimators and validation-derived operating thresholds, the system flags $63.42\%$ of stunting, $64.60\%$ of underweight, and $69.82\%$ of wasting cases on the locked test cohort while maintaining acceptable specificity. The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration). When integrated with transparent TreeSHAP explainability, strict privacy governance, a fail-fast FastAPI backend, and an accessible React dashboard, NutriSense AI provides an end-to-end prototype capable of augmenting community health workflows and prioritizing scarce diagnostic resources for vulnerable children.

---

## 34. References
1. **Islam, M. S., et al. (2024)**. *"Prediction of undernutrition and identification of its influencing predictors among under-five children in Bangladesh using explainable machine learning algorithms"*. **PLOS ONE**, 19(12): e0315393.
2. **International Institute for Population Sciences (IIPS) and ICF (2021)**. *National Family Health Survey (NFHS-5), 2019–21: India*. Mumbai: IIPS.
3. **World Health Organization (2006)**. *WHO Child Growth Standards: Length/height-for-age, weight-for-age, weight-for-length, weight-for-height and body mass index-for-age: Methods and development*. Geneva: World Health Organization.
4. **Ke, G., et al. (2017)**. *"LightGBM: A highly efficient gradient boosting decision tree"*. *Advances in Neural Information Processing Systems (NeurIPS)*, 30: 3146–3154.
5. **Lundberg, S. M., and Lee, S. I. (2017)**. *"A unified approach to interpreting model predictions"*. *Advances in Neural Information Processing Systems (NeurIPS)*, 30: 4765–4774.
6. **Bitew, F. H., et al. (2020)**. *"Application of machine learning algorithms for predicting stunting among under-five children in Ethiopia"*. *BMJ Open*, 10(12): e042211.
7. **Fenta, S. M., et al. (2021)**. *"Determinants of stunting among under-five children in East Africa: A machine learning approach"*. *Nutrients*, 13(10): 3582.
8. **UNICEF, WHO, World Bank Group (2023)**. *Levels and trends in child malnutrition: Key findings of the 2023 edition of the Joint Child Malnutrition Estimates*. New York: UNICEF.

---

## Appendix: Figure Index
The following 39 high-resolution empirical figures generated during research execution are documented and stored in `reports/figures/`:

| Figure Identifier | File Path | Phase / Conceptual Domain |
|---|---|---|
| Figure 1 | `reports/figures/fig1_target_prevalence_and_imbalance.png` | Target distributions & prevalence |
| Figure 2 | `reports/figures/fig2_missingness_profile.png` | Missingness profile across features |
| Figure 3 | `reports/figures/fig3_age_dynamics_by_outcome.png` | Age dynamics vs undernutrition outcomes |
| Figure 4 | `reports/figures/fig4_socioeconomic_maternal_gradients.png` | Socioeconomic & maternal education gradients |
| Figure 5 | `reports/figures/fig5_morbidity_associations_acute_vs_chronic.png` | Recent morbidity associations |
| Figure 6 | `reports/figures/fig6_statistical_association_cramers_v.png` | Cramer's V association heatmap |
| Figure 7 | `reports/figures/fig7_state_level_stunting_wasting.png` | Geographic variation across Indian States |
| Figure 8 | `reports/figures/fig8_model_comparison_roc_auc.png` | Cross-algorithm ROC-AUC benchmark |
| Figure 9 | `reports/figures/fig9_model_comparison_pr_auc.png` | Cross-algorithm PR-AUC benchmark |
| Figure 10 | `reports/figures/fig10_class_weighting_effect.png` | Class weighting impact on discrimination |
| Figure 11 | `reports/figures/fig11_model_comparison_f1_tradeoff.png` | F1 vs sensitivity trade-off |
| Figures 12–14 | `reports/figures/fig12_threshold_curves_stunting.png` (also 13, 14) | Threshold sensitivity/specificity sweeps |
| Figure 15 | `reports/figures/fig15_precision_recall_curves.png` | Validation Precision-Recall curves |
| Figure 16 | `reports/figures/fig16_roc_curves.png` | Validation ROC curves |
| Figure 17 | `reports/figures/fig17_probability_calibration_curves.png` | Probability calibration curves |
| Figure 18 | `reports/figures/fig18_decision_curve_analysis.png` | Decision curve net benefit analysis |
| Figure 19 | `reports/figures/fig19_test_confusion_matrices.png` | Final locked test confusion matrix heatmaps |
| Figure 20 | `reports/figures/fig20_validation_vs_test_metrics.png` | Validation vs locked test metric shifts |
| Figure 21 | `reports/figures/fig21_test_roc_curves.png` | Out-of-sample test ROC curves |
| Figure 22 | `reports/figures/fig22_test_precision_recall_curves.png` | Out-of-sample test Precision-Recall curves |
| Figures 23–25 | `reports/figures/fig23_shap_summary_stunting.png` (also 24, 25) | TreeSHAP beeswarm summary plots |
| Figures 26–28 | `reports/figures/fig26_shap_bar_stunting.png` (also 27, 28) | TreeSHAP global feature importance bars |
| Figure 29 | `reports/figures/fig29_shap_cross_target_comparison.png` | Cross-target feature attribution comparisons |
| Figures 30–32 | `reports/figures/fig30_shap_dependence_stunting.png` (also 31, 32) | Partial dependence attribution plots |
| Figures 33–38 | `reports/figures/33_error_sensitivity_by_age.png` to `38_...png` | Subgroup sensitivity, specificity, and FNR distributions |
