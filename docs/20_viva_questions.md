# NutriSense AI: Comprehensive Viva Voce & Oral Defense Examination Guide

A rigorous, exhaustive compendium of 52 technical questions and verified answers organized across categories A through Z.

---

### A. Project Basics & Scope
#### Q1: What is NutriSense AI, and what problem does it address?
**Answer**: NutriSense AI is an explainable machine learning risk intelligence system designed for community pre-screening of early childhood undernutrition (stunting, underweight, wasting) under **Scenario A: Scale-Free Community Triage**. It addresses the challenge of equipment scarcity (missing or uncalibrated scales and stadiometers) in rural and remote Anganwadi centers across India by predicting nutritional risk from 34 non-invasive demographic, maternal, household, and morbidity recall features.

#### Q2: What is the core difference between screening and clinical diagnosis in this project?
**Answer**: Screening evaluates statistical risk within an asymptomatic or community population to prioritize individuals for confirmatory examination, whereas diagnosis establishes confirmed clinical presence through gold-standard physical measurement. NutriSense AI outputs `screen_positive` classifications against validation-derived operating thresholds; it never outputs clinical diagnoses. The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration).

---

### B. Dataset & Provenance
#### Q3: What dataset is used, and what is its sample size?
**Answer**: The project uses the **India National Family Health Survey (NFHS-5, 2019–21)** Standard Demographic and Health Surveys (DHS) Children's Recode (KR) dataset (`IAKR7EFL.DTA`). The raw file contains 232,920 records (441,380,745 bytes). Filtering for living under-five children (`b5 == 1`) yields an analytical baseline cohort of 220,460 children aged 0–59 months.

#### Q4: Why was NFHS-5 selected instead of hospital or clinic records?
**Answer**: NFHS-5 provides nationally representative, population-level probability sampling spanning all 28 Indian States and 8 Union Territories. Hospital-based datasets introduce severe selection bias (over-representing acutely ill children), whereas NFHS-5 captures true community epidemiological prevalence and diverse socioeconomic strata.

---

### C. DHS Data Governance & Ethics
#### Q5: What are the ethical and data governance requirements for using DHS microdata?
**Answer**: DHS data access requires formal research authorization from ICF/The DHS Program. Microdata redistribution is strictly prohibited. Raw microdata (`IAKR7EFL.DTA`) contains cluster and household identifiers and must be quarantined: it is never committed to GitHub, bundled with client software, exposed through APIs, or made publicly downloadable.

---

### D. Target Variables & Clinical Standards
#### Q6: How are the three malnutrition targets defined?
**Answer**: Following WHO 2006 Child Growth Standards:
- **Stunting** (`stunting`): Height-for-Age Z-score $\text{HAZ} < -2.00\text{ SD}$ (chronic linear growth faltering; test prevalence: $35.62\%$).
- **Underweight** (`underweight`): Weight-for-Age Z-score $\text{WAZ} < -2.00\text{ SD}$ (composite deficit; test prevalence: $30.88\%$).
- **Wasting** (`wasting`): Weight-for-Height Z-score $\text{WHZ} < -2.00\text{ SD}$ (acute tissue/muscle depletion; test prevalence: $18.66\%$).

#### Q7: Why are Z-scores outside $\pm 6.0\text{ SD}$ excluded?
**Answer**: In standard WHO and DHS data cleaning protocols, anthropometric Z-scores outside $[-6.0, +6.0]\text{ SD}$ are classified as biologically implausible measurement or transcription errors and are coded as missing (NaN) to prevent training distortion.

---

### E. Data Preprocessing
#### Q8: How did you handle structural missingness in the preceding birth interval?
**Answer**: For firstborn children, a preceding birth interval does not biologically exist ($N = 75{,}609$ children in the dataset). Dropping these cases would bias the dataset against firstborns. We introduced an explicit binary flag `is_firstborn = 1` and imputed the missing interval with the cohort median (32 months), allowing tree estimators to partition firstborns cleanly.

#### Q9: How were categorical variables processed?
**Answer**: Categorical features were standardized using public health taxonomies (e.g. WHO/UNICEF JMP classifications for water and sanitation) and encoded via one-hot encoding within a scikit-learn `ColumnTransformer` pipeline. Numerical features underwent median imputation and standard scaling.

---

### F. Feature Engineering
#### Q10: How many features are in Scenario A, and what categories do they cover?
**Answer**: Exactly **34 approved non-invasive candidate features** spanning 8 domains:
1. Child demographics (age, sex, birth order, birth interval, multiple birth, firstborn)
2. Birth characteristics (subjective birth size, birth weight, delivery place)
3. Feeding practices (current breastfeeding status)
4. Recent morbidities (2-week recall of diarrhea, fever, acute cough)
5. Maternal characteristics (age, age at first birth, education, maternal BMI, ANC checkups, total children born)
6. Household socioeconomics (wealth quintile, residence rural/urban, caste, religion)
7. Household environment / WASH (drinking water, sanitation facility, electricity, cooking fuel, household size, female head)
8. Geographic context (state ID).

---

### G. Data Leakage Prevention
#### Q11: What variables were explicitly prohibited to prevent data leakage?
**Answer**: All direct anthropometric physical measurements (`hw2` weight, `hw3` height, `hw4`–`hw12` measurement flags, `hw57` hemoglobin, `hw70`–`hw73` Z-scores, `hw13` edema), direct target labels, and survey sampling weights (`v005`, `sample_weight`). Including any of these would cause mathematical target leakage and destroy real-world pre-screening validity.

---

### H. Train / Validation / Test Strategy
#### Q12: Why did you use household-level clustered splitting instead of random splitting?
**Answer**: Standard random row splitting would place siblings from the same household into both training and test sets. Sibling pairs share identical household wealth, maternal education, sanitation, water source, and geographic state, causing optimistic data leakage. Grouping by composite household key `(v001, v002)` guaranteed **0 shared households** across partitions (70% train: 154,238; 15% val: 33,153; 15% test: 33,069).

---

### I. Baseline Logistic Regression
#### Q13: What was the purpose of the baseline Logistic Regression model?
**Answer**: Regularized Logistic Regression (L2 penalty) served as the linear benchmark (Step 9). It established minimum discrimination baselines (Validation ROC-AUC: 0.6582 for stunting, 0.6710 for underweight, 0.6120 for wasting) against which non-linear tree-based ensembles were evaluated.

---

### J. LightGBM Estimator Architecture
#### Q14: Why was LightGBM chosen as the champion model family?
**Answer**: LightGBM uses leaf-wise (best-first) tree growth with histogram-based binning. In our multi-algorithm comparison (Step 10), LightGBM achieved the highest validation discrimination (ROC-AUC 0.6699 for stunting, 0.6812 for underweight, 0.6253 for wasting) while exhibiting faster training and sub-5 ms inference compared to XGBoost and CatBoost.

#### Q15: Did you use class weighting or unweighted models in the final pipeline?
**Answer**: We selected **LightGBM Unweighted** with post-hoc threshold calibration. Step 10 and Step 11 proved that training-time class weighting (`balanced`) severely distorts probability calibration (inflating predicted probabilities) without improving rank discrimination (ROC-AUC was identical to 4 decimal places). Unweighted models preserve smooth, monotonic probability calibration.

---

### K. Model Comparison
#### Q16: Which five algorithms were compared, and what were the findings?
**Answer**: Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM. Tree-based gradient boosting models consistently outperformed linear models by $\sim 0.010 - 0.013$ in ROC-AUC and $\sim 0.015 - 0.018$ in PR-AUC, demonstrating that non-linear feature interactions (e.g. child age interacting with breastfeeding and maternal BMI) carry predictive signal.

---

### L. Threshold Selection Methodology
#### Q17: Why is default threshold $\tau = 0.50$ inappropriate for community screening?
**Answer**: Undernutrition targets have prevalence between $18.7\%$ and $35.6\%$. At default $\tau = 0.50$, models optimize overall accuracy by predicting negative for nearly all cases, yielding disastrously low sensitivity ($20\%-40\%$). In public health screening, missing a malnourished child (false negative) is far costlier than performing a quick confirmatory measurement on an unaffected child (false positive).

#### Q18: What are the locked operating thresholds, and how were they chosen?
**Answer**: Validation-derived operating thresholds were selected using the validation cohort in Step 11 via F1 maximization and Youden's $J$ balancing, and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration):
- **Stunting**: $\tau = 0.35$ (targets $\ge 60\%$ sensitivity)
- **Underweight**: $\tau = 0.31$ (targets $\ge 60\%$ sensitivity)
- **Wasting**: $\tau = 0.17$ (targets $\ge 65\%$ sensitivity for acute risk).

---

### M. ROC-AUC Interpretation
#### Q19: What were the locked test ROC-AUC scores, and what do they mean?
**Answer**: Stunting: **0.6671** [95% CI: 0.6606–0.6729]; Underweight: **0.6848** [95% CI: 0.6784–0.6915]; Wasting: **0.6349** [95% CI: 0.6274–0.6428]. An ROC-AUC of 0.6848 means that given a randomly chosen underweight child and a randomly chosen normal child, the model assigns a higher risk probability to the underweight child $68.48\%$ of the time using non-invasive features alone.

---

### N. PR-AUC in Imbalanced Regimes
#### Q20: Why is PR-AUC particularly important for wasting?
**Answer**: Wasting has lower prevalence ($18.66\%$). In imbalanced settings, ROC-AUC can appear optimistic because a large number of true negatives keeps the false positive rate low. PR-AUC evaluates precision against recall directly. The wasting test PR-AUC was **0.2752** (substantially higher than the non-informative base rate of 0.1866).

---

### O. Sensitivity (Recall)
#### Q21: What test sensitivities were achieved at the locked thresholds?
**Answer**:
- Stunting: **63.42%** [95% CI: 62.46%–64.31%] ($6,952 / 10,961$ stunted children detected)
- Underweight: **64.60%** [95% CI: 63.56%–65.57%] ($6,270 / 9,706$ underweight children detected)
- Wasting: **69.82%** [95% CI: 68.64%–70.97%] ($3,929 / 5,627$ wasted children detected).

---

### P. Specificity
#### Q22: What were the corresponding specificities?
**Answer**: Stunting: **61.18%**; Underweight: **61.79%**; Wasting: **48.42%**. For chronic conditions, approximately $61\%$ of unaffected children are correctly ruled out, reducing Anganwadi workload by over $60\%$. For wasting, lower specificity ($48.4\%$) was accepted to preserve nearly $70\%$ sensitivity for acute vulnerability.

---

### Q. Positive Predictive Value (PPV)
#### Q23: Why is PPV for wasting 23.70%, and is that acceptable?
**Answer**: PPV is intrinsically bounded by baseline disease prevalence. In an $18.66\%$ prevalence population with $70\%$ sensitivity, PPV mathematically stabilizes around $24\%$. In community triage, a PPV of $24\%$ means 1 in 4 flagged children has true acute wasting upon confirmatory measurement—an acceptable yield for an inexpensive 5-minute physical assessment.

---

### R. F1-Score & Balanced Accuracy
#### Q24: What were the test F1 scores and Balanced Accuracies?
**Answer**:
- Stunting: F1 = **0.5431**; Balanced Accuracy = **62.30%**
- Underweight: F1 = **0.5165**; Balanced Accuracy = **63.19%**
- Wasting: F1 = **0.3539**; Balanced Accuracy = **59.12%**.

---

### S. Brier Score & Probability Verification
#### Q25: What is the Brier score, and what did it demonstrate?
**Answer**: The Brier score measures mean squared error between predicted probabilities and binary outcomes. Lower scores indicate better probability verification against empirical frequencies (we do not claim probability calibration):
- Stunting: **0.2111** (Null baseline: 0.2293)
- Underweight: **0.1938** (Null baseline: 0.2134)
- Wasting: **0.1466** (Null baseline: 0.1518).
All models outperformed non-informative base rate baselines.

---

### T. TreeSHAP Explainability
#### Q26: What is TreeSHAP, and why was it chosen over permutation importance?
**Answer**: TreeSHAP (Lundberg & Lee, 2017) applies cooperative game theory to compute exact Shapley feature attributions in polynomial time $\mathcal{O}(T L D^2)$ by tracking conditional expectations down tree paths. It guarantees local additivity and consistency, avoiding sampling artifacts inherent in permutation methods.

#### Q27: What are the top 3 predictive features for stunting?
**Answer**:
1. `child_age_months` (mean absolute SHAP = 0.2032)
2. `wealth_quintile` (mean absolute SHAP = 0.1699)
3. `birth_weight_kg` (mean absolute SHAP = 0.1446).

#### Q28: How does the effect of child age differ between stunting and wasting?
**Answer**: In stunting, child age has a positive SHAP correlation ($\rho = +0.366$) because linear growth faltering is cumulative, compounding over the first 24–36 months. In wasting, child age has a negative SHAP correlation ($\rho = -0.921$) because acute wasting risk peaks in early infancy (0–12 months) during weaning and transition from exclusive breastfeeding.

#### Q29: What directional effects were observed for maternal BMI and wealth?
**Answer**: Higher maternal BMI and higher household wealth exhibited strong negative SHAP contributions ($\rho < -0.96$), confirming that maternal nutritional reserve and socioeconomic living conditions serve as protective buffers across all three undernutrition conditions.

---

### U. Error Analysis
#### Q30: What constitutes a false positive vs. a false negative in this triage setting?
**Answer**: A **false positive** is an unaffected child flagged for confirmatory anthropometric assessment who is found to be normal. A **false negative** is a malnourished child flagged as low risk who misses referral. Triage thresholds prioritize minimizing false negatives.

---

### V. Subgroup Disparities & Fairness
#### Q31: How does model sensitivity vary across household wealth quintiles?
**Answer**: Stunting sensitivity is highest in Poorest households ($74.5\%$, miss rate $25.5\%$) and lowest in Richest households ($43.8\%$, miss rate $56.2\%$). In wealthy households, undernutrition is rare and driven by unmeasured idiosyncratic conditions (e.g. congenital illness), making non-invasive survey features less predictive.

#### Q32: How does model sensitivity vary across child age groups?
**Answer**: In infants aged 0–5 months, stunting sensitivity is lower ($41.2\%$) because linear deficits take several months to manifest anthropometrically. For children aged 24–59 months, stunting sensitivity exceeds $68\%$.

---

### W. FastAPI Backend & Pydantic Validation
#### Q33: How does the backend enforce the approved feature schema?
**Answer**: The backend uses Pydantic v2 `ChildScreeningRequest` configured with `model_config = ConfigDict(extra="forbid")`. Any extra or undeclared fields, as well as prohibited variable names (`hw70`, `hw2`, `hw3`, `hw57`, `stunting`), are rejected immediately with HTTP 422 Unprocessable Content.

#### Q34: What happens during application startup?
**Answer**: A FastAPI `lifespan` context manager calls `ScreeningService.initialize()`. It verifies the model registry, computes SHA-256 hashes of all 3 LightGBM model files, verifies feature count (34) and thresholds, and preloads pipelines into memory. If any checksum fails, the application aborts startup immediately.

---

### X. React Frontend & UX Design
#### Q35: How is the React frontend structured, and what technology was used?
**Answer**: The frontend is a Vite-powered React 18 Single-Page Application utilizing React Router 6 and a custom vanilla CSS design system. It contains 5 pages: Home (`/`), Screen Child (`/screen`), Results (`/results`), Methodology (`/about`), and System Health (`/system`).

#### Q36: How does the frontend handle the 34 features without overwhelming the user?
**Answer**: The 34 features are organized into 7 logical sections: Child Demographics, Recent Morbidities, Feeding/Delivery, Maternal History, Household Socioeconomics, WASH Environment, and State Location. It includes a "Fill Sample Data" button for rapid demonstration.

#### Q37: Does the React frontend contain any ML logic or thresholds?
**Answer**: **Zero.** The frontend does not calculate probabilities, does not store thresholds, and does not evaluate trees. It submits data to `POST /api/v1/screen` and renders the structured response returned by FastAPI.

---

### Y. Privacy, Security & Isolation
#### Q38: How does the API guarantee that sensitive child data is not logged?
**Answer**: An operational logging middleware logs only high-level telemetry (`req_id`, `method`, `path`, `status`, `duration_ms`). Request bodies containing child demographics, maternal BMI, or caste are strictly excluded from logging.

#### Q39: Can an API client select a custom model file or modify thresholds?
**Answer**: **No.** Model paths, thresholds, and feature schemas are strictly server-controlled. Client injection of model parameters is rejected with HTTP 422.

---

### Z. Limitations & Future Directions
#### Q40: What are the primary scientific limitations of NutriSense AI?
**Answer**:
1. Cross-sectional observational survey data (associations do not establish causality).
2. Lower discrimination for acute wasting (ROC-AUC 0.6349) compared to chronic stunting.
3. Lack of prospective clinical validation in operational field settings.
4. Model parameters reflect NFHS-5 baseline conditions (temporal shifts may require recalibration).

#### Q41: Can NutriSense AI be deployed today as a diagnostic replacement for scales?
**Answer**: **Absolutely not.** It is designed strictly as a pre-screening triage filter to prioritize children for physical measurement when equipment is constrained. It must never replace direct clinical anthropometry.

#### Q42: What is the most immediate prospective research extension?
**Answer**: Conducting an empirical pilot study in select Anganwadi centers comparing NutriSense AI triage prioritization against standard routine growth charting to evaluate real-world sensitivity, health worker usability, and time-to-intervention.

---

### Bonus Technical Questions
#### Q43: What was the exact raw DHS DTA file size, and why does it matter?
**Answer**: Exactly $441{,}380{,}745$ bytes. It serves as an immutable cryptographic benchmark to verify that the raw survey dataset was never modified or overwritten during research execution.

#### Q44: What are the exact SHA-256 hashes of the three champion model artifacts?
**Answer**:
- Stunting: `3a1a9c0be5d8eb35483857812da2d77441d85ec3d233952c8a650b3e03f5b736`
- Underweight: `2fc16e1a56d0b036398dd6d428ee7231cc09cfdec34c3dcb3cfd104caf2a3eef`
- Wasting: `7900278ce83da2aafacccd8d77a924a82cf0d7b2fdfab20aa37a1ac6bdd7e919`.

#### Q45: How did you verify that the API does not touch the raw DHS file at runtime?
**Answer**: In unit test `test_14_dhs_raw_file_never_required`, we mocked `builtins.open` and asserted that zero file access calls targeted `IAKR7EFL.DTA` during inference requests.

#### Q46: What is Youden's J statistic?
**Answer**: $J = \text{Sensitivity} + \text{Specificity} - 1$. It quantifies the vertical distance between the ROC curve and the chance diagonal. For Stunting, Youden's $J = 0.2461$; for Underweight, $J = 0.2639$; for Wasting, $J = 0.1824$.

#### Q47: Why did Stunting have an analytical prevalence of 35.53% while Wasting had 18.71%?
**Answer**: Stunting represents cumulative, long-term chronic malnutrition and has high endemic prevalence across India. Wasting represents acute, rapid weight loss resulting from recent infection or acute food deprivation, which exhibits lower baseline prevalence and higher seasonal volatility.

#### Q48: How does the system handle an unmeasured birth weight or maternal BMI?
**Answer**: It uses an explicit indicator approach: `birth_weight_missing = 1` and `mother_bmi_missing = 1`, with missing values imputed to the cohort median. Tree algorithms can split on the missingness flag directly, treating unmeasured status as an informative socioeconomic signal.

#### Q49: What is the purpose of the health endpoints `/health` vs `/api/v1/health/model`?
**Answer**: `/health` is a lightweight liveness probe for load balancers returning `{"status": "healthy"}` without touching models. `/api/v1/health/model` is a readiness probe that verifies model residency in memory and returns 503 Service Unavailable if models fail integrity verification.

#### Q50: How does the system ensure zero Python stack traces are exposed to frontend users?
**Answer**: Custom exception handlers in `main.py` intercept `RequestValidationError`, `HTTPException`, and generic `Exception`, formatting errors into a structured JSON schema `{ success: false, error: { code, message, details } }`.

#### Q51: How many total automated tests exist across the repository?
**Answer**: Exactly **115 Python unit tests** (Steps 0–18) and **9 Node frontend tests** (Step 19), totaling 124 passing automated tests with zero regressions.

#### Q52: What is the single most important scientific takeaway of NutriSense AI?
**Answer**: Non-invasive survey indicators contain sufficient statistical signal to detect approximately $65\%$ of undernourished children with $61\%$ specificity, providing a practical, scalable decision-support filter for low-resource community triage when physical measuring instruments are absent.
