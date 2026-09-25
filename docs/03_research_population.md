# Step 3 — Research Population Definition

## Objective
The objective of Step 3 is to establish the formal, epidemiologically grounded eligibility criteria that define the analytical research population for **NutriSense AI** using the India NFHS-5 Children's Recode (`IAKR7EFL.DTA`). 

Under **Scenario A (Community Pre-Screening)**, we identify the exact cohort of children eligible for malnutrition risk screening without relying on physical height or weight measurements as inputs. We trace sample retention and attrition sequentially from the raw dataset, evaluate target-specific versus unified complete cohorts, distinguish between structural and measurement-related missingness, and compare our population boundaries with the base research paper (Islam et al., 2024, PLOS ONE).

---

## Operational Prediction Setting: Scenario A (Community Pre-Screening)

The operational setting of NutriSense AI is **frontline community screening**:
- **Target User**: Frontline healthcare workers (e.g., ASHA workers, Anganwadi workers, Auxiliary Nurse Midwives) or community caregivers in rural and underserved Indian districts.
- **Workflow**: Frontline workers gather maternal, child demographic, dietary, birth history, and household living condition observations during routine home visits *prior to* or *in the absence of* specialized clinical equipment (e.g., digital infant scales or portable length measuring boards).
- **Leakage Prevention Rule**: 
  - Physical measurements `hw2` (weight in kg), `hw3` (height in cm), percentiles `hw4`–`hw12`, and standardized indices `hw70`–`hw73` **must never become predictive features** in the feature matrix $X$.
  - These variables are reserved exclusively for defining and validating ground-truth outcome labels in Step 5.

---

## Research Population Derivation Pipeline

We evaluate eligibility through four sequential stages:

```
[1. Full Raw Survey Roster (NFHS-5 KR)]
        N = 232,920 records
                 |
                 v  Exclude deceased children (b5 == 0): 8,702 records
[2. Living Children Cohort]
        N = 224,218 records (96.26%)
                 |
                 v  Exclude missing age in completed months (hw1 is NaN): 2,955 records
[3. Living Under-Five Children (0-59 Months)]
        N = 221,263 records (94.99%)
                 |
                 +-------------------------------------------------------+
                 |                                                       |
                 v                                                       v
[Option 1: Complete Unified Cohort]                    [Option 2: Target-Specific Cohorts]
All three outcomes simultaneously valid:               - Stunting (hw70 valid):    206,025
        N = 198,802 records (85.35%)                   - Underweight (hw71 valid): 210,524
                                                       - Wasting (hw72 valid):     201,687
```

---

## Detailed Eligibility Criteria & Empirical Counts

### 1. Initial Dataset Population
- **Starting Record Count**: Exactly **232,920** birth records in `IAKR7EFL.DTA`.
- Represents all children born to interviewed women aged 15–49 in the five years preceding the NFHS-5 survey (2019–21).

---

### 2. Criterion 1: Child Survival Status (`b5`)
- **Rule**: Retain only children who are currently alive at the time of survey interview (`b5 == 1`).
- **Empirical Counts**:
  - Alive (`b5 == 1`): **224,218** (96.26%)
  - Deceased (`b5 == 0`): **8,702** (3.74%)
- **Methodological Reasoning**:
  - Childhood malnutrition risk screening is a prospective healthcare intervention designed to identify living children at high risk of physiological growth faltering.
  - Deceased children cannot be screened for current nutritional deficit.
  - Crucially, questions regarding current age (`b8`), recent morbidity (diarrhea `h11`, fever `h22`, cough `h31`), and current feeding (`v404`) are structurally non-applicable (NaN) for deceased children. Retaining them would introduce massive structural missingness and confounding mortality with chronic nutritional deficit.

---

### 3. Criterion 2: Age Eligibility (0–59 Completed Months via `hw1`)
- **Rule**: Retain living children whose age is verified between 0 and 59 completed months using `hw1`.
- **Empirical Counts**:
  - Valid `hw1` $\in [0, 59]$: **221,263** (98.68% of living children)
  - Missing `hw1` (NaN): **2,955** (1.32% of living children)
  - Minimum `hw1`: $0.0\text{ months}$; Maximum `hw1`: $59.0\text{ months}$.
- **Methodological Reasoning**:
  - The international epidemiological definition of under-five malnutrition (WHO, UNICEF, World Bank) is strictly calibrated for children aged 0 to 59 completed months (up to the day before the child's fifth birthday).
  - Variable `hw1` provides exact age in completed months calculated from day/month/year of birth against interview date.
  - The 2,955 living children with missing `hw1` correspond to households where the height/weight anthropometry module was not administered (code 7 in `hw13` = "No measurement found in household schedule"). Without exact age in months, reference growth z-scores cannot be evaluated against WHO standards.

---

### 4. Criterion 3: Anthropometric Ground-Truth Availability & Plausibility
- **Rule**: A child is eligible for outcome evaluation only if their measured anthropometric z-score is within biologically plausible WHO limits ($< 9000$) and not unmeasured (`NaN`).
- **Handling Sentinel & Flag Codes**:
  - `9996`: Height out of plausible biological limits (documented in `hw72`).
  - `9997`: Age in days out of plausible limits.
  - `9998`: Flagged cases. WHO growth standards flag z-scores beyond biologically plausible human extremes ($\text{HAZ} < -6\text{ SD}$ or $> +6\text{ SD}$; $\text{WHZ} < -5\text{ SD}$ or $> +5\text{ SD}$) as measurement, recording, or transcription errors.
  - **Decision**: These cases are excluded from model training targets because training an ML model on physically impossible data corrupted by measurement error damages model generalization.

#### Empirical Attrition Breakdown by Outcome (among 221,263 Living Under-5 Children)

| Metric | Stunting (`hw70` - HAZ) | Underweight (`hw71` - WAZ) | Wasting (`hw72` - WHZ) |
| :--- | :---: | :---: | :---: |
| **Valid Biologically Plausible (< 9000)** | **206,025** | **210,524** | **201,687** |
| Flagged 9998 (WHO Biological Flag) | 5,045 | 1,141 | 7,661 |
| Flagged 9996 (Height Out of Limits) | 0 | 0 | 5,183 |
| Flagged 9997 (Age Out of Limits) | 19 | 19 | 0 |
| Unmeasured / Absent / Refused (`NaN`) | 10,174 | 9,579 | 6,732 |
| **Total Living Under-5 Children** | **221,263** | **221,263** | **221,263** |

---

## Comparison: Unified Complete Cohort vs. Target-Specific Cohorts

An essential design decision is whether all models must be trained on the exact same children, or if each condition should use all available valid cases.

### Set-Theoretic Overlap Analysis
- **Complete Unified Cohort** (Valid `hw70` AND `hw71` AND `hw72` simultaneously): **198,802** children.
- **Union Cohort** (At least one valid measurement): **211,646** children.
- Children with valid Stunting but missing/flagged Wasting: 7,176.
- Children with valid Underweight but missing/flagged Stunting: 5,183.
- Children with valid Wasting but missing/flagged Stunting: 2,838.

| Formulation Strategy | Sample Size ($N$) | Methodological Advantages | Trade-Offs / Disadvantages |
| :--- | :---: | :--- | :--- |
| **Strategy 1: Complete Unified Cohort (Recommended)** | **198,802** | - Exactly matches base paper methodology (Islam et al., 2024).<br>- Enables direct head-to-head performance comparison across stunting, wasting, and underweight on identical children.<br>- Supports multi-task and multi-label learning pipelines.<br>- Guarantees consistent evaluation across all three outcomes. | - Excludes ~7,200 valid stunting cases and ~2,800 valid wasting cases due to flags in another anthropometric indicator. |
| **Strategy 2: Separate Target-Specific Cohorts** | Stunting: 206,025<br>Underweight: 210,524<br>Wasting: 201,687 | - Maximizes data retention for each individual outcome.<br>- Avoids discarding valid height data simply because a weight measurement was flagged. | - Evaluates models on slightly different sub-samples, preventing strict head-to-head comparison.<br>- Incompatible with joint multi-target architectures. |

*Research Decision*: We retain both definitions. For our primary comparative benchmarking against the base paper, we utilize the **Complete Unified Cohort ($N = 198{,}802$)**, while retaining the capability to evaluate target-specific pipelines ($N = 201{,}687$ to $210{,}524$).

---

## Taxonomy of Missingness in NFHS-5

A common beginner mistake in machine learning is executing `df.dropna()` across the entire table. We formally distinguish between five types of missingness:

1. **Structural Non-Applicability**:
   - Variables that do not apply to a child by survey design logic.
   - *Example*: Preceding birth interval (`b11`) is missing for all firstborn children (`bord == 1`). Morbidity questions (`h11`, `h22`, `h31`) are missing for deceased children (`b5 == 0`).
   - *Handling*: Structural missingness must be preserved or encoded with dedicated category levels (e.g., "Firstborn"), never blindly dropped.
2. **Outcome Missingness**:
   - Cases where the child was absent, sick, or parents refused measurement (`hw13` codes 2, 3, 4, 5).
   - *Handling*: These children must be excluded from supervised model training because ground truth cannot be verified, but their features remain informative for population audits.
3. **Measurement / Transcription Flagging**:
   - Sentinel codes `9996`, `9997`, `9998` in anthropometry.
   - *Handling*: Excluded from valid outcome ground truth.
4. **Predictor Missingness**:
   - Unanswered questions in candidate features (e.g., missing mother's BMI `v445`, missing antenatal visits `m14`).
   - *Handling*: **Do NOT drop rows.** These records remain eligible. Missing values will be handled during Step 4 (Data Cleaning) and Step 7 (Feature Engineering) using tree-native missingness paths or principled imputation.
5. **Special Categorical Sentinel Codes**:
   - Codes `98` ("Don't know") or `99` in survey questions.
   - *Handling*: Recoded as explicit categorical categories representing informational uncertainty.

---

## Base Paper Comparison (Islam et al., 2024, PLOS ONE)

| Dimension | Base Research Paper (Islam et al., 2024) | NutriSense AI (Our Project) | Comparison & Contextual Impact |
| :--- | :--- | :--- | :--- |
| **Country & Survey** | Bangladesh DHS (BDHS 2017–18) | India NFHS-5 (2019–21) | Different national demographic profiles, geographic scale, and administrative structures. |
| **Raw Survey Records** | 8,759 births | **232,920** records | **>26x larger sample size**, enabling high statistical power and robust subgroup validation. |
| **Deceased Exclusions** | 457 deceased children excluded | **8,702** deceased children excluded | Identical methodological eligibility rule. |
| **Age Eligibility** | Under-five children (0–59 months) | Under-five children (0–59 months) | Exact methodological match using DHS `hw1`. |
| **Anthropometric Exclusions** | Children with missing or flagged HAZ, WAZ, or WHZ excluded | Flagged (9996, 9997, 9998) and unmeasured excluded | Exact methodological alignment with WHO growth standards. |
| **Final Analyzed Population** | **7,859** children | **198,802** children (Unified Complete) | Our extension scales the research approach from ~7.8k cases to ~198.8k cases across 36 Indian States and Union Territories. |
| **Cohort Strategy** | Single complete cohort (all three valid) | Unified Complete Cohort ($N = 198{,}802$) alongside Target-Specific ($N \ge 201{,}687$) | We reproduce the base paper's unified approach while documenting the target-specific alternative. |

---

## Research Decisions Made in Step 3

1. **Excluded Deceased Children**: 8,702 records (`b5 == 0`) excluded from the analytical population on epidemiological grounds.
2. **Excluded Missing/Out-of-Range Age**: 2,955 records with missing `hw1` excluded from the core under-five growth assessment cohort.
3. **Excluded Biologically Implausible Z-scores**: All sentinel values (`9996`, `9997`, `9998`) excluded from target eligibility.
4. **Primary Benchmark Cohort Established**: The Complete Unified Cohort ($N = 198{,}802$) is adopted as the primary research population to maintain methodological parity with Islam et al. (2024).
5. **Preserved Predictor Incompleteness**: Zero rows were dropped due to missing candidate features (such as `v445`, `m14`, or `hw57`).

---

## Limitations
- Frontline pre-screening models trained on this population are representative of children whose parents consented to survey measurement; children in extremely isolated or conflict areas who could not be measured by DHS fieldworkers (`hw13 == 7`) may experience unobserved structural vulnerabilities.
- Regional variation in measurement refusal rates exists across Indian districts, which will be evaluated during spatial EDA (Step 6).

---

## Interview / Viva Questions & Answers

### Q1: Why did you exclude deceased children from your analytical population rather than treating mortality as an extreme malnutrition outcome?
**Sample Answer**:
"Childhood mortality and childhood undernutrition represent related but distinct epidemiological endpoints. In our project, the goal is to build an early intervention risk intelligence system that screens living under-five children during frontline health visits to detect stunting, wasting, or underweight before irreversible cognitive or metabolic damage occurs. Deceased children cannot receive nutritional rehabilitation. Furthermore, in survey microdata, maternal morbidity recall (such as diarrhea or fever in the past two weeks) and current feeding practices are structurally non-applicable for deceased children. Including them would introduce severe structural missingness and confound survival prediction with nutritional growth faltering."

### Q2: What is the difference between the Unified Complete Cohort ($N = 198{,}802$) and the Target-Specific Cohorts ($N = 201{,}687$ to $210{,}524$)?
**Sample Answer**:
"In the Unified Complete Cohort, a child is eligible only if their Height-for-Age (HAZ), Weight-for-Age (WAZ), and Weight-for-Height (WHZ) z-scores are all simultaneously valid and unflagged, resulting in 198,802 children. This is the exact design used by Islam et al. (2024) and enables strict head-to-head comparison of model performance across all three nutritional outcomes on identical patients. In contrast, target-specific cohorts retain children who have valid data for one specific target (e.g., 206,025 for stunting) even if another anthropometric measurement was flagged as biologically implausible. In our research, we adopt the unified cohort for base-paper comparability while documenting the target-specific alternative."

### Q3: Why did you refuse to drop records with missing predictor variables during the population definition step?
**Sample Answer**:
"Defining the research population must be governed solely by study eligibility criteria (target population definition and outcome ground-truth availability), never by feature completeness. Dropping rows because a predictor like mother's BMI or antenatal visits is missing induces severe selection bias (often called complete-case bias), disproportionately dropping marginalized or illiterate populations who have higher missingness rates. Handling incomplete predictors is properly addressed downstream in data cleaning and feature engineering using tree-native surrogate splits, missingness indicator flags, or principled imputation."

---

## Commands Used
```bash
# Evaluate exact NFHS-5 population attrition counts
python src/data/evaluate_population.py

# Run Step 3 research population verification test suite
python tests/test_research_population.py
```

---

## Next Step
**Step 4 — Data Cleaning**: Implement systematic cleaning pipelines on the defined cohort, including recoding of sentinel values, handling duplicate records, type casting, and structured categorical encoding, without deleting rows for incomplete predictors.
