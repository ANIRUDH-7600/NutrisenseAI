# Step 2 — Data Documentation & Data Dictionary

## Objective
The objective of Step 2 is to construct a rigorous, auditable, and comprehensive Data Dictionary for the **NutriSense AI** project, directly derived from the official India NFHS-5 microdata files (`IAKR7EFL.DTA`, `IAKR7EFL.DO`, `IAKR7EFL.DCT`, and `IAKR7EFL.MAP`). 

We systematically document every candidate variable across child, maternal, household, and survey-design domains without guessing variable names or inventing values. We establish a clear epidemiological distinction between raw anthropometric measurements and standardized z-scores, perform a formal target-leakage analysis across potential prediction scenarios, map all concepts to our base research paper (Islam et al., 2024, PLOS ONE), and verify metadata integrity through automated unit tests.

---

## What We Did
1. **Automated Metadata Extraction (`src/data/generate_data_dictionary.py`)**: Built an extraction script parsing binary Stata variable labels and `.DO` value label definitions directly from `IAKR7EDT/`, verifying labels and counts across all 232,920 records.
2. **Artifact Generation (`data_dictionary.csv`)**: Produced an auditable machine-readable data dictionary containing 59 candidate variables with official labels, data types, units, scaling factors, sentinel codes, empirical valid/missing counts, roles, and inclusion/exclusion rationale.
3. **Anthropometric Foundations Formulation**: Formulated the mathematical and epidemiological distinctions between raw physical measurements (height, weight), standardized growth indices (HAZ, WAZ, WHZ), and diagnostic malnutrition thresholds.
4. **Target Leakage Risk Matrix**: Analyzed variables at risk of causing data leakage (`hw2`, `hw3`, `hw4`–`hw12`, `hw70`–`hw73`, `hw13`), defining the operational prediction scenario under which each variable is strictly prohibited or conditionally permissible.
5. **Base Paper Methodological Alignment**: Mapped every predictor from Islam et al. (2024, PLOS ONE) against actual NFHS-5 variables, identifying exact matches, available proxies, and India-specific extensions.
6. **Automated Verification Suite (`tests/test_data_dictionary.py`)**: Executed automated unit tests confirming that all documented variables exist in `IAKR7EFL.DTA`, labels match verbatim, and row sums satisfy invariants ($N_{\text{valid}} + N_{\text{missing}} = 232{,}920$).

---

## Files Created
- `data_dictionary.csv`: Machine-readable data dictionary covering 59 candidate variables.
- `src/data/generate_data_dictionary.py`: Automated metadata extraction and empirical profiling script.
- `tests/test_data_dictionary.py`: Automated verification suite asserting dictionary correctness.
- `docs/02_data_documentation.md`: Comprehensive Step 2 documentation.

## Files Modified
- None (all raw DHS files remain strictly read-only).

---

## Deep Dive: Anthropometric Variables & Growth Standards

Understanding the difference between raw measurements, standardized z-scores, and malnutrition classification is fundamental to avoiding fatal data leakage and methodological errors.

```
+-----------------------------------------------------------------------------+
|                            1. RAW MEASUREMENTS                              |
|           hw2: Weight in kg (1 dec)       hw3: Height in cm (1 dec)         |
|                          hw1: Age in months (0-59)                          |
+-----------------------------------------------------------------------------+
                                       |
                                       v  [Transformed against WHO 2006 Reference]
+-----------------------------------------------------------------------------+
|                      2. STANDARDIZED Z-SCORES (SD x 100)                     |
|  hw70: Height-for-Age (HAZ)   hw71: Weight-for-Age (WAZ)   hw72: Weight-for-Height (WHZ) |
+-----------------------------------------------------------------------------+
                                       |
                                       v  [Clinical Diagnostic Cutoffs]
+-----------------------------------------------------------------------------+
|                         3. MALNUTRITION CLASSIFICATION                      |
|      Stunting: HAZ < -2 SD         Underweight: WAZ < -2 SD        Wasting: WHZ < -2 SD     |
+-----------------------------------------------------------------------------+
```

### 1. Raw Height and Weight Measurements (`hw2`, `hw3`)
- **`hw2` (Child weight)**: Measured in kilograms with 1 implied decimal place (`125` represents $12.5\text{ kg}$). Measures total body mass (muscle, fat, water, skeleton).
- **`hw3` (Child height/length)**: Measured in centimeters with 1 implied decimal place (`855` represents $85.5\text{ cm}$). Measured recumbent (lying down) for children $<24$ months, and standing for children $\ge 24$ months.
- **Limitation**: A raw height of $80\text{ cm}$ has completely different biological implications for a 6-month-old infant (exceptionally tall) versus a 48-month-old child (severely stunted). Raw physical metrics cannot indicate malnutrition without reference to age and sex.

### 2. Standardized Anthropometric Z-Scores (HAZ, WAZ, WHZ)
To evaluate growth, the World Health Organization (WHO) 2006 Multicentre Growth Reference Study established international growth standards for healthy children raised under optimal conditions. Z-scores represent the standard deviation difference between an observed child and the median of an internationally healthy reference population of the exact same age and sex:

$$Z_i = \frac{\text{Observed Value}_i - \text{Median}_{\text{WHO Reference}}}{\sigma_{\text{WHO Reference}}}$$

In the DHS NFHS-5 dataset:
- **`hw70` (HAZ - Height-for-Age Z-score)**: Measures linear growth faltering. Reflects **chronic, long-term undernutrition** caused by sustained poverty, recurrent infections, and prolonged inadequate nutrient intake.
- **`hw71` (WAZ - Weight-for-Age Z-score)**: Composite index of body mass relative to age. Reflects **overall undernutrition** (both chronic stunting and acute wasting combined).
- **`hw72` (WHZ - Weight-for-Height Z-score)**: Measures body weight relative to achieved height. Reflects **acute malnutrition (wasting)**, typically triggered by recent acute enteric illness, severe diarrhea, or sudden food shortages.
- **`hw73` (BMI-for-Age Z-score)**: Measures body mass index relative to age; clinically complementary to WHZ.

### 3. Coding Specifics of `hw70`, `hw71`, and `hw72` in NFHS-5
- **Implied Decimal Scaling**: DHS stores z-scores multiplied by $100$. A value of `-224` in the data file corresponds to a true z-score of $-2.24\text{ SD}$. A value of `-140` is $-1.40\text{ SD}$.
- **Biologically Plausible Ranges**:
  - `hw70` (HAZ): Plausible range is $-6.00\text{ SD}$ to $+6.00\text{ SD}$ (file values `-600` to `+600`).
  - `hw71` (WAZ): Plausible range is $-6.00\text{ SD}$ to $+5.00\text{ SD}$ (file values `-600` to `+500`).
  - `hw72` (WHZ): Plausible range is $-5.00\text{ SD}$ to $+5.00\text{ SD}$ (file values `-500` to `+500`).
- **Official DHS Sentinel Flag Codes**:
  - `9996`: Height out of plausible biological limits (documented in `hw72`).
  - `9997`: Age in days out of plausible limits.
  - `9998`: Flagged cases (WHO biological implausibility flag). Values exceeding WHO standard flags are classified as measurement or data-entry errors.
  - `NaN`: Child not measured, deceased, or absent.

### 4. Malnutrition Classification Thresholds
Based on internationally accepted WHO and UNICEF standards:
- **Stunting** (from `hw70`):
  - Normal: $\text{HAZ} \ge -2.00\text{ SD}$ (`hw70 >= -200`)
  - Moderate Stunting: $-3.00\text{ SD} \le \text{HAZ} < -2.00\text{ SD}$ (`-300 <= hw70 < -200`)
  - Severe Stunting: $\text{HAZ} < -3.00\text{ SD}$ (`hw70 < -300`)
- **Underweight** (from `hw71`):
  - Normal: $\text{WAZ} \ge -2.00\text{ SD}$ (`hw71 >= -200`)
  - Moderate Underweight: $-3.00\text{ SD} \le \text{WAZ} < -2.00\text{ SD}$ (`-300 <= hw71 < -200`)
  - Severe Underweight: $\text{WAZ} < -3.00\text{ SD}$ (`hw71 < -300`)
- **Wasting** (from `hw72`):
  - Normal: $\text{WHZ} \ge -2.00\text{ SD}$ (`hw72 >= -200`)
  - Moderate Wasting: $-3.00\text{ SD} \le \text{WHZ} < -2.00\text{ SD}$ (`-300 <= hw72 < -200`)
  - Severe Acute Malnutrition (SAM): $\text{WHZ} < -3.00\text{ SD}$ (`hw72 < -300`)

---

## Target Leakage Analysis & Prediction Scenarios

A critical trap in healthcare machine learning is failing to define the **prediction operational context**, resulting in catastrophic data leakage.

### Defining Our Prediction Scenarios
1. **Scenario A: Community Pre-Screening & Early Risk Intelligence (Our Primary Goal)**
   - *Setting*: An ASHA worker, Anganwadi worker, or mother in a remote rural community fills out a mobile screening questionnaire with basic demographic, birth history, maternal, and household factors.
   - *Physical Equipment Available*: **No digital stadiometer board or precision infant scale.**
   - *Goal*: Flag children at high risk of developing stunting or wasting *before* or *without* clinical anthropometric measurement.
   - *Leakage Rule*: **Direct child height (`hw3`), child weight (`hw2`), and derived z-scores are strictly prohibited as input features.**
2. **Scenario B: Cross-Condition Diagnostic Assistance (Secondary/Alternative)**
   - *Setting*: A clinic has an infant weighing scale but lacks a calibrated length board, or vice versa.
   - *Goal*: Predict stunting using weight and demographic features, or predict wasting using weight and symptoms.
   - *Leakage Rule*: If predicting Stunting (`hw70`), child weight (`hw2`) might be available, but child height (`hw3`) is 100% prohibited.

### Formal Target Leakage Risk Matrix

| Variable | DHS Label | Why It Causes Target Leakage | Permissible in Scenario A (Pre-Screening)? | Permissible in Scenario B (Cross-Condition)? | Decision & Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`hw70`** | Height/Age SD (new WHO) | **Direct mathematical target** for Stunting. | **NO (Target Only)** | **NO (Target Only)** | Target label only; strictly excluded from feature matrix $X$. |
| **`hw71`** | Weight/Age SD (new WHO) | **Direct mathematical target** for Underweight. | **NO (Target Only)** | **NO (Target Only)** | Target label only; strictly excluded from feature matrix $X$. |
| **`hw72`** | Weight/Height SD (new WHO) | **Direct mathematical target** for Wasting. | **NO (Target Only)** | **NO (Target Only)** | Target label only; strictly excluded from feature matrix $X$. |
| **`hw73`** | BMI SD (new WHO) | Direct collinear transformation of height and weight. | **NO** | **NO** | Alternative target; leaks all anthropometric deficit classes. |
| **`hw3`** | Child height in cm | Direct numerator in HAZ; direct denominator in WHZ. | **NO (Severe Leakage)** | **NO** | If height is measured, stunting is already diagnosed by lookup table. Using it in ML makes the task trivial and useless. |
| **`hw2`** | Child weight in kg | Direct numerator in WAZ and WHZ. | **NO (Severe Leakage)** | Conditionally (for Stunting only) | Excluded in Scenario A because community pre-screening aims to identify risk without requiring scales. |
| **`hw4`–`hw6`** | Height/Age percentiles & % median | Mathematical transformations of child height. | **NO** | **NO** | Direct mathematical surrogates of `hw70`. Strictly excluded. |
| **`hw7`–`hw9`** | Weight/Age percentiles & % median | Mathematical transformations of child weight. | **NO** | **NO** | Direct mathematical surrogates of `hw71`. Strictly excluded. |
| **`hw10`–`hw12`** | Weight/Height percentiles & % median | Mathematical transformations of weight/height. | **NO** | **NO** | Direct mathematical surrogates of `hw72`. Strictly excluded. |
| **`hw13`** | Result of measurement | Flags whether child was measured, dead, or refused. | **NO (Data Audit Only)** | **NO (Data Audit Only)** | Perfectly correlates with missingness in the target; used in Step 3 for population filtering, excluded from features. |

---

## Base Paper Comparison (Islam et al., 2024, PLOS ONE)

We systematically compared the variables and concepts analyzed by Islam et al. (2024) in the Bangladesh DHS (BDHS 2017–18) against our India NFHS-5 (2019–21) KR dataset.

| Base-Paper Concept | BDHS Variable | NFHS-5 Variable | Exact Match? | Available in NFHS-5? | Methodological & Contextual Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Child Age** | `b8` / `hw1` | `hw1` | Yes | **Yes** | BDHS grouped age into 5 bins; NFHS-5 provides exact age in months (`hw1`, 0–59) which enables finer continuous modeling. |
| **Child Sex** | `b4` | `b4` | **Yes** | **Yes** | Exact standard DHS recode match (1=Male, 2=Female). |
| **Birth Order** | `bord` | `bord` | **Yes** | **Yes** | Exact match (continuous integer 1 to 16). |
| **Size of Child at Birth** | `m18` | `m18` | **Yes** | **Yes** | Exact match (1=Very large to 5=Very small, 8=Don't know). |
| **Birth Weight** | `m19` | `m19` | **Yes** | **Yes** | Exact match in grams (3 decimals); subject to sentinel handling (9996=not weighed). |
| **Child Twin Status** | `b0` | `b0` | **Yes** | **Yes** | Exact match (0=Single, 1=Multiple). |
| **Preceding Birth Interval** | `b11` | `b11` | **Yes** | **Yes** | Exact match in months; missing for firstborns. |
| **Currently Breastfeeding** | `v404` | `v404` | **Yes** | **Yes** | Exact match (0=No, 1=Yes). |
| **Breastfeeding Duration** | `m4` | `m4` | **Yes** | **Yes** | Exact match in months. |
| **Recent Diarrhea** | `h11` | `h11` | **Yes** | **Yes** | Exact match (0=No, 2=Yes). |
| **Recent Fever** | `h22` | `h22` | **Yes** | **Yes** | Exact match (0=No, 2=Yes). |
| **Recent Cough / ARI** | `h31` | `h31` | **Yes** | **Yes** | Exact match (0=No, 2=Yes). |
| **Child Anemia Level** | `hw57` | `hw57` | **Yes** | **Yes** | Exact match (1=Severe, 2=Moderate, 3=Mild, 4=Not anemic). |
| **Mother's Current Age** | `v012` | `v012` | **Yes** | **Yes** | Exact match (continuous years 15–49). |
| **Mother's Age at First Birth** | `v212` | `v212` | **Yes** | **Yes** | Exact match (continuous years). |
| **Mother's BMI** | `v445` | `v445` | **Yes** | **Yes** | Exact match; 2 implied decimals in DHS. |
| **Mother's Education** | `v106` | `v106` | **Yes** | **Yes** | Exact match (0=None, 1=Primary, 2=Secondary, 3=Higher). |
| **Mother's Employment** | `v714` | `v714` | **Yes** | **Yes** | Exact match (0=No, 1=Yes). |
| **Partner's Education** | `v701` | `v701` | **Yes** | **Yes** | Exact match (0=None, 1=Primary, 2=Secondary, 3=Higher). |
| **Antenatal Care (ANC) Visits** | `m14` | `m14` | **Yes** | **Yes** | Exact match (continuous count 0–90+; 98=Don't know). |
| **Place of Delivery** | `m15` | `m15` | **Yes** | **Yes** | Exact match (categorized into Health facility vs Home). |
| **Wealth Index Quintile** | `v190` | `v190` | **Yes** | **Yes** | Exact match (1=Poorest to 5=Richest). |
| **Place of Residence** | `v025` | `v025` | **Yes** | **Yes** | Exact match (1=Urban, 2=Rural). |
| **Subnational Region** | `v024` (Divisions) | `v024` (States) | Conceptual | **Yes** | BDHS had 8 administrative divisions; NFHS-5 has **36 Indian States and Union Territories**, offering substantially richer geographical variance. |
| **Social Group / Caste** | Not in BDHS | `s116` | **Novel Extension** | **Yes** | India-specific social stratification (SC, ST, OBC, General) highly predictive of structural health disparity. |
| **Drinking Water Source** | `v113` | `v113` | **Yes** | **Yes** | Standard WHO/UNICEF JMP classification into improved vs unimproved. |
| **Toilet Facility Type** | `v116` | `v116` | **Yes** | **Yes** | Standard WHO/UNICEF JMP classification into improved vs unimproved. |
| **Household Electricity** | `v119` | `v119` | **Yes** | **Yes** | Exact match (0=No, 1=Yes). |
| **Cooking Fuel Type** | `v161` | `v161` | **Yes** | **Yes** | Exact match (Clean fuel vs solid biomass). |

---

## Data Dictionary Summary: Candidate Variables Documented

A total of **59 candidate variables** were extracted and documented in [data_dictionary.csv](file:///d:/finalyearproj/Nutrisense-Ai/data_dictionary.csv). Below is the domain-wise summary:

### 1. Child Anthropometry & Targets (16 Variables)
- **Targets**: `hw70` (HAZ - Stunting), `hw71` (WAZ - Underweight), `hw72` (WHZ - Wasting), `hw73` (BMI-for-Age).
- **Leakage / Audit**: `hw2` (Weight), `hw3` (Height), `hw4`–`hw12` (Percentiles and % medians), `hw13` (Measurement result).

### 2. Child Demographics, Birth History & Feeding (11 Variables)
- **Variables**: `hw1` (Age in months), `b4` (Sex), `b5` (Alive status), `b8` (Age in years), `bord` (Birth order), `b0` (Twin indicator), `b11` (Preceding birth interval), `m18` (Birth size), `m19` (Birth weight), `v404` (Currently breastfeeding), `m4` (Breastfeeding duration).

### 3. Child Morbidity & Health (4 Variables)
- **Variables**: `hw57` (Anemia level), `h11` (Recent diarrhea), `h22` (Recent fever), `h31` (Recent cough/ARI).

### 4. Maternal Characteristics & Healthcare (10 Variables)
- **Variables**: `v012` (Mother's current age), `v212` (Age at first birth), `v445` (Mother's BMI), `v457` (Mother's anemia level), `v106` (Mother's education), `v701` (Partner's education), `v714` (Mother working status), `m14` (ANC visits count), `m15` (Place of delivery), `v201` (Children ever born).

### 5. Household Socioeconomic, Environment & Sanitation (12 Variables)
- **Variables**: `v190` (Wealth index quintile), `v025` (Urban/Rural residence), `v024` (State/UT - 36 categories), `sdist` (District code), `s116` (Social group / Caste), `v130` (Religion), `v113` (Drinking water source), `v116` (Toilet facility), `v119` (Electricity), `v161` (Cooking fuel), `v136` (Household size), `v151` (Sex of household head).

### 6. Survey Design & Identifiers (6 Variables)
- **Variables**: `v001` (Cluster / PSU number), `v002` (Household number), `v003` (Respondent line number), `v005` (Sample weight), `v021` (Primary sampling unit), `v022` (Sampling strata).

---

## Important ML & Epidemiological Concepts

1. **Construct Validity**:
   A machine learning model only predicts what its target represents. If our target is stunting (`hw70 < -200`), the model is predicting cumulative linear growth failure over months and years, NOT acute nutritional deficit over the last two weeks. Distinguishing between chronic stunting (HAZ), acute wasting (WHZ), and composite underweight (WAZ) ensures clinical interpretability.
2. **Operational Prediction Context vs. Retrospective Association**:
   Many published papers commit the error of including all available variables in an ML model and reporting an AUC of 0.98. When inspected, they included child height to predict stunting or child weight to predict wasting. By defining **Scenario A (Pre-Screening)**, we enforce that our model only uses predictors accessible to a community health worker without clinical measurement tools.
3. **Data Immutability & Metadata Auditing**:
   We generated [data_dictionary.csv](file:///d:/finalyearproj/Nutrisense-Ai/data_dictionary.csv) programmatically using Python and regex from the official `.DTA` and `.DO` files. This creates a fully auditable data engineering artifact that guarantees zero hallucinated column names.

---

## Research Reasoning
- We refrained from selecting final model features at Step 2 because feature selection must follow population eligibility filtering (Step 3), target definition (Step 5), and exploratory data analysis (Step 6).
- Documenting candidate predictors early prevents ad-hoc, unprincipled variable dropping later in the pipeline.

---

## Quality Check & Automated Verification Results

We implemented and executed [tests/test_data_dictionary.py](file:///d:/finalyearproj/Nutrisense-Ai/tests/test_data_dictionary.py).

**Verification Command**:
```powershell
python tests/test_data_dictionary.py
```

**Measured Output**:
```text
[*] Running Step 2 Data Dictionary Integrity Tests...
  [OK] data_dictionary.csv exists with >= 50 variables.
  [OK] All variables and official labels strictly verified against raw .DTA.
  [OK] All required roles, domains, and candidate targets verified.
  [OK] Observation totals strictly equal 232,920 across all variables.
[SUCCESS] All Step 2 data dictionary tests passed successfully!
```

---

## Limitations
- Step 2 documents candidate variables but does not perform categorical encoding, imputation, or row filtering.
- Implied decimal scalings (such as `hw70 / 100` or `v005 / 1000000`) are documented here, but raw values in the underlying dataset remain unchanged until preprocessing.

---

## Interview / Viva Questions & Answers

### Q1: What is the biological and clinical difference between HAZ, WAZ, and WHZ, and why can't they be combined into a single target?
**Sample Answer**:
HAZ (Height-for-Age Z-score), WAZ (Weight-for-Age Z-score), and WHZ (Weight-for-Height Z-score) capture distinct physiological manifestations of undernutrition. HAZ measures linear growth faltering (stunting), reflecting chronic, cumulative nutritional deprivation and repeated infections over months or years. WHZ measures body mass relative to height (wasting), reflecting acute, rapid weight loss or failure to gain weight due to recent severe illness or acute starvation. WAZ is a composite index of both linear and acute deficits. Because their underlying etiologies and intervention strategies differ (e.g., immediate therapeutic feeding for wasting vs long-term dietary and environmental interventions for stunting), combining them into a single generic 'undernutrition' label obscures critical clinical distinctions.

### Q2: How did you identify target leakage in the DHS dataset, and why is child height considered a leakage variable when predicting stunting?
**Sample Answer**:
Target leakage occurs when an input feature contains information that directly derives from or calculates the target outcome. Stunting is medically defined as having a Height-for-Age Z-score (HAZ) below -2.00 SD, which is calculated directly from a child's measured height, age, and sex against the WHO growth tables. If a machine learning model is given measured child height (`hw3`) to predict stunting, the model simply learns the WHO growth chart arithmetic, producing an artificially perfect model that provides zero clinical utility in pre-screening settings where height has not yet been measured.

### Q3: Why does `v005` have six implied decimals, and what role does it play in machine learning pipelines?
**Sample Answer**:
The DHS dataset stores women's individual sample weights as large integers (`v005`) with six implied decimal places to maintain high numerical precision across statistical platforms without rounding. The true normalized sampling weight is $W_i = \text{v005} / 1{,}000{,}000$. In machine learning, sample weights should not be passed as input features ($X$) because the model would learn sampling artifacts. Instead, they can be used during sample-weighted loss function optimization or for calculating nationally representative evaluation metrics and prevalence estimates.

---

## Commands Used
```bash
# Generate data_dictionary.csv programmatically from NFHS-5 metadata
python src/data/generate_data_dictionary.py

# Run Step 2 data dictionary automated verification test suite
python tests/test_data_dictionary.py
```

---

## Next Step
**Step 3 — Define the Research Population**: Establish the explicit, epidemiologically grounded eligibility criteria for our analysis cohort (e.g., child age 0–59 months, living status, valid anthropometric measurements), compare our cohort selection with Islam et al. (2024), and compute the exact retention and exclusion numbers.
