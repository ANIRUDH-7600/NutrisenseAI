# Step 4 — Systematic Data Cleaning

## Objective
The objective of Step 4 is to execute a rigorous, deterministic, and auditable data cleaning pipeline on the research population of living under-five children ($N = 221{,}263$), resolving DHS sentinel flag codes, applying documented decimal scalings, performing a taxonomy-driven missingness audit, and preparing an intermediate cleaned representation under `data/interim/` without ever altering the raw source dataset or exposing confidential microdata.

In strict adherence to research integrity, **no machine learning models were trained, no feature selection was performed, no SMOTE/oversampling was applied, no imputation was executed, and no binary target labels were constructed yet.**

---

## Methodological Classification Framework

To maintain scientific rigor and avoid conflating distinct concepts, every decision in this document is classified into one of four distinct categories:
1. **Documented DHS Cleaning Rule**: Explicitly defined in official DHS/NFHS-5 documentation, `.DO` value labels, or `.MAP` codebooks (e.g., sentinel codes `9996`, `9997`, `9998`, and implied decimal scaling).
2. **Observed Data Range**: Empirical minimums and maximums measured directly from the data after removing documented sentinel codes (e.g., observed maternal BMI range of $12.02$ to $59.99\text{ kg/m}^2$).
3. **Modeling Decision**: Design choices made for NutriSense AI's specific research formulation (e.g., defining `still_breastfeeding` and separating completed from right-censored breastfeeding duration).
4. **Future Preprocessing Decision**: Transformations intentionally deferred to Step 5+ (e.g., target binarization, imputation, one-hot encoding, and scaling).

---

## Raw vs. Cleaned Data Architecture

```
[Raw Immutable DHS Source]
  IAKR7EDT/IAKR7EFL.DTA (441,380,745 bytes, 232,920 rows x 1,644 cols)
  Status: READ-ONLY (Single source of truth)
              |
              v  Filtered to Living Under-5 Children (hw1 in [0, 59] & b5 == 1)
              |
              v  Deterministic Cleaning Pipeline (src/data/clean_data.py)
              |
[Interim Cleaned Representation]
  data/interim/cleaned_u5_data.csv.gz (221,263 rows x 66 columns, 18.17 MB)
  Status: Gitignored / Internal storage only
              |
              +---> Target Eligibility Flags (Stunting: 206,025; Underweight: 210,524; Wasting: 201,687; Unified: 198,802)
              +---> Continuous Cleaned Z-scores (hw70_clean, hw71_clean, hw72_clean, hw73_clean)
              +---> Cleaned Predictor Candidates (demographics, maternal, socioeconomic, sanitation)
              +---> Survey Weights & Identifiers (sample_weight, v001, v002, v024)
              +---> Physical Measurements (hw2_clean, hw3_clean) - Reserved for audit / target validation only
```

---

## 1. Documented DHS Special-Code Handling Matrix

The DHS coding system uses explicit numerical sentinel codes to represent measurement anomalies, biological implausibility, and informational uncertainty. Treating these codes as ordinary continuous numbers is a critical error in epidemiological data science.

| Variable | Special Code | Official DHS Meaning (from `.DO` / `.MAP`) | Cleaning Action | Decision Category | Methodological Justification |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`hw70`, `hw71`, `hw72`, `hw73`** | `9998` | Flagged cases (WHO biological implausibility: HAZ $<-6$ or $>+6$; WHZ $<-5$ or $>+5$) | Recode to `NaN` | **Documented DHS Rule** | Biologically impossible growth measurements caused by recording or measurement error; damages model generalization. |
| **`hw72`** | `9996` | Height out of plausible limits | Recode to `NaN` | **Documented DHS Rule** | Cannot compute valid weight-for-height index if height is unmeasured/invalid. |
| **`hw70`, `hw71`** | `9997` | Age in days out of plausible limits | Recode to `NaN` | **Documented DHS Rule** | Exact day of birth inconsistent; cannot calibrate against daily WHO reference tables. |
| **`v445`** (Mother BMI) | `9998` | Flagged cases | Recode to `NaN` | **Documented DHS Rule** | Official DHS flag for biologically implausible maternal anthropometry. |
| **`m14`** (ANC visits) | `98` | Don't know | Recode to `NaN` | **Documented DHS Rule** | Informational uncertainty; mother does not recall exact count of prenatal visits. **No arbitrary cap imposed.** |
| **`m19`** (Birth weight) | `9996` | Not weighed at birth | Recode to `NaN` | **Documented DHS Rule** | Institutional delivery did not occur or scales unavailable in home delivery. |
| **`m19`** (Birth weight) | `9998` | Don't know | Recode to `NaN` | **Documented DHS Rule** | Mother cannot recall infant birth weight. |
| **`m18`** (Size at birth) | `8` | Don't know | Recode to `NaN` | **Documented DHS Rule** | Mother unable to report relative size. |
| **`m4`** (Breastfeeding) | `94` | Never breastfed | Recoded to `0.0` months | **Documented DHS Rule** | Child never received breast milk; duration is exactly zero. |
| **`m4`** (Breastfeeding) | `95` | Still breastfeeding | Separated into `still_breastfeeding = 1`, `m4_completed = NaN`, and right-censored `m4_censored = hw1` | **Modeling Decision** | **Official DHS Convention**: Code 95 signifies ongoing breastfeeding. For weaned children, completed duration is unknown (censored). We provide separate variables to avoid conflating completed duration with right-censored child age. |
| **`m4`** (Breastfeeding) | `98` | Don't know | Recode to `NaN` | **Documented DHS Rule** | Informational uncertainty. |
| **`h11`, `h22`, `h31`** | `8` | Don't know (morbidity) | Recode to `NaN` | **Documented DHS Rule** | Mother uncertain if child had fever, diarrhea, or cough in preceding two weeks. |
| **`v701`** (Partner edu) | `8` | Don't know | Recode to `NaN` | **Documented DHS Rule** | Partner's schooling level unknown to respondent. |
| **`s116`** (Caste) | `8` | Don't know | Recode to `NaN` | **Documented DHS Rule** | Social group unstated. |

---

## 2. Implied Decimals & Mathematical Transformations

DHS avoids floating-point precision loss across different statistical tools (Stata, SAS, SPSS) by storing decimal numbers multiplied by powers of 10. We systematically reversed this scaling to restore standard epidemiological units.

| Raw Variable | Cleaned Column | Scale Factor | Transformation Applied | Decision Category | Clinical / Analytical Unit | Observed Range After Cleaning |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **`hw70`** | `hw70_clean` | $10^{-2}$ | $\text{hw70} / 100.0$ (if $<9000$) | Documented DHS Rule | Standard Deviations ($\text{SD}$) | $[-6.00, +6.00]$ |
| **`hw71`** | `hw71_clean` | $10^{-2}$ | $\text{hw71} / 100.0$ (if $<9000$) | Documented DHS Rule | Standard Deviations ($\text{SD}$) | $[-6.00, +5.00]$ |
| **`hw72`** | `hw72_clean` | $10^{-2}$ | $\text{hw72} / 100.0$ (if $<9000$) | Documented DHS Rule | Standard Deviations ($\text{SD}$) | $[-5.00, +5.00]$ |
| **`hw73`** | `hw73_clean` | $10^{-2}$ | $\text{hw73} / 100.0$ (if $<9000$) | Documented DHS Rule | Standard Deviations ($\text{SD}$) | $[-5.00, +5.00]$ |
| **`v445`** | `v445_clean` | $10^{-2}$ | $\text{v445} / 100.0$ (if $<9000$) | Documented DHS Rule | $\text{kg/m}^2$ (Mother's BMI) | $[12.02, 59.99]$ |
| **`v005`** | `sample_weight` | $10^{-6}$ | $\text{v005} / 1{,}000{,}000.0$ | Documented DHS Rule | Normalized Probability Weight | $[0.003, 37.59]$ (Mean $\approx 1.0$) |
| **`m19`** | `birth_weight_kg` | $10^{-3}$ | $\text{m19} / 1000.0$ (if $\in [500, 6000]$) | Documented DHS Rule | Kilograms ($\text{kg}$) | $[0.50, 6.00]$ |
| **`hw2`** | `hw2_clean` | $10^{-1}$ | $\text{hw2} / 10.0$ (if $<9000$) | Documented DHS Rule | Kilograms ($\text{kg}$, child weight) | $[1.2, 35.0]$ |
| **`hw3`** | `hw3_clean` | $10^{-1}$ | $\text{hw3} / 10.0$ (if $<9000$) | Documented DHS Rule | Centimeters ($\text{cm}$, child height) | $[45.0, 130.0]$ |

*Note: Categorical variables (e.g., `v106` education, `v190` wealth quintile, `b4` sex) were NOT scaled.*

---

## 3. Deep Dive on Specific Review Inquiries

### A. Clarification on `m4` (Breastfeeding Duration)
- **Official Definition in `.DO` / `.MAP`**:
  Variable label: *"Duration of breastfeeding"*. Defined in value label `M4`:
  - `93`: *"Ever breastfed, not currently breastfeeding"*
  - `94`: *"Never breastfed"*
  - `95`: *"Still breastfeeding"*
  - `96`: *"Breastfed until died"*
  - `97`: *"Inconsistent"*
  - `98`: *"Don't know"*
  - Values $0$ to $59$: Completed months of breastfeeding among weaned children.
- **Methodological Analysis**:
  For an infant who is actively breastfeeding (`m4 == 95`), their completed duration is unknown/unfinished (right-censored in epidemiological survival analysis). Substituting `hw1` directly into `m4` treats right-censored ongoing exposure as if it were finished duration, confounding age with weaning age.
- **Our Resolution**:
  We provide two methodologically clean, separate variables:
  1. `still_breastfeeding`: Binary indicator ($1=\text{Yes}$, $0=\text{No}$, $\text{NaN}=\text{Missing}$).
  2. `m4_completed_months`: Completed duration strictly for children who have concluded breastfeeding ($94 \rightarrow 0.0\text{ mo}$; $95 \rightarrow \text{NaN}$ because duration is ongoing; $98 \rightarrow \text{NaN}$).
  3. `m4_censored_months`: Duration where active breastfeeding is right-censored at current child age (`hw1`), provided for survival/hazard modeling.

### B. Removal of Arbitrary `m14` Capping (Antenatal Visits)
- **Official Definition in `.DO`**:
  Variable label: *"Number of antenatal visits during pregnancy"*. Defined in value label `M14`:
  - `0`: *"No antenatal visits"*
  - `98`: *"Don't know"*
- **Correction Made**:
  An earlier script draft tentatively capped `m14` at 40 visits. **This arbitrary cap was completely removed.** 
  DHS official metadata specifies only that code `98` is "Don't know". We recoded code `98` to `NaN` and preserved all valid observed visits.
- **Observed Range**: Minimum $0.0$ visits, Maximum **$95.0$ visits**. All valid reported visits ($>40$) are fully preserved.

### C. Clarification of `v445` Range ($12.02$ to $59.99\text{ kg/m}^2$)
- **Official Definition in `.DO`**:
  Variable label: *"Body Mass Index"*. Value label `V445` defines:
  - `9998`: *"Flagged cases"*
- **Correction & Verification**:
  The range $12.02$ to $59.99\text{ kg/m}^2$ is **NOT** an imposed validity filter. Our cleaning script applies strictly:
  $$\text{v445\_clean} = \begin{cases} \text{v445} / 100.0 & \text{if } \text{v445} < 9000 \\ \text{NaN} & \text{if } \text{v445} \ge 9000 \text{ or missing} \end{cases}$$
  The values $12.02$ and $59.99\text{ kg/m}^2$ represent the **observed empirical minimum and maximum after removing official DHS sentinel flag 9998**. No valid maternal BMI observation was clipped or discarded.

---

## 4. Missingness Audit Across Candidate Variables ($N = 221{,}263$)

We refused to apply naive complete-case deletion (`dropna()`). The table below documents the empirical missingness structure across all candidate predictors and targets.

| Variable Name | Role | Total Records | Valid Count | Missing Count | Missing % | Missingness Category | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| `hw70_clean` | Target (HAZ) | 221,263 | 206,025 | 15,238 | 6.89% | E. Outcome Unavailable / Flagged | -6.00 | 6.00 |
| `hw71_clean` | Target (WAZ) | 221,263 | 210,524 | 10,739 | 4.85% | E. Outcome Unavailable / Flagged | -6.00 | 5.00 |
| `hw72_clean` | Target (WHZ) | 221,263 | 201,687 | 19,576 | 8.85% | E. Outcome Unavailable / Flagged | -5.00 | 5.00 |
| `hw73_clean` | Target (BMI-Z) | 221,263 | 202,215 | 19,048 | 8.61% | E. Outcome Unavailable / Flagged | -5.00 | 5.00 |
| `hw1` | Feature (Age mo) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 0.00 | 59.00 |
| `b4` | Feature (Sex) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 2.00 |
| `bord` | Feature (Birth order) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 16.00 |
| `b0` | Feature (Twin) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 0.00 | 3.00 |
| `b11` | Feature (Spacing) | 221,263 | 136,133 | 85,130 | 38.47% | **A. Structural (Firstborn)** | 6.00 | 224.00 |
| `m18_clean` | Feature (Birth size) | 221,263 | 218,383 | 2,880 | 1.30% | C. Refused / Don't Know | 1.00 | 5.00 |
| `birth_weight_kg` | Feature (Birth wt) | 221,263 | 200,439 | 20,824 | 9.41% | C. Refused / Unweighed | 0.50 | 6.00 |
| `hw57` | Feature (Anemia) | 221,263 | 183,855 | 37,408 | 16.91% | E. Outcome Unavailable / Consent | 1.00 | 4.00 |
| `still_breastfeeding`| Feature (BF status)| 221,263 | 220,870 | 393 | 0.18% | C. Refused / Don't Know | 0.00 | 1.00 |
| `m4_completed_months`| Feature (Weaned BF)| 221,263 | 71,282 | 149,981 | 67.78% | **A. Structural (Ongoing BF: 106k; Unasked: 43k)**| 0.00 | 59.00 |
| `m4_censored_months` | Feature (Censored) | 221,263 | 177,820 | 43,443 | 19.63% | C. Unasked in Roster / Don't Know | 0.00 | 59.00 |
| `diarrhea_recent` | Feature (Diarrhea) | 221,263 | 221,023 | 240 | 0.11% | C. Refused / Don't Know | 0.00 | 1.00 |
| `fever_recent` | Feature (Fever) | 221,263 | 221,119 | 144 | 0.07% | C. Refused / Don't Know | 0.00 | 1.00 |
| `cough_recent` | Feature (Cough) | 221,263 | 221,006 | 257 | 0.12% | C. Refused / Don't Know | 0.00 | 1.00 |
| `v012` | Feature (Mother age) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 15.00 | 49.00 |
| `v212` | Feature (Age 1st birth)| 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 12.00 | 48.00 |
| `v445_clean` | Feature (Mother BMI)| 221,263 | 215,671 | 5,592 | 2.53% | E. Unmeasured / Sentinel 9998 | 12.02 | 59.99 |
| `v457` | Feature (Mother an.)| 221,263 | 213,293 | 7,970 | 3.60% | E. Unmeasured / Consent | 1.00 | 4.00 |
| `v106` | Feature (Mother edu)| 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 0.00 | 3.00 |
| `v701_clean` | Feature (Partner edu)| 221,263 | 210,796 | 10,467 | 4.73% | C. Don't Know / Unmarried | 0.00 | 3.00 |
| `v714` | Feature (Mother work)| 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 0.00 | 1.00 |
| `m14_clean` | Feature (ANC visits)| 221,263 | 168,553 | 52,710 | 23.82% | C. Unasked in Roster / Sentinel 98 | 0.00 | 95.00 |
| `m15` | Feature (Delivery) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 11.00 | 96.00 |
| `v201` | Feature (Parity) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 16.00 |
| `v190` | Feature (Wealth) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 5.00 |
| `v025` | Feature (Residence) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 2.00 |
| `v024` | Feature (State) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 37.00 |
| `caste_clean` | Feature (Caste/Tribe)| 221,263 | 218,858 | 2,405 | 1.09% | C. Refused / Don't Know | 1.00 | 4.00 |
| `v130` | Feature (Religion) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 96.00 |
| `v113` | Feature (Water) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 11.00 | 97.00 |
| `v116` | Feature (Toilet) | 221,263 | 221,262 | 1 | 0.00% | B. Genuine Missing | 11.00 | 97.00 |
| `v119` | Feature (Electricity)| 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 0.00 | 7.00 |
| `v161` | Feature (Cooking fuel)| 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 96.00 |
| `v136` | Feature (Family size)| 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 35.00 |
| `v151` | Feature (Head sex) | 221,263 | 221,263 | 0 | 0.00% | B. Genuine Missing | 1.00 | 2.00 |
| `sample_weight` | Survey Weight | 221,263 | 221,263 | 0 | 0.00% | Survey Design Weight | 0.003 | 37.59 |

---

## 5. Range Validation: Observed Ranges vs. Imposed Bounds

| Metric | Cleaning Rule Applied | Imposed Upper/Lower Bound? | Observed Empirical Range | Clinical Plausibility Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **`hw70_clean`** | Recode $\ge 9000$ to NaN, $\div 100$ | Documented WHO Limits ($-6$ to $+6$) | $[-6.00, +6.00]\text{ SD}$ | Plausible biological distribution under WHO growth standards. |
| **`hw71_clean`** | Recode $\ge 9000$ to NaN, $\div 100$ | Documented WHO Limits ($-6$ to $+5$) | $[-6.00, +5.00]\text{ SD}$ | Plausible biological distribution. |
| **`hw72_clean`** | Recode $\ge 9000$ to NaN, $\div 100$ | Documented WHO Limits ($-5$ to $+5$) | $[-5.00, +5.00]\text{ SD}$ | Plausible biological distribution. |
| **`v445_clean`** | Recode sentinel 9998 to NaN, $\div 100$ | **NO imposed filter** | $[12.02, 59.99]\text{ kg/m}^2$ | Observed maternal BMI from extreme emaciation to severe obesity. |
| **`m14_clean`** | Recode sentinel 98 to NaN | **NO imposed filter** | $[0.0, 95.0]\text{ visits}$ | Observed prenatal visits from zero to intensive high-risk care. |
| **`birth_weight_kg`** | Recode sentinel $\ge 9000$ to NaN, $\div 1000$ | Plausible clinical limits ($0.5$ to $6.0\text{ kg}$) | $[0.50, 6.00]\text{ kg}$ | Standard pediatric range for infant birth weight. |

---

## 6. Physical Measurements Governance (Scenario A Compliance)

The following variables were cleaned and retained in `data/interim/cleaned_u5_data.csv.gz` exclusively for target verification and secondary clinical audit:
- `hw2_clean` (Child weight in kg)
- `hw3_clean` (Child height in cm)
- `hw4`–`hw12` (Percentiles and % reference medians)
- `hw13` (Measurement result)

> [!IMPORTANT]
> In accordance with our **Scenario A (Community Pre-Screening)** contract, **none of these physical measurement variables will ever enter the model input feature matrix $X$.**

---

## 7. Before vs. After Aggregate Counts

| Cohort / Target Dimension | Raw Dataset Count | Cleaned Living U5 Dataset Count | Population Retention % |
| :--- | :---: | :---: | :---: |
| **Total Child Records** | 232,920 | **221,263** | 94.99% |
| **Living Children** | 224,218 | **221,263** | 98.68% |
| **Eligible Stunting (`hw70_clean` valid)** | 206,025 | **206,025** | 100.0% (Retained) |
| **Eligible Underweight (`hw71_clean` valid)** | 210,524 | **210,524** | 100.0% (Retained) |
| **Eligible Wasting (`hw72_clean` valid)** | 201,687 | **201,687** | 100.0% (Retained) |
| **Eligible Complete Unified Cohort** | 198,802 | **198,802** | 100.0% (Retained) |

---

## 8. Decisions Intentionally Deferred to Step 5+

To maintain research integrity and modular design, the following operations were intentionally NOT performed in Step 4:
1. **Binary Target Creation**: Binarizing $Z < -2\text{ SD}$ and $Z < -3\text{ SD}$ is deferred to **Step 5 (Target Creation)**.
2. **Missing Feature Imputation**: Deciding median/mode vs KNN vs tree-native handling is deferred to **Step 7 (Feature Engineering)**.
3. **Categorical Encoding**: One-hot or ordinal encoding is deferred to **Step 7**.
4. **Feature Scaling / Normalization**: Standard scaling or min-max normalization is deferred to **Step 7**.
5. **Class Resampling (SMOTE)**: Resampling is strictly forbidden before data splitting and is deferred to **Step 11**.

---

## 9. Viva / Interview Questions & Answers

### Q1: Why is it methodologically essential to separate `still_breastfeeding` from completed breastfeeding duration?
**Sample Answer**:
"In epidemiological surveys, children who are still breastfeeding at the time of the interview have an unfinished, right-censored exposure duration. If a researcher naively replaces the 'still breastfeeding' sentinel code (code 95) with the child's current age, they create a synthetic variable that perfectly tracks age rather than nutritional weaning practices. In our pipeline, we separate this into two variables: a binary indicator `still_breastfeeding = 1` capturing current protective feeding status, and `m4_completed_months` capturing completed duration strictly among weaned children. This prevents duration from being conflated with current infant age."

### Q2: Why did you remove the arbitrary cap of 40 on antenatal visits (`m14`)?
**Sample Answer**:
"Data cleaning must be strictly grounded in documented survey codebooks rather than arbitrary heuristic thresholds. In the NFHS-5 questionnaire, `m14` captures the exact reported count of antenatal care visits, and the official value label `M14` defines only code `0` ('No visits') and code `98` ('Don't know'). Imposing an arbitrary cap at 40 visits would truncate valid empirical observations (such as high-risk pregnancies receiving weekly monitoring). By removing the arbitrary cap, we recode only the documented special code `98` to `NaN` while preserving all valid reported visits up to the observed empirical maximum of 95 visits."

### Q3: Is maternal BMI (`v445`) filtered between 12.02 and 59.99 kg/m², or is that an observed range?
**Sample Answer**:
"The range 12.02 to 59.99 kg/m² is purely the observed empirical data range after removing documented DHS sentinel code 9998 ('Flagged cases'). No arbitrary mathematical bounds were imposed on maternal BMI. After scaling raw values by 100 (reversing the two implied decimals), the minimum observed value is 12.02 kg/m² (severe maternal wasting) and the maximum is 59.99 kg/m² (extreme maternal obesity). Both represent genuine, clinically plausible extremes in a nationwide survey of over 200,000 Indian mothers."

---

## Commands Used
```bash
# Run systematic data cleaning pipeline with review corrections
python src/data/clean_data.py

# Run Step 4 verification test suite asserting review corrections
python tests/test_data_cleaning.py
```

---

## Next Step
**Step 5 — Target Creation**: Systematically define and construct the clinical malnutrition outcome indicators (moderate and severe stunting, wasting, underweight, and composite anthropometric failure) based on the cleaned continuous z-scores (`hw70_clean`, `hw71_clean`, `hw72_clean`) following WHO 2006 guidelines.
