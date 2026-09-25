# Step 1 — Dataset Inspection & Metadata Discovery

## Objective
The objective of Step 1 is to safely inspect the raw, authorized India NFHS-5 (2019–21) Children's Recode (KR) dataset without modifying or cleaning any data, verify file integrity, record exact dataset dimensions, examine data types and missing-value patterns, understand standard DHS coding structures (including implied decimal scales and sentinel flag codes), discover anthropometric outcome variables and candidate demographic predictors, and document potential data leakage risks.

---

## What We Did
1. **File Identification**: Located and inventoried all raw files under `IAKR7EDT/`:
   - `IAKR7EFL.DTA` (441.38 MB Stata 13/14+ binary dataset).
   - `IAKR7EFL.DO` (324.69 KB Stata syntax file with variable and value label definitions).
   - `IAKR7EFL.DCT` (62.54 KB Stata dictionary definition file).
   - `IAKR7EFL.MAP` (595.24 KB variable layout and column mapping codebook).
   - `IAKR7EFL.FRQ` and `IAKR7EFL.FRW` (unweighted and weighted frequency distribution tables).
2. **Metadata Header Inspection**: Read the binary Stata header using `pandas.io.stata.StataReader` without inflating the entire file in memory. Discovered exact dimensions: **232,920 rows** and **1,644 columns**.
3. **Automated Metadata Parsing (`src/data/inspect_metadata.py`)**: Developed an automated parser extracting human-readable variable descriptions and value label dictionaries directly from `IAKR7EFL.DO`.
4. **Data Type & Missingness Profiling (`src/data/inspect_data_types.py`)**: Profiled key anthropometric targets, demographic predictors, household indicators, and survey weights across all 232,920 records.
5. **Coding Convention Discovery**: Identified critical DHS encoding patterns:
   - Z-scores (`hw70`, `hw71`, `hw72`, `hw73`) are scaled by 100 with two implied decimals (e.g., `-224.0` corresponds to a standard deviation of `-2.24 SD`).
   - Flag codes `9996`, `9997`, and `9998` represent out-of-range biological values or WHO-flagged errors.
   - Child survival status (`b5 == 0`) explains exactly 8,702 structural missing values across morbidity and age questions.
6. **Automated Verification Suite (`tests/test_inspection.py`)**: Built and executed automated unit tests verifying file existence, exact dimensions, and core anthropometric variable presence.

---

## Files Created
- `src/data/inspect_metadata.py`: Automated parser for Stata `.DO` label dictionaries and dataset dimensions.
- `src/data/inspect_data_types.py`: Quantitative profiler measuring missingness, data types, and DHS sentinel flag codes.
- `tests/test_inspection.py`: Automated verification tests confirming dataset presence, 232,920 x 1,644 dimensions, and essential variables.
- `docs/01_dataset_inspection.md`: Comprehensive Step 1 documentation.

---

## Files Modified
- None. The raw dataset files in `IAKR7EDT/` remain strictly read-only and unaltered.

---

## Libraries Used

### 1. `pandas` (specifically `pandas.io.stata.StataReader` and `pandas.read_stata`)
- **What it does**: Reads Stata binary format (`.dta`), parses header metadata (variable names, types, labels), and loads selected column subsets into DataFrames.
- **Why we used it**: DHS microdata is distributed in Stata format. `StataReader` allows inspecting header metadata and reading specific column subsets (`columns=[...]`) without loading all 1,644 columns into RAM at once, preventing memory overflow.
- **Alternatives**: Full `pd.read_stata(path)` without column filtering, `pyreadstat`.
- **Why we didn't use alternatives**: Loading all 1,644 columns of 232,920 rows consumes >3 GB RAM and slows initial inspection; `pandas.io.stata.StataReader` is native, built into pandas, and requires no external C compiler dependencies.

### 2. `numpy`
- **What it does**: Provides vectorized numeric checks (`np.issubdtype`), sentinel value filtering, and statistical summaries.
- **Why we used it**: Rapid calculation of numeric column minimums, maximums, and sentinel count identification.
- **Alternatives**: Built-in Python `math` module.
- **Why we didn't use alternatives**: `numpy` operates directly on memory contiguous array blocks, executing operations in milliseconds across 232,920 records.

### 3. Python standard library (`re`, `os`)
- **What it does**: Regular expression matching and filesystem path checks.
- **Why we used it**: Parsed Stata syntax definitions (`label define`, `label variable`) directly from `IAKR7EFL.DO` to produce an automated, programmatic mapping of integer codes to labels.
- **Alternatives**: Manual copy-pasting from PDF codebooks.
- **Why we didn't use alternatives**: Manual transcription is prone to human error and violates reproducibility.

---

## Important Code Explanation

### 1. Header Metadata Extraction without Memory Ingestion (`src/data/inspect_metadata.py`)
```python
with StataReader(DTA_PATH) as reader:
    vlabels = reader.variable_labels()
    nobs = getattr(reader, "_nobs", 0)
    nvar = getattr(reader, "_nvar", 0)
```
- `StataReader(DTA_PATH)` opens a streaming pointer to the binary Stata file.
- Calling `reader.variable_labels()` reads the header dictionary segment.
- This populates `_nobs` (number of observations = 232,920) and `_nvar` (number of variables = 1,644) in less than 2 seconds, completely bypassing the multi-hundred-megabyte data matrix.

### 2. Stata `.DO` File Label Parsing (`src/data/inspect_metadata.py`)
```python
var_match = re.search(rf'label variable {var}\s+"([^"]+)"', content, re.IGNORECASE)
val_lbl_match = re.search(rf'label values {var}\s+([A-Za-z0-9_]+)', content, re.IGNORECASE)
```
- Scans `IAKR7EFL.DO` for Stata commands:
  - `label variable hw70 "Height/Age standard deviation (new WHO)"`
  - `label values hw70 HW70`
  - `label define HW70 9996 "Height out of plausible limits" 9997 "Age in days out of plausible limits" 9998 "Flagged cases"`
- Programmatically extracts the exact clinical meaning of every code directly from the official survey definitions.

### 3. Column Profiling & Sentinel Discovery (`src/data/inspect_data_types.py`)
```python
if col in ["hw70", "hw71", "hw72", "hw73"]:
    n_flagged = int(series.isin([9996, 9997, 9998]).sum())
```
- Detects that values like `9998` are NOT large positive z-scores (+99.98 SD), but DHS sentinel codes denoting WHO biological implausibility flags.

---

## Data Flow
In Step 1:
1. `IAKR7EFL.DTA` header was read into `StataReader` to capture dimensions and variable lists.
2. `IAKR7EFL.DO` text was parsed by regex to extract official variable descriptions and value label dictionaries.
3. A focused 37-variable subset was loaded into RAM via `pd.read_stata(columns=...)`.
4. Summary metrics were extracted and verified.
5. **The raw files on disk remained untouched.**

---

## Dataset Dimensions & Measured Results

- **Source File**: `IAKR7EDT/IAKR7EFL.DTA`
- **Total Observations (Rows)**: **232,920** children born in the 5 years preceding the survey.
- **Total Variables (Columns)**: **1,644** survey variables.
- **Dead Children in History**: **8,702** (`b5 == 0`).
- **Living Children**: **224,218** (`b5 == 1`).

### Profile of Key Inspected Variables (37 Columns)

| Variable | Dtype | Meaning | NaN Count | NaN % | Sentinel / Flagged | Valid Range Min | Valid Range Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hw70` | float64 | Height/Age standard deviation (new WHO) | 21,831 | 9.37% | 5,064 | -600.0 (-6.0 SD) | 599.0 (+5.99 SD) |
| `hw71` | float64 | Weight/Age standard deviation (new WHO) | 21,236 | 9.12% | 1,160 | -600.0 (-6.0 SD) | 495.0 (+4.95 SD) |
| `hw72` | float64 | Weight/Height standard deviation (new WHO) | 18,389 | 7.89% | 12,844 | -500.0 (-5.0 SD) | 499.0 (+4.99 SD) |
| `hw73` | float64 | BMI standard deviation (new WHO) | 21,917 | 9.41% | 8,788 | -500.0 (-5.0 SD) | 499.0 (+4.99 SD) |
| `hw13` | float64 | Result of measurement - height/weight | 118 | 0.05% | 0 | 0.0 (Measured) | 7.0 (No meas. found) |
| `hw1` | float64 | Child's age in months | 11,657 | 5.00% | 0 | 0.0 months | 59.0 months |
| `b4` | int8 | Sex of child (1=Male, 2=Female) | 0 | 0.00% | 0 | 1 | 2 |
| `b5` | int8 | Child is alive (0=No, 1=Yes) | 0 | 0.00% | 0 | 0 | 1 |
| `b8` | float64 | Current age of child (single years) | 8,702 | 3.74% | 0 | 0 | 4 |
| `bord` | int8 | Birth order number | 0 | 0.00% | 0 | 1 | 16 |
| `m18` | int8 | Size of child at birth | 0 | 0.00% | 3,545 (code 8: don't know) | 1 (Very large) | 5 (Very small) |
| `m19` | int16 | Birth weight in kg (3 decimals) | 0 | 0.00% | 23,654 (9996/9998: unweighed) | 500 (0.5 kg) | 6500 (6.5 kg) |
| `hw57` | float64 | Child anemia level (1=Severe, 4=None) | 49,065 | 21.07% | 0 | 1 | 4 |
| `v012` | int8 | Respondent's (mother's) current age | 0 | 0.00% | 0 | 15 years | 49 years |
| `v445` | float64 | Mother's Body Mass Index (2 decimals) | 5,910 | 2.54% | 0 | 1202 (12.02) | 5990 (59.90) |
| `v457` | float64 | Mother's anemia level (1=Severe, 4=None) | 8,537 | 3.67% | 0 | 1 | 4 |
| `m14` | float64 | Antenatal care visits during pregnancy | 56,077 | 24.08% | 0 | 0 | 98 (Don't know) |
| `m15` | int8 | Place of delivery | 0 | 0.00% | 0 | 11 (Home) | 96 (Other) |
| `m4` | float64 | Duration of breastfeeding (months) | 46,738 | 20.07% | 0 | 0 | 98 (Don't know) |
| `v404` | int8 | Currently breastfeeding (0=No, 1=Yes) | 0 | 0.00% | 0 | 0 | 1 |
| `v001` | int32 | Primary Sampling Unit / Cluster number | 0 | 0.00% | 0 | 101 | 93242 |
| `v005` | int32 | Sample weight (6 implied decimals) | 0 | 0.00% | 0 | 3,462 | 37,585,909 |
| `v021` | int32 | Primary sampling unit | 0 | 0.00% | 0 | 101 | 93242 |
| `v022` | int32 | Sample strata for sampling errors | 0 | 0.00% | 0 | 11 | 93223 |
| `v024` | int8 | State / Union Territory (36 states) | 0 | 0.00% | 0 | 1 | 37 |
| `v025` | int8 | Type of place of residence (1=Urban, 2=Rural) | 0 | 0.00% | 0 | 1 | 2 |
| `v106` | int8 | Mother's education (0=None, 3=Higher) | 0 | 0.00% | 0 | 0 | 3 |
| `v190` | int8 | Wealth index quintile (1=Poorest, 5=Richest) | 0 | 0.00% | 0 | 1 | 5 |
| `v113` | int8 | Source of drinking water | 0 | 0.00% | 0 | 11 | 97 |
| `v116` | float64 | Type of toilet facility | 1 | 0.00% | 0 | 11 | 97 |
| `v119` | int8 | Household has electricity (0=No, 1=Yes) | 0 | 0.00% | 0 | 0 | 7 |
| `v136` | int8 | Number of household members | 0 | 0.00% | 0 | 1 | 35 |
| `v151` | int8 | Sex of head of household (1=Male, 2=Female) | 0 | 0.00% | 0 | 1 | 3 |
| `v152` | int8 | Age of head of household | 0 | 0.00% | 0 | 15 years | 98 years |
| `h11` | float64 | Child had diarrhea recently (0=No, 2=Yes) | 8,702 | 3.74% | 0 | 0 | 8 (Don't know) |
| `h22` | float64 | Child had fever in last two weeks (0=No, 2=Yes) | 8,702 | 3.74% | 0 | 0 | 8 (Don't know) |
| `h31` | float64 | Child had cough in last two weeks (0=No, 2=Yes) | 8,702 | 3.74% | 0 | 0 | 8 (Don't know) |

---

## Detailed Breakdown of Anthropometric Targets

### 1. Stunting: Height-for-Age (`hw70`)
- **Valid Measured Range**: 206,025 children have valid measurements between `-600` (-6.00 SD) and `+599` (+5.99 SD).
- **Flagged Cases (`9998`)**: 5,045 records were flagged by DHS as biologically implausible based on WHO flags (HAZ < -6 SD or > +6 SD).
- **Out of Range Age (`9997`)**: 19 cases.
- **Unmeasured / Missing (`NaN`)**: 21,831 records (due to child absence, refusal, or deceased status).

### 2. Underweight: Weight-for-Age (`hw71`)
- **Valid Measured Range**: 210,524 children have valid measurements between `-600` (-6.00 SD) and `+495` (+4.95 SD).
- **Flagged Cases (`9998`)**: 1,141 records.
- **Out of Range Age (`9997`)**: 19 cases.
- **Unmeasured / Missing (`NaN`)**: 21,236 records.

### 3. Wasting: Weight-for-Height (`hw72`)
- **Valid Measured Range**: 201,687 children have valid measurements between `-500` (-5.00 SD) and `+499` (+4.99 SD).
- **Out of Range Height (`9996`)**: 5,183 records.
- **Flagged Cases (`9998`)**: 7,661 records.
- **Unmeasured / Missing (`NaN`)**: 18,389 records.

---

## Important DHS Coding Conventions Discovered

1. **Implicit Decimal Places**:
   - Anthropometric standard deviations (`hw70`, `hw71`, `hw72`, `hw73`) have **two implied decimal places**. `-200` represents `-2.00 SD`. A value of `-140` is `-1.40 SD`.
   - Respondent's BMI (`v445`) has **two implied decimal places**. `2150` represents a BMI of `21.50 kg/m²`.
   - Sample weights (`v005`) have **six implied decimal places**. To obtain the true normalized sampling weight $W_i$, one must compute $W_i = \text{v005} / 1{,}000{,}000$.
   - Birth weight (`m19`) has **three implied decimal places**. A value of `3200` represents `3.200 kg`.
2. **Sentinel / Special Missing Codes**:
   - `9996`: Height / measurement out of plausible limits.
   - `9997`: Age in days out of plausible limits.
   - `9998`: Flagged by WHO standard as biologically implausible.
   - `9999` / `98`: "Don't know" / Missing.
   - **Critical Takeaway**: If treated as continuous numbers, `9998` would skew model means and tree splits. They must be handled cleanly as missing or invalid indicator flags in subsequent steps.
3. **Structural Missingness vs. Random Missingness**:
   - Variables `b8` (current child age), `h11` (diarrhea), `h22` (fever), and `h31` (cough) have **exactly 8,702 missing values**, which matches the count of deceased children (`b5 == 0`). These questions are logically omitted in the DHS questionnaire for deceased children.

---

## Identifiers, Survey Weights, and Multilevel Variables

- **`v001`**: Cluster Number (Primary Sampling Unit - PSU). Represents the specific village or census enumeration block where the survey was conducted. Crucial for understanding spatial clustering.
- **`v002`**: Household Number. Identifies individual families within a cluster.
- **`v003`**: Respondent's line number in the household roster.
- **`v005`**: Women's sample weight (integer with 6 implied decimals). Must be used if calculating nationally representative population statistics, but handled carefully during machine learning loss optimization.
- **`v021`, `v022`, `v024`**: Primary sampling unit, sampling strata, and state identifiers (36 states and Union Territories).

---

## Potential Data Leakage Concerns Discovered

In Step 1 inspection, we identified variables that **must NEVER be used as features** to predict malnutrition:
1. **`hw2` (Weight in kg) and `hw3` (Height in cm)**:
   - If predicting stunting (Height-for-Age), using `hw3` (child height) would be severe target leakage.
   - If predicting wasting (Weight-for-Height), using `hw2` (child weight) or `hw3` would be severe target leakage.
2. **`hw70`, `hw71`, `hw72`, `hw73` (Z-scores)**:
   - These are direct representations of the target outcomes. Using `hw70` to predict stunting or `hw72` to predict wasting would be direct label leakage.
3. **`hw4` through `hw12` (Percentiles and Percent of Reference Median)**:
   - Derived directly from child height and weight against the WHO reference median.
4. **`hw13` (Result of Measurement)**:
   - Encodes whether the child was measured or refused; directly correlates with missingness in the target variable.

---

## What We Know vs. What Still Needs to be Verified

### What We Know with Certainty (Empirically Measured)
- The raw dataset has exactly **232,920 rows** and **1,644 columns**.
- Stunting, wasting, and underweight indicators exist in standard DHS WHO format (`hw70`, `hw71`, `hw72`).
- 8,702 records correspond to deceased children (`b5 == 0`).
- Valid anthropometric measurements exist for >200,000 children.
- Sentinel codes `9996`, `9997`, `9998` exist and require systematic exclusion or recoding.

### What Still Needs to be Verified (For Steps 2 and 3)
- Exactly how many living children aged 0–59 months have complete anthropometric measurements for all three conditions simultaneously vs. individually.
- How the base paper (Islam et al., 2024) handled children with one valid anthropometric measurement and one missing measurement.
- Which specific socioeconomic, dietary, and healthcare predictors should be included in the candidate feature pool.

---

## Base Paper Comparison
- **Base Paper (Islam et al., 2024, PLOS ONE)**:
  - Dataset: Bangladesh DHS 2017–18 (BDHS) KR file.
  - Sample size analyzed: 8,759 births, filtered down to 7,859 living children with valid anthropometric measurements.
  - Predictors: Focused on ~20 selected variables (child age, sex, birth order, maternal BMI, education, antenatal visits, wealth index, place of residence).
- **Our Project**:
  - Dataset: India NFHS-5 2019–21 (IAKR7EFL.DTA).
  - Total records: **232,920**, representing an order-of-magnitude larger cohort (>25x the size of the BDHS dataset).
  - Extends the analysis to evaluate national and regional heterogeneity across 36 Indian states and union territories.

---

## Limitations
- Step 1 only inspects the dataset; it does not clean, filter, or impute missing data.
- Row counts reported here represent the full raw survey roster, not the final eligible research cohort.

---

## Interview Questions & Answers

### Q1: What is the India NFHS-5 Children's Recode (KR) dataset, and what does each row represent?
**Sample Answer**:
The KR (Children's Recode) file in NFHS-5 contains individual survey records for every child born in the five years preceding the survey to interviewed women aged 15–49. Each row represents a single child, capturing birth history, immunization, anthropometric measurements (height, weight, z-scores), maternal characteristics, and household living conditions.

### Q2: Why are z-scores like `hw70` and `hw72` stored as large integers like `-224` or `9998` instead of floats like `-2.24`?
**Sample Answer**:
DHS uses legacy fixed-width and optimized database storage conventions where standard deviations are scaled by 100 to avoid floating-point rounding errors across different statistical packages. A value of `-224` represents `-2.24 SD`. Values such as `9996`, `9997`, and `9998` are sentinel codes denoting out-of-range heights, impossible ages, or biological implausibility flags defined by the WHO 2006 Child Growth Standards. Treating `9998` as a numeric value would ruin model training and must be parsed during preprocessing.

### Q3: What is target leakage in the context of childhood malnutrition modeling, and how did you detect it during Step 1?
**Sample Answer**:
Target leakage occurs when a model is trained using features that directly contain or are derived from the target variable, artificially inflating evaluation metrics while rendering the model useless in real clinical deployment. During Step 1, we identified variables like child height (`hw3`), weight (`hw2`), percent of reference median (`hw6`, `hw9`), and direct WHO standard deviations (`hw70`, `hw71`, `hw72`). In a risk intelligence screening system, raw anthropometric measurements cannot be used as input predictors to predict anthropometric deficits like stunting or wasting.

---

## Commands Used
```bash
# Run metadata inspection parser
python src/data/inspect_metadata.py

# Run data types, missingness, and sentinel code profiler
python src/data/inspect_data_types.py

# Run Step 1 automated verification test suite
python tests/test_inspection.py
```

---

## Next Step
**Step 2 — Data Documentation**: Create a formal, auditable Data Dictionary for the variables selected for our research cohort, documenting their variable names, meanings, data types, DHS coding, missing-value representations, and role (target, predictor, survey weight, or identifier).
