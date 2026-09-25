# Step 0 — Project Initialization & Architecture Setup

## Objective
The objective of Step 0 is to initialize a clean, professional, and reproducible directory structure for the **NutriSense AI** project, establish a strict data protection protocol for confidential DHS/NFHS survey microdata, configure dependencies, and verify that the computing environment is fully ready for rigorous machine learning research.

---

## What We Did
1. **Workspace Inspection**: Inspected the root workspace to locate existing assets without altering or prematurely loading raw survey files.
2. **Directory Architecture Creation**: Created a modular directory structure separating confidential raw data, interim cleaned files, final processed features, serialized models, report visualizations, source packages, and testing suites.
3. **Strict Data Protection (`.gitignore`)**: Enacted privacy rules preventing any confidential DHS/NFHS microdata files (such as `IAKR7EDT/`, `.dta`, `.sav`, `.dat`, `.csv`, and `.parquet`), serialized weights, and local environment variables from ever being tracked or committed to version control.
4. **Dependency Specification (`requirements.txt`)**: Defined the core scientific, machine learning, and explainability stack with explicit version bounds suitable for Python 3.10+.
5. **Modular Source Packages**: Initialized Python package namespaces under `src/` (`src.data`, `src.features`, `src.models`, `src.explainability`, `src.utils`) and `tests/`.
6. **Verification Test (`tests/test_setup.py`)**: Built an automated verification script to validate directory presence, `.gitignore` privacy enforcement, and library importability.

---

## Files Created
- `.gitignore`: Specifies privacy exclusion rules for DHS survey microdata, binary models, cache files, and virtual environments.
- `requirements.txt`: Specifies core dependencies required across the data processing, modeling, and explainability pipeline.
- `data/raw/.gitkeep`: Preserves raw data folder in Git while keeping confidential contents untracked.
- `data/interim/.gitkeep`: Preserves interim transformation storage.
- `data/processed/.gitkeep`: Preserves processed datasets storage.
- `models/.gitkeep`: Preserves model artifact directory.
- `reports/figures/.gitkeep`: Preserves directory for generated analytical plots.
- `src/__init__.py`: Marks `src` as a top-level Python package.
- `src/data/__init__.py`: Package namespace for data extraction and cleaning.
- `src/features/__init__.py`: Package namespace for feature engineering and target derivation.
- `src/models/__init__.py`: Package namespace for baseline and candidate ML models.
- `src/explainability/__init__.py`: Package namespace for SHAP interpretability modules.
- `src/utils/__init__.py`: Package namespace for survey weight utils, logging, and helpers.
- `tests/__init__.py`: Package namespace for project test suites.
- `tests/test_setup.py`: Standalone automated verification test for Step 0.
- `docs/00_project_setup.md`: Comprehensive documentation for Step 0.

---

## Files Modified
- None (this is the initialization step).

---

## Libraries Used

### 1. `numpy`
- **What it does**: Provides high-performance multidimensional array objects, vectorized mathematical routines, and linear algebra primitives.
- **Why we used it**: Required by scikit-learn, XGBoost, and SHAP for numerical operations, array indexing, matrix computations, and NaN handling.
- **Alternatives**: Pure Python lists / `math` module, or PyTorch/JAX tensors.
- **Why we didn't use alternatives**: Pure Python is orders of magnitude slower and lacks vectorized SIMD operations; PyTorch/JAX add unnecessary multi-gigabyte GPU runtime overhead for tabular epidemiological modeling.

### 2. `pandas`
- **What it does**: Provides tabular DataFrame and Series structures with indexing, slicing, aggregation, and native support for reading Stata (`.dta`), CSV, and Parquet files.
- **Why we used it**: The DHS Children's Recode contains hundreds of survey columns and hundreds of thousands of child records; pandas is the industry standard for tabular data wrangling and Stata import.
- **Alternatives**: Polars, Dask, Modin.
- **Why we didn't use alternatives**: Polars is fast but has varying compatibility with legacy Stata categorical labels and metadata; Dask/Modin are distributed tools suited for terabyte-scale datasets and introduce cluster overhead unnecessary for a single NFHS-5 recode file (~500MB to 1GB).

### 3. `scipy`
- **What it does**: Provides scientific computing, statistical distributions, hypothesis testing, and sparse matrix operations.
- **Why we used it**: Required by scikit-learn for probability distributions, statistical significance checks, and handling sparse feature matrices.
- **Alternatives**: `statsmodels`.
- **Why we didn't use alternatives**: `scipy` is lighter and already a direct dependency of scikit-learn; `statsmodels` is excellent for econometric regression tables but redundant for building predictive ML pipelines.

### 4. `pyreadstat`
- **What it does**: Reads and writes Stata (`.dta`), SAS (`.sas7bdat`), and SPSS (`.sav`) files using compiled C libraries (`ReadStat`), preserving column labels, value label dictionaries, and missing-value flags.
- **Why we used it**: DHS files rely heavily on encoded integers mapped to variable value labels (e.g., `1 = Male`, `2 = Female`, `98 = Don't know`). `pyreadstat` extracts both numeric codes and variable label metadata without data loss.
- **Alternatives**: Native `pandas.read_stata()`.
- **Why we didn't use alternatives**: `pandas.read_stata()` is good for reading values, but `pyreadstat` provides superior performance and direct access to metadata dictionaries, which are vital for Step 1 inspection and Step 2 codebook creation.

### 5. `scikit-learn`
- **What it does**: Provides robust implementations of classification algorithms (Logistic Regression, Random Forest), data preprocessing scalers/encoders, stratified k-fold splitting, cross-validation, and performance evaluation metrics.
- **Why we used it**: Standard, audited, reproducible library for tabular machine learning with pipeline support (`sklearn.pipeline.Pipeline`) preventing data leakage.
- **Alternatives**: Custom implementations from scratch, Weka, or deep learning libraries.
- **Why we didn't use alternatives**: Custom algorithms are error-prone and unoptimized; deep learning on tabular demographic data often underperforms tree ensembles and adds needless training instability.

### 6. `xgboost` & `lightgbm`
- **What it does**: Optimized gradient boosted decision tree (GBDT) frameworks using histogram-based splitting and tree pruning.
- **Why we used it**: Tree ensembles consistently achieve state-of-the-art predictive performance on tabular survey data, naturally handling non-linear interactions, skewed demographic features, and missing values.
- **Alternatives**: CatBoost, AdaBoost, or Multi-Layer Perceptrons (MLP).
- **Why we didn't use alternatives**: XGBoost and LightGBM provide the best balance of speed, hyperparameter flexibility, and seamless integration with the TreeSHAP explainability algorithm.

### 7. `shap` (SHapley Additive exPlanations)
- **What it does**: Implements game-theoretic Shapley values to provide mathematically grounded, local and global feature attribution for machine learning models.
- **Why we used it**: Health and epidemiological screening systems require trustworthy, explainable outputs to reveal *why* a child is flagged at high risk of undernutrition.
- **Alternatives**: LIME (Local Interpretable Model-agnostic Explanations), Permutation Feature Importance.
- **Why we didn't use alternatives**: LIME constructs local surrogate approximations that can be unstable; Permutation Importance only yields global importance and fails to explain individual patient predictions. SHAP provides local accuracy and consistency.

### 8. `joblib`
- **What it does**: Provides lightweight pipelining and disk serialization optimized for Python objects containing large NumPy arrays.
- **Why we used it**: Essential for persisting trained models, scalers, and encoders to disk for Step 18 and Step 19 backend deployment.
- **Alternatives**: Standard Python `pickle`.
- **Why we didn't use alternatives**: `joblib` is significantly faster and more memory-efficient than `pickle` when serializing scikit-learn models and large array matrices.

### 9. `matplotlib` & `seaborn`
- **What it does**: Plotting libraries for creating static, publication-quality figures, distribution plots, ROC curves, and confusion matrices.
- **Why we used it**: Required for visual inspection during EDA (Step 6), evaluation curves (Step 13), and research reports.
- **Alternatives**: Plotly, Bokeh.
- **Why we didn't use alternatives**: Plotly and Bokeh produce heavy interactive JavaScript bundles; static vector/raster plots (`matplotlib`/`seaborn`) are preferred for research papers and thesis documentation.

---

## Important Code Explanation

### 1. Privacy Enforcement in `.gitignore`
```gitignore
# --- DHS / NFHS CONFIDENTIAL MICRODATA (CRITICAL PRIVACY RULE) ---
IAKR7EDT/
data/raw/*
!data/raw/.gitkeep
data/interim/*
!data/interim/.gitkeep
data/processed/*
!data/processed/.gitkeep
*.dta
*.DTA
*.sav
*.SAV
*.dat
*.DAT
*.sas7bdat
*.csv
*.parquet
*.feather
```
- **Logical Flow**:
  - `IAKR7EDT/` specifically ignores the extracted folder of the India NFHS-5 Children's Recode.
  - `data/raw/*` followed by `!data/raw/.gitkeep` tells Git: "Ignore all files inside `data/raw`, but preserve the directory itself by tracking `.gitkeep`."
  - Extension wildcards (`*.dta`, `*.csv`, etc.) ensure that even if a developer accidentally saves an extracted survey file in a subfolder or project root, Git will refuse to stage or commit it.

### 2. Automated Verification in `tests/test_setup.py`
```python
def test_gitignore_protects_confidential_data():
    """Verify that .gitignore exists and contains critical rules for DHS/NFHS privacy."""
    assert os.path.exists(".gitignore"), ".gitignore does not exist"
    with open(".gitignore", "r") as f:
        content = f.read()
    
    assert "IAKR7EDT" in content, ".gitignore must ignore raw DHS folder IAKR7EDT"
    assert "*.dta" in content or "*.DTA" in content, ".gitignore must ignore Stata microdata"
    assert "data/raw/*" in content, ".gitignore must ignore raw data contents"
    assert "models/*" in content, ".gitignore must ignore trained model weights"
```
- **Logical Flow**:
  - Automatically asserts that the privacy rules are physically present in `.gitignore`. If someone alters `.gitignore` and accidentally un-ignores microdata, the test will fail immediately.

---

## Data Flow
In Step 0, **no microdata moves**. 
The only data flow is architectural:
1. Directory structure is created on the local filesystem.
2. Version control boundaries are established via `.gitignore`.
3. The environment dependencies are verified.
4. The raw dataset remains isolated and untouched inside `IAKR7EDT/`.

---

## Important Concepts

1. **Microdata Confidentiality**:
   DHS and NFHS survey datasets contain individual-level responses from mothers and children across Indian districts. Even though names are removed, combinations of demographic variables (village/cluster, age, birth dates, family size) could theoretically allow re-identification. The DHS terms of use strictly forbid distributing or uploading raw microdata.
2. **Modular Code Architecture**:
   Instead of writing a single monolithic script (`all_in_one.py`), splitting code into `src/data/`, `src/features/`, and `src/models/` ensures code reusability, testability, and prevents training/inference discrepancy.
3. **Data Leakage Prevention by Architecture**:
   By strictly separating `data/raw/` (immutable read-only), `data/interim/` (cleaned subsets), and `data/processed/` (final train/test features), we ensure that raw survey data is never contaminated.

---

## Research Reasoning
- We established the project structure **before** touching any data to avoid the common pitfall of accidentally staging confidential survey files into Git history. Once pushed to a remote repository, DHS microdata cannot easily be removed without rewriting Git history.
- We selected Python 3.10+ and standard tabular ML libraries because our research foundation relies on tree ensembles and explainable AI (SHAP), which are mature and stable in the scikit-learn/pandas ecosystem.

---

## Base Paper Comparison
- **Base Paper (Islam et al., 2024, PLOS ONE)**:
  - Studied undernutrition in Bangladesh using the 2017–18 BDHS dataset.
  - Used Stata/R/Python pipelines for preprocessing and evaluated algorithms (LR, RF, LightGBM, XGBoost, CatBoost) with SHAP explainability.
  - Research focused on BDHS-specific variables and sample sizes (~7,800 to ~8,500 children).
- **Our Project**:
  - We adapt and extend the methodology to the Indian context using the NFHS-5 (2019–21) Children's Recode (KR dataset), which contains over 200,000 child records and distinct regional, dietary, and socioeconomic features.
  - In Step 0, we established the foundational engineering structure to support this large-scale epidemiological dataset safely.

---

## Results
- Directory structure successfully generated:
  - `data/raw/`, `data/interim/`, `data/processed/`, `docs/`, `src/` (with 5 subpackages), `models/`, `reports/figures/`, `tests/`.
- Confidentiality rules verified in `.gitignore`.
- Core scientific imports verified:
  - Python: `3.13.5`
  - NumPy: `2.3.2`
  - Pandas: `2.2.3`
  - Scikit-Learn: `1.6.1`
  - SciPy: `1.15.2`
  - XGBoost: `3.3.0`
  - SHAP: `0.52.0`
- Execution of `python tests/test_setup.py`: **Passed (Exit code 0)**.

---

## Limitations
- Step 0 does not inspect or validate the contents, encoding, or validity of the NFHS-5 dataset files.
- It only confirms directory layout, Git protection rules, and package availability.

---

## Interview Questions

### Q1: Why is a strict `.gitignore` the first thing created in an epidemiological ML project rather than jumping straight into code?
**Sample Answer**:
In healthcare and public health data science, research ethics and data governance come first. Datasets like DHS/NFHS contain confidential microdata protected by user agreements and privacy laws. Staging raw survey data into a Git repository risks irreversible public exposure. Setting up strict ignore rules on day zero guarantees that microdata, intermediate transformations, and secret credentials can never be committed accidentally.

### Q2: Why did we choose tree-based ensembles (XGBoost, LightGBM) and linear baselines rather than deep neural networks for this tabular problem?
**Sample Answer**:
Empirical research across tabular benchmarks (e.g., Grinsztajn et al., 2022) demonstrates that tree-based ensembles consistently outperform deep neural networks on tabular datasets characterized by heterogeneous feature types (categorical survey codes, continuous anthropometrics), unnormalized scales, and extreme class imbalance. Furthermore, tree models provide exact, computationally feasible TreeSHAP computations for clinical explainability, whereas deep networks require costly approximations.

### Q3: Why do we separate `data/raw/`, `data/interim/`, and `data/processed/`?
**Sample Answer**:
This follows the principle of data immutability. Raw data must remain an untouched single source of truth. If a transformation error occurs in downstream cleaning, the raw data remains uncorrupted. `data/interim/` holds intermediate stages (e.g., filtered population cohorts), and `data/processed/` holds the exact, leakage-free feature matrices ready for model training.

---

## Commands Used
```bash
# Check Python environment
python --version

# Run Step 0 verification test
python tests/test_setup.py
```

---

## Next Step
**Step 1 — Dataset Inspection**: Safely inspect the extracted NFHS-5 KR dataset inside `IAKR7EDT/` to identify file formats, column dimensions, missing value representations, and data types without modifying the original raw files.
