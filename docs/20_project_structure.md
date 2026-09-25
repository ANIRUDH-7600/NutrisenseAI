# NutriSense AI: Complete Project Structure & Repository Inventory

An exhaustive architectural map and directory inventory of the NutriSense AI repository.

---

## 1. Top-Level Repository Directory Map

```
d:/finalyearproj/Nutrisense-Ai/
├── data/                           # Data storage (raw metadata, interim, and processed splits)
├── models/                         # Champion LightGBM pipelines & cryptographic model registry
├── src/                            # Core research, feature engineering, and backend API source code
├── frontend/                       # React 18 / Vite 6 Single-Page Application
├── docs/                           # Exhaustive academic reports and step-by-step documentation (Steps 00–20)
├── tests/                          # 115 automated Python unit and integration regression tests
├── reports/                        # 39 generated high-resolution research plots and evaluation figures
├── IAKR7EDT/                       # Quarantined raw DHS NFHS-5 survey extract (IAKR7EFL.DTA)
├── requirements.txt                # Complete Python dependencies (scientific + API)
├── data_dictionary.csv             # Full feature schema & survey variable mapping
├── README.md                       # Main project landing page and quickstart guide
└── .gitignore                      # Git exclusion rules (quarantining raw DTA and build artifacts)
```

---

## 2. Directory Purpose & Sub-Tree Breakdown

### 2.1 `models/` — Machine Learning Artifacts & Provenance
Stores the locked, champion model pipelines and cryptographic registration contracts.
```
models/
├── model_registry.json                                         # Machine-readable cryptographic registry
├── model_comparison_lightgbm_stunting_unweighted.joblib       # Champion LightGBM pipeline for Stunting (tau = 0.35)
├── model_comparison_lightgbm_underweight_unweighted.joblib    # Champion LightGBM pipeline for Underweight (tau = 0.31)
└── model_comparison_lightgbm_wasting_unweighted.joblib        # Champion LightGBM pipeline for Wasting (tau = 0.17)
```
- **Purpose**: Fully self-contained scikit-learn/LightGBM pipelines comprising `ColumnTransformer` (median imputers, scalers, one-hot encoders) and `LGBMClassifier`.
- **Integrity**: Audited by SHA-256 digests. No model retraining or parameter modification occurs at runtime.

---

### 2.2 `src/` — Scientific Code & FastAPI Backend
Separated into research/feature modules and the web API application layer.
```
src/
├── data/                           # Data inspection, cleaning, and household-grouped splitting
│   ├── clean_data.py               # Filtering living children and biological range validation
│   ├── split_data.py               # Household-grouped train/val/test splitting (seed = 42)
│   └── eda.py                      # Exploratory data analysis routines
├── features/                       # Feature construction and leakage prevention
│   └── feature_engineering.py      # Construction of the 34 approved Scenario-A features
├── models/                         # Model training, threshold calibration, and inference logic
│   ├── baseline_model.py           # Regularized Logistic Regression baseline
│   ├── model_comparison.py        # 5-algorithm benchmarking (LR, RF, XGB, CatBoost, LightGBM)
│   ├── threshold_analysis.py       # Validation decision threshold selection
│   ├── final_test_evaluation.py    # Out-of-sample locked test evaluation
│   ├── inference_pipeline.py       # Step-16 production inference engine
│   └── model_registry.py           # Step-17 registry management & hash audit utilities
├── explainability/                 # Explainable AI and interpretability
│   └── shap_analysis.py            # TreeSHAP computation, beeswarm, and dependence plots
├── utils/                          # Cross-module helper functions and logging
└── api/                            # Step-18 FastAPI Application Layer
    ├── config.py                   # Environment-controlled settings, CORS, and metadata
    ├── main.py                     # FastAPI application, lifespan context, privacy logging, error handlers
    ├── schemas.py                  # Pydantic v2 schemas (ConfigDict(extra="forbid"), biological consistency)
    ├── routes/
    │   ├── health.py               # GET /health (liveness) & GET /api/v1/health/model (readiness)
    │   ├── metadata.py             # GET /api/v1/metadata (public model provenance)
    │   └── screening.py            # POST /api/v1/screen (primary triage inference)
    └── services/
        └── screening_service.py    # Singleton service managing in-memory pipeline cache
```

---

### 2.3 `frontend/` — React 18 / Vite 6 Web Application
Modern single-page application providing an intuitive interface for frontline health workers.
```
frontend/
├── index.html                      # HTML5 entrypoint with Google Fonts
├── package.json                    # Node dependencies and build scripts
├── vite.config.js                  # Vite dev server with reverse-proxying to FastAPI
├── .env.development                # API base URL configuration
├── tests/
│   └── frontend.test.js            # Automated frontend test suite (9 passing tests)
├── src/
│   ├── main.jsx                    # Root ReactDOM mount
│   ├── App.jsx                     # Route definitions & shared state container
│   ├── index.css                   # Custom Vanilla CSS design system (healthcare purple/slate)
│   ├── components/
│   │   ├── Navbar.jsx              # Responsive navigation with mobile drawer
│   │   ├── Footer.jsx              # Provenance & academic attribution
│   │   └── DisclaimerBanner.jsx    # Persistent epidemiological disclaimer banner
│   ├── pages/
│   │   ├── HomePage.jsx            # Value proposition, tri-target cards, triage workflow
│   │   ├── ScreeningPage.jsx       # 7-section form (A–G) for 34 Scenario-A features
│   │   ├── ResultsPage.jsx         # Tri-target risk dashboard & triage guidance
│   │   ├── AboutPage.jsx           # Methodology, dataset, and SHAP explainability
│   │   └── SystemPage.jsx          # Live health telemetry & model specifications
│   ├── services/
│   │   └── api.js                  # Centralized HTTP client for Step-18 endpoints
│   └── utils/
│       └── constants.js            # Dropdown options, 36 Indian states, sample data
└── dist/                           # Production static bundle (247 kB JS)
```

---

### 2.4 `docs/` — Comprehensive Research & Step Documentation
Exhaustive technical documentation covering every step of the research and development lifecycle.
```
docs/
├── 00_project_setup.md             # Environment setup and directory structure
├── 01_dataset_inspection.md        # Raw DHS KR variable inventory
├── 02_data_documentation.md        # Comprehensive data dictionary
├── 03_research_population.md       # Target population and exclusion rules
├── 04_data_cleaning.md             # Cleaning protocols and missingness handling
├── 05_target_creation.md           # WHO 2006 Z-score target definitions
├── 06_eda.md                       # Exploratory data analysis and epidemiological distributions
├── 07_feature_engineering.md       # Approved 34 Scenario-A candidate features
├── 08_train_validation_test_split.md # Household-grouped 70/15/15 partition
├── 09_baseline_model.md            # Regularized Logistic Regression baseline
├── 10_model_comparison.md          # Multi-algorithm comparison (LR, RF, XGB, CatBoost, LightGBM)
├── 11_threshold_calibration.md     # Decision boundary optimization
├── 12_final_model_selection.md     # Selection of LightGBM Unweighted champion models
├── 13_final_test_evaluation.md     # Locked out-of-sample test results and CIs
├── 14_shap_explainability.md       # TreeSHAP feature attributions and dependence plots
├── 15_error_analysis.md            # Subgroup performance disparities and error analysis
├── 16_inference_pipeline.md        # Production end-to-end inference specification
├── 17_model_registry.md            # Cryptographic packaging and integrity auditing
├── 18_backend_api.md               # FastAPI application, OpenAPI documentation, and security
├── 19_frontend.md                  # React architecture, components, and viva Q&A
├── 20_final_research_report.md     # Complete 34-section academic research paper with 10 tables
├── 20_system_architecture.md       # Global system architecture specification
├── 20_viva_questions.md            # 52-question oral defense examination guide (A–Z)
├── 20_project_explanation.md       # 30-second, 2-minute, and 5-minute spoken scripts
├── 20_demo_script.md               # Interactive step-by-step evaluation script
├── 20_reproducibility.md           # Exact environment replication protocol
└── 20_project_structure.md         # Repository tree and directory inventory (this file)
```

---

### 2.5 `tests/` — Automated Regression Test Suite
Comprehensive automated test suite guaranteeing zero regressions across all steps.
```
tests/
├── test_setup.py                   # Environment and path verification
├── test_inspection.py              # Raw data accessibility
├── test_data_dictionary.py         # Variable mapping verification
├── test_research_population.py     # Cohort exclusion criteria
├── test_data_cleaning.py           # Missing data and outlier filtering
├── test_target_creation.py         # WHO Z-score cutoff boundaries
├── test_eda.py                     # Correlation and prevalence consistency
├── test_feature_engineering.py     # 34-feature extraction and leakage audit
├── test_split_data.py              # Zero household overlap verification
├── test_baseline_model.py          # Logistic regression baseline sanity
├── test_model_comparison.py        # Multi-model evaluation contracts
├── test_threshold_analysis.py      # Threshold boundary assertions
├── test_final_model_selection.py   # Champion model loading
├── test_final_test_evaluation.py   # Test partition metric assertions
├── test_shap_analysis.py           # TreeSHAP output validity
├── test_error_analysis.py          # Subgroup sample size and rate calculations
├── test_inference_pipeline.py      # Step-16 end-to-end inference assertions
├── test_model_registry.py          # Step-17 SHA-256 hash audits
└── test_api.py                     # Step-18 FastAPI endpoints, security, and privacy
```

---

### 2.6 `reports/figures/` — High-Resolution Visual Artifacts
39 publication-grade figures documenting every phase of the research:
- `fig1_target_prevalence_and_imbalance.png`
- `fig2_missingness_profile.png`
- `fig3_age_dynamics_by_outcome.png`
- `fig4_socioeconomic_maternal_gradients.png`
- `fig5_morbidity_associations_acute_vs_chronic.png`
- `fig6_statistical_association_cramers_v.png`
- `fig7_state_level_stunting_wasting.png`
- `fig8_model_comparison_roc_auc.png`
- `fig9_model_comparison_pr_auc.png`
- `fig10_class_weighting_effect.png`
- `fig11_model_comparison_f1_tradeoff.png`
- `fig12_threshold_curves_stunting.png` to `fig14_threshold_curves_wasting.png`
- `fig15_precision_recall_curves.png`
- `fig16_roc_curves.png`
- `fig17_probability_calibration_curves.png`
- `fig18_decision_curve_analysis.png`
- `fig19_test_confusion_matrices.png`
- `fig20_validation_vs_test_metrics.png`
- `fig21_test_roc_curves.png`
- `fig22_test_precision_recall_curves.png`
- `fig23_shap_summary_stunting.png` to `fig25_shap_summary_wasting.png`
- `fig26_shap_bar_stunting.png` to `fig28_shap_bar_wasting.png`
- `fig29_shap_cross_target_comparison.png`
- `fig30_shap_dependence_stunting.png` to `fig32_shap_dependence_wasting.png`
- `33_error_sensitivity_by_age.png` to `38_error_distribution_by_age.png`.
