# NutriSense AI: System Architecture Specification

## 1. Architectural Philosophy
NutriSense AI enforces a strict architectural boundary between two operational domains:
1. **The Research & Training Subsystem**: An offline, batch scientific computing environment where raw DHS microdata is ingested, cleaned, engineered, split, and used to train and calibrate LightGBM models.
2. **The Community Screening Deployment Subsystem**: A lightweight, real-time, privacy-preserving web application stack (FastAPI + React) that executes inference using in-memory model artifacts without accessing raw survey data.

---

## 2. Global Architecture Diagram

```
========================================================================================================
                                 RESEARCH & TRAINING PIPELINE (OFFLINE)
========================================================================================================

India NFHS-5 Microdata (IAKR7EFL.DTA, 441 MB)
               │
               ▼
      src/data/cleaning.py
               │   • Restricted to living under-five children (b5 == 1)
               │   • Cleaned plausible ranges, handled structural missingness
               ▼
    src/features/engineering.py
               │   • 34 approved non-invasive candidate features (Scenario A)
               │   • Strict isolation: ZERO physical measurements (hw2-hw12, hw70-hw73, hw57)
               ▼
       src/data/split.py
               │   • Household-grouped clustered split: 70% Train, 15% Val, 15% Test
               │   • Zero shared households (seed = 42)
               ▼
      src/models/training.py
               │   • Multi-algorithm benchmarking (LR, RF, XGB, CatBoost, LightGBM)
               │   • LightGBM Unweighted selected as champion
               ▼
     src/models/thresholds.py
               │   • Validation-derived operating thresholds (selected Step 11, locked Step 12):
               │     Stunting: tau = 0.35 | Underweight: tau = 0.31 | Wasting: tau = 0.17
               │   • (Evaluated on held-out test cohort in Step 13; no probability calibration claimed)
               ▼
      src/models/registry.py
               │   • Hashes computed (SHA-256), metadata frozen
               ▼
  models/model_registry.json + models/*.joblib (Champion Artifacts)
========================================================================================================
                          FROZEN ARTIFACT BOUNDARY (NO MICRODATA TRANSFERRED)
========================================================================================================
                                 DEPLOYMENT & TRIAGE RUNTIME (REAL-TIME)
========================================================================================================

Frontline Community Health Worker / ASHA / Anganwadi Kiosk
               │
               ▼  (HTTP HTTPS / Browser Interface)
React 18 / Vite 6 Single-Page Application (frontend/)
  ├── src/pages/ScreeningPage.jsx (7-section structured form: 34 features)
  ├── src/pages/ResultsPage.jsx (Tri-target risk dashboard + triage advice)
  ├── src/pages/SystemPage.jsx (Live health telemetry & registry audit)
  └── src/services/api.js (Centralized HTTP client)
               │
               ▼  REST / JSON (POST /api/v1/screen)
FastAPI Backend Application (src/api/)
  ├── main.py (Lifespan manager, privacy middleware, exception handlers)
  ├── schemas.py (Pydantic v2: ConfigDict(extra="forbid"), biological validation)
  ├── routes/ (health.py, metadata.py, screening.py)
  └── services/screening_service.py (Singleton in-memory pipeline manager)
               │
               ▼  In-Memory Execution (0 ms disk latency)
Step-16 Inference Pipeline (src/models/inference_pipeline.py)
               │
               ▼  Audited against SHA-256 Checksums
Step-17 Model Registry & Champion Pipelines (models/*.joblib)
               │
               ▼
Structured Screening Predictions JSON:
  {
    "success": true,
    "model_version": "nutrisense-scenario-a-v1.0.0",
    "predictions": {
      "stunting": { "probability": 0.4133, "threshold": 0.35, "screen_positive": true },
      "underweight": { "probability": 0.3751, "threshold": 0.31, "screen_positive": true },
      "wasting": { "probability": 0.2535, "threshold": 0.17, "screen_positive": true }
    }
  }
```

---

## 3. Component Breakdown

### 3.1 Research & Training Subsystem
- **Quarantined Microdata**: The Standard DHS Phase 7 Children's Recode file (`IAKR7EFL.DTA`) resides strictly in `IAKR7EDT/`. It is never deployed or read by the web service.
- **Data Engineering**: Data cleaning scripts filter non-viable records, normalize categories (JMP water/sanitation standards), and build the 34-feature matrix.
- **Model Checkpointing**: The approved estimators (`LGBMClassifier`) and `ColumnTransformer` preprocessors are saved as self-contained `.joblib` files under `models/`.
- **Model Registry (`models/model_registry.json`)**: Acts as a cryptographic contract defining model versions, feature names, decision thresholds, and SHA-256 file hashes.

### 3.2 Backend Service Subsystem (FastAPI)
- **Zero Raw Data Requirement**: The backend requires exclusively `models/*.joblib` and `models/model_registry.json`.
- **Fail-Fast Lifespan Startup**: Upon server boot, `ScreeningService.initialize()` computes SHA-256 hashes of the `.joblib` files and compares them with the registry. If any file is missing, altered, or corrupt, startup halts with a critical exception.
- **Zero Repeated Disk I/O**: Validated pipelines are retained in memory (`inference_pipeline._CACHED_MODELS`), delivering sub-10 ms inference per child.
- **Operational Privacy Middleware**: Ingests requests, assigns random UUID4 request IDs, and logs only operational telemetry (`METHOD`, `path`, `status`, `duration_ms`). Request payloads containing child/maternal features are strictly excluded from logs.
- **Strict Pydantic Validation**: `ChildScreeningRequest` enforces `extra = "forbid"`, rejecting undeclared or prohibited variables (`hw70`, `hw2`, `hw3`, `hw57`, `stunting`) with HTTP 422.

### 3.3 Frontend Subsystem (React + Vite)
- **Decoupled Client**: Single-Page Application (SPA) compiled to static HTML, CSS, and JavaScript.
- **Accessible Form Design**: 7 logical sections (Child Demographics, Recent Morbidities, Feeding/Delivery, Maternal History, Household Socioeconomics, WASH Environment, State Location) translating 34 features into human-friendly inputs.
- **Results Dashboard**: Displays unrounded probabilities, percentage metrics, decision thresholds, visual comparison bars, and secondary triage guidance.
- **Zero Persistent Storage**: Screening data is retained strictly in in-memory React state, never stored in `localStorage`, cookies, or external databases.

---

## 4. Key Architectural Guarantees
1. **Decoupled Authority**: The backend is the sole authority for predictions. The frontend never computes probabilities, fits thresholds, or loads models.
2. **Deterministic Triage**: All predictions are deterministic functions of the 34 input features and locked thresholds.
3. **Immutability**: Neither API requests nor frontend actions can alter model artifacts, feature schemas, or decision rules.
4. **Data Isolation**: Raw survey microdata is quarantined and inaccessible to client applications.
