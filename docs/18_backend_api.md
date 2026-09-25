# Step 18: NutriSense AI Backend API Documentation

## 1. Backend Purpose
The NutriSense AI Backend API exposes the approved Step-16 inference pipeline and Step-17 model registry for **Scenario A: Community Pre-Screening** as a high-performance, asynchronous RESTful web service. It enables frontends, health workers, and field triage applications to submit non-invasive child, maternal, household, and morbidity profiles and obtain immediate, unrounded screening risk probabilities alongside locked decision rules for childhood undernutrition (**Stunting**, **Underweight**, and **Wasting**).

> [!IMPORTANT]
> **Strict Scientific Charter**: The API provides non-invasive risk intelligence and pre-screening triage. Outputs represent statistical risk probabilities against validation-calibrated operating thresholds and **do not constitute clinical diagnoses**.

---

## 2. Architecture
The backend is structured as an decoupled application/API layer strictly separated from model training, validation, and serialization.

```
Frontend / Client
       │ (HTTP JSON Request)
       ▼
FastAPI Application (src/api/main.py)
       │
       ├─► Operational Logging & Privacy Middleware (UUID request_id, 0% PII)
       ├─► CORS Middleware (Environment controlled)
       │
Pydantic Request Validation (src/api/schemas.py)
       │ (Forbids extra/arbitrary fields & anthropometric leakage)
       ▼
Screening Service Layer (src/api/services/screening_service.py)
       │ (Singleton: verifies integrity & manages in-memory pipeline cache)
       ▼
Inference Pipeline Engine (src/models/inference_pipeline.py)
       │ (Step 16 approved inference logic)
       ▼
Model Registry & Cryptographic Verification (src/models/model_registry.py)
       │ (Step 17 model registry: SHA-256 audits, locked thresholds)
       ▼
Approved LightGBM Champion Artifacts (models/*.joblib)
       │
       ▼
Structured Screening Response (JSON)
```

Directory structure:
```
src/api/
├── __init__.py
├── config.py                     # Environment, CORS, and metadata configuration
├── main.py                       # FastAPI entrypoint, lifespan manager, middleware, error handlers
├── schemas.py                    # Pydantic v2 validation and serialization schemas
├── routes/
│   ├── __init__.py
│   ├── health.py                 # /health and /api/v1/health/model
│   ├── metadata.py               # /api/v1/metadata
│   └── screening.py              # /api/v1/screen
└── services/
    ├── __init__.py
    └── screening_service.py      # Thin service layer binding registry & inference
```

---

## 3. Technology Stack
- **Python**: 3.10+ (tested on Python 3.13)
- **FastAPI**: 0.110+ (asynchronous, OpenAPI-native, high performance)
- **Pydantic**: 2.6+ (strict request validation with `ConfigDict(extra="forbid")`)
- **Uvicorn**: 0.28+ (ASGI server)
- **Scikit-learn / LightGBM**: In-memory fitted inference pipelines
- **HTTPX & TestClient**: Quality assurance and automated testing

---

## 4. API Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/health` | Service liveness probe. Returns 200 without touching models or DHS data. | None |
| `GET` | `/api/v1/health/model` | Model readiness and cryptographic integrity verification. | None |
| `GET` | `/api/v1/metadata` | Public model provenance, feature schema, targets, and locked thresholds. | None |
| `POST` | `/api/v1/screen` | Primary child screening evaluation under Scenario A. | None |
| `GET` | `/docs` | Interactive Swagger UI API documentation. | None |
| `GET` | `/redoc` | ReDoc API documentation. | None |
| `GET` | `/openapi.json` | OpenAPI 3.1 machine-readable specification. | None |

---

## 5. Request Schema (`ChildScreeningRequest`)
The screening endpoint strictly enforces the approved 34 Scenario-A input features. Extra or arbitrary fields are forbidden.

```json
{
  "child_age_months": 24.0,
  "child_age_group": "12_23_mo",
  "child_sex_male": 1,
  "birth_order": 2.0,
  "is_multiple_birth": 0,
  "is_firstborn": 0,
  "preceding_birth_interval_months": 28.0,
  "birth_size_ordinal": 3.0,
  "birth_weight_kg": 2.8,
  "birth_weight_missing": 0,
  "delivery_place_type": "Public_Facility",
  "still_breastfeeding": 1.0,
  "diarrhea_recent": 0.0,
  "fever_recent": 0.0,
  "cough_recent": 0.0,
  "mother_age_years": 26.0,
  "mother_age_first_birth": 22.0,
  "mother_education_level": 2.0,
  "mother_bmi": 21.5,
  "mother_bmi_missing": 0,
  "total_children_born": 2.0,
  "anc_visits_count": 4.0,
  "anc_visits_missing": 0,
  "wealth_quintile": 2.0,
  "is_rural": 1,
  "caste_category": "OBC",
  "religion_category": "Hindu",
  "drinking_water_type": "Improved_Piped",
  "sanitation_facility_type": "Pit_Latrine",
  "has_electricity": 1,
  "clean_cooking_fuel": 1,
  "household_size": 5.0,
  "household_head_female": 0,
  "state_id": 10
}
```

---

## 6. Response Schema (`ScreeningResponse`)
All predictions report unrounded statistical probabilities, locked thresholds, deterministic screening positive classifications, and provenance:

```json
{
  "success": true,
  "model_version": "nutrisense-scenario-a-v1.0.0",
  "feature_schema_version": "scenario-a-34-v1",
  "predictions": {
    "stunting": {
      "probability": 0.4133094,
      "threshold": 0.35,
      "screen_positive": true,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    },
    "underweight": {
      "probability": 0.3751241,
      "threshold": 0.31,
      "screen_positive": true,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    },
    "wasting": {
      "probability": 0.2534787,
      "threshold": 0.17,
      "screen_positive": true,
      "decision_rule": "probability >= threshold",
      "model_family": "LightGBM",
      "condition": "unweighted"
    }
  }
}
```

---

## 7. Model Loading
- **Startup Loading**: Models are loaded once at startup via the FastAPI `lifespan` context manager.
- **Fail-Fast Initialization**: If any model file is missing or its SHA-256 hash does not match `models/model_registry.json`, the startup process raises an exception and halts immediately.
- **Zero Disk I/O per Request**: Pre-verified pipelines reside in memory and are injected directly into `inference_pipeline._CACHED_MODELS`.

---

## 8. Registry Integration
The API integrates directly with Step 17's `models/model_registry.json`:
- `registry_version`: `1.0.0`
- `model_version`: `nutrisense-scenario-a-v1.0.0`
- `feature_schema_version`: `scenario-a-34-v1`
- `scenario`: `A` (Community Pre-Screening)
- `model_family`: `LightGBM` (Unweighted)
- `thresholds`: `stunting=0.35`, `underweight=0.31`, `wasting=0.17`

---

## 9. Validation
The backend enforces multi-tiered validation:
1. **Schema Validation**: Types, ranges, nullability, and required fields.
2. **Extra Field Rejection**: Configured with `extra = "forbid"`. Any extraneous field returns HTTP 422.
3. **Anthropometric Leakage Rejection**: Variables `hw70`, `hw71`, `hw72`, `hw73`, `hw2`, `hw3`, `hw4-hw12`, `hw13`, `hw57`, `stunting`, `underweight`, `wasting`, `sample_weight`, etc. are forbidden and trigger HTTP 422.
4. **Biological Consistency**:
   - `mother_age_first_birth <= mother_age_years`
   - `birth_order <= total_children_born`
   - `child_age_months` within `[0, 59]`
   - `state_id` within `[1, 36]`

---

## 10. Error Handling
All error responses adhere to a unified JSON structure without stack traces:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Validation failed for screening request.",
    "details": [
      "body -> child_age_months: Field required"
    ]
  }
}
```

HTTP status codes:
- **400 Bad Request**: Biological inconsistency or logical conflict.
- **422 Unprocessable Content**: Schema validation error, missing field, or prohibited feature.
- **500 Internal Server Error**: Unexpected exception (stack traces concealed from clients).
- **503 Service Unavailable**: Model service uninitialized or cryptographic integrity degraded.

---

## 11. Privacy
- **Zero Microdata**: The API never touches raw DHS survey files (`IAKR7EFL.DTA`) at runtime.
- **Zero Personally Identifiable Information (PII)**: The schema does not accept child names, parent names, addresses, phone numbers, GPS coordinates, or household IDs.
- **Synthetic Random Request IDs**: Every HTTP request is assigned a unique UUID4 (`X-Request-ID`) containing zero client-derived entropy.

---

## 12. Logging Policy
Application logging is strictly limited to non-sensitive operational telemetry:
- **Logged**: HTTP Method, URL path, HTTP status code, request duration in ms, and random `req_id`.
- **FORBIDDEN and NEVER LOGGED**:
  - Request body payloads
  - Child age, sex, birth order, or birth weight
  - Maternal characteristics or BMI
  - Household wealth, caste, religion, or location
  - Predicted probabilities or screening results

---

## 13. CORS (Cross-Origin Resource Sharing)
Configured dynamically via `src/api/config.py`:
- **Development (`NUTRISENSE_ENV=development`)**:
  - `http://localhost:3000`
  - `http://localhost:5173`
  - `http://127.0.0.1:3000`
  - `http://127.0.0.1:5173`
- **Production (`NUTRISENSE_ENV=production`)**:
  - Explicit origins supplied via `NUTRISENSE_ALLOWED_ORIGINS` environment variable (comma-delimited).
  - Wildcard `*` is strictly forbidden in production.

---

## 14. Configuration
Configured through environment variables:
- `NUTRISENSE_ENV`: `development` (default) or `production`.
- `NUTRISENSE_ALLOWED_ORIGINS`: Comma-separated list of allowed origins.
- `MODEL_REGISTRY_PATH`: Path to model registry JSON (default: `models/model_registry.json`).

---

## 15. OpenAPI Documentation
FastAPI automatically serves interactive, schema-verified OpenAPI 3.1 documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

All endpoints and Pydantic fields include descriptive documentation and clinical disclaimers.

---

## 16. Security
- **Immutability**: Models and registries are read-only. The API provides no mutation or upload endpoints.
- **Model Tampering Prevention**: Startup audits SHA-256 hashes against cryptographic signatures.
- **No Path Traversal**: Clients cannot pass file paths, model paths, or threshold overrides.
- **No Data Leakage**: Internal exception stack traces and server filesystem paths are never exposed.

---

## 17. Testing
The test suite in `tests/test_api.py` includes 20 comprehensive unit, security, and privacy tests:
- Liveness, readiness, and public metadata verification
- Complete valid screening payload evaluation
- Missing, invalid-type, and out-of-range field rejections
- Strict rejection of prohibited anthropometric leakage
- Probability bounds `[0.0, 1.0]` and locked threshold verification
- Model integrity degradation simulation (503 response)
- Zero DHS raw microdata dependency verification
- Logger privacy audit (confirms payload fields do not appear in logs)
- Idempotence / deterministic output verification
- Rejection of client-supplied model paths and arbitrary fields
- Concealment of stack traces and internal filesystem paths

Run commands:
```bash
# Run API test suite
python -m unittest tests/test_api.py -v

# Run full project regression suite (Steps 0–18)
python -m unittest discover tests -p "test_*.py" -v
```

---

## 18. Running Locally

### Starting the Server
Run Uvicorn with auto-reload:
```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Accessing Documentation
Open your browser at:
- **Interactive Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## 19. Example cURL Requests

### 1. Health Check
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response:**
```json
{"status": "healthy"}
```

### 2. Model Health & Readiness Check
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/health/model"
```
**Response:**
```json
{
  "status": "healthy",
  "model_version": "nutrisense-scenario-a-v1.0.0",
  "registry_version": "1.0.0",
  "feature_schema_version": "scenario-a-34-v1",
  "model_family": "LightGBM",
  "targets": ["stunting", "underweight", "wasting"],
  "model_integrity_verified": true
}
```

### 3. Application Metadata
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/metadata"
```

### 4. Child Undernutrition Screening
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/screen" \
  -H "Content-Type: application/json" \
  -d '{
    "child_age_months": 24.0,
    "child_sex_male": 1,
    "birth_order": 2.0,
    "is_multiple_birth": 0,
    "is_firstborn": 0,
    "preceding_birth_interval_months": 28.0,
    "birth_size_ordinal": 3.0,
    "birth_weight_kg": 2.8,
    "birth_weight_missing": 0,
    "delivery_place_type": "Public_Facility",
    "still_breastfeeding": 1.0,
    "diarrhea_recent": 0.0,
    "fever_recent": 0.0,
    "cough_recent": 0.0,
    "mother_age_years": 26.0,
    "mother_age_first_birth": 22.0,
    "mother_education_level": 2.0,
    "mother_bmi": 21.5,
    "mother_bmi_missing": 0,
    "total_children_born": 2.0,
    "anc_visits_count": 4.0,
    "anc_visits_missing": 0,
    "wealth_quintile": 2.0,
    "is_rural": 1,
    "caste_category": "OBC",
    "religion_category": "Hindu",
    "drinking_water_type": "Improved_Piped",
    "sanitation_facility_type": "Pit_Latrine",
    "has_electricity": 1,
    "clean_cooking_fuel": 1,
    "household_size": 5.0,
    "household_head_female": 0,
    "state_id": 10
  }'
```

---

## 20. Limitations
1. **Screening Tool Only**: This API is designed strictly for community pre-screening and triage to identify children at elevated risk of undernutrition who lack access to calibrated stadiometers or scales. It does not provide clinical diagnoses.
2. **Secondary Triage Required**: Any child flagged with `screen_positive: true` must be referred for direct clinical anthropometry (height, weight, MUAC, and bilateral pitting edema assessment).
3. **Scenario A Scope**: Models are calibrated specifically for non-invasive community screening. Invasive biomedical or laboratory biomarkers are not incorporated.
4. **Zero Intervention Logic**: In accordance with system specifications, treatment, supplementation, and prescription protocols are not generated by this API layer.
