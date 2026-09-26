# NutriSense AI: Childhood Malnutrition Risk Intelligence System

NutriSense AI is an epidemiological and clinical machine learning risk intelligence system designed for community pre-screening of early childhood undernutrition (Stunting, Underweight, and Wasting) under **Scenario A: Scale-Free Community Triage**.

> [!WARNING]
> **RESEARCH & EPIDEMIOLOGICAL SCREENING SYSTEM ONLY**:
> All outputs provided by NutriSense AI are statistical screening risk predictions based on non-invasive community indicators. **They do not constitute medical or clinical diagnoses.** Any child flagged as `screen_positive` requires secondary triage and direct clinical anthropometric measurement.

> [!IMPORTANT]
> **NO DHS MICRODATA IN DEPLOYMENT**:
> Raw DHS microdata (`IAKR7EFL.DTA`) is strictly for offline research, model training, and validation. It must **never** be packaged, exposed, or included in backend API production deployments.

---

## React Frontend (Step 19)

A modern, responsive React Single-Page Application (SPA) built with Vite, React Router, and a vanilla CSS design system. It allows community health workers to input non-invasive community indicators and view risk predictions and secondary triage guidance.

### Starting the Frontend
```bash
# Navigate to frontend directory and start Vite development server
cd frontend
npm run dev
```
The dashboard will be available at: [http://127.0.0.1:5173](http://127.0.0.1:5173)

### Building for Production
```bash
cd frontend
npm run build
```

### Running Frontend Tests
```bash
npm test --prefix frontend
```

---

## Backend API (Step 18)

The backend provides a FastAPI application exposing the approved Step-16 inference pipeline and Step-17 model registry.

### Starting the Backend
Run the following command from the workspace root:
```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### API Documentation
Interactive OpenAPI documentation is automatically available when the backend is running:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### Core Endpoints
- `GET /health`: Service liveness probe.
- `GET /api/v1/health/model`: Model integrity check and readiness probe.
- `GET /api/v1/metadata`: Public model provenance and locked decision thresholds.
- `POST /api/v1/screen`: Child community pre-screening risk evaluation.

### Example Screening Request (30-Feature Canonical v1.0.0 Schema)
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
    "mother_bmi": 21.5,
    "mother_bmi_missing": 0,
    "total_children_born": 2.0,
    "anc_visits_count": 4.0,
    "anc_visits_missing": 0,
    "wealth_quintile": 2.0,
    "is_rural": 1,
    "drinking_water_type": "Improved_Piped",
    "sanitation_facility_type": "Pit_Latrine",
    "has_electricity": 1,
    "clean_cooking_fuel": 1,
    "household_size": 5.0,
    "state_id": 10
  }'
```

---

## Research Documentation & Transition Guide

NutriSense AI is fully documented with a comprehensive academic paper, system architecture, viva defense preparation, demo scripts, reproducibility guides, and the 30-feature feature transition report:

- **[30-Feature Model Transition Guide](file:///d:/finalyearproj/Nutrisense-Ai/docs/22_30_feature_model_transition.md)**: Comprehensive transition report documenting the 30-feature v1.0.0 release, validation operating thresholds, metric comparisons, SHAP updates, and model registry hashes.
- **[Final Research Report](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_final_research_report.md)**: Publication-ready manuscript with tables, Islam et al. (2024) base paper benchmarks, and figure index.
- **[System Architecture](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_system_architecture.md)**: Complete structural and runtime specification separating offline DHS research pipelines from the FastAPI/React production runtime.
- **[Oral Viva & Defense Guide](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_viva_questions.md)**: 52 categorized viva questions and model answers spanning ML theory, ethics, epidemiology, software architecture, and system limitations.
- **[Project Explanation Narratives](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_project_explanation.md)**: Structured oral pitches (30-second elevator, 2-minute executive, 5-minute technical deep-dive) tailored for evaluators.
- **[Demonstration Script](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_demo_script.md)**: Step-by-step evaluation guide featuring synthetic community case studies.
- **[Reproducibility Guide](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_reproducibility.md)**: Complete environment configuration, dependency matrix, and verification commands.
- **[Project Structure Directory](file:///d:/finalyearproj/Nutrisense-Ai/docs/20_project_structure.md)**: Comprehensive repository audit and file inventory.

---

## Running the Application Locally

### 1. Start the FastAPI Backend Server
```bash
cd backend
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **API URL**: `http://127.0.0.1:8000`
* **Swagger Documentation**: `http://127.0.0.1:8000/docs`
* **Health Check**: `http://127.0.0.1:8000/health`

### 2. Start the React Frontend Web Application
```bash
cd frontend
npm run dev
```
* **Web UI**: `http://localhost:5173`

---

## Running Automated Tests
```bash
# Backend pytest regression test suite (280 unit & integration tests)
cd backend
python -m pytest

# Frontend automated test suite (9 Node specification tests)
npm test --prefix frontend
```
