# NutriSense AI: Complete Reproducibility Protocol

A comprehensive technical guide for cloning, configuring, verifying, and executing the NutriSense AI project from source code across research, backend, and frontend environments.

---

## 1. System Requirements

| Environment Component | Minimum Requirement | Verified Test Baseline |
|---|---|---|
| **Operating System** | Windows 10/11, macOS 12+, or Ubuntu 20.04+ | Windows 11 Enterprise (64-bit) |
| **Python** | Python 3.10 to 3.13 | Python 3.13.0 |
| **Node.js** | Node v18+ | Node v22.15.0 |
| **npm** | npm v9+ | npm v10.9.2 |
| **RAM** | 8 GB minimum (16 GB recommended for batch training) | 16 GB RAM |
| **Disk Space** | 2 GB for software + models (500 MB without raw DTA) | 2.5 GB available |

---

## 2. Python Environment Setup

### Step 2.1: Clone and Create Virtual Environment
```bash
# Navigate to workspace root
cd d:/finalyearproj/Nutrisense-Ai

# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on Linux/macOS
source venv/bin/activate
```

### Step 2.2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Key Python Dependencies (`requirements.txt`)
- `numpy>=1.26.0`
- `pandas>=2.2.0`
- `scipy>=1.12.0`
- `pyreadstat>=1.2.0` (for reading Stata DTA survey files)
- `scikit-learn==1.6.1`
- `xgboost>=2.0.0`
- `lightgbm>=4.3.0`
- `catboost>=1.2.0`
- `imbalanced-learn>=0.12.0`
- `shap>=0.45.0`
- `joblib>=1.3.0`
- `fastapi>=0.110.0`
- `uvicorn>=0.28.0`
- `pydantic>=2.6.0`
- `httpx>=0.27.0`
- `pytest>=8.0.0`

---

## 3. Node.js Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install exact npm packages
npm install
```

### Key Frontend Dependencies (`frontend/package.json`)
- `react: ^18.3.1`
- `react-dom: ^18.3.1`
- `react-router-dom: ^6.28.0`
- `lucide-react: ^1.16.0`
- `vite: ^6.0.1`
- `@vitejs/plugin-react: ^4.3.4`

---

## 4. Model Registry & Checksum Verification

Before running inference, verify the cryptographic integrity of the approved model artifacts:

```bash
python -c "
import json, hashlib

with open('models/model_registry.json') as f:
    reg = json.load(f)

for target, info in reg['models'].items():
    with open(info['artifact'], 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert sha == info['sha256'], f'Hash mismatch on {target}'
    print(f'[OK] {target}: SHA-256 matches registered signature ({sha[:12]}...)')
print('All model artifacts verified successfully.')
"
```

### Locked Model Checksums (`models/model_registry.json`)
- **Stunting** (`models/model_comparison_lightgbm_stunting_unweighted.joblib`):
  `3a1a9c0be5d8eb35483857812da2d77441d85ec3d233952c8a650b3e03f5b736`
- **Underweight** (`models/model_comparison_lightgbm_underweight_unweighted.joblib`):
  `2fc16e1a56d0b036398dd6d428ee7231cc09cfdec34c3dcb3cfd104caf2a3eef`
- **Wasting** (`models/model_comparison_lightgbm_wasting_unweighted.joblib`):
  `7900278ce83da2aafacccd8d77a924a82cf0d7b2fdfab20aa37a1ac6bdd7e919`

---

## 5. Execution Commands

### 5.1 Starting the FastAPI Backend
From the workspace root:
```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Interactive ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- OpenAPI Specification: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### 5.2 Starting the React Frontend
In a separate terminal:
```bash
cd frontend
npm run dev
```
- Frontend Dashboard: [http://127.0.0.1:5173](http://127.0.0.1:5173)

### 5.3 Building the Frontend for Production
```bash
cd frontend
npm run build
```
Generates optimized static assets under `frontend/dist/`.

---

## 6. Automated Testing Suite

### 6.1 Backend API Unit Tests (Step 18)
```bash
python -m unittest tests/test_api.py -v
```
*(Runs 20 automated tests: liveness, readiness, schema validation, extra-field rejection, prohibited variable isolation, privacy logging, deterministic evaluation, and 500 error sanitization).*

### 6.2 Model Registry & Inference Pipeline Tests (Steps 16 & 17)
```bash
python -m unittest tests/test_model_registry.py tests/test_inference_pipeline.py -v
```
*(Runs 38 automated tests: schema integrity, threshold checks, fail-fast integrity audits, and inference output contracts).*

### 6.3 Full Project Regression Test Suite (Steps 0 through 18)
```bash
python -m unittest discover tests -p "test_*.py" -v
```
*(Runs all 115 unit and integration tests across data cleaning, feature engineering, modeling, threshold analysis, error analysis, registry, and API).*

### 6.4 Frontend Automated Test Suite (Step 19)
```bash
npm test --prefix frontend
```
*(Runs 9 automated Node tests validating 34-feature schema conformance, biological constraints, non-clinical terminology rules, and sensitive data logging exclusions).*

---

## 7. Data Access & Governance Requirements

### DHS Microdata Policy
1. The raw survey data (`IAKR7EFL.DTA`) is **not included in public git repositories** in strict adherence to ICF International data governance agreements.
2. Researchers seeking to reproduce the offline training pipeline from scratch must register at [https://dhsprogram.com](https://dhsprogram.com) and request formal access to the **India NFHS-5 (2019–21) Children's Recode (KR)** dataset.
3. Upon receiving approved access credentials, download `IAKR7EFL.DTA` and place it in the `IAKR7EDT/` directory.
4. Verify file authenticity using file size:
   ```bash
   # Exact expected byte size:
   441,380,745 bytes
   ```

### Runtime Independence
For executing the FastAPI backend and React frontend, the raw DHS file is **not required**. The backend runs entirely on the pre-trained champion artifacts in `models/`.
