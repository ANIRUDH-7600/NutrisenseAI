# Step 19: NutriSense AI React Frontend Documentation

## 1. Objective
Step 19 delivers a modern, responsive, accessible React frontend for NutriSense AI. The application provides community health workers, epidemiological researchers, and field practitioners with a clean interface to input non-invasive community attributes under **Scenario A (Community Pre-Screening)** and inspect risk probabilities and calibrated decision rules for early childhood undernutrition (**Stunting**, **Underweight**, and **Wasting**).

> [!IMPORTANT]
> **Strict Scientific Charter**: The frontend exclusively uses non-diagnostic screening language (`Screen Positive`, `Screen Negative`, `Predicted Probability`, `Community Screening Threshold`). It explicitly disclaims any clinical diagnosis or medical treatment claims.

---

## 2. Architecture
The frontend is decoupled from the ML pipeline and interacts solely with the Step-18 FastAPI backend via REST/JSON.

```
Frontline User / Health Worker
       │
       ▼
React Frontend (Vite Single-Page Application)
       │
       ├─► React Router (Client-side routing: /, /screen, /results, /about, /system)
       ├─► Local UI State (In-memory, zero persistent storage of sensitive data)
       ├─► Client Validation (Pre-submission range & biological plausibility checks)
       │
       ▼ REST / JSON (HTTP POST /api/v1/screen)
FastAPI Backend (Step 18)
       │
       ▼
Inference Pipeline Engine (Step 16) & Model Registry (Step 17)
       │
       ▼
Approved LightGBM Champion Models (models/*.joblib)
       │
       ▼
Structured Screening Predictions (JSON)
       │
       ▼
Results Dashboard (Probability progress bars, thresholds, triage guidance)
```

---

## 3. Technology Choices
- **React 18 / 19**: Component-based reactive UI architecture with efficient virtual DOM reconciliation.
- **Vite 6**: Ultra-fast module-bundling, rapid Hot Module Replacement (HMR), and lightweight production builds (under 250 kB gzip).
- **React Router 6**: Client-side declarative routing with clean URL semantics (`/`, `/screen`, `/results`, `/about`, `/system`).
- **Vanilla CSS (Design Tokens)**: Zero-overhead, highly customizable design system avoiding bulky Tailwind dependencies and ensuring strict control over typography, spacing, and animations.
- **Lucide React**: Accessible, lightweight icon primitives for clinical visual indicators.
- **Fetch API**: Native browser HTTP client wrapped in a resilient service layer (`apiService`) without external transport overhead.

---

## 4. Folder Structure
```
frontend/
├── index.html                      # HTML entrypoint with metadata and fonts
├── package.json                    # Dependencies & build scripts
├── vite.config.js                  # Vite server & reverse-proxy configuration
├── .env.development                # VITE_API_BASE_URL=http://127.0.0.1:8000
├── tests/
│   └── frontend.test.js            # Automated Node/Vitest frontend test suite
├── src/
│   ├── main.jsx                    # Application bootstrapping
│   ├── App.jsx                     # Route definitions & shared state container
│   ├── index.css                   # Comprehensive design system & utilities
│   ├── components/
│   │   ├── Navbar.jsx              # Navigation with active states & mobile menu
│   │   ├── Footer.jsx              # Provenance & academic attribution
│   │   └── DisclaimerBanner.jsx    # Persistent epidemiological disclaimer
│   ├── pages/
│   │   ├── HomePage.jsx            # Landing page & workflow explanation
│   │   ├── ScreeningPage.jsx       # 7-section form collecting 34 Scenario-A features
│   │   ├── ResultsPage.jsx         # Tri-target risk dashboard & triage guidance
│   │   ├── AboutPage.jsx           # Methodology, dataset, and SHAP explainability
│   │   └── SystemPage.jsx          # Live health telemetry & model specifications
│   ├── services/
│   │   └── api.js                  # Centralized HTTP client for Step-18 endpoints
│   └── utils/
│       └── constants.js            # Dropdown options, 36 Indian states, sample data
└── dist/                           # Production-ready optimized static bundle
```

---

## 5. Components
1. **`Navbar`**: Fixed responsive navigation bar. Features brand icon, Scenario-A badge, active link highlighting, and an accessible mobile hamburger drawer.
2. **`Footer`**: Displays project provenance (India NFHS-5 2019–21), academic attribution, and data isolation statements.
3. **`DisclaimerBanner`**: Reusable alert component prominently displayed on every page, emphasizing that predictions are pre-screening indicators and not clinical diagnoses.

---

## 6. Pages
1. **Home (`/`)**: Introduces NutriSense AI, explains the rationale behind non-invasive community pre-screening, outlines the 3 pediatric targets, and presents the 3-step triage workflow.
2. **Screen Child (`/screen`)**: Structured input form divided into 7 logical sections (A through G) matching the 34 approved Scenario-A features. Includes a "Fill Sample Data" button for rapid demonstration.
3. **Results (`/results`)**: Comprehensive results dashboard presenting Stunting, Underweight, and Wasting cards. Features unrounded probabilities, percentage displays, decision threshold markers, and secondary clinical triage protocols.
4. **Methodology (`/about`)**: Detailed academic overview covering the NFHS-5 dataset, under-five population, LightGBM classifier rationale, TreeSHAP explainability, and why anthropometric outcome variables are excluded from inputs.
5. **System Status (`/system`)**: Real-time diagnostic monitor that polls `/health`, `/api/v1/health/model`, and `/api/v1/metadata` to show service uptime, in-memory model verification, and locked decision boundaries.

---

## 7. API Integration
The frontend connects to the Step-18 FastAPI endpoints:
- `POST /api/v1/screen`: Primary child screening evaluation.
- `GET /health`: Service liveness probe.
- `GET /api/v1/health/model`: In-memory model readiness and SHA-256 integrity verification.
- `GET /api/v1/metadata`: Public model provenance, feature schema, targets, and thresholds.

Base URL is configured via `import.meta.env.VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`).

---

## 8. Request Flow
1. User enters demographic, maternal, household, and symptom recall data into the 7 form sections on `/screen`.
2. User clicks **"Evaluate Child Undernutrition Risk"**.
3. Client-side validation checks input ranges and biological consistency.
4. If valid, `apiService.screenChild(payload)` dispatches an HTTP POST request to `/api/v1/screen`.
5. The button displays a spinning loading indicator ("Running Screening...") and disables double submissions.
6. Upon successful HTTP 200 response, the client updates in-memory state and routes to `/results`.

---

## 9. Response Flow
1. FastAPI validates input via Pydantic schema `ChildScreeningRequest` (`extra = "forbid"`).
2. Step-16 `predict_child_screening` executes inference across in-memory LightGBM pipelines.
3. FastAPI returns a structured `ScreeningResponse`.
4. The frontend unpacks probabilities, thresholds, and `screen_positive` boolean flags.
5. Visual progress bars render predicted probability alongside the locked threshold line marker.
6. The status banner dynamically reflects whether any of the 3 indicators flagged screen-positive.

---

## 10. Form Validation
The frontend provides responsive pre-submission validation to assist users while respecting the backend as the authoritative validator:
- **Child Age**: Enforces `0 <= age <= 59` completed months.
- **Biological Plausibility**:
  - `mother_age_first_birth <= mother_age_years`
  - `birth_order <= total_children_born`
- **Firstborn Logic**: If `birth_order === 1`, automatically flags `is_firstborn = 1` and disables `preceding_birth_interval_months`.
- **Numeric Ranges**: Constrains birth weight (`0.5 - 7.0 kg`), maternal BMI (`10 - 60 kg/m²`), household size (`>= 1`), and state ID (`1 - 36`).

---

## 11. Error Handling
All error states are caught and presented as human-readable UI alerts without technical stack traces:
- **Network Failure / API Offline**: "Unable to connect to the NutriSense AI backend. Please verify that the FastAPI server is running on http://127.0.0.1:8000."
- **422 Validation Error**: Highlights invalid fields with specific error messages.
- **500 Server Error**: "The screening service encountered an unexpected error. Please try again."
- **503 Model Unavailable**: "Model integrity verification failed or models are not loaded."

---

## 12. Privacy
- **Zero Raw Microdata**: The frontend never touches, downloads, or requests raw survey microdata.
- **No Persistent PII Storage**: Child profiles and screening predictions reside strictly in in-memory React state. They are never written to `localStorage`, `sessionStorage`, or cookies.
- **Zero Feature Logging**: No form payloads or child demographics are logged to `console.log` or browser analytics.

---

## 13. Accessibility
- Semantic HTML5 structure (`<nav>`, `<main>`, `<section>`, `<footer>`, `<form>`).
- Explicit `<label>` associations for every input via matching `id` and `htmlFor`.
- Visible keyboard focus rings (`outline: 2px solid var(--color-primary)`).
- Standard ARIA attributes (`role="alert"`, `aria-expanded`, `aria-label`).
- High-contrast color choices adhering to WCAG 2.1 AA standards.

---

## 14. Responsive Design
- Mobile-first CSS media queries (`max-width: 768px`).
- Form cards collapse seamlessly from a multi-column grid to a clean single-column layout on mobile.
- Results grid stacks vertically on tablets and smartphones.
- Mobile drawer navigation toggles smoothly via hamburger button.

---

## 15. Testing
The automated test suite in `frontend/tests/frontend.test.js` covers:
1. Conformance of client payload structures to the 34 approved Scenario-A features.
2. Complete absence of prohibited anthropometric leakage variables (`hw70`, `hw2`, `hw3`, `hw57`, `stunting`).
3. Child age validation constraints (0–59 months).
4. Maternal biological consistency checks (`first_birth <= current_age`).
5. Parity constraints (`birth_order <= total_children_born`).
6. Automatic firstborn handling for preceding birth interval.
7. Indian State ID dropdown completeness (36 states/UTs).
8. Strict non-clinical screening terminology enforcement.
9. Verification that operational telemetry excludes sensitive data.

Run command:
```bash
npm test --prefix frontend
```

---

## 16. How to Run Frontend
From the workspace root:
```bash
# Navigate to frontend and start Vite development server
cd frontend
npm run dev
```
Open your browser at:
`http://127.0.0.1:5173`

To build the production bundle:
```bash
npm run build
```

---

## 17. How Frontend Connects to FastAPI
1. **Direct Connection**: Frontend dispatches requests to `VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`).
2. **Reverse Proxy (Development)**: `vite.config.js` configures local proxy routes `/api` and `/health` forwarding to `http://127.0.0.1:8000`.
3. **CORS Alignment**: FastAPI's `src/api/config.py` allows origins `http://localhost:5173` and `http://127.0.0.1:5173` in development mode.

---

## 18. What the Frontend Does NOT Do
- Does **NOT** calculate probabilities, compute logits, or execute tree evaluations.
- Does **NOT** calculate or alter decision thresholds.
- Does **NOT** load `.joblib`, `.pkl`, or LightGBM model weights.
- Does **NOT** access, parse, or download raw DHS files.
- Does **NOT** duplicate Step-7 feature engineering or anthropometric calculators.
- Does **NOT** output clinical diagnoses or drug/supplement prescriptions.

---

## 19. Known Limitations
1. **Screening Setting Only**: Designed specifically for community pre-screening triage in resource-constrained environments where weighing scales and stadiometers are absent.
2. **Secondary Evaluation Mandatory**: A `Screen Positive` result requires confirmation through formal clinical anthropometry (recumbent length/height, digital weight, MUAC tape, bilateral edema check).
3. **Network Dependent**: In its current form, the frontend requires HTTP connectivity to the Step-18 FastAPI service.

---

## 20. Viva / Evaluator Q&A

### Q1: Why React and Vite instead of basic HTML/JS or server-rendered templates?
**Answer**: React provides modular, declarative component architecture and clean state management between the multi-section screening form and the results dashboard. Vite offers blazing-fast ES-module bundling, instant HMR, and ultra-compact production builds (~247 kB JS) suitable for deployment on low-bandwidth rural health kiosks.

### Q2: Why validate on the frontend if the FastAPI backend already validates?
**Answer**: Frontend validation provides immediate visual feedback to the health worker (e.g. flagging impossible maternal ages or birth orders before network transmission). However, frontend validation is purely a user-experience enhancement; the FastAPI backend remains the authoritative gatekeeper, strictly enforcing Pydantic schemas with `extra = "forbid"`.

### Q3: Why must ML inference remain strictly on the backend?
**Answer**: Housing model pipelines in the backend ensures cryptographic integrity auditing (SHA-256 verification via model registry), prevents client-side model tampering, avoids downloading heavy Python scikit-learn/LightGBM dependencies into the browser, and keeps the frontend lightweight and portable.

### Q4: Why are raw DHS survey files never sent to or loaded by the frontend?
**Answer**: Under international DHS privacy protocols, raw survey microdata contains detailed cluster and household identifiers that must remain protected and quarantined. Deploying microdata to a client browser violates privacy standards and introduces severe data bloat (441 MB). The frontend requires only statistical model predictions.

### Q5: Why is the term "screen_positive" used instead of "diagnosed with malnutrition"?
**Answer**: Epidemiologically and legally, machine learning models predicting risk from non-invasive community markers are screening filters, not clinical diagnostic tests. A child flagged as screen-positive is statistically prioritized for direct anthropometric measurement, avoiding false diagnostic labeling.

### Q6: Why are decision thresholds not calculated or adjusted by the frontend?
**Answer**: Decision thresholds (&tau; = 0.35 for stunting, 0.31 for underweight, 0.17 for wasting) were formally optimized on the independent validation split in Step 11 to achieve pre-specified sensitivity and specificity balance. Allowing frontend threshold manipulation would invalidate model calibration and compromise epidemiological safety.

---

## 21. Next Step
The project frontend (Step 19) is complete and verified. Steps 0–18 remain fully intact and passing. No further action is required for Step 19.
