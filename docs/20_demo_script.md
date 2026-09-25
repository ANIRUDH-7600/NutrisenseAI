# NutriSense AI: Live Demonstration & Evaluation Script

A step-by-step presentation script for conducting an interactive demonstration of NutriSense AI during oral project examinations and viva voce defense.

> **Synthetic Demonstration Data Notice**: All patient profiles and values used in this demonstration are **entirely synthetic test cases** designed to illustrate triage decision boundaries. No real-world personally identifiable children or health records are utilized.

---

## 1. System Setup & Startup

### Step 1.1: Launch the FastAPI Backend
Open a terminal in the project root:
```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
**Presenter commentary**:
> *"Here, the FastAPI backend boots up. Notice the startup logs: `ScreeningService` initializes automatically, audits the SHA-256 hashes of all three LightGBM model files against `models/model_registry.json`, confirms that all 34 features and locked thresholds match, and preloads the pipelines into memory. If any model artifact were corrupted or missing, the server would fail to boot immediately."*

### Step 1.2: Launch the React Frontend
Open a second terminal:
```bash
cd frontend
npm run dev
```
**Presenter commentary**:
> *"We start our Vite development server, which binds to port 5173 with automatic reverse-proxying configured for the FastAPI backend."*

---

## 2. Navigating the User Interface

### Step 2.1: Open the Landing Page
Open a browser to: `http://127.0.0.1:5173`

**What to point out**:
1. **Brand & Scenario Tag**: Highlight "NutriSense AI" and the "Scenario A — Scale-Free Community Pre-Screening" badge.
2. **Clinical Disclaimer Banner**: Note the prominent amber disclaimer:
   > *"This tool provides an AI-based community pre-screening result and is not a medical diagnosis. Screening results should be interpreted by an appropriately qualified healthcare professional."*
3. **Tri-Target Overview Cards**: Explain that NutriSense AI assesses three distinct pediatric conditions simultaneously:
   - Stunting (Height-for-Age Z-score < -2 SD; Chronic linear deficit; Threshold: 35%)
   - Underweight (Weight-for-Age Z-score < -2 SD; Composite deficit; Threshold: 31%)
   - Wasting (Weight-for-Height Z-score < -2 SD; Acute deficit; Threshold: 17%)
4. **The 3-Step Workflow**: Walk through: (1) Collect Non-Invasive Data -> (2) LightGBM Model Inference -> (3) Validation-Derived Operating Threshold Triage. Explain that operating thresholds were selected on the validation cohort in Step 11 and locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration).

---

## 3. Demonstrating Child Screening

### Step 3.1: Navigate to "Screen Child"
Click **"Start Child Screening"** in the hero section or the **"Screen Child"** link in the navbar.

**What to point out**:
1. **Form Organization (7 Logical Sections)**: Explain that rather than displaying 34 raw inputs in a single overwhelming column, the form organizes features into clear clinical domains:
   - **Section A**: Child Demographics & Birth Characteristics (Age, Sex, Birth order, Birth size, Birth weight)
   - **Section B**: Recent Morbidities (2-week recall of Diarrhea, Fever, Cough)
   - **Section C**: Feeding & Delivery Setting (Breastfeeding continuation, Delivery place)
   - **Section D**: Maternal History (Mother's age, age at first birth, parity, education, BMI, ANC visits)
   - **Section E**: Household Socioeconomics (Wealth quintile, residence, caste, religion, household size)
   - **Section F**: Household Environment / WASH (JMP water source, sanitation facility, electricity, cooking fuel)
   - **Section G**: Geographic State / UT (Administrative State ID).
2. **Firstborn Logic**: Change "Birth Order" to `1`. Point out that the form automatically disables the "Preceding Birth Interval" field with the helper text *"N/A (Firstborn)"*.

### Step 3.2: Populate Demonstration Data
Click the **"Fill Sample Data"** button at the top right of the form.

**Synthetic Profile Values Loaded**:
- Child Age: 24.0 months | Male | Birth Order: 2 | Single Birth | Preceding Interval: 28 months
- Birth Size: Average (3.0) | Birth Weight: 2.8 kg | Public Facility Delivery
- Currently Breastfeeding: Yes | Diarrhea: No | Fever: No | Cough: No
- Mother Age: 26 years | Age at First Birth: 22 years | Parity: 2 children | Secondary Education | BMI: 21.5 kg/m² | ANC Visits: 4
- Wealth Quintile: Q2 (Poorer) | Rural | Caste: OBC | Religion: Hindu | Household Size: 5 | Electricity: Yes | Clean Fuel: Yes | State: Bihar (10).

**Presenter commentary**:
> *"This profile represents a typical 2-year-old child living in a rural household. Notice that all 34 inputs are strictly non-invasive—no stadiometers or hanging scales were required to solicit these values."*

### Step 3.3: Submit Screening Evaluation
Click **"Evaluate Child Undernutrition Risk"**.

**What to point out**:
- The button enters a loading state (`"Running Screening..."`) with an animated spinner, preventing accidental double submissions.
- The request completes in under 100 milliseconds via the local FastAPI backend.

---

## 4. Explaining the Screening Results Dashboard

The application automatically transitions to `/results`.

### Step 4.1: Top Status Banner
Point to the dynamic triage banner:
> *"The banner identifies that all three undernutrition conditions were evaluated simultaneously. For this test case, Stunting and Underweight flag as **Screen Positive**, while Wasting flags as **Screen Positive**."*

### Step 4.2: Inspecting Individual Target Cards
Direct attention to the **Stunting Card**:
1. **Predicted Probability**: $41.3\%$ (unrounded: $0.413309$).
2. **Operating Threshold**: $35.0\%$ (validation-derived operating threshold selected in Step 11 and locked in Step 12; no probability calibration claimed).
3. **Screening Badge**: Amber/Red badge labeled **"Screen Positive"**.
4. **Visual Comparison Progress Bar**: The colored fill bar visibly extends beyond the dark vertical threshold line at $35\%$.
5. **Non-Diagnostic Wording**: Point out the interpretation text:
   > *"The predicted probability is above the community screening threshold, indicating elevated statistical risk. Secondary clinical anthropometric triage is indicated."*
   > *(Crucially, it does NOT state 'Child has malnutrition' or 'Diagnosed with stunting'.)*

### Step 4.3: Secondary Clinical Triage Guidance Box
Scroll to the bottom of the page and explain the triage box:
> *"Because this is a pre-screening filter, every screen-positive result directs the frontline worker to standard clinical follow-up protocols:
> 1. Direct measurement of length/height using an infantometer or stadiometer.
> 2. Direct body mass weighing using a digital hanging scale.
> 3. MUAC tape measurement.
> 4. Physical inspection for bilateral pitting edema."*

---

## 5. Demonstrating Methodology & Explainability
Click **"Methodology"** in the navbar (`/about`).

**What to point out**:
1. **Dataset Overview**: India NFHS-5 (2019–21) Children's Recode ($N = 220{,}460$ analytical cohort).
2. **Strict Feature Isolation**: Re-emphasize that direct anthropometric variables (`hw2`–`hw12`, `hw70`–`hw73`, `hw57`) are permanently isolated.
3. **TreeSHAP Findings**: Explain the top global feature rankings:
   - Stunting is driven primarily by child age and household wealth.
   - Underweight is driven by birth weight and maternal BMI.
   - Wasting is driven by infant age (higher risk at younger ages) and maternal nutritional status.

---

## 6. Demonstrating System Health & Model Provenance
Click **"System Status"** in the navbar (`/system`).

**What to point out**:
1. **Real-Time Telemetry**: Click the **"Refresh Status"** button. The page polls `/health`, `/api/v1/health/model`, and `/api/v1/metadata`.
2. **API Service Liveness**: Displays "Healthy" (green indicator).
3. **Model Integrity Audit**: Displays "Verified & In-Memory" (confirming that all 3 model files match their SHA-256 signatures).
4. **Registered Specifications**:
   - Model Release Version: `nutrisense-scenario-a-v1.0.0`
   - Feature Schema Version: `scenario-a-34-v1`
   - Model Family: `LightGBM` (Unweighted)
   - Locked Decision Boundaries: Stunting ($35\%$), Underweight ($31\%$), Wasting ($17\%$).
5. **Security Verification**: Highlight that no internal filesystem paths or secret tokens are disclosed.

---

## 7. Interactive API Documentation (FastAPI Docs)
Open a new browser tab to: `http://127.0.0.1:8000/docs`

**What to point out**:
1. **OpenAPI 3.1 Swagger UI**: Built natively by FastAPI.
2. **Interactive `POST /api/v1/screen` Endpoint**: Expand the endpoint to show the comprehensive Pydantic documentation, input field constraints, and strict schema validation.
3. **Liveness & Metadata Endpoints**: Show `/health` and `/api/v1/metadata`.

---

## 8. Concluding the Demonstration
Return to the React application and summarize:
> *"In summary, NutriSense AI bridges the gap between complex survey epidemiology and frontline community healthcare. By combining 34 non-invasive NFHS-5 indicators with LightGBM estimators and validation-derived operating thresholds, TreeSHAP interpretability, a secure FastAPI backend, and an intuitive React interface, we demonstrate a practical, deployable decision-support system for pediatric undernutrition pre-screening in India. Thank you."*
