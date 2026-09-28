# Authorization & IDOR/BOLA Prevention — NutriSense AI

## 1. Overview
In NutriSense AI, healthcare workers perform sensitive pediatric malnutrition assessments in community triage environments. Screening assessments contain household demographic, socioeconomic, and morbidity records of young children.

To prevent Insecure Direct Object References (**IDOR**) and Broken Object Level Authorization (**BOLA** / OWASP API Security Top 10 API1:2023), NutriSense AI establishes a strict server-side resource ownership model starting with **Phase 6**.

---

## 2. Resource Ownership Architecture

```
+-------------------------------------------------------------+
|                  Client Request + Bearer JWT                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|            FastAPI Dependency: get_current_user             |
|  - Cryptographically verifies HS256 signature               |
|  - Validates expiry and absence from revoked_tokens store   |
|  - Resolves active user profile from MongoDB                |
|  - Extracted caller identity: current_user["user_id"]       |
+-------------------------------------------------------------+
                              |
                              | authenticated user_id
                              v
+-------------------------------------------------------------+
|                  Endpoint Authorization Layer               |
|                                                             |
|  POST /api/v1/screen                                        |
|    --> Forces document["user_id"] = current_user["user_id"] |
|    --> Ignores/forbids any client-supplied user_id          |
|                                                             |
|  GET /api/v1/screenings                                     |
|    --> Query filter: {"user_id": current_user["user_id"]}   |
|    --> Orders by created_at DESC (1-100 pagination)         |
|                                                             |
|  GET /api/v1/screenings/{screening_id}                      |
|    --> Query filter: {"screening_id": id, "user_id": uid}  |
|    --> Returns 404 SCREENING_NOT_FOUND if non-owner         |
|                                                             |
|  DELETE /api/v1/screenings/{screening_id}                   |
|    --> Delete filter: {"screening_id": id, "user_id": uid} |
|    --> Returns 404 SCREENING_NOT_FOUND if non-owner         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     MongoDB (nutrisense_ai)                 |
|  screenings collection:                                     |
|    - Index: {"screening_id": 1} (unique)                    |
|    - Compound Index: {"user_id": 1, "created_at": -1}       |
+-------------------------------------------------------------+
```

---

## 3. Screening Document Ownership Model

Every screening assessment persisted in MongoDB contains the immutable server-assigned `user_id`:

```json
{
  "_id": "ObjectId('68b1a2c3d4e5...')",
  "screening_id": "scr_9f8b2c4e1a0d",
  "user_id": "usr_alpha_111",
  "child_name": "Anonymous Child",
  "created_at": "2026-09-28T12:00:00Z",
  "inputs": {
    "child_age_months": 24.0,
    "child_sex_male": 1,
    "mother_age_years": 26.0,
    "state_id": 10
  },
  "predictions": {
    "stunting": { "probability": 0.42, "threshold": 0.36, "screen_positive": true },
    "underweight": { "probability": 0.28, "threshold": 0.30, "screen_positive": false },
    "wasting": { "probability": 0.15, "threshold": 0.17, "screen_positive": false }
  },
  "model_version": "nutrisense-scenario-a-v1.0.0",
  "feature_schema_version": "scenario-a-30-v1"
}
```

### Key Ownership Guarantees:
1. **Server-Derived Identity**: The `user_id` is extracted strictly from `current_user["user_id"]` (populated by verified JWT claims).
2. **Zero Client Spoofing**: Even if a malicious request attempts to inject `{"user_id": "usr_victim_999"}` in the payload or query parameters, `ChildScreeningRequest` forbids extra fields (`extra = "forbid"`), and `save_screening_record()` explicitly overwrites any payload value with the verified `user_id`.

---

## 4. Route-by-Route Authorization Enforcement

### 1. `POST /api/v1/screen` (Screening Creation)
- **Authentication**: Required (`Depends(get_current_user)`).
- **Behavior**: Runs the Step-16 Scenario A LightGBM inference pipeline. Persists assessment with `user_id = current_user["user_id"]`.
- **Fail-Soft Persistence**: If MongoDB is unreachable, the ML predictions are still returned to the clinician with HTTP 200 without leaking stack traces or halting patient screening.

### 2. `GET /api/v1/screenings` (Screening History)
- **Authentication**: Required (`Depends(get_current_user)`).
- **Behavior**: Retrieves only screenings where `user_id == current_user["user_id"]`.
- **Database Query**: `_db.screenings.find({"user_id": user_id}).sort("created_at", -1)`.
- **Pagination Bounds**: Validates `limit` (1 to 100) and `skip` (>= 0). Users cannot browse or paginate into other healthcare workers' histories.

### 3. `GET /api/v1/screenings/{screening_id}` (Single Record Retrieval)
- **Authentication**: Required (`Depends(get_current_user)`).
- **Behavior**: Searches MongoDB with `{"screening_id": screening_id, "user_id": current_user["user_id"]}`.
- **Enumeration Defense (Safe 404)**: If the record belongs to another healthcare worker, the endpoint returns **HTTP 404 `SCREENING_NOT_FOUND`** (identical to a non-existent record) rather than HTTP 403 Forbidden. This prevents attackers from probing and enumerating valid screening IDs across users.

### 4. `DELETE /api/v1/screenings/{screening_id}` (Record Deletion)
- **Authentication**: Required (`Depends(get_current_user)`).
- **Behavior**: Executes `delete_one({"screening_id": screening_id, "user_id": current_user["user_id"]})`.
- **Isolation Guarantee**: If User A attempts to delete User B's screening, 0 records are deleted and HTTP 404 is returned. User B's record remains completely intact in MongoDB.

---

## 5. Database Indexes & Performance
To guarantee high-throughput, low-latency lookups without full-collection scans:
- **`screening_id` (Unique)**: `{"screening_id": 1}`, unique index for single-record lookups.
- **Compound User History Index**: `[("user_id", 1), ("created_at", -1)]`, accelerates filtered list queries and chronological pagination.

---

## 6. Legacy & Unowned Screening Records
In development or pre-Phase-6 databases, historical test or demo records may exist without a `user_id` field:
- **Backward Compatibility**: `ScreeningRecord` schema sets `user_id: Optional[str] = None`.
- **Strict Isolation**: Because all queries explicitly filter by `{"user_id": current_user["user_id"]}`, unowned legacy records are never exposed to authenticated users and never randomly assigned to arbitrary accounts.

---

## 7. Mandatory Two-User Security Verification Matrix

Automated verification in `backend/tests/test_ownership_phase6.py` and `backend/tests/test_security_hardening_phase7.py`:

| Action | User A (usr_alpha_111) | User B (usr_beta_222) | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| Create Record | Creates Screening A | Creates Screening B | Records tagged with respective user IDs | **PASS** |
| List History | Sees Screening A | Sees Screening B | Zero cross-user records visible | **PASS** |
| Cross-Get | Requests Screening B | Requests Screening A | HTTP 404 `SCREENING_NOT_FOUND` | **PASS** |
| Cross-Delete | Deletes Screening B | Deletes Screening A | HTTP 404; Target record remains intact | **PASS** |
| Own Delete | Deletes Screening A | Deletes Screening B | HTTP 200 `success: true` | **PASS** |
