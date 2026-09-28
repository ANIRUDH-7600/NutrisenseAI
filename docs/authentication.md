# Authentication & Authorization Architecture — NutriSense AI

## 1. Overview
NutriSense AI serves community healthcare workers, Anganwadi coordinators, and pediatric triage clinicians assessing early childhood undernutrition risk under Scenario A (scale-free community pre-screening).

To ensure patient data privacy, prevent unauthorized access, and eliminate Insecure Direct Object References (IDOR/BOLA), an end-to-end authentication and access control layer is established over the existing FastAPI and MongoDB architecture.

---

## 2. Authentication Architecture

```
+-------------------------------------------------------------+
|                     Client (React Frontend)                 |
|  - Login / Signup forms with client validation              |
|  - Secure storage of JWT Bearer token                       |
|  - Passes Authorization: Bearer <token>                     |
+-------------------------------------------------------------+
                               |
                               v  HTTPS / TLS
+-------------------------------------------------------------+
|                    FastAPI Backend Router                   |
|  POST /api/v1/auth/register  --> Create User in MongoDB     |
|  POST /api/v1/auth/login     --> Verify & Issue JWT Token   |
|  GET  /api/v1/auth/me        --> Return Current User        |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 FastAPI Dependency Layer                    |
|  get_current_user (Bearer Token Verification via PyJWT)     |
|  - Validates token signature (HS256)                        |
|  - Enforces expiration (exp) and issued-at (iat)            |
|  - Resolves user from MongoDB and verifies active status   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Protected Endpoints                     |
|  POST /api/v1/screen               --> Injects user_id      |
|  GET  /api/v1/screenings           --> Filters by user_id   |
|  GET  /api/v1/screenings/{id}      --> Verifies ownership   |
|  DELETE /api/v1/screenings/{id}    --> Verifies ownership   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     MongoDB (nutrisense_ai)                 |
|  Collections:                                               |
|    - users       (unique email, unique user_id)             |
|    - screenings  (indexed by user_id and created_at)        |
+-------------------------------------------------------------+
```

---

## 3. User Document Design (MongoDB `users` Collection)

The `users` collection stores account credentials and healthcare worker profiles. Passwords are **NEVER** stored in plaintext; only salted cryptographic bcrypt hashes are persisted.

```json
{
  "_id": "ObjectId('...')",
  "user_id": "usr_9f8b2c4e1a0d",
  "name": "Dr. Ananya Sharma",
  "email": "ananya.sharma@clinic.gov.in",
  "password_hash": "$2b$12$e8x...[60-character bcrypt hash]...",
  "role": "health_worker",
  "is_active": true,
  "created_at": "2026-09-28T10:00:00Z",
  "updated_at": "2026-09-28T10:00:00Z"
}
```

### Key Field Design Decisions:
- **`user_id`**: A stable, URL-safe random identifier (`usr_<12_hex_chars>`). MongoDB's native `_id` is retained for internal indexing, but `user_id` is exposed to APIs to prevent internal database implementation leakage.
- **`email`**: Normalized to lowercase, whitespace-trimmed, validated via RFC 5322 regex. A unique index (`{"email": 1}`, `unique=True`) guarantees zero duplicate accounts.
- **`password_hash`**: Bcrypt hash string with cost factor 12. Strictly excluded from all API response models.
- **`is_active`**: Boolean flag allowing immediate administrative account deactivation without requiring token blocklists.
- **`role`**: Authorization role (default: `"health_worker"`, extensible to `"administrator"`).

---

## 4. Cryptographic Password Hashing (bcrypt)

- **Algorithm**: `bcrypt` (Blowfish-based cipher with adaptive cost factor).
- **Work Factor / Rounds**: Cost factor `12` (~250ms per hash on production hardware), providing strong resistance against brute-force and offline dictionary attacks.
- **Salt Generation**: Automatically generates a cryptographically random 128-bit salt per password hash, preventing precomputed rainbow table attacks.
- **Constant-Time Verification**: `verify_password()` utilizes constant-time comparison (`bcrypt.checkpw`) to neutralize side-channel timing attacks.
- **Password Complexity Constraints**:
  - Minimum length: 8 characters (maximum: 128 characters to prevent DoS via algorithm saturation).
  - Must include at least one letter and one numeral.
  - Plaintext passwords are never logged, echoed, or included in error traces.

---

## 5. JWT Authentication & Token Lifecycle

- **Library**: `PyJWT` (v2.8.0+).
- **Signing Algorithm**: `HS256` (HMAC with SHA-256).
- **Token Claims**:
  - `sub`: `user_id` (string identifier of the user).
  - `email`: Normalized email address of the account.
  - `role`: Authorization role (`health_worker`).
  - `iat`: Timestamp (UTC) when the token was issued.
  - `exp`: Timestamp (UTC) after which the token is invalid.
  - `jti`: Unique token identifier (UUID hex) enabling tracking and revocation.
- **Token Expiration**: Default lifetime is `1440` minutes (24 hours) to facilitate uninterrupted community healthcare fieldwork in rural and semi-urban settings.
- **Validation Behavior**:
  - Validates cryptographic signature using `JWT_SECRET`.
  - Enforces required presence of `sub`, `exp`, and `iat`.
  - Rejects expired tokens with HTTP `401 Unauthorized` (`TOKEN_EXPIRED`).
  - Rejects tampered tokens with HTTP `401 Unauthorized` (`INVALID_TOKEN`).

---

## 6. Authentication Dependency (`get_current_user`)

FastAPI routes declare user authentication requirements using dependency injection:

```python
from fastapi import Depends
from src.api.security import get_current_user

@router.post("/screen")
async def screen_child(
    request: ChildScreeningRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    ...
```

- When the `Authorization: Bearer <token>` header is absent, the dependency immediately returns HTTP `401 Unauthorized` with `{ "code": "AUTHENTICATION_REQUIRED" }`.
- When valid, the user record is verified in MongoDB (checking `is_active: true`) and returned without `password_hash`.

---

## 7. User-Owned Screening Relationship & IDOR Protection

In Phase 6, screening documents in MongoDB will be explicitly associated with the authenticated user:

```json
{
  "screening_id": "scr_a1b2c3d4e5f6",
  "user_id": "usr_9f8b2c4e1a0d",
  "child_name": "Aarav Sharma",
  "created_at": "2026-09-28T10:15:00Z",
  "inputs": { ... },
  "predictions": { ... },
  "model_version": "nutrisense-scenario-a-v1.0.0",
  "feature_schema_version": "scenario-a-30-v1"
}
```

### IDOR / BOLA Prevention Plan:
1. **Creation**: When `POST /api/v1/screen` is called, `user_id` is extracted strictly from the verified JWT token (`current_user["user_id"]`), never from the request body.
2. **List Retrieval**: `GET /api/v1/screenings` automatically filters by `{"user_id": current_user["user_id"]}`. Users cannot view other health workers' screenings.
3. **Single Record Retrieval**: `GET /api/v1/screenings/{screening_id}` queries `{"screening_id": screening_id, "user_id": current_user["user_id"]}`. If the record belongs to another user, the API responds with `404 Not Found` (or `403 Forbidden`), preventing ID enumeration.
4. **Deletion**: `DELETE /api/v1/screenings/{screening_id}` enforces identical ownership checks.

---

## 8. Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `JWT_SECRET` | string | *dev key in dev mode* | Cryptographic HMAC secret key (must be >= 32 characters in production). |
| `JWT_ALGORITHM` | string | `HS256` | JWT signing algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | integer | `1440` | Token validity period in minutes (24 hours). |
| `BCRYPT_ROUNDS` | integer | `12` | Bcrypt hashing work factor. |

> **Security Rule**: Real secrets are never checked into version control. `.env` is ignored in `.gitignore`, and `.env.example` contains only template placeholders.

---

## 9. Implemented Registration Flow (`POST /api/v1/auth/register`)

The registration endpoint is live and fully tested:

- **Endpoint**: `POST /api/v1/auth/register`
- **Request Format**:
  ```json
  {
    "name": "Dr. Priya Sen",
    "email": "priya.sen@district-hospital.org",
    "password": "SecurePassword2026!",
    "confirm_password": "SecurePassword2026!",
    "role": "health_worker"
  }
  ```
- **Response Format (HTTP 201 Created)**:
  ```json
  {
    "user_id": "usr_9f8b2c4e1a0d",
    "name": "Dr. Priya Sen",
    "email": "priya.sen@district-hospital.org",
    "role": "health_worker",
    "created_at": "2026-09-28T10:00:00Z",
    "is_active": true
  }
  ```

### Processing Pipeline:
1. **Input Normalization & Validation**:
   - `email`: Stripped of leading/trailing whitespace and converted strictly to lowercase.
   - `password`: Enforced length (8–128 chars), required character types (at least one letter and one number), and confirmed matching.
   - `name`: Whitespace trimmed, minimum 2 characters.
2. **Pre-Insert Uniqueness Check**:
   - Queries `db.users.find_one({"email": normalized_email})`.
   - If an account exists, returns HTTP `409 Conflict` (`EMAIL_ALREADY_EXISTS`).
3. **Cryptographic Salted Hashing**:
   - Computes `bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(12))`.
   - Raw plaintext password is discarded immediately.
4. **Asynchronous Persistence**:
   - Inserts document into `db.users` with unique indexes on `email` and `user_id`.
   - Concurrent duplicate race conditions caught via `DuplicateKeyError` and mapped to HTTP `409 Conflict`.
5. **Fail-Soft Safe Response**:
   - Returns sanitized `UserResponse` with HTTP `201 Created`.
   - Strictly excludes `password`, `password_hash`, or internal system keys.

---

## 10. Implementation Status Tracker

### IMPLEMENTED:
- [x] **MongoDB `users` collection**: Initialized with unique indexes on `email` and `user_id`.
- [x] **Email uniqueness enforcement**: Database index and pre-check validation preventing duplicate accounts.
- [x] **Email normalization**: Consistent lowercase trimming across all checks and storage.
- [x] **Password hashing**: Salted bcrypt hashing (work factor 12), zero plaintext persistence.
- [x] **Registration endpoint (`POST /api/v1/auth/register`)**: Returns HTTP 201 with sanitized `UserResponse`.
- [x] **Automated registration tests**: 11 unit/integration test cases covering all validation, duplicate, and database failure modes.

### NOT YET IMPLEMENTED (Scheduled for subsequent phases):
- [ ] **Login (`POST /api/v1/auth/login`)**: Phase 3
- [ ] **Current User Profile (`GET /api/v1/auth/me`)**: Phase 3
- [ ] **Logout**: Phase 4
- [ ] **Server-side route protection (`get_current_user`) enforcement on screening endpoints**: Phase 5
- [ ] **User-owned screening records & IDOR filtering**: Phase 6
- [ ] **Frontend authentication UI (Signup, Login, Dashboard)**: Phases 11–13

