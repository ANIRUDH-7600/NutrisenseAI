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

---

## 10. Implemented Login Flow (`POST /api/v1/auth/login`)

The login endpoint authenticates registered health workers and issues a cryptographically signed JWT bearer token:

- **Endpoint**: `POST /api/v1/auth/login`
- **HTTP Method**: `POST`
- **Request Format**:
  ```json
  {
    "email": "priya.sen@district-hospital.org",
    "password": "SecurePassword2026!"
  }
  ```
- **Successful Response (HTTP 200 OK)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 86400,
    "user": {
      "user_id": "usr_9f8b2c4e1a0d",
      "name": "Dr. Priya Sen",
      "email": "priya.sen@district-hospital.org",
      "role": "health_worker",
      "created_at": "2026-09-28T10:00:00Z",
      "is_active": true
    }
  }
  ```

### Processing Pipeline:
1. **Email Normalization**:
   - Normalized exactly identically to registration (`request.email.strip().lower()`).
2. **User Document Retrieval**:
   - Queries MongoDB `users` collection using the normalized email.
3. **Bcrypt Password Verification**:
   - Uses constant-time `verify_password()` (`bcrypt.checkpw`) against the stored `password_hash`.
4. **Active Account Enforcement**:
   - Checks `is_active: true`.
5. **Zero User Enumeration (Generic 401)**:
   - For nonexistent email, incorrect password, or inactive account, returns an identical generic HTTP 401 response:
     ```json
     {
       "success": false,
       "error": {
         "code": "INVALID_CREDENTIALS",
         "message": "Invalid email or password.",
         "details": []
       }
     }
     ```
6. **JWT Issuance**:
   - Signs a standard JWT with claims `sub`, `email`, `role`, `iat`, `exp`, `jti` using `JWT_SECRET` and `JWT_ALGORITHM` (`HS256`).
   - Lifetime defaults to `ACCESS_TOKEN_EXPIRE_MINUTES * 60` seconds (24 hours / 86400 seconds).

---

## 11. Current User Profile Endpoint (`GET /api/v1/auth/me`)

The `/auth/me` endpoint returns safe profile information for the authenticated health worker:

- **Endpoint**: `GET /api/v1/auth/me`
- **Required Header**: `Authorization: Bearer <JWT>`
- **Response Format (HTTP 200 OK)**:
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

### Verification Pipeline:
1. **Header Parsing**: Extracts Bearer token via `HTTPBearer(auto_error=False)` dependency.
2. **Cryptographic Validation**: Decodes and verifies HS256 signature with `JWT_SECRET`.
3. **Expiration Enforcement**: Rejects expired tokens with HTTP 401 (`TOKEN_EXPIRED`).
4. **Subject Verification**: Extracts `sub` (user_id) from verified claims.
5. **Active Account Verification**: Queries MongoDB `users` collection to confirm the user exists and is active.
6. **Safe Response Guarantee**:
   - Strictly returns `UserResponse`.
   - Never exposes `password`, `password_hash`, database `_id`, or `JWT_SECRET`.

---

---

## 12. Implemented Logout & Token Revocation Flow (`POST /api/v1/auth/logout`)

The logout endpoint securely invalidates the caller's JWT token session by storing its unique cryptographic identifier (`jti`) in a MongoDB revocation store:

- **Endpoint**: `POST /api/v1/auth/logout`
- **HTTP Method**: `POST`
- **Required Header**: `Authorization: Bearer <JWT>`
- **Successful Response (HTTP 200 OK)**:
  ```json
  {
    "success": true,
    "message": "Successfully logged out."
  }
  ```

### Revocation Architecture:
1. **Token Identification**:
   - The token's unique `jti` (UUID hex) and subject `user_id` are extracted from verified claims.
2. **Revocation Persistence (`revoked_tokens` Collection)**:
   - Persists a minimal metadata document strictly omitting the raw JWT:
     ```json
     {
       "jti": "8f1a2c3e4b5d6e7f8a9b0c1d2e3f4a5b",
       "user_id": "usr_9f8b2c4e1a0d",
       "revoked_at": "2026-09-28T12:00:00Z",
       "expires_at": "2026-09-29T12:00:00Z"
     }
     ```
   - Raw tokens and Authorization headers are never stored in the database or written to logs.
3. **Session Isolation**:
   - **Multi-user isolation**: Logging out User A invalidates only `jti_A`. User B's token remains completely valid.
   - **Multi-session isolation**: Each login issues a fresh unique `jti`. Logging out one device/session does not terminate other concurrent sessions belonging to the same health worker.
4. **Automated Pruning & TTL Cleanup**:
   - A MongoDB TTL index (`expireAfterSeconds=0` on `expires_at`) automatically removes expired revocation documents once the underlying JWT reaches natural expiration.
   - A programmatic cleanup function (`clean_expired_revocations()`) provides explicit pruning support.
5. **Idempotency**:
   - Repeated logout requests with the same token succeed idempotently (`200 OK`) without crashing or duplicating records (`upsert=True`).
6. **Authentication Dependency Enforcement**:
   - When a revoked token is presented to [`get_current_user`](file:///d:/finalyearproj/Nutrisense-Ai/backend/src/api/security.py#L135), the dependency detects the revoked `jti` in MongoDB and immediately halts execution with HTTP 401 (`TOKEN_REVOKED`).

## 14. Protected Screening Endpoints (Phase 5 — Route Protection)

### Overview & Security Scope
In Phase 5, all four operational screening endpoints are protected using FastAPI's `Depends(get_current_user)`. Unauthenticated or invalid requests are rejected immediately at the gateway layer before reaching screening or database logic.

```
+-------------------------------------------------------------+
|                  Incoming Client Request                    |
|  POST /api/v1/screen                                        |
|  GET  /api/v1/screenings                                    |
|  GET  /api/v1/screenings/{screening_id}                     |
|  DELETE /api/v1/screenings/{screening_id}                  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              FastAPI Dependency: get_current_user            |
|  1. Extract Bearer token from Authorization header           |
|     - Missing / empty / non-Bearer  --> HTTP 401            |
|  2. Decode & verify JWT signature (HS256)                   |
|     - Tampered / malformed / expired --> HTTP 401           |
|  3. Check revocation status in MongoDB revoked_tokens       |
|     - Revoked (logged out)          --> HTTP 401            |
|  4. Fetch user document from MongoDB users collection        |
|     - User not found / inactive     --> HTTP 401            |
|     - Database connection error     --> HTTP 503            |
+-------------------------------------------------------------+
                              |
                              | [Authenticated & Active User]
                              v
+-------------------------------------------------------------+
|               Existing Screening Business Logic             |
|  - Validates 30-feature Scenario A input                     |
|  - Executes LightGBM unweighted ensemble inference           |
|  - Applies locked clinical thresholds                        |
|  - Returns risk scores, probabilities, and SHAP explanations |
|  - Saves/retrieves screening records                         |
+-------------------------------------------------------------+
```

### Protected Endpoints:
1. `POST /api/v1/screen`: Requires valid Bearer JWT. Validates input child profile and generates ML predictions while rejecting unauthenticated traffic.
2. `GET /api/v1/screenings`: Requires valid Bearer JWT. Retrieves recent screening histories.
3. `GET /api/v1/screenings/{screening_id}`: Requires valid Bearer JWT. Retrieves screening record by identifier.
4. `DELETE /api/v1/screenings/{screening_id}`: Requires valid Bearer JWT. Removes screening record by identifier.

### 401 Rejection Behavior Matrix:
All unauthenticated requests receive standardized JSON error envelopes without leaking internal details:

| Request Condition | HTTP Status | Error Code | Response Message |
| :--- | :--- | :--- | :--- |
| Missing `Authorization` header | 401 | `AUTHENTICATION_REQUIRED` | Authentication required. Please provide a valid Bearer token. |
| Empty Bearer token (`Bearer `) | 401 | `AUTHENTICATION_REQUIRED` | Authentication required. Please provide a valid Bearer token. |
| Non-Bearer scheme (`Basic xyz`) | 401 | `AUTHENTICATION_REQUIRED` | Authentication required. Please provide a valid Bearer token. |
| Malformed token string | 401 | `INVALID_TOKEN` | Invalid authentication token. |
| Tampered token signature | 401 | `INVALID_TOKEN` | Invalid authentication token. |
| Expired token (`exp` elapsed) | 401 | `TOKEN_EXPIRED` | Authentication token has expired. |
| Revoked token (logged out) | 401 | `TOKEN_REVOKED` | Authentication token has been revoked. |
| User inactive (`is_active=false`)| 401 | `USER_INACTIVE` | User account is inactive. |
| Nonexistent user (`sub` not in DB)| 401 | `USER_NOT_FOUND` | User account not found. |

### Critical Architectural Distinction: Authentication vs. Authorization
> [!IMPORTANT]
> **Phase 5 provides AUTHENTICATION protection.**
> It verifies *who* the caller is and ensures that only authenticated, active healthcare workers with valid, unrevoked JWT credentials can reach the screening API endpoints.
>
> **Phase 6 will provide RESOURCE OWNERSHIP / AUTHORIZATION protection.**
> Phase 6 will associate each screening document with the caller (`screening.user_id = authenticated_user.user_id`) and enforce strict ownership filtering so users can only view or delete their own screening records.
>
> **Do not claim IDOR/BOLA protection is complete until Phase 6 is implemented.**

---

## 15. Security Considerations & Error Codes

### Standardized Error Responses:
| Scenario | HTTP Status | Error Code | Description |
| :--- | :--- | :--- | :--- |
| Nonexistent email on login | 401 | `INVALID_CREDENTIALS` | Generic message to prevent email enumeration. |
| Wrong password on login | 401 | `INVALID_CREDENTIALS` | Generic message to prevent brute-force timing clues. |
| Inactive user on login | 401 | `INVALID_CREDENTIALS` | Generic message preventing deactivation enumeration. |
| Missing Authorization header | 401 | `AUTHENTICATION_REQUIRED` | Bearer token required. |
| Malformed Authorization header | 401 | `AUTHENTICATION_REQUIRED` | Non-Bearer or empty token. |
| Tampered JWT signature | 401 | `INVALID_TOKEN` | Cryptographic signature mismatch. |
| Malformed JWT payload | 401 | `INVALID_TOKEN` | Unparseable token structure or missing jti. |
| Expired JWT | 401 | `TOKEN_EXPIRED` | Expired lifetime. |
| Revoked JWT token | 401 | `TOKEN_REVOKED` | Token invalidated via logout. |
| Nonexistent user token | 401 | `USER_NOT_FOUND` | User deleted or absent from DB. |
| Deactivated account token | 401 | `USER_INACTIVE` | Account deactivated in DB. |
| Database unreachable | 503 | `DATABASE_UNAVAILABLE` | Fail-safe without leaking stack traces. |

### Operational Privacy Policy:
- **Zero Credential Logging**: Plaintext passwords, password hashes, JWT secrets, and `Authorization` headers are never logged.
- **Zero Feature/Body Logging**: Child screening request attributes and predictions remain strictly unlogged in production telemetry.
- **Security Limitation**: Token revocation requires a low-latency check against MongoDB `revoked_tokens`. If the database is disconnected, authentication endpoints fail safe with HTTP 503 rather than permitting potentially revoked tokens.

---

## 16. Implementation Status Tracker

### IMPLEMENTED:
- [x] **Signup (`POST /api/v1/auth/register`)**: Account creation with input validation, password complexity, and duplicate rejection (Phase 2).
- [x] **Login (`POST /api/v1/auth/login`)**: Generic 401 anti-enumeration defenses, bcrypt verification, JWT generation (Phase 3).
- [x] **Current User Profile (`GET /api/v1/auth/me`)**: Token authentication dependency resolving safe active user identity from MongoDB (Phase 3).
- [x] **Logout (`POST /api/v1/auth/logout`)**: Session token revocation via MongoDB `revoked_tokens` store (Phase 4).
- [x] **Token Revocation (`jti` tracking & TTL pruning)**: Granular per-session revocation with automated MongoDB TTL index cleanup (Phase 4).
- [x] **Protected Screening Routes (`POST /screen`, `GET /screenings`, `GET /screenings/{id}`, `DELETE /screenings/{id}`)**: Phase 5 complete with `Depends(get_current_user)`.
- [x] **User-Owned Screening Records & IDOR/BOLA Protection**: Phase 6 complete with strict resource ownership filtering and composite indexing (`user_id` + `created_at`).
- [x] **Security Hardening**: Phase 7 complete with NoSQL injection guards, sliding-window rate limiting (HTTP 429), defensive HTTP security headers, CORS restrictions, and pagination bounds.
- [x] **Automated Test Suites**:
  - Phase 1 Security Tests: 8 passing
  - Phase 2 Registration Tests: 11 passing
  - Phase 3 Login Tests: 22 passing
  - Phase 4 Logout Tests: 15 passing
  - Phase 5 Route Protection Tests: 44 passing
  - Phase 6 Ownership / IDOR Tests: 14 passing
  - Phase 7 Security Hardening Tests: 11 passing
  - Existing API, Database & Inference Tests: 46 passing
  - **Total Backend Tests**: 171 passing (100%)
  - **Total Frontend Tests**: 9 passing (100%)

### NOT YET IMPLEMENTED (Scheduled for subsequent phases):
- [ ] **Frontend Authentication UI (Login, Signup, User Menu)**: Phases 11–13
- [ ] **Final Deployment & Production Environment Configuration**: Phase 14



