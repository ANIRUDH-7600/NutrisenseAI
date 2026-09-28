# Security Architecture & Hardening — NutriSense AI

## 1. Overview & Threat Model
NutriSense AI is an AI-powered childhood malnutrition risk intelligence platform for community pre-screening under Scenario A. Because the system processes health and demographic indicators of children and identifies risk for Stunting, Underweight, and Wasting, security and privacy are fundamental engineering priorities.

This document details the security controls, defensive countermeasures, and architectural boundaries implemented across **Phases 1 through 7**.

> [!NOTE]
> In accordance with security engineering best practices, NutriSense AI does not claim to be "100% hack-proof" or "absolutely secure." Security is a continuous process of defense-in-depth, least-privilege access, rigorous automated testing, and threat surface minimization.

---

## 2. Defensive Controls Implemented

### A. Authentication & Credential Security
- **Bcrypt Password Hashing**: Passwords are never stored in plaintext. They are hashed using `bcrypt` with a cost factor of `12` (~250ms per derivation) and cryptographically random 128-bit salts.
- **Constant-Time Verification**: Verification utilizes `bcrypt.checkpw` to neutralize side-channel timing attacks.
- **Password Complexity**: Minimum 8 characters, maximum 128 characters (preventing DoS via hash saturation), requiring letters and numbers.
- **JWT Cryptographic Integrity**: Standardized JSON Web Tokens signed with `HS256` and environment-controlled `JWT_SECRET`.
- **Token Claims**: Contains `sub` (user_id), `email`, `role`, `iat`, `exp` (24h expiry), and unique `jti` (UUID).
- **Session Revocation & Logout**: Upon calling `POST /api/v1/auth/logout`, the token's `jti` is stored in MongoDB `revoked_tokens` with automated TTL pruning (`expireAfterSeconds=0`). Revoked tokens are immediately rejected across all endpoints with HTTP 401 (`TOKEN_REVOKED`).
- **Fail-Safe Offline Mode**: If the database is disconnected during token validation, requests fail safe with HTTP 503 (`DATABASE_UNAVAILABLE`) rather than allowing potentially revoked tokens.

### B. Authorization & IDOR / BOLA Prevention
- **Resource Ownership**: Every screening is permanently associated with `user_id = current_user["user_id"]`.
- **Database Query Scoping**:
  - `GET /api/v1/screenings` queries strictly by `{"user_id": current_user["user_id"]}`.
  - `GET /api/v1/screenings/{id}` and `DELETE /api/v1/screenings/{id}` query by `{"screening_id": id, "user_id": uid}`.
- **Safe 404 (Anti-Enumeration)**: Cross-user access attempts return HTTP 404 (`SCREENING_NOT_FOUND`) rather than HTTP 403, preventing malicious probing or enumeration of screening IDs.
- **Client Identity Spoofing Protection**: `ChildScreeningRequest` forbids unexpected fields (`extra = "forbid"`), and database persistence explicitly forces the verified caller's `user_id`.

### C. NoSQL Injection Defenses
- **Type-Enforced Queries**: In `backend/src/api/database.py`, all query arguments (`screening_id`, `user_id`, `email`, `jti`) are validated via `isinstance(arg, str)` checks before building MongoDB filters.
- **Dict & Operator Rejection**: Any attempted injection of MongoDB operators (e.g. `{"$gt": ""}`, `{"$ne": null}`) via JSON bodies is rejected immediately by Pydantic schema validation with HTTP 422.

### D. Brute-Force & Rate Limiting
- **Sliding-Window Limiter**: Implemented in `backend/src/api/rate_limiter.py` per client IP address.
- **Protected Routes**:
  - `POST /api/v1/auth/login` (configurable, default 60 requests/minute per IP)
  - `POST /api/v1/auth/register` (configurable, default 60 requests/minute per IP)
- **429 Response**: Exceeding rate limits triggers HTTP 429 (`TOO_MANY_REQUESTS`) with a standard `Retry-After` header.

### E. Request Schema Hardening & Boundary Enforcement
- **Strict Pydantic Validation**: All API request models forbid extraneous fields (`extra = "forbid"`).
- **Pagination Boundary Enforcement**: `limit` is bounded between 1 and 100 (`ge=1, le=100`), and `skip` is bounded (`ge=0, le=10000`).
- **Input Sanitation**: Categorical and numeric inputs are strictly validated against approved Scenario A clinical domains.

### F. HTTP Security Headers
All responses automatically include defensive HTTP security headers injected via middleware in `backend/src/api/main.py`:
- `X-Content-Type-Options: nosniff` (prevents MIME-sniffing attacks)
- `X-Frame-Options: DENY` (clickjacking protection)
- `X-XSS-Protection: 1; mode=block` (legacy XSS filtering)
- `Referrer-Policy: strict-origin-when-cross-origin` (prevents referrer leakage)
- `Permissions-Policy: geolocation=(), camera=(), microphone=()` (disables unused browser capabilities)
- `Content-Security-Policy: default-src 'self'` (restricts resource loading)
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` (enforces HTTPS)

### G. CORS Configuration
- **Environment-Controlled Origins**: Allowed origins are read from `NUTRISENSE_ALLOWED_ORIGINS` (defaults to local development origins `http://localhost:5173`, `http://127.0.0.1:5173`).
- **No Wildcard Credentials**: Wildcard `allow_origins=["*"]` is strictly avoided with `allow_credentials=True`.
- **Restricted Methods**: Preflight strictly restricts allowed methods to `["GET", "POST", "DELETE", "OPTIONS"]`.

### H. Error Handling & Privacy
- **Standardized Error Envelopes**: Structured JSON errors (`code`, `message`, `details`).
- **Zero Information Leakage**: Exception handlers catch unhandled exceptions and return generic HTTP 500 responses without exposing stack traces, Python internals, local file paths (`D:\...`), or MongoDB connection strings.

### I. Logging Privacy
- **Operational Telemetry Only**: Telemetry logs record only request ID, method, path, HTTP status, and duration in milliseconds.
- **Strict Prohibition**: Plaintext passwords, password hashes, JWT tokens, `Authorization` headers, and child screening features are strictly excluded from logs.

### J. Secret Management
- **Environment Variables**: JWT secrets and MongoDB connection URIs are loaded exclusively from `.env` via `python-dotenv`.
- **Git Hygiene**: `.gitignore` strictly excludes `.env` and `.env.*` files. No credentials or secrets are committed to version control.
- **Frontend Isolation**: React frontend communicates exclusively through REST API and contains no database credentials or backend secrets.

### K. DHS / NFHS-5 Research Microdata Protection
- **Research Data Isolation**: Raw survey data (`IAKR7EDT/`, `*.dta`, `*.dat`, `data/raw/*`) is strictly excluded from git and inaccessible via the REST API.
- **Runtime Independence**: The production LightGBM screening pipeline operates entirely in-memory and does not require raw DHS files to execute predictions.

### L. Machine Learning Model Integrity
- **Cryptographic Hashes**: Production model artifacts are verified against precomputed SHA-256 hashes defined in `models/model_registry.json`. Startup fails immediately if checksums do not match:
  - Stunting: `09aa09beb3d79d9b336afe585714ec05c7ae6618e1a0db197858129f9dc95191`
  - Underweight: `282e4fb0f2b294fb3bd2e1ea2f0f9c3d8a0b87f386475afc31c9800d78506159`
  - Wasting: `a68b5dc29cb98f2806458acefaf934aa44e57f9e8e81038dcd2fce5900287bb2`

---

## 3. Known Limitations & Future Work
1. **Distributed Rate Limiting**: The current sliding-window rate limiter is stored in-memory within the FastAPI process. In a horizontally scaled multi-worker deployment, an external Redis or MongoDB-backed rate limiter should be employed.
2. **Account Lockout Policies**: While rate limiting mitigates rapid brute-force attacks, progressive lockout after repeated consecutive credential failures per email will be considered in future releases.
3. **Multi-Factor Authentication (MFA)**: MFA / TOTP may be evaluated for higher-privileged clinical coordinator roles.
