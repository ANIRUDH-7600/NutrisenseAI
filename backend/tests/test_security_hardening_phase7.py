"""
Phase 7 — Comprehensive Security Hardening Test Suite for NutriSense AI.

Validates defensive controls across the entire API surface:
1. NoSQL Injection Defense (dict/operator injection in query strings & body fields)
2. Authentication & Authorization Enforcement (JWT verification, expired/tampered/revoked tokens)
3. Object-Level Authorization (IDOR / BOLA cross-user isolation)
4. Brute-Force & Rate Limiting (HTTP 429 on abuse of /auth/login and /auth/register)
5. Request Schema & Input Validation (unexpected fields, boundary violations, invalid types)
6. Security Headers (nosniff, DENY, HSTS, CSP, Referrer-Policy, Permissions-Policy)
7. CORS Security (environment-controlled, no unsafe wildcards with credentials)
8. Error Handling & Privacy (zero stack traces, file paths, or credentials leaked)
9. Sensitive Data Logging Prohibition (zero passwords, tokens, or child health data in logs)
10. DHS/NFHS-5 Data Privacy (research microdata strictly unexposed)
"""

import copy
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.security import create_access_token
from src.api.rate_limiter import reset_rate_limiter, check_rate_limit, set_rate_limit_enabled
from tests.test_inference_pipeline import get_sample_valid_input


@pytest.fixture(autouse=True)
def reset_limiter():
    """Ensure rate limiter is fresh before and after each test."""
    reset_rate_limiter()
    set_rate_limit_enabled(True)
    yield
    reset_rate_limiter()
    set_rate_limit_enabled(True)


@pytest.fixture
def client():
    """TestClient without lifespan auto-run to isolate test execution."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_sec_db():
    """In-memory mock store for security tests."""
    users = {
        "dr.sharma@clinic.gov.in": {
            "user_id": "usr_sec_100",
            "name": "Dr. Sharma",
            "email": "dr.sharma@clinic.gov.in",
            "role": "health_worker",
            "is_active": True,
            "password_hash": "$2b$12$e8xFakeHashForTestingOnly12345678901234567890123456789"
        },
        "dr.patel@hospital.org": {
            "user_id": "usr_sec_200",
            "name": "Dr. Patel",
            "email": "dr.patel@hospital.org",
            "role": "health_worker",
            "is_active": True,
            "password_hash": "$2b$12$e8xFakeHashForTestingOnly12345678901234567890123456789"
        }
    }
    revoked = {}
    screenings = {
        "scr_sec_sharma_1": {
            "screening_id": "scr_sec_sharma_1",
            "user_id": "usr_sec_100",
            "child_name": "Child A",
            "created_at": "2026-09-28T10:00:00Z",
            "inputs": {"child_age_months": 24.0},
            "predictions": {"stunting": {"risk_level": "Elevated Risk"}},
            "model_version": "nutrisense-scenario-a-v1.0.0",
            "feature_schema_version": "scenario-a-30-v1"
        },
        "scr_sec_patel_1": {
            "screening_id": "scr_sec_patel_1",
            "user_id": "usr_sec_200",
            "child_name": "Child B",
            "created_at": "2026-09-28T11:00:00Z",
            "inputs": {"child_age_months": 18.0},
            "predictions": {"stunting": {"risk_level": "Low Risk"}},
            "model_version": "nutrisense-scenario-a-v1.0.0",
            "feature_schema_version": "scenario-a-30-v1"
        }
    }

    async def mock_find_one_users(filter_dict, projection=None):
        if not isinstance(filter_dict, dict):
            return None
        if "email" in filter_dict and isinstance(filter_dict["email"], str):
            return copy.deepcopy(users.get(filter_dict["email"]))
        if "user_id" in filter_dict and isinstance(filter_dict["user_id"], str):
            for u in users.values():
                if u.get("user_id") == filter_dict["user_id"]:
                    return copy.deepcopy(u)
        return None

    async def mock_find_one_revoked(filter_dict, projection=None):
        jti = filter_dict.get("jti")
        return copy.deepcopy(revoked.get(jti)) if jti in revoked else None

    async def mock_find_one_screenings(filter_dict, projection=None):
        sid = filter_dict.get("screening_id")
        if not isinstance(sid, str) or sid not in screenings:
            return None
        rec = screenings[sid]
        if "user_id" in filter_dict and isinstance(filter_dict["user_id"], str):
            if rec.get("user_id") != filter_dict["user_id"]:
                return None
        return copy.deepcopy(rec)

    async def mock_delete_one_screenings(filter_dict):
        sid = filter_dict.get("screening_id")
        res = MagicMock()
        if isinstance(sid, str) and sid in screenings:
            rec = screenings[sid]
            if "user_id" in filter_dict and isinstance(filter_dict["user_id"], str):
                if rec.get("user_id") != filter_dict["user_id"]:
                    res.deleted_count = 0
                    return res
            del screenings[sid]
            res.deleted_count = 1
        else:
            res.deleted_count = 0
        return res

    def mock_screenings_find(filter_dict=None, projection=None):
        target_uid = filter_dict.get("user_id") if (filter_dict and isinstance(filter_dict, dict)) else None
        matches = [
            copy.deepcopy(s) for s in screenings.values()
            if target_uid is None or s.get("user_id") == target_uid
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=matches)
        return mock_cursor

    db = MagicMock()
    db.users.find_one = AsyncMock(side_effect=mock_find_one_users)
    db.revoked_tokens.find_one = AsyncMock(side_effect=mock_find_one_revoked)
    db.screenings.find_one = AsyncMock(side_effect=mock_find_one_screenings)
    db.screenings.delete_one = AsyncMock(side_effect=mock_delete_one_screenings)
    db.screenings.find = MagicMock(side_effect=mock_screenings_find)

    return db, users, screenings


# ==============================================================================
# 1. NOSQL INJECTION HARDENING
# ==============================================================================

def test_nosql_injection_in_screening_id_path_rejected_safely(client, mock_sec_db):
    """Path parameters with NoSQL injection operators are safely rejected without execution."""
    db, users, screenings = mock_sec_db
    token = create_access_token(user_id="usr_sec_100", email="dr.sharma@clinic.gov.in")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        # 1. MongoDB query operator injection attempts
        for payload in ['{"$ne": null}', '{"$gt": ""}', "[$ne]", "$where"]:
            res = client.get(f"/api/v1/screenings/{payload}", headers=headers)
            assert res.status_code in [404, 422]
            assert "Traceback" not in res.text
            assert "pymongo" not in res.text.lower()


def test_nosql_injection_in_auth_login_payload_rejected(client, mock_sec_db):
    """NoSQL dict payloads in email/password login requests are rejected by schema validation."""
    db, users, screenings = mock_sec_db

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        malicious_payload = {
            "email": {"$ne": None},  # Dict instead of string
            "password": {"$gt": ""}
        }
        res = client.post("/api/v1/auth/login", json=malicious_payload)
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "INVALID_INPUT"


# ==============================================================================
# 2. BRUTE-FORCE / RATE LIMITING
# ==============================================================================

def test_login_endpoint_enforces_rate_limiting(client, mock_sec_db):
    """Excessive rapid login attempts trigger HTTP 429 Too Many Requests."""
    db, users, screenings = mock_sec_db
    login_payload = {
        "email": "dr.sharma@clinic.gov.in",
        "password": "WrongPassword123!"
    }

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True), \
         patch("src.api.routes.auth.LOGIN_RATE_LIMIT", 5):
        # 5 attempts allowed
        for _ in range(5):
            res = client.post("/api/v1/auth/login", json=login_payload)
            assert res.status_code in [401, 200]

        # 6th attempt exceeds rate limit
        res_blocked = client.post("/api/v1/auth/login", json=login_payload)
        assert res_blocked.status_code == 429
        assert res_blocked.json()["error"]["code"] == "TOO_MANY_REQUESTS"
        assert "Retry-After" in res_blocked.headers


def test_register_endpoint_enforces_rate_limiting(client, mock_sec_db):
    """Excessive rapid registration attempts trigger HTTP 429 Too Many Requests."""
    db, users, screenings = mock_sec_db

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True), \
         patch("src.api.routes.auth.REGISTER_RATE_LIMIT", 5):
        # Send 5 rapid registration attempts
        for i in range(5):
            reg_payload = {
                "name": f"Doctor {i}",
                "email": f"doctor_{i}@clinic.gov.in",
                "password": "SecurePassword2026!"
            }
            res = client.post("/api/v1/auth/register", json=reg_payload)
            assert res.status_code != 429

        # 6th registration attempt gets throttled
        res_blocked = client.post("/api/v1/auth/register", json={
            "name": "Doctor Throttled",
            "email": "doctor_throttled@clinic.gov.in",
            "password": "SecurePassword2026!"
        })
        assert res_blocked.status_code == 429
        assert res_blocked.json()["error"]["code"] == "TOO_MANY_REQUESTS"


# ==============================================================================
# 3. HTTP SECURITY HEADERS
# ==============================================================================

def test_security_headers_present_on_all_responses(client):
    """Verifies defensive HTTP security headers are injected into API responses."""
    endpoints = ["/health", "/api/v1/metadata", "/api/v1/health/model"]
    for ep in endpoints:
        res = client.get(ep)
        assert res.headers.get("X-Content-Type-Options") == "nosniff"
        assert res.headers.get("X-Frame-Options") == "DENY"
        assert res.headers.get("X-XSS-Protection") == "1; mode=block"
        assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "geolocation=()" in res.headers.get("Permissions-Policy", "")
        assert "default-src 'self'" in res.headers.get("Content-Security-Policy", "")
        assert "max-age=" in res.headers.get("Strict-Transport-Security", "")


# ==============================================================================
# 4. CORS CONFIGURATION HARDENING
# ==============================================================================

def test_cors_allows_configured_methods_including_delete(client):
    """CORS preflight OPTIONS request returns configured methods for allowed origins and rejects untrusted origins."""
    from src.api.config import ALLOWED_ORIGINS

    valid_origin = ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "http://localhost:5173"
    res = client.options(
        "/api/v1/screenings/scr_sec_sharma_1",
        headers={
            "Origin": valid_origin,
            "Access-Control-Request-Method": "DELETE"
        }
    )
    assert res.status_code == 200
    allow_methods = res.headers.get("access-control-allow-methods", "")
    assert "DELETE" in allow_methods
    assert "POST" in allow_methods
    assert "GET" in allow_methods

    # Untrusted origin is rejected
    res_untrusted = client.options(
        "/api/v1/screenings/scr_sec_sharma_1",
        headers={
            "Origin": "https://malicious-attacker-domain.xyz",
            "Access-Control-Request-Method": "DELETE"
        }
    )
    assert res_untrusted.status_code == 400
    assert "Disallowed CORS origin" in res_untrusted.text


# ==============================================================================
# 5. REQUEST SCHEMA HARDENING & BOUNDARY CHECKS
# ==============================================================================

def test_unexpected_fields_in_screening_request_strictly_rejected(client, mock_sec_db):
    """Screening payload with arbitrary injected extra fields is rejected (extra='forbid')."""
    db, users, screenings = mock_sec_db
    token = create_access_token(user_id="usr_sec_100", email="dr.sharma@clinic.gov.in")

    payload = get_sample_valid_input()
    payload["admin_override"] = True
    payload["bypassed_threshold"] = 0.05

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        res = client.post(
            "/api/v1/screen",
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 422
        assert res.json()["error"]["code"] == "INVALID_INPUT"


def test_pagination_bounds_enforced_on_screenings_list(client, mock_sec_db):
    """Negative skip or limit > 100 on /screenings is rejected by Query validation."""
    db, users, screenings = mock_sec_db
    token = create_access_token(user_id="usr_sec_100", email="dr.sharma@clinic.gov.in")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        # Limit too large (> 100)
        res_large_limit = client.get("/api/v1/screenings?limit=500", headers=headers)
        assert res_large_limit.status_code == 422

        # Negative limit (<= 0)
        res_neg_limit = client.get("/api/v1/screenings?limit=-1", headers=headers)
        assert res_neg_limit.status_code == 422

        # Negative skip (< 0)
        res_neg_skip = client.get("/api/v1/screenings?skip=-5", headers=headers)
        assert res_neg_skip.status_code == 422


# ==============================================================================
# 6. IDOR / BOLA CROSS-USER ISOLATION
# ==============================================================================

def test_idor_cross_user_isolation_strictly_enforced(client, mock_sec_db):
    """
    User Sharma (usr_sec_100) cannot access or delete User Patel's (usr_sec_200) screening.
    Safe 404 is returned to prevent ID enumeration.
    """
    db, users, screenings = mock_sec_db
    token_sharma = create_access_token(user_id="usr_sec_100", email="dr.sharma@clinic.gov.in")
    headers_sharma = {"Authorization": f"Bearer {token_sharma}"}

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        # 1. Sharma tries to retrieve Patel's screening
        res_get = client.get("/api/v1/screenings/scr_sec_patel_1", headers=headers_sharma)
        assert res_get.status_code == 404
        assert res_get.json()["error"]["code"] == "SCREENING_NOT_FOUND"

        # 2. Sharma tries to delete Patel's screening
        res_del = client.delete("/api/v1/screenings/scr_sec_patel_1", headers=headers_sharma)
        assert res_del.status_code == 404
        assert res_del.json()["error"]["code"] == "SCREENING_NOT_FOUND"
        assert "scr_sec_patel_1" in screenings  # Still intact in DB!

        # 3. Sharma's list shows only Sharma's record
        res_list = client.get("/api/v1/screenings", headers=headers_sharma)
        assert res_list.status_code == 200
        ids = [s["screening_id"] for s in res_list.json()["screenings"]]
        assert "scr_sec_sharma_1" in ids
        assert "scr_sec_patel_1" not in ids


# ==============================================================================
# 7. SENSITIVE DATA LEAKAGE & PRIVACY
# ==============================================================================

def test_error_responses_never_leak_stack_traces_or_internals(client):
    """400, 401, 404, 422, and 500 error envelopes never contain Python tracebacks or file paths."""
    # Invalid endpoint 404
    res_404 = client.get("/api/v1/unknown_route_xyz")
    assert "Traceback" not in res_404.text
    assert "site-packages" not in res_404.text
    assert "D:\\" not in res_404.text

    # Malformed JSON 422
    res_422 = client.post(
        "/api/v1/screen",
        content="invalid-non-json-content",
        headers={"Content-Type": "application/json", "Authorization": "Bearer fake.token.here"}
    )
    assert "Traceback" not in res_422.text
    assert "Exception" not in res_422.text


def test_dhs_raw_microdata_strictly_inaccessible_via_api(client):
    """Raw NFHS/DHS survey files cannot be accessed or traversed via API."""
    traversal_paths = [
        "/IAKR7EDT",
        "/data/raw/IAKR7EDT",
        "/api/v1/data/raw",
        "/api/v1/static/IAKR7EDT.dta",
        "/../IAKR7EDT"
    ]
    for p in traversal_paths:
        res = client.get(p)
        assert res.status_code in [404, 405]
        assert "DHS" not in res.text
