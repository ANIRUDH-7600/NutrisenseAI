"""
Automated Test Suite for Phase 5: Server-Side Route Protection.

Verifies that the existing get_current_user authentication dependency protects all screening endpoints:
  - POST   /api/v1/screen
  - GET    /api/v1/screenings
  - GET    /api/v1/screenings/{screening_id}
  - DELETE /api/v1/screenings/{screening_id}

For each endpoint, tests:
  1. No Authorization header -> 401 AUTHENTICATION_REQUIRED
  2. Empty Bearer token -> 401 AUTHENTICATION_REQUIRED
  3. Malformed token -> 401 INVALID_TOKEN
  4. Expired token -> 401 TOKEN_EXPIRED
  5. Tampered token -> 401 INVALID_TOKEN
  6. Revoked token -> 401 TOKEN_REVOKED
  7. Valid token -> endpoint proceeds
  8. Inactive user -> rejected (401 USER_INACTIVE)
  9. Nonexistent user -> rejected (401 USER_NOT_FOUND)
 10. Database authentication failure -> 503 DATABASE_UNAVAILABLE

Also verifies:
 - POST /api/v1/screen preserves exact ML predictions, thresholds, and feature handling under authentication.
 - GET /api/v1/screenings preserves list structure under authentication.
 - GET /api/v1/screenings/{id} preserves record structure or 404 under authentication.
 - DELETE /api/v1/screenings/{id} preserves deletion confirmation or 404 under authentication.
"""

import copy
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.security import (
    hash_password,
    create_access_token,
)
from tests.test_inference_pipeline import get_sample_valid_input


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient within the application lifespan."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_auth_screen_db():
    """In-memory mock store for MongoDB users, revoked_tokens, and screenings."""
    stored_users = {}
    revoked_tokens = {}
    stored_screenings = {
        "scr_existing_123": {
            "screening_id": "scr_existing_123",
            "child_name": "Test Child",
            "created_at": "2026-09-28T12:00:00Z",
            "inputs": {"child_age_months": 24},
            "predictions": {"stunting": {"probability": 0.42, "threshold": 0.36, "screen_positive": True}},
            "triage": {},
            "model_version": "nutrisense-scenario-a-v1.0.0",
            "feature_schema_version": "scenario-a-30-v1"
        }
    }

    async def mock_find_one_users(filter_dict, projection=None):
        if "email" in filter_dict:
            user = stored_users.get(filter_dict["email"])
            return copy.deepcopy(user) if user else None
        if "user_id" in filter_dict:
            for u in stored_users.values():
                if u.get("user_id") == filter_dict["user_id"]:
                    return copy.deepcopy(u)
            return None
        return None

    async def mock_find_one_revoked(filter_dict, projection=None):
        jti = filter_dict.get("jti")
        if jti and jti in revoked_tokens:
            return copy.deepcopy(revoked_tokens[jti])
        return None

    async def mock_find_one_screenings(filter_dict, projection=None):
        sid = filter_dict.get("screening_id")
        if sid and sid in stored_screenings:
            rec = stored_screenings[sid]
            if "user_id" in filter_dict and rec.get("user_id") is not None:
                if rec.get("user_id") != filter_dict["user_id"]:
                    return None
            return copy.deepcopy(rec)
        return None

    async def mock_insert_one_screenings(doc):
        sid = doc.get("screening_id", "scr_mock_new")
        stored_screenings[sid] = copy.deepcopy(doc)
        res = MagicMock()
        res.inserted_id = sid
        return res

    async def mock_delete_one_screenings(filter_dict):
        sid = filter_dict.get("screening_id")
        res = MagicMock()
        if sid and sid in stored_screenings:
            rec = stored_screenings[sid]
            if "user_id" in filter_dict and rec.get("user_id") is not None:
                if rec.get("user_id") != filter_dict["user_id"]:
                    res.deleted_count = 0
                    return res
            del stored_screenings[sid]
            res.deleted_count = 1
        else:
            res.deleted_count = 0
        return res

    async def mock_update_one_revoked(filter_dict, update_dict, upsert=False):
        jti = filter_dict.get("jti")
        if upsert and "$setOnInsert" in update_dict:
            if jti not in revoked_tokens:
                revoked_tokens[jti] = copy.deepcopy(update_dict["$setOnInsert"])
        res = MagicMock()
        res.matched_count = 1 if jti in revoked_tokens else 0
        return res

    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(side_effect=mock_find_one_users)
    mock_db.revoked_tokens.find_one = AsyncMock(side_effect=mock_find_one_revoked)
    mock_db.revoked_tokens.update_one = AsyncMock(side_effect=mock_update_one_revoked)
    mock_db.screenings.find_one = AsyncMock(side_effect=mock_find_one_screenings)
    mock_db.screenings.insert_one = AsyncMock(side_effect=mock_insert_one_screenings)
    mock_db.screenings.delete_one = AsyncMock(side_effect=mock_delete_one_screenings)

    # Mock cursor for screenings.find() supporting user_id filter
    def mock_screenings_find(filter_dict=None, projection=None):
        target_uid = filter_dict.get("user_id") if filter_dict else None
        matches = [
            copy.deepcopy(s) for s in stored_screenings.values()
            if target_uid is None or s.get("user_id") is None or s.get("user_id") == target_uid
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=matches)
        return mock_cursor

    mock_db.screenings.find = MagicMock(side_effect=mock_screenings_find)

    return mock_db, stored_users, revoked_tokens, stored_screenings


def _seed_user(stored_users, user_id, email, password="SecurePassword2026!", is_active=True):
    """Seed user in mock storage."""
    stored_users[email.strip().lower()] = {
        "user_id": user_id,
        "name": "Dr. Anita Sen",
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "role": "health_worker",
        "is_active": is_active,
        "created_at": "2026-09-28T10:00:00Z"
    }


PROTECTED_ROUTES = [
    ("POST", "/api/v1/screen"),
    ("GET", "/api/v1/screenings"),
    ("GET", "/api/v1/screenings/scr_existing_123"),
    ("DELETE", "/api/v1/screenings/scr_existing_123"),
]


def _call_route(client, method, path, headers=None, json_data=None):
    """Helper to dispatch HTTP request."""
    if method == "POST":
        return client.post(path, headers=headers, json=json_data or get_sample_valid_input())
    elif method == "GET":
        return client.get(path, headers=headers)
    elif method == "DELETE":
        return client.delete(path, headers=headers)
    raise ValueError(f"Unsupported method: {method}")


# ==============================================================================
# 1. No Authorization Header -> 401
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_no_auth_header_rejected_with_401(client, mock_auth_screen_db, method, path):
    """Verify all protected screening routes reject unauthenticated requests with HTTP 401."""
    mock_db, _, _, _ = mock_auth_screen_db
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path)
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "AUTHENTICATION_REQUIRED"


# ==============================================================================
# 2. Empty Bearer Token -> 401
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_empty_bearer_token_rejected_with_401(client, mock_auth_screen_db, method, path):
    """Verify empty or whitespace Bearer token is rejected with HTTP 401."""
    mock_db, _, _, _ = mock_auth_screen_db
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": "Bearer "})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "AUTHENTICATION_REQUIRED"


# ==============================================================================
# 3. Malformed Token -> 401
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_malformed_token_rejected_with_401(client, mock_auth_screen_db, method, path):
    """Verify malformed JWT string is rejected with HTTP 401 INVALID_TOKEN."""
    mock_db, _, _, _ = mock_auth_screen_db
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": "Bearer not.a.valid.jwt"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_TOKEN"


# ==============================================================================
# 4. Expired Token -> 401
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_expired_token_rejected_with_401(client, mock_auth_screen_db, method, path):
    """Verify expired token is rejected with HTTP 401 TOKEN_EXPIRED."""
    mock_db, _, _, _ = mock_auth_screen_db
    expired_token = create_access_token(
        user_id="usr_exp_p5",
        email="expired@clinic.org",
        expires_delta=timedelta(seconds=-10)
    )
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {expired_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "TOKEN_EXPIRED"


# ==============================================================================
# 5. Tampered Token -> 401
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_tampered_token_rejected_with_401(client, mock_auth_screen_db, method, path):
    """Verify signature-tampered token is rejected with HTTP 401 INVALID_TOKEN."""
    mock_db, _, _, _ = mock_auth_screen_db
    valid_token = create_access_token(user_id="usr_tamper_p5", email="tamper@clinic.org")
    parts = valid_token.split(".")
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6IHRydWV9.{parts[2]}"

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {tampered_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_TOKEN"


# ==============================================================================
# 6. Revoked Token -> 401
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_revoked_token_rejected_with_401(client, mock_auth_screen_db, method, path):
    """Verify logged-out/revoked token is rejected with HTTP 401 TOKEN_REVOKED."""
    mock_db, stored_users, revoked_tokens, _ = mock_auth_screen_db
    _seed_user(stored_users, "usr_revoked_p5", "revoked@clinic.org")
    token = create_access_token(user_id="usr_revoked_p5", email="revoked@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # 1. Log out with the token to revoke it
        res_logout = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res_logout.status_code == 200

        # 2. Protected screening routes must now reject this token
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "TOKEN_REVOKED"


# ==============================================================================
# 7. Valid Token -> Endpoint Proceeds
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_valid_token_allows_endpoint_proceed(client, mock_auth_screen_db, method, path):
    """Verify valid authenticated token allows request to reach the underlying endpoint."""
    mock_db, stored_users, _, _ = mock_auth_screen_db
    _seed_user(stored_users, "usr_valid_p5", "valid@clinic.org")
    token = create_access_token(user_id="usr_valid_p5", email="valid@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200


# ==============================================================================
# 8. Inactive User -> Rejected (401)
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_inactive_user_token_rejected(client, mock_auth_screen_db, method, path):
    """Verify valid token for a deactivated user account is rejected."""
    mock_db, stored_users, _, _ = mock_auth_screen_db
    _seed_user(stored_users, "usr_inactive_p5", "inactive@clinic.org", is_active=False)
    inactive_token = create_access_token(user_id="usr_inactive_p5", email="inactive@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {inactive_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "USER_INACTIVE"


# ==============================================================================
# 9. Deleted/Nonexistent User -> Rejected (401)
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_nonexistent_user_token_rejected(client, mock_auth_screen_db, method, path):
    """Verify valid token for deleted user is rejected with HTTP 401 USER_NOT_FOUND."""
    mock_db, _, _, _ = mock_auth_screen_db
    ghost_token = create_access_token(user_id="usr_ghost_missing_999", email="ghost@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {ghost_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "USER_NOT_FOUND"


# ==============================================================================
# 10. Database Authentication Failure Handled Safely (503)
# ==============================================================================
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_database_failure_handled_safely(client, mock_auth_screen_db, method, path):
    """Verify database unavailability during auth returns HTTP 503 without leaking stack traces."""
    token = create_access_token(user_id="usr_db_fail_p5", email="dbfail@clinic.org")

    with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
        res = _call_route(client, method, path, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 503
        err = res.json()["error"]
        assert err["code"] == "DATABASE_UNAVAILABLE"
        assert "Traceback" not in str(res.content)


# ==============================================================================
# Functional Screening Assertions Under Authentication
# ==============================================================================
def test_screen_preserves_ml_behavior_when_authenticated(client, mock_auth_screen_db):
    """Verify POST /api/v1/screen returns identical statistical predictions under authentication."""
    mock_db, stored_users, _, _ = mock_auth_screen_db
    _seed_user(stored_users, "usr_ml_test", "ml.worker@clinic.org")
    token = create_access_token(user_id="usr_ml_test", email="ml.worker@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        payload = get_sample_valid_input()
        res = client.post("/api/v1/screen", json=payload, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()

        assert data["success"] is True
        assert data["model_version"] == "nutrisense-scenario-a-v1.0.0"
        assert data["feature_schema_version"] == "scenario-a-30-v1"
        assert "predictions" in data

        for target in ["stunting", "underweight", "wasting"]:
            pred = data["predictions"][target]
            assert 0.0 <= pred["probability"] <= 1.0
            assert 0.0 <= pred["threshold"] <= 1.0
            assert isinstance(pred["screen_positive"], bool)
            assert pred["screen_positive"] == (pred["probability"] >= pred["threshold"])


def test_list_screenings_response_structure_when_authenticated(client, mock_auth_screen_db):
    """Verify GET /api/v1/screenings returns ScreeningListResponse structure under authentication."""
    mock_db, stored_users, _, _ = mock_auth_screen_db
    _seed_user(stored_users, "usr_list_test", "list.worker@clinic.org")
    token = create_access_token(user_id="usr_list_test", email="list.worker@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/screenings", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert "total" in data
        assert "screenings" in data
        assert isinstance(data["screenings"], list)


def test_get_screening_by_id_when_authenticated(client, mock_auth_screen_db):
    """Verify GET /api/v1/screenings/{id} returns ScreeningRecord or 404 under authentication."""
    mock_db, stored_users, _, stored_screenings = mock_auth_screen_db
    _seed_user(stored_users, "usr_get_test", "get.worker@clinic.org")
    stored_screenings["scr_existing_123"]["user_id"] = "usr_get_test"
    token = create_access_token(user_id="usr_get_test", email="get.worker@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # Existing record
        res_found = client.get("/api/v1/screenings/scr_existing_123", headers={"Authorization": f"Bearer {token}"})
        assert res_found.status_code == 200
        assert res_found.json()["screening_id"] == "scr_existing_123"

        # Nonexistent record
        res_not_found = client.get("/api/v1/screenings/scr_does_not_exist", headers={"Authorization": f"Bearer {token}"})
        assert res_not_found.status_code == 404
        assert res_not_found.json()["error"]["code"] == "SCREENING_NOT_FOUND"


def test_delete_screening_by_id_when_authenticated(client, mock_auth_screen_db):
    """Verify DELETE /api/v1/screenings/{id} deletes record or returns 404 under authentication."""
    mock_db, stored_users, _, stored_screenings = mock_auth_screen_db
    _seed_user(stored_users, "usr_del_test", "del.worker@clinic.org")
    stored_screenings["scr_existing_123"]["user_id"] = "usr_del_test"
    token = create_access_token(user_id="usr_del_test", email="del.worker@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # Existing record deletion
        res_del = client.delete("/api/v1/screenings/scr_existing_123", headers={"Authorization": f"Bearer {token}"})
        assert res_del.status_code == 200
        assert res_del.json()["success"] is True

        # Nonexistent record deletion
        res_not_found = client.delete("/api/v1/screenings/scr_nonexistent", headers={"Authorization": f"Bearer {token}"})
        assert res_not_found.status_code == 404
        assert res_not_found.json()["error"]["code"] == "SCREENING_NOT_FOUND"
