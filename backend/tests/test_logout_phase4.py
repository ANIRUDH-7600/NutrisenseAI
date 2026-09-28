"""
Automated Test Suite for Phase 4: Logout & Token Revocation.

Covers:
 1. Successful logout
 2. Logout requires authentication
 3. Missing token rejected
 4. Invalid token rejected
 5. Expired token rejected
 6. Tampered token rejected
 7. Token becomes unusable after logout
 8. Logout does not invalidate another user's token
 9. Logout does not expose the JWT
10. Logout does not store the raw JWT
11. Revoked token rejected by protected authentication dependency
12. Database failure handled safely
13. Repeated logout handled safely
14. Revocation record contains jti/user_id but not raw JWT
15. Expired revocation records can be cleaned up according to the chosen design
"""

import copy
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError

from src.api.main import app
from src.api.security import (
    hash_password,
    create_access_token,
    get_current_user,
)
from src.api.database import clean_expired_revocations


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient within the application lifespan."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_auth_db():
    """In-memory mock store for MongoDB users and revoked_tokens collections."""
    stored_users = {}
    revoked_tokens = {}

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

    async def mock_update_one_revoked(filter_dict, update_dict, upsert=False):
        jti = filter_dict.get("jti")
        if upsert and "$setOnInsert" in update_dict:
            if jti not in revoked_tokens:
                revoked_tokens[jti] = copy.deepcopy(update_dict["$setOnInsert"])
        res = MagicMock()
        res.matched_count = 1 if jti in revoked_tokens else 0
        return res

    async def mock_delete_many_revoked(filter_dict):
        lte_val = filter_dict.get("expires_at", {}).get("$lte")
        deleted_count = 0
        to_delete = []
        for j, rec in revoked_tokens.items():
            if lte_val and rec.get("expires_at") <= lte_val:
                to_delete.append(j)
        for j in to_delete:
            del revoked_tokens[j]
            deleted_count += 1
        res = MagicMock()
        res.deleted_count = deleted_count
        return res

    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(side_effect=mock_find_one_users)
    mock_db.revoked_tokens.find_one = AsyncMock(side_effect=mock_find_one_revoked)
    mock_db.revoked_tokens.update_one = AsyncMock(side_effect=mock_update_one_revoked)
    mock_db.revoked_tokens.delete_many = AsyncMock(side_effect=mock_delete_many_revoked)

    return mock_db, stored_users, revoked_tokens


def _seed_user(stored_users, user_id, email, password="SecurePassword2026!", role="health_worker", is_active=True):
    """Helper to seed user record into mock database."""
    stored_users[email.strip().lower()] = {
        "user_id": user_id,
        "name": "Dr. Sunita Sen",
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "role": role,
        "is_active": is_active,
        "created_at": "2026-09-28T10:00:00Z",
        "updated_at": "2026-09-28T10:00:00Z"
    }


# ==============================================================================
# 1. Successful Logout
# ==============================================================================
def test_successful_logout(client, mock_auth_db):
    """Test 1: Verify successful logout returns HTTP 200 with LogoutResponse."""
    mock_db, stored_users, revoked_tokens = mock_auth_db
    _seed_user(stored_users, "usr_logout_001", "worker.logout@clinic.org")
    token = create_access_token(user_id="usr_logout_001", email="worker.logout@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["message"] == "Successfully logged out."
        assert len(revoked_tokens) == 1


# ==============================================================================
# 2. Logout Requires Authentication
# ==============================================================================
def test_logout_requires_authentication(client, mock_auth_db):
    """Test 2: Calling logout with no token returns HTTP 401 AUTHENTICATION_REQUIRED."""
    mock_db, _, _ = mock_auth_db
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout")
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "AUTHENTICATION_REQUIRED"


# ==============================================================================
# 3. Missing Token Rejected
# ==============================================================================
def test_missing_token_rejected(client, mock_auth_db):
    """Test 3: Calling logout with empty Bearer header returns HTTP 401."""
    mock_db, _, _ = mock_auth_db
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer "})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "AUTHENTICATION_REQUIRED"


# ==============================================================================
# 4. Invalid Token Rejected
# ==============================================================================
def test_invalid_token_rejected(client, mock_auth_db):
    """Test 4: Calling logout with malformed token returns HTTP 401 INVALID_TOKEN."""
    mock_db, _, _ = mock_auth_db
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer totally.invalid.token"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_TOKEN"


# ==============================================================================
# 5. Expired Token Rejected
# ==============================================================================
def test_expired_token_rejected(client, mock_auth_db):
    """Test 5: Calling logout with expired token returns HTTP 401 TOKEN_EXPIRED."""
    mock_db, _, _ = mock_auth_db
    expired_token = create_access_token(
        user_id="usr_exp_005",
        email="expired@clinic.org",
        expires_delta=timedelta(seconds=-10)
    )

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {expired_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "TOKEN_EXPIRED"


# ==============================================================================
# 6. Tampered Token Rejected
# ==============================================================================
def test_tampered_token_rejected(client, mock_auth_db):
    """Test 6: Calling logout with signature-tampered token returns HTTP 401 INVALID_TOKEN."""
    mock_db, _, _ = mock_auth_db
    valid_token = create_access_token(user_id="usr_tamper_006", email="tamper@clinic.org")
    parts = valid_token.split(".")
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6IHRydWV9.{parts[2]}"

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {tampered_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_TOKEN"


# ==============================================================================
# 7. Token Becomes Unusable After Logout
# ==============================================================================
def test_token_becomes_unusable_after_logout(client, mock_auth_db):
    """Test 7: Verify token works prior to logout, but returns HTTP 401 TOKEN_REVOKED after logout."""
    mock_db, stored_users, _ = mock_auth_db
    _seed_user(stored_users, "usr_cycle_007", "lifecycle@clinic.org")
    token = create_access_token(user_id="usr_cycle_007", email="lifecycle@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # 1. Verify token works on /auth/me
        res_before = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_before.status_code == 200

        # 2. Log out
        res_logout = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res_logout.status_code == 200

        # 3. Token is now unusable
        res_after = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_after.status_code == 401
        err = res_after.json()["error"]
        assert err["code"] == "TOKEN_REVOKED"


# ==============================================================================
# 8. Logout Does Not Invalidate Another User's Token
# ==============================================================================
def test_logout_does_not_invalidate_another_user_token(client, mock_auth_db):
    """Test 8: Logging out User A must NOT revoke User B's token or session."""
    mock_db, stored_users, _ = mock_auth_db
    _seed_user(stored_users, "usr_a_008", "user.a@clinic.org")
    _seed_user(stored_users, "usr_b_008", "user.b@clinic.org")

    token_a = create_access_token(user_id="usr_a_008", email="user.a@clinic.org")
    token_b = create_access_token(user_id="usr_b_008", email="user.b@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # User A logs out
        res_logout_a = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token_a}"})
        assert res_logout_a.status_code == 200

        # User A is revoked
        res_me_a = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token_a}"})
        assert res_me_a.status_code == 401
        assert res_me_a.json()["error"]["code"] == "TOKEN_REVOKED"

        # User B remains completely valid and active
        res_me_b = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token_b}"})
        assert res_me_b.status_code == 200
        assert res_me_b.json()["email"] == "user.b@clinic.org"


# ==============================================================================
# 9. Logout Does Not Expose the JWT
# ==============================================================================
def test_logout_does_not_expose_jwt(client, mock_auth_db):
    """Test 9: Verify raw JWT is strictly absent from the logout response body and response headers."""
    mock_db, stored_users, _ = mock_auth_db
    _seed_user(stored_users, "usr_audit_009", "audit.jwt@clinic.org")
    token = create_access_token(user_id="usr_audit_009", email="audit.jwt@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert token not in str(res.content)
        for h_val in res.headers.values():
            assert token not in h_val


# ==============================================================================
# 10. Logout Does Not Store the Raw JWT
# ==============================================================================
def test_logout_does_not_store_raw_jwt(client, mock_auth_db):
    """Test 10: Verify the database revocation store contains only jti metadata, never raw JWT."""
    mock_db, stored_users, revoked_tokens = mock_auth_db
    _seed_user(stored_users, "usr_db_check_010", "db.check@clinic.org")
    token = create_access_token(user_id="usr_db_check_010", email="db.check@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

        # Inspect stored revocation records
        for record in revoked_tokens.values():
            assert "access_token" not in record
            assert "token" not in record
            assert "jwt" not in record
            assert token not in str(record)


# ==============================================================================
# 11. Revoked Token Rejected by Protected Authentication Dependency
# ==============================================================================
def test_revoked_token_rejected_by_auth_dependency(mock_auth_db):
    """Test 11: Unit test verifying get_current_user directly rejects revoked token."""
    import asyncio
    from fastapi.security import HTTPAuthorizationCredentials
    from fastapi import HTTPException

    mock_db, stored_users, revoked_tokens = mock_auth_db
    _seed_user(stored_users, "usr_dep_011", "dep.check@clinic.org")
    token = create_access_token(user_id="usr_dep_011", email="dep.check@clinic.org")

    async def run():
        with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
            # First verify valid token passes
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            user_data = await get_current_user(credentials=creds)
            assert user_data["user_id"] == "usr_dep_011"

            # Manually revoke the token's jti
            from src.api.security import decode_access_token
            payload = decode_access_token(token)
            revoked_tokens[payload["jti"]] = {
                "jti": payload["jti"],
                "user_id": "usr_dep_011",
                "revoked_at": datetime.now(timezone.utc),
                "expires_at": datetime.fromtimestamp(payload["exp"], timezone.utc)
            }

            # Now verify get_current_user rejects with TOKEN_REVOKED
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(credentials=creds)
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail.get("code") == "TOKEN_REVOKED"

    asyncio.run(run())


# ==============================================================================
# 12. Database Failure Handled Safely
# ==============================================================================
def test_database_failure_handled_safely(client, mock_auth_db):
    """Test 12: Verify database unavailability returns HTTP 503 without leaking stack traces or crashing."""
    token = create_access_token(user_id="usr_fail_012", email="fail@clinic.org")

    # Subtest A: Database disconnected on logout
    with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
        res_disc = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res_disc.status_code == 503
        err = res_disc.json()["error"]
        assert err["code"] == "DATABASE_UNAVAILABLE"
        assert "Traceback" not in str(res_disc.content)

    # Subtest B: MongoDB cluster failure during revocation write
    mock_db = MagicMock()
    mock_db.revoked_tokens.update_one = AsyncMock(side_effect=PyMongoError("Cluster write quorum unavailable"))
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res_err = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res_err.status_code == 503
        err = res_err.json()["error"]
        assert err["code"] == "DATABASE_UNAVAILABLE"
        assert "Traceback" not in str(res_err.content)


# ==============================================================================
# 13. Repeated Logout Handled Safely
# ==============================================================================
def test_repeated_logout_handled_safely(client, mock_auth_db):
    """Test 13: Calling logout multiple times with same token succeeds idempotently without error."""
    mock_db, stored_users, _ = mock_auth_db
    _seed_user(stored_users, "usr_repeat_013", "repeat@clinic.org")
    token = create_access_token(user_id="usr_repeat_013", email="repeat@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res1 = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res1.status_code == 200

        res2 = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res2.status_code == 200
        assert res2.json()["success"] is True


# ==============================================================================
# 14. Revocation Record Contains jti/user_id But Not Raw JWT
# ==============================================================================
def test_revocation_record_contents(client, mock_auth_db):
    """Test 14: Verify exact fields of stored revocation record: jti, user_id, revoked_at, expires_at."""
    mock_db, stored_users, revoked_tokens = mock_auth_db
    _seed_user(stored_users, "usr_fields_014", "fields@clinic.org")
    token = create_access_token(user_id="usr_fields_014", email="fields@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

        record = list(revoked_tokens.values())[0]
        assert "jti" in record
        assert "user_id" in record
        assert record["user_id"] == "usr_fields_014"
        assert "revoked_at" in record
        assert "expires_at" in record
        assert "token" not in record
        assert "access_token" not in record


# ==============================================================================
# 15. Expired Revocation Records Cleaned Up
# ==============================================================================
def test_clean_expired_revocations_removes_old_entries(mock_auth_db):
    """Test 15: Verify clean_expired_revocations prunes expired revocation documents."""
    import asyncio

    mock_db, _, revoked_tokens = mock_auth_db
    now = datetime.now(timezone.utc)

    # Entry 1: Expired 1 hour ago
    revoked_tokens["jti_expired_1"] = {
        "jti": "jti_expired_1",
        "user_id": "usr_old",
        "revoked_at": now - timedelta(hours=2),
        "expires_at": now - timedelta(hours=1),
    }

    # Entry 2: Still active (expires in 12 hours)
    revoked_tokens["jti_active_2"] = {
        "jti": "jti_active_2",
        "user_id": "usr_current",
        "revoked_at": now,
        "expires_at": now + timedelta(hours=12),
    }

    async def run():
        with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
            deleted = await clean_expired_revocations()
            assert deleted == 1
            assert "jti_expired_1" not in revoked_tokens
            assert "jti_active_2" in revoked_tokens

    asyncio.run(run())
