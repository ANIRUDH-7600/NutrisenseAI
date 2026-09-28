"""
Automated Test Suite for Phase 3: Login & Current User Profile.

Covers:
 1. Successful login
 2. Returned JWT is valid
 3. JWT contains expected claims
 4. Correct user_id in sub
 5. Correct normalized email
 6. Correct role
 7. exp exists and is valid
 8. iat exists and is valid
 9. jti exists and is unique
10. Wrong password rejected (generic 401)
11. Nonexistent email rejected (generic 401, zero email enumeration)
12. Inactive account rejected on login (generic 401, zero enumeration)
13. Malformed token rejected by /auth/me (HTTP 401)
14. Expired token rejected by /auth/me (HTTP 401)
15. Tampered token rejected (HTTP 401)
16. Missing token rejected (HTTP 401)
17. Malformed Authorization header rejected (HTTP 401)
18. Nonexistent user token rejected by /auth/me (HTTP 401)
19. Inactive user token rejected by /auth/me (HTTP 401)
20. /auth/me does not expose password_hash
21. /auth/me does not expose password or JWT secret
22. Database failure handled safely (HTTP 503, zero stack trace leaked)
"""

import copy
import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError
import jwt

from src.api.main import app
from src.api.security import (
    hash_password,
    create_access_token,
    decode_access_token,
)
from src.api.config import JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient within the application lifespan."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_user_db():
    """In-memory mock store for MongoDB users collection."""
    stored_users = {}

    async def mock_find_one(filter_dict, projection=None):
        if "email" in filter_dict:
            user = stored_users.get(filter_dict["email"])
            return copy.deepcopy(user) if user else None
        if "user_id" in filter_dict:
            for u in stored_users.values():
                if u.get("user_id") == filter_dict["user_id"]:
                    return copy.deepcopy(u)
            return None
        return None

    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(side_effect=mock_find_one)
    mock_db.revoked_tokens.find_one = AsyncMock(return_value=None)
    return mock_db, stored_users


def _seed_user(stored_users, user_id, email, password, role="health_worker", is_active=True):
    """Helper to seed a user record directly into the test storage."""
    stored_users[email.strip().lower()] = {
        "user_id": user_id,
        "name": "Dr. Aarohi Patel",
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "role": role,
        "is_active": is_active,
        "created_at": "2026-09-28T10:00:00Z",
        "updated_at": "2026-09-28T10:00:00Z"
    }


# ==============================================================================
# 1. Successful Login
# ==============================================================================
def test_successful_login(client, mock_user_db):
    """Test 1: Verify successful login returns HTTP 200 with TokenResponse structure."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_login_001", "aarohi.patel@clinic.gov.in", "SecurePassword2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "aarohi.patel@clinic.gov.in",
            "password": "SecurePassword2026!"
        })
        assert res.status_code == 200

        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert "user" in data
        assert data["user"]["user_id"] == "usr_login_001"
        assert data["user"]["email"] == "aarohi.patel@clinic.gov.in"
        assert data["user"]["name"] == "Dr. Aarohi Patel"
        assert data["user"]["role"] == "health_worker"
        assert data["user"]["is_active"] is True
        assert "password" not in data["user"]
        assert "password_hash" not in data["user"]


# ==============================================================================
# 2. Returned JWT is Valid
# ==============================================================================
def test_returned_jwt_is_valid(client, mock_user_db):
    """Test 2: Verify the returned JWT token can be cryptographically verified using secret."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_valid_002", "valid.jwt@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "valid.jwt@clinic.org",
            "password": "Passcode2026!"
        })
        assert res.status_code == 200
        token = res.json()["access_token"]

        # Validate with pyjwt directly
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        assert isinstance(payload, dict)


# ==============================================================================
# 3. JWT Contains Expected Claims
# ==============================================================================
def test_jwt_contains_expected_claims(client, mock_user_db):
    """Test 3: Verify the JWT token payload contains all planned claims."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_claims_003", "claims.user@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "claims.user@clinic.org",
            "password": "Passcode2026!"
        })
        payload = jwt.decode(res.json()["access_token"], JWT_SECRET, algorithms=[JWT_ALGORITHM])

        expected_claims = {"sub", "email", "role", "iat", "exp", "jti"}
        assert expected_claims.issubset(set(payload.keys()))


# ==============================================================================
# 4. Correct user_id in sub
# ==============================================================================
def test_correct_user_id_in_sub(client, mock_user_db):
    """Test 4: Verify the 'sub' claim matches the user's permanent user_id."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_custom_sub_777", "sub.test@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "sub.test@clinic.org",
            "password": "Passcode2026!"
        })
        payload = decode_access_token(res.json()["access_token"])
        assert payload["sub"] == "usr_custom_sub_777"


# ==============================================================================
# 5. Correct Normalized Email
# ==============================================================================
def test_correct_normalized_email(client, mock_user_db):
    """Test 5: Verify login normalizes email (whitespace and mixed case) identically to registration."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_norm_005", "normalized.user@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "  NORMALIZED.User@Clinic.Org  ",
            "password": "Passcode2026!"
        })
        assert res.status_code == 200
        payload = decode_access_token(res.json()["access_token"])
        assert payload["email"] == "normalized.user@clinic.org"


# ==============================================================================
# 6. Correct Role
# ==============================================================================
def test_correct_role(client, mock_user_db):
    """Test 6: Verify role claim matches user role."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_role_006", "role.admin@clinic.org", "Passcode2026!", role="administrator")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "role.admin@clinic.org",
            "password": "Passcode2026!"
        })
        assert res.status_code == 200
        payload = decode_access_token(res.json()["access_token"])
        assert payload["role"] == "administrator"


# ==============================================================================
# 7. exp Exists
# ==============================================================================
def test_exp_exists_and_valid(client, mock_user_db):
    """Test 7: Verify exp claim exists, is integer, and is in future."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_exp_007", "exp.test@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "exp.test@clinic.org",
            "password": "Passcode2026!"
        })
        payload = decode_access_token(res.json()["access_token"])
        assert "exp" in payload
        assert isinstance(payload["exp"], int)
        assert payload["exp"] > payload["iat"]


# ==============================================================================
# 8. iat Exists
# ==============================================================================
def test_iat_exists_and_valid(client, mock_user_db):
    """Test 8: Verify iat claim exists and is integer."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_iat_008", "iat.test@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "iat.test@clinic.org",
            "password": "Passcode2026!"
        })
        payload = decode_access_token(res.json()["access_token"])
        assert "iat" in payload
        assert isinstance(payload["iat"], int)


# ==============================================================================
# 9. jti Exists
# ==============================================================================
def test_jti_exists_and_unique(client, mock_user_db):
    """Test 9: Verify jti claim exists and is unique across subsequent logins."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_jti_009", "jti.test@clinic.org", "Passcode2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res1 = client.post("/api/v1/auth/login", json={"email": "jti.test@clinic.org", "password": "Passcode2026!"})
        res2 = client.post("/api/v1/auth/login", json={"email": "jti.test@clinic.org", "password": "Passcode2026!"})

        payload1 = decode_access_token(res1.json()["access_token"])
        payload2 = decode_access_token(res2.json()["access_token"])

        assert "jti" in payload1 and isinstance(payload1["jti"], str) and len(payload1["jti"]) > 0
        assert "jti" in payload2 and isinstance(payload2["jti"], str) and len(payload2["jti"]) > 0
        assert payload1["jti"] != payload2["jti"]


# ==============================================================================
# 10. Wrong Password Rejected
# ==============================================================================
def test_wrong_password_rejected(client, mock_user_db):
    """Test 10: Verify incorrect password returns HTTP 401 with generic INVALID_CREDENTIALS."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_wrong_010", "correct.email@clinic.org", "CorrectPassword2026!")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "correct.email@clinic.org",
            "password": "WrongPassword999!"
        })
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_CREDENTIALS"
        assert err["message"] == "Invalid email or password."


# ==============================================================================
# 11. Nonexistent Email Rejected
# ==============================================================================
def test_nonexistent_email_rejected(client, mock_user_db):
    """Test 11: Verify nonexistent email returns identical generic 401 error (no user enumeration)."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "does.not.exist@clinic.org",
            "password": "AnyPassword123!"
        })
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_CREDENTIALS"
        assert err["message"] == "Invalid email or password."


# ==============================================================================
# 12. Inactive Account Rejected on Login
# ==============================================================================
def test_inactive_account_rejected_on_login(client, mock_user_db):
    """Test 12: Verify inactive account credentials return identical generic 401 (no status enumeration)."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_inactive_012", "inactive.worker@clinic.org", "ValidPassword2026!", is_active=False)

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/login", json={
            "email": "inactive.worker@clinic.org",
            "password": "ValidPassword2026!"
        })
        assert res.status_code == 401
        err = res.json()["error"]
        # Must match nonexistent and wrong password exactly to prevent account probing
        assert err["code"] == "INVALID_CREDENTIALS"
        assert err["message"] == "Invalid email or password."


# ==============================================================================
# 13. Malformed Token Rejected by /auth/me
# ==============================================================================
def test_malformed_token_rejected_by_auth_me(client, mock_user_db):
    """Test 13: Verify malformed JWT format returns HTTP 401 INVALID_TOKEN."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_TOKEN"


# ==============================================================================
# 14. Expired Token Rejected by /auth/me
# ==============================================================================
def test_expired_token_rejected_by_auth_me(client, mock_user_db):
    """Test 14: Verify expired JWT token returns HTTP 401 TOKEN_EXPIRED."""
    mock_db, _ = mock_user_db
    expired_token = create_access_token(
        user_id="usr_expired_014",
        email="expired@clinic.org",
        expires_delta=timedelta(seconds=-10)
    )

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "TOKEN_EXPIRED"


# ==============================================================================
# 15. Tampered Token Rejected
# ==============================================================================
def test_tampered_token_rejected(client, mock_user_db):
    """Test 15: Verify signature-tampered JWT token returns HTTP 401 INVALID_TOKEN."""
    mock_db, _ = mock_user_db
    valid_token = create_access_token(user_id="usr_tamper_015", email="tamper@clinic.org")
    parts = valid_token.split(".")
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6IHRydWV9.{parts[2]}"

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tampered_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "INVALID_TOKEN"


# ==============================================================================
# 16. Missing Token Rejected
# ==============================================================================
def test_missing_token_rejected(client, mock_user_db):
    """Test 16: Verify request without Authorization header returns HTTP 401 AUTHENTICATION_REQUIRED."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me")
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "AUTHENTICATION_REQUIRED"


# ==============================================================================
# 17. Malformed Authorization Header Rejected
# ==============================================================================
def test_malformed_authorization_header_rejected(client, mock_user_db):
    """Test 17: Verify various non-Bearer or whitespace-only Authorization headers return HTTP 401."""
    mock_db, _ = mock_user_db

    malformed_headers = [
        "Bearer",
        "Bearer    ",
        "Basic dXNlcjpwYXNz",
        "Token some_random_token",
        "JustATokenWithoutScheme",
    ]

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        for bad_header in malformed_headers:
            res = client.get("/api/v1/auth/me", headers={"Authorization": bad_header})
            assert res.status_code == 401, f"Expected 401 for header '{bad_header}', got {res.status_code}"


# ==============================================================================
# 18. Nonexistent User Token Rejected
# ==============================================================================
def test_nonexistent_user_token_rejected(client, mock_user_db):
    """Test 18: Verify validly signed token for a deleted/nonexistent user returns HTTP 401 USER_NOT_FOUND."""
    mock_db, stored_users = mock_user_db
    ghost_token = create_access_token(user_id="usr_deleted_ghost_999", email="ghost@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {ghost_token}"})
        assert res.status_code == 401
        err = res.json()["error"]
        assert err["code"] == "USER_NOT_FOUND"


# ==============================================================================
# 19. Inactive User Token Rejected
# ==============================================================================
def test_inactive_user_token_rejected(client, mock_user_db):
    """Test 19: Verify validly signed token for a deactivated account is rejected by /auth/me."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_deactivated_019", "deactivated@clinic.org", "Passcode2026!", is_active=False)
    inactive_token = create_access_token(user_id="usr_deactivated_019", email="deactivated@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {inactive_token}"})
        assert res.status_code in (401, 403)
        err = res.json()["error"]
        assert err["code"] == "USER_INACTIVE"


# ==============================================================================
# 20. /auth/me Does Not Expose password_hash
# ==============================================================================
def test_auth_me_does_not_expose_password_hash(client, mock_user_db):
    """Test 20: Verify /auth/me strictly omits password_hash from the response payload."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_safe_020", "safe.user@clinic.org", "Passcode2026!")
    token = create_access_token(user_id="usr_safe_020", email="safe.user@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert "password_hash" not in data
        assert "password_hash" not in str(res.content)


# ==============================================================================
# 21. /auth/me Does Not Expose Password or Secret
# ==============================================================================
def test_auth_me_does_not_expose_password_or_secret(client, mock_user_db):
    """Test 21: Verify /auth/me strictly omits plaintext passwords and JWT secret."""
    mock_db, stored_users = mock_user_db
    _seed_user(stored_users, "usr_secret_021", "secret.audit@clinic.org", "SecretPassword2026!")
    token = create_access_token(user_id="usr_secret_021", email="secret.audit@clinic.org")

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert "password" not in data
        assert JWT_SECRET not in str(res.content)
        assert "SecretPassword2026!" not in str(res.content)


# ==============================================================================
# 22. Database Failure Handled Safely
# ==============================================================================
def test_database_failure_handled_safely(client, mock_user_db):
    """Test 22: Verify database unavailability returns HTTP 503 without leaking stack traces or crashing."""
    # Subtest A: Database disconnected on login
    with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
        res_login_disc = client.post("/api/v1/auth/login", json={
            "email": "offline@clinic.org",
            "password": "Password123!"
        })
        assert res_login_disc.status_code == 503
        err = res_login_disc.json()["error"]
        assert err["code"] == "DATABASE_UNAVAILABLE"
        assert "Traceback" not in str(res_login_disc.content)

    # Subtest B: Database error exception on login
    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(side_effect=PyMongoError("Simulated MongoDB cluster read failure"))
    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res_login_err = client.post("/api/v1/auth/login", json={
            "email": "dberror@clinic.org",
            "password": "Password123!"
        })
        assert res_login_err.status_code == 503
        err = res_login_err.json()["error"]
        assert err["code"] == "DATABASE_UNAVAILABLE"
        assert "Traceback" not in str(res_login_err.content)

    # Subtest C: Database disconnected on /auth/me
    token = create_access_token(user_id="usr_db_fail_022", email="dbfail@clinic.org")
    with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
        res_me_disc = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res_me_disc.status_code == 503
        err = res_me_disc.json()["error"]
        assert err["code"] == "DATABASE_UNAVAILABLE"
        assert "Traceback" not in str(res_me_disc.content)
