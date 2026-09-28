"""
Automated Test Suite for Phase 2: Sign Up (User Registration).

Covers:
- Successful registration (HTTP 201 Created)
- Secure password hashing with bcrypt
- Strict absence of plaintext password or password_hash in database and API responses
- Duplicate email rejection (HTTP 409 Conflict)
- Email case and whitespace normalization
- Input validation (email format, password length, complexity, confirmation match, name)
- Database unavailability handling (HTTP 503 Service Unavailable)
- DuplicateKeyError race condition handling (HTTP 409 Conflict)
- Security audit against information and secret leakage
"""

import copy
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from src.api.main import app
from src.api.security import verify_password


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient within the application lifespan."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_user_db():
    """In-memory mock store for MongoDB users collection."""
    stored_users = {}

    async def mock_insert_one(document):
        email = document["email"]
        if email in stored_users:
            raise DuplicateKeyError(f"E11000 duplicate key error collection: users index: email_1 dup key: {email}")
        stored_users[email] = copy.deepcopy(document)
        res = MagicMock()
        res.inserted_id = f"mock_id_{len(stored_users)}"
        return res

    async def mock_find_one(filter_dict, projection=None):
        if "email" in filter_dict:
            return copy.deepcopy(stored_users.get(filter_dict["email"]))
        if "user_id" in filter_dict:
            for u in stored_users.values():
                if u.get("user_id") == filter_dict["user_id"]:
                    return copy.deepcopy(u)
        return None

    mock_db = MagicMock()
    mock_db.users.insert_one = AsyncMock(side_effect=mock_insert_one)
    mock_db.users.find_one = AsyncMock(side_effect=mock_find_one)
    return mock_db, stored_users


def test_successful_registration(client, mock_user_db):
    """Verify standard valid registration returns HTTP 201 with safe UserResponse."""
    mock_db, stored_users = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        payload = {
            "name": "Dr. Priya Sen",
            "email": "priya.sen@district-hospital.org",
            "password": "SecurePassword2026!",
            "confirm_password": "SecurePassword2026!",
            "role": "health_worker"
        }

        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201

        data = response.json()
        assert "user_id" in data
        assert data["user_id"].startswith("usr_")
        assert data["name"] == "Dr. Priya Sen"
        assert data["email"] == "priya.sen@district-hospital.org"
        assert data["role"] == "health_worker"
        assert data["is_active"] is True
        assert "created_at" in data

        # Security check: NEVER expose password or password_hash
        assert "password" not in data
        assert "password_hash" not in data


def test_password_is_bcrypt_hashed_and_not_plaintext(client, mock_user_db):
    """Verify that stored database record contains valid bcrypt hash and never plaintext password."""
    mock_db, stored_users = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        plain_password = "FieldWorkerPassword99!"
        payload = {
            "name": "Kavita Rao",
            "email": "kavita.rao@anganwadi.gov.in",
            "password": plain_password,
            "confirm_password": plain_password
        }

        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201

        # Inspect the record persisted in database
        saved_doc = stored_users["kavita.rao@anganwadi.gov.in"]
        assert "password" not in saved_doc
        assert "password_hash" in saved_doc
        assert saved_doc["password_hash"] != plain_password
        assert saved_doc["password_hash"].startswith("$2b$") or saved_doc["password_hash"].startswith("$2a$")

        # Cryptographically verify the hash matches the plain password
        assert verify_password(plain_password, saved_doc["password_hash"]) is True
        assert verify_password("IncorrectPassword123!", saved_doc["password_hash"]) is False


def test_duplicate_email_rejected_with_409(client, mock_user_db):
    """Verify that registering with an already existing email returns HTTP 409 Conflict."""
    mock_db, stored_users = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        payload = {
            "name": "First User",
            "email": "duplicate.test@clinic.org",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }

        first_res = client.post("/api/v1/auth/register", json=payload)
        assert first_res.status_code == 201

        # Attempt duplicate registration
        duplicate_res = client.post("/api/v1/auth/register", json=payload)
        assert duplicate_res.status_code == 409

        err = duplicate_res.json()
        assert err["success"] is False
        assert err["error"]["code"] == "EMAIL_ALREADY_EXISTS"
        assert "already exists" in err["error"]["message"].lower()


def test_email_case_and_whitespace_normalization(client, mock_user_db):
    """Verify that email addresses are normalized to lowercase trimmed format."""
    mock_db, stored_users = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # Register with mixed case and leading/trailing whitespace
        payload1 = {
            "name": "Dr. Anita Desai",
            "email": "  Anita.Desai@StateHealth.GOV.IN  ",
            "password": "Password1234!",
            "confirm_password": "Password1234!"
        }

        res1 = client.post("/api/v1/auth/register", json=payload1)
        assert res1.status_code == 201
        assert res1.json()["email"] == "anita.desai@statehealth.gov.in"

        # Attempt registration with all-lowercase version of the same email
        payload2 = {
            "name": "Anita Desai Duplicate",
            "email": "anita.desai@statehealth.gov.in",
            "password": "DifferentPassword123!",
            "confirm_password": "DifferentPassword123!"
        }
        res2 = client.post("/api/v1/auth/register", json=payload2)
        assert res2.status_code == 409
        assert res2.json()["error"]["code"] == "EMAIL_ALREADY_EXISTS"


def test_invalid_email_formats_rejected(client, mock_user_db):
    """Verify that malformed email addresses are rejected with HTTP 422."""
    mock_db, _ = mock_user_db

    invalid_emails = [
        "notanemail",
        "@missingusername.com",
        "missingatsign.org",
        "spaces in@email.com",
        "user@nodomain",
        "",
    ]

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        for bad_email in invalid_emails:
            res = client.post("/api/v1/auth/register", json={
                "name": "Test User",
                "email": bad_email,
                "password": "Password123!",
                "confirm_password": "Password123!"
            })
            assert res.status_code == 422, f"Expected 422 for invalid email: '{bad_email}', got {res.status_code}"


def test_password_length_validation(client, mock_user_db):
    """Verify password length constraints (min 8, max 128 chars)."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # Too short (< 8 chars)
        res_short = client.post("/api/v1/auth/register", json={
            "name": "Test User",
            "email": "short.pass@clinic.org",
            "password": "Pass1!",
            "confirm_password": "Pass1!"
        })
        assert res_short.status_code == 422

        # Too long (> 128 chars)
        too_long = "LongPass" * 20 + "1!"
        res_long = client.post("/api/v1/auth/register", json={
            "name": "Test User",
            "email": "long.pass@clinic.org",
            "password": too_long,
            "confirm_password": too_long
        })
        assert res_long.status_code == 422


def test_password_character_complexity_validation(client, mock_user_db):
    """Verify that password requires at least one letter and one number."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # Letters only (no digits)
        res_no_digits = client.post("/api/v1/auth/register", json={
            "name": "Test User",
            "email": "nodigits@clinic.org",
            "password": "OnlyLettersNoDigits!",
            "confirm_password": "OnlyLettersNoDigits!"
        })
        assert res_no_digits.status_code == 422

        # Digits only (no letters)
        res_no_letters = client.post("/api/v1/auth/register", json={
            "name": "Test User",
            "email": "noletters@clinic.org",
            "password": "123456789012!",
            "confirm_password": "123456789012!"
        })
        assert res_no_letters.status_code == 422


def test_password_confirmation_mismatch(client, mock_user_db):
    """Verify that mismatched password confirmation is rejected with HTTP 422."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/register", json={
            "name": "Test User",
            "email": "mismatch@clinic.org",
            "password": "Password123!",
            "confirm_password": "CompletelyDifferent456!"
        })
        assert res.status_code == 422


def test_invalid_name_validation(client, mock_user_db):
    """Verify that name must be at least 2 characters long."""
    mock_db, _ = mock_user_db

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        # Single char
        res = client.post("/api/v1/auth/register", json={
            "name": "A",
            "email": "valid.email@clinic.org",
            "password": "Password123!",
            "confirm_password": "Password123!"
        })
        assert res.status_code == 422

        # Whitespace only
        res_spaces = client.post("/api/v1/auth/register", json={
            "name": "   ",
            "email": "valid.email@clinic.org",
            "password": "Password123!",
            "confirm_password": "Password123!"
        })
        assert res_spaces.status_code == 422


def test_database_unavailable_handling(client):
    """Verify that when database is disconnected, registration returns HTTP 503 Service Unavailable."""
    with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
        res = client.post("/api/v1/auth/register", json={
            "name": "Test User",
            "email": "offline.test@clinic.org",
            "password": "Password123!",
            "confirm_password": "Password123!"
        })
        assert res.status_code == 503
        err = res.json()
        assert err["success"] is False
        assert err["error"]["code"] == "DATABASE_UNAVAILABLE"


def test_concurrent_duplicate_key_error_handled(client):
    """Verify that MongoDB DuplicateKeyError race condition is caught and returned as HTTP 409."""
    mock_db = MagicMock()
    # First find_one returns None (simulate race before write)
    mock_db.users.find_one = AsyncMock(return_value=None)
    # Then insert_one throws DuplicateKeyError
    mock_db.users.insert_one = AsyncMock(side_effect=DuplicateKeyError("E11000 duplicate key error"))

    with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
        res = client.post("/api/v1/auth/register", json={
            "name": "Race Condition User",
            "email": "race@clinic.org",
            "password": "Password123!",
            "confirm_password": "Password123!"
        })
        assert res.status_code == 409
        assert res.json()["error"]["code"] == "EMAIL_ALREADY_EXISTS"
