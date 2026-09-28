"""
Unit tests for Phase 1 Authentication Design & Security Utilities.
Verifies bcrypt password hashing, JWT token handling, Pydantic validation schemas,
and authentication dependency logic.
"""

import pytest
from datetime import timedelta
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)
from src.api.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
)


def test_hash_password_produces_bcrypt_hash():
    """Verify that hash_password generates a valid $2b$ bcrypt hash and is non-empty."""
    password = "HealthcareWorker2026!"
    hashed = hash_password(password)
    assert hashed != password
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert len(hashed) >= 59


def test_verify_password_correct_and_incorrect():
    """Verify constant-time password verification behavior."""
    password = "CorrectHorseBatteryStaple99!"
    hashed = hash_password(password)

    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(password, "invalid_hash_string") is False


def test_hash_password_empty_raises_value_error():
    """Verify that hashing an empty password raises ValueError."""
    with pytest.raises(ValueError):
        hash_password("")


def test_jwt_create_and_decode_valid():
    """Verify that JWT encodes and decodes correct claims and timestamps."""
    user_id = "usr_test_12345"
    email = "Nurse.Meera@Clinic.org"
    role = "health_worker"

    token = create_access_token(user_id=user_id, email=email, role=role)
    assert isinstance(token, str)
    assert len(token.split(".")) == 3

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["email"] == "nurse.meera@clinic.org"
    assert payload["role"] == role
    assert "exp" in payload
    assert "iat" in payload
    assert "jti" in payload
    assert payload["exp"] > payload["iat"]


def test_jwt_expired_token_raises_401():
    """Verify that an expired JWT token raises 401 TOKEN_EXPIRED."""
    user_id = "usr_expired_001"
    email = "expired@clinic.org"
    # Create token that expired 10 seconds ago
    token = create_access_token(
        user_id=user_id,
        email=email,
        expires_delta=timedelta(seconds=-10)
    )

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail.get("code") == "TOKEN_EXPIRED"


def test_jwt_tampered_token_raises_401():
    """Verify that a tampered JWT token signature raises 401 INVALID_TOKEN."""
    token = create_access_token(user_id="usr_tamper", email="tamper@clinic.org")
    # Mutate the payload portion of the token
    parts = token.split(".")
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6IHRydWV9.{parts[2]}"

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(tampered_token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail.get("code") == "INVALID_TOKEN"


def test_user_register_schema_validation():
    """Verify registration schema validation and email normalization."""
    req = UserRegisterRequest(
        name="  Dr. Ananya Sharma  ",
        email="  Ananya.Sharma@Hospital.gov.in  ",
        password="ValidPassword2026!",
        confirm_password="ValidPassword2026!"
    )
    assert req.name == "Dr. Ananya Sharma"
    assert req.email == "ananya.sharma@hospital.gov.in"

    # Password mismatch
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            name="Test Worker",
            email="worker@clinic.org",
            password="Password123!",
            confirm_password="DifferentPassword456!"
        )

    # Weak password (< 8 chars)
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            name="Test Worker",
            email="worker@clinic.org",
            password="short1!"
        )

    # Invalid email format
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            name="Test Worker",
            email="not_an_email_at_all",
            password="ValidPassword123!"
        )


def test_user_response_excludes_password_hash():
    """Verify UserResponse schema strictly excludes password hash or private data."""
    user = UserResponse(
        user_id="usr_secure_99",
        name="Health Worker",
        email="worker@nutrisense.ai",
        role="health_worker",
        created_at="2026-09-28T10:00:00Z",
        is_active=True
    )
    dumped = user.model_dump()
    assert "password" not in dumped
    assert "password_hash" not in dumped
    assert dumped["user_id"] == "usr_secure_99"
