"""
Security and authentication utilities for NutriSense AI.

Features:
- Cryptographic password hashing and verification using bcrypt (cost factor 12).
- JWT (JSON Web Token) creation, signing, and signature verification using PyJWT.
- Bearer token extraction and authentication dependency for FastAPI.
- Robust protection against timing attacks, token tampering, and signature forgery.
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.api.config import (
    JWT_SECRET,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    BCRYPT_ROUNDS,
)

logger = logging.getLogger("nutrisense.security")

# HTTP Bearer scheme for Authorization header (auto_error=False allows customizable 401 response)
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(plain_password: str) -> str:
    """
    Hashes a plaintext password using bcrypt with a configurable work factor.
    Bcrypt automatically generates a unique 128-bit salt per hash to prevent rainbow table attacks.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    
    password_bytes = plain_password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a stored bcrypt hash in constant time.
    Safely returns False on corrupted hashes or type mismatches.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        password_bytes = plain_password.encode("utf-8")
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        logger.warning(f"Password verification failure: {e}")
        return False


def create_access_token(
    user_id: str,
    email: str,
    role: str = "health_worker",
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Creates a cryptographically signed JWT access token.
    Claims include:
    - sub (subject): user_id
    - email: normalized user email
    - role: authorization role
    - iat: issued-at timestamp (UTC)
    - exp: expiration timestamp (UTC)
    - jti: unique token identifier for tracking/revocation
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),
        "email": str(email).lower().strip(),
        "role": str(role),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": uuid.uuid4().hex,
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and verifies a JWT access token signature and expiration.
    Raises standardized HTTPException(401) on invalid or expired tokens.
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "exp", "iat"]}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "TOKEN_EXPIRED",
                "message": "Authentication token has expired. Please log in again.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token encountered: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_TOKEN",
                "message": "Invalid authentication token. Authorization denied.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    FastAPI dependency that enforces authentication on protected routes.
    Extracts Bearer token from header, validates signature, and resolves the user
    from the database (or verified claims).
    
    Returns safe user dict strictly without password_hash.
    """
    if credentials is None or not credentials.credentials or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTHENTICATION_REQUIRED",
                "message": "Authentication token is required to access this resource.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials.strip()
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_TOKEN_SUBJECT",
                "message": "Authentication token contains invalid user subject.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_TOKEN",
                "message": "Authentication token is missing required jti identifier.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Verify against MongoDB
    import src.api.database as db_mod
    if not db_mod._is_connected or db_mod.get_db() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database service is temporarily unavailable. Please try again later.",
                "details": []
            }
        )

    # Verify token has not been revoked
    try:
        if await db_mod.is_token_revoked(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "TOKEN_REVOKED",
                    "message": "Authentication token has been revoked. Please log in again.",
                    "details": []
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Database error checking token revocation for jti '{jti}': {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database service is temporarily unavailable. Please try again later.",
                "details": []
            }
        )

    db = db_mod.get_db()
    try:
        user_doc = await db.users.find_one({"user_id": user_id})
    except Exception as e:
        logger.warning(f"Database error resolving user: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database service is temporarily unavailable. Please try again later.",
                "details": []
            }
        )

    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "USER_NOT_FOUND",
                "message": "User associated with token no longer exists.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    if not user_doc.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "USER_INACTIVE",
                "message": "User account has been deactivated.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Return safe dictionary strictly without password_hash or internal secrets
    return {
        "user_id": user_doc.get("user_id", str(user_doc.get("_id"))),
        "name": user_doc.get("name", "User"),
        "email": user_doc.get("email"),
        "role": user_doc.get("role", "health_worker"),
        "created_at": user_doc.get("created_at"),
        "is_active": user_doc.get("is_active", True),
    }
