"""
Authentication routes for NutriSense AI.
Handles health worker registration, credential validation, and account provisioning.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials
from pymongo.errors import DuplicateKeyError, PyMongoError

from src.api.rate_limiter import (
    check_rate_limit,
    LOGIN_RATE_LIMIT,
    REGISTER_RATE_LIMIT,
)

from src.api.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    LogoutResponse,
    ErrorResponse,
)
from src.api.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    get_current_user,
    bearer_scheme,
)
from src.api.database import (
    get_user_by_email,
    create_user,
)
from src.api.config import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
logger = logging.getLogger("nutrisense_api")


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"model": UserResponse, "description": "User account successfully registered"},
        400: {"model": ErrorResponse, "description": "Invalid registration request"},
        409: {"model": ErrorResponse, "description": "Account with email already exists"},
        422: {"model": ErrorResponse, "description": "Validation error (password complexity, invalid email)"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Register a New Health Worker Account",
    description=(
        "Registers a new community health worker account with name, email, and password. "
        "Normalizes email, checks for duplicates, hashes password using bcrypt (work factor 12), "
        "and creates a record in the MongoDB 'users' collection. "
        "Strictly never stores or returns plaintext passwords or password hashes."
    )
)
async def register_user_endpoint(
    request: UserRegisterRequest,
    http_request: Request
) -> UserResponse:
    """Creates a new health worker account with salted bcrypt password hashing."""
    check_rate_limit(http_request, action="register", max_requests=REGISTER_RATE_LIMIT)
    normalized_email = request.email.strip().lower()

    try:
        # 1. Check if user already exists with this normalized email
        existing_user = await get_user_by_email(normalized_email)
        if existing_user is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "EMAIL_ALREADY_EXISTS",
                    "message": "An account with this email address already exists.",
                    "details": []
                }
            )

        # 2. Hash password securely using Phase 1 bcrypt implementation
        password_hash = hash_password(request.password)

        # 3. Create user record asynchronously in MongoDB
        user_doc = await create_user({
            "name": request.name,
            "email": normalized_email,
            "password_hash": password_hash,
            "role": request.role or "health_worker",
        })

        logger.info(f"User registered successfully: {user_doc['user_id']}")

        # 4. Return safe UserResponse (strictly excluding password_hash)
        return UserResponse(
            user_id=user_doc["user_id"],
            name=user_doc["name"],
            email=user_doc["email"],
            role=user_doc["role"],
            created_at=user_doc.get("created_at"),
            is_active=user_doc.get("is_active", True),
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions (e.g. 409 Conflict)
        raise
    except DuplicateKeyError:
        # Concurrent registration conflict on unique index
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "EMAIL_ALREADY_EXISTS",
                "message": "An account with this email address already exists.",
                "details": []
            }
        )
    except (PyMongoError, RuntimeError) as e:
        logger.warning(f"Registration failure due to database unavailability: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database service is temporarily unavailable. Please try again later.",
                "details": []
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during user registration: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during registration.",
                "details": []
            }
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": TokenResponse, "description": "User successfully authenticated and token issued"},
        400: {"model": ErrorResponse, "description": "Invalid login request structure"},
        401: {"model": ErrorResponse, "description": "Invalid credentials or inactive account"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Authenticate User and Issue JWT Token",
    description=(
        "Authenticates a health worker using email and password credentials. "
        "Normalizes email, verifies bcrypt password hash, and enforces active account status. "
        "Returns a signed JWT bearer token with safe UserResponse. "
        "Prevents user enumeration by returning a standardized generic 401 message for all authentication failures."
    )
)
async def login_user_endpoint(
    request: UserLoginRequest,
    http_request: Request
) -> TokenResponse:
    """Authenticates health worker credentials and returns a signed JWT access token."""
    check_rate_limit(http_request, action="login", max_requests=LOGIN_RATE_LIMIT)
    normalized_email = request.email.strip().lower()

    def _generic_auth_failure():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_CREDENTIALS",
                "message": "Invalid email or password.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        # Check database connectivity
        import src.api.database as db_mod
        if not db_mod._is_connected or db_mod.get_db() is None:
            raise RuntimeError("Database is currently disconnected or unavailable.")

        # 1. Retrieve user by normalized email
        user = await get_user_by_email(normalized_email)
        if user is None:
            # Generic 401: Nonexistent email
            _generic_auth_failure()

        # 2. Verify password hash using constant-time bcrypt verification
        if not verify_password(request.password, user.get("password_hash", "")):
            # Generic 401: Incorrect password
            _generic_auth_failure()

        # 3. Reject inactive accounts
        if not user.get("is_active", True):
            # Generic 401: Inactive account (same message, no enumeration)
            _generic_auth_failure()

        # 4. Create cryptographically signed JWT token using Phase 1 implementation
        user_id = user["user_id"]
        role = user.get("role", "health_worker")
        access_token = create_access_token(
            user_id=user_id,
            email=user["email"],
            role=role,
        )

        expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60

        logger.info(f"User logged in successfully: {user_id}")

        # 5. Return TokenResponse strictly excluding password or password_hash
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserResponse(
                user_id=user["user_id"],
                name=user["name"],
                email=user["email"],
                role=role,
                created_at=user.get("created_at"),
                is_active=user.get("is_active", True),
            )
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions (e.g. 401 Unauthorized)
        raise
    except (PyMongoError, RuntimeError) as e:
        logger.warning(f"Login failure due to database unavailability: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database service is temporarily unavailable. Please try again later.",
                "details": []
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during user login: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during login.",
                "details": []
            }
        )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": UserResponse, "description": "Current authenticated user profile"},
        401: {"model": ErrorResponse, "description": "Missing, invalid, expired, or inactive user token"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Get Current User Profile",
    description=(
        "Retrieves profile information for the currently authenticated health worker. "
        "Requires a valid JWT Bearer token in the Authorization header. "
        "Verifies user existence and active status in MongoDB and returns safe UserResponse."
    )
)
async def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> UserResponse:
    """Returns safe user identity information strictly excluding password_hash and secrets."""
    return UserResponse(
        user_id=current_user["user_id"],
        name=current_user["name"],
        email=current_user["email"],
        role=current_user.get("role", "health_worker"),
        created_at=current_user.get("created_at"),
        is_active=current_user.get("is_active", True),
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": LogoutResponse, "description": "Successfully logged out and token invalidated"},
        401: {"model": ErrorResponse, "description": "Missing, invalid, expired, or tampered token"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Log Out Health Worker and Invalidate Token",
    description=(
        "Invalidates the specific JWT session token presented in the Authorization header. "
        "Records the token's unique jti in MongoDB revocation store. "
        "Subsequent requests using this token will be rejected with HTTP 401 TOKEN_REVOKED. "
        "Does not invalidate unrelated sessions belonging to the user or other users."
    )
)
async def logout_user_endpoint(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> LogoutResponse:
    """Invalidates the caller's JWT token session by storing its unique jti in MongoDB."""
    if credentials is None or not credentials.credentials or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTHENTICATION_REQUIRED",
                "message": "Authentication token is required to log out.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials.strip()
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    jti = payload.get("jti")
    exp = payload.get("exp")

    if not user_id or not jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_TOKEN",
                "message": "Authentication token is malformed or missing claims.",
                "details": []
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    expires_at = (
        datetime.fromtimestamp(exp, timezone.utc)
        if exp
        else datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    try:
        import src.api.database as db_mod
        if not db_mod._is_connected or db_mod.get_db() is None:
            raise RuntimeError("Database is currently disconnected or unavailable.")

        await db_mod.revoke_token(jti=jti, user_id=user_id, expires_at=expires_at)
        logger.info(f"User logged out successfully: user_id={user_id}")

        return LogoutResponse(
            success=True,
            message="Successfully logged out."
        )

    except HTTPException:
        raise
    except (PyMongoError, RuntimeError) as e:
        logger.warning(f"Logout failure due to database unavailability: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database service is temporarily unavailable. Please try again later.",
                "details": []
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during logout: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during logout.",
                "details": []
            }
        )
