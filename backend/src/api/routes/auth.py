"""
Authentication routes for NutriSense AI.
Handles health worker registration, credential validation, and account provisioning.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from src.api.schemas import (
    UserRegisterRequest,
    UserResponse,
    ErrorResponse,
)
from src.api.security import hash_password
from src.api.database import (
    get_user_by_email,
    create_user,
)

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
async def register_user_endpoint(request: UserRegisterRequest) -> UserResponse:
    """Creates a new health worker account with salted bcrypt password hashing."""
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
    except RuntimeError as e:
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
