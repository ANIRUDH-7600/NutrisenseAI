"""
Screening endpoint for Scenario A Community Pre-Screening.
Thin route delegating strictly to ScreeningService and Step-16 inference pipeline.
Protected by server-side JWT authentication dependency (Phase 5).
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Query

from src.api.schemas import (
    ChildScreeningRequest,
    ScreeningResponse,
    ErrorResponse,
    ScreeningRecord,
    ScreeningListResponse,
)
from src.api.services.screening_service import ScreeningService
from src.api.security import get_current_user
from src.api.database import (
    save_screening_record,
    get_screening_records,
    get_screening_record_by_id,
    delete_screening_record,
)

router = APIRouter(prefix="/api/v1", tags=["Screening"])
logger = logging.getLogger("nutrisense_api")


@router.post(
    "/screen",
    response_model=ScreeningResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input logic or biological violation"},
        401: {"model": ErrorResponse, "description": "Authentication required, invalid, expired, or revoked token"},
        422: {"model": ErrorResponse, "description": "Validation error or prohibited variable supplied"},
        500: {"model": ErrorResponse, "description": "Unexpected internal error"},
        503: {"model": ErrorResponse, "description": "Model service unavailable or integrity check failure"}
    },
    summary="Evaluate Community Child Undernutrition Screening Risk",
    description=(
        "Accepts non-invasive demographic, maternal, household, and recent morbidity indicators "
        "under Scenario A (Community Pre-Screening). Evaluates risk for Stunting, Underweight, and Wasting "
        "using approved LightGBM models and pre-specified decision thresholds.\n\n"
        "Requires valid JWT Bearer authentication.\n\n"
        "**CRITICAL NOTICE**: Predictions represent statistical screening probabilities and do NOT "
        "constitute clinical diagnoses."
    )
)
async def screen_child_endpoint(
    request: ChildScreeningRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScreeningResponse:
    """Executes validated child undernutrition risk screening and persists record to MongoDB."""
    service = ScreeningService.get_instance()
    
    try:
        data_dict = request.model_dump()
        result = service.screen_child(data_dict)

        # Persist assessment asynchronously to MongoDB (fail-soft with user ownership)
        user_id = current_user.get("user_id")
        screening_id = await save_screening_record({
            "child_name": result.get("child_name", "Anonymous Child"),
            "inputs": data_dict,
            "predictions": result.get("predictions", {}),
            "triage": result.get("triage", {}),
            "model_version": result.get("model_version"),
            "feature_schema_version": result.get("feature_schema_version"),
            "disclaimer": result.get("disclaimer"),
        }, user_id=user_id)
        if screening_id:
            logger.info(f"Screening persisted to MongoDB with ID: {screening_id} for user: {user_id}")

        return ScreeningResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_INPUT",
                "message": "Invalid screening input.",
                "details": str(e).split("; ") if isinstance(e, ValueError) else [str(e)]
            }
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Model service unavailable or uninitialized.",
                "details": [str(e)]
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in screen_child_endpoint: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
                "details": []
            }
        )


@router.get(
    "/screenings",
    response_model=ScreeningListResponse,
    responses={
        200: {"model": ScreeningListResponse, "description": "List of recent screening assessments"},
        401: {"model": ErrorResponse, "description": "Authentication required, invalid, expired, or revoked token"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="List Recent Screening Records",
    description="Fetches recent screening assessments persisted in MongoDB belonging to the authenticated user, ordered newest first. Requires JWT authentication."
)
async def list_screenings_endpoint(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records to return (1-100)"),
    skip: int = Query(0, ge=0, le=10000, description="Offset for pagination"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScreeningListResponse:
    """Retrieves recent screening assessments from MongoDB scoped strictly to current_user."""
    user_id = current_user.get("user_id")
    records = await get_screening_records(user_id=user_id, limit=limit, skip=skip)
    return ScreeningListResponse(total=len(records), screenings=records)


@router.get(
    "/screenings/{screening_id}",
    response_model=ScreeningRecord,
    responses={
        200: {"model": ScreeningRecord, "description": "Screening assessment details"},
        401: {"model": ErrorResponse, "description": "Authentication required, invalid, expired, or revoked token"},
        404: {"model": ErrorResponse, "description": "Screening assessment record not found"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Get Specific Screening Assessment Record",
    description="Retrieves a specific screening record by its unique screening_id, verifying user ownership. Returns 404 if record does not exist or belongs to another user (preventing ID enumeration)."
)
async def get_screening_endpoint(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScreeningRecord:
    """Retrieves a single screening assessment by screening_id, enforcing ownership by current_user."""
    user_id = current_user.get("user_id")
    record = await get_screening_record_by_id(screening_id, user_id=user_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCREENING_NOT_FOUND",
                "message": f"Screening assessment '{screening_id}' not found.",
                "details": []
            }
        )
    return ScreeningRecord(**record)


@router.delete(
    "/screenings/{screening_id}",
    responses={
        200: {"description": "Screening assessment successfully deleted"},
        401: {"model": ErrorResponse, "description": "Authentication required, invalid, expired, or revoked token"},
        404: {"model": ErrorResponse, "description": "Screening assessment not found"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Delete a Screening Assessment Record",
    description="Deletes a screening record by its unique screening_id, enforcing user ownership. Returns 404 if record does not exist or belongs to another user."
)
async def delete_screening_endpoint(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Deletes a screening assessment by screening_id scoped strictly to current_user."""
    user_id = current_user.get("user_id")
    success = await delete_screening_record(screening_id, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCREENING_NOT_FOUND",
                "message": f"Screening assessment '{screening_id}' could not be deleted (not found or database offline).",
                "details": []
            }
        )
    return {"success": True, "message": f"Screening record '{screening_id}' deleted."}
