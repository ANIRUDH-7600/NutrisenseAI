"""
Screening endpoint for Scenario A Community Pre-Screening.
Thin route delegating strictly to ScreeningService and Step-16 inference pipeline.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from src.api.schemas import ChildScreeningRequest, ScreeningResponse, ErrorResponse
from src.api.services.screening_service import ScreeningService

router = APIRouter(prefix="/api/v1", tags=["Screening"])
logger = logging.getLogger("nutrisense_api")


from typing import Optional, List
from src.api.database import (
    save_screening_record,
    get_screening_records,
    get_screening_record_by_id,
    delete_screening_record,
)
from src.api.schemas import (
    ChildScreeningRequest,
    ScreeningResponse,
    ErrorResponse,
    ScreeningRecord,
    ScreeningListResponse,
)


@router.post(
    "/screen",
    response_model=ScreeningResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input logic or biological violation"},
        422: {"model": ErrorResponse, "description": "Validation error or prohibited variable supplied"},
        500: {"model": ErrorResponse, "description": "Unexpected internal error"},
        503: {"model": ErrorResponse, "description": "Model service unavailable or integrity check failure"}
    },
    summary="Evaluate Community Child Undernutrition Screening Risk",
    description=(
        "Accepts non-invasive demographic, maternal, household, and recent morbidity indicators "
        "under Scenario A (Community Pre-Screening). Evaluates risk for Stunting, Underweight, and Wasting "
        "using approved LightGBM models and pre-specified decision thresholds.\n\n"
        "**CRITICAL NOTICE**: Predictions represent statistical screening probabilities and do NOT "
        "constitute clinical diagnoses."
    )
)
async def screen_child_endpoint(request: ChildScreeningRequest) -> ScreeningResponse:
    """Executes validated child undernutrition risk screening and persists record to MongoDB."""
    service = ScreeningService.get_instance()
    
    try:
        data_dict = request.model_dump()
        result = service.screen_child(data_dict)

        # Persist assessment asynchronously to MongoDB (fail-soft)
        screening_id = await save_screening_record({
            "inputs": data_dict,
            "predictions": result.get("predictions", {}),
            "triage": result.get("triage", {}),
            "model_version": result.get("model_version"),
            "feature_schema_version": result.get("feature_schema_version"),
            "disclaimer": result.get("disclaimer"),
        })
        if screening_id:
            logger.info(f"Screening persisted to MongoDB with ID: {screening_id}")

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
    summary="List Recent Screening Records",
    description="Fetches recent screening assessments persisted in MongoDB, ordered newest first."
)
async def list_screenings_endpoint(limit: int = 20, skip: int = 0) -> ScreeningListResponse:
    """Retrieves recent screening assessments from MongoDB."""
    records = await get_screening_records(limit=limit, skip=skip)
    return ScreeningListResponse(total=len(records), screenings=records)


@router.get(
    "/screenings/{screening_id}",
    response_model=ScreeningRecord,
    summary="Get Specific Screening Assessment Record",
    description="Retrieves a specific screening record by its unique screening_id."
)
async def get_screening_endpoint(screening_id: str) -> ScreeningRecord:
    """Retrieves a single screening assessment by screening_id."""
    record = await get_screening_record_by_id(screening_id)
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
    summary="Delete a Screening Assessment Record",
    description="Deletes a screening record by its unique screening_id."
)
async def delete_screening_endpoint(screening_id: str):
    """Deletes a screening assessment by screening_id."""
    success = await delete_screening_record(screening_id)
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
