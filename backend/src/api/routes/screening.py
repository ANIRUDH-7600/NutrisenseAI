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
def screen_child_endpoint(request: ChildScreeningRequest) -> ScreeningResponse:
    """Executes validated child undernutrition risk screening."""
    service = ScreeningService.get_instance()
    
    try:
        data_dict = request.model_dump()
        result = service.screen_child(data_dict)
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
