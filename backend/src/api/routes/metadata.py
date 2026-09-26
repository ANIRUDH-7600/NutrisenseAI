"""
Metadata endpoint for safe public model and application provenance.
"""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas import MetadataResponse, ErrorResponse
from src.api.services.screening_service import ScreeningService

router = APIRouter(prefix="/api/v1", tags=["Metadata"])


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model service unavailable or uninitialized"}
    },
    summary="Retrieve Public Model and Application Provenance Metadata",
    description=(
        "Returns non-sensitive metadata including application name, model versions, "
        "target designations, locked decision thresholds, and regulatory disclaimers. "
        "Strictly excludes DHS microdata, filesystem paths, or internal parameters."
    )
)
def get_metadata_endpoint() -> MetadataResponse:
    """Returns safe public model and application metadata."""
    service = ScreeningService.get_instance()
    try:
        metadata = service.get_metadata()
        return MetadataResponse(**metadata)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Model service unavailable or uninitialized.",
                "details": [str(e)]
            }
        )
