"""Health check endpoints for service and model integrity."""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas import HealthResponse, ModelHealthResponse
from src.api.services.screening_service import ScreeningService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service Liveness Check")
def health_check() -> HealthResponse:
    """Returns basic service health status without touching datasets or models."""
    return HealthResponse(status="healthy")


@router.get("/api/v1/health/model", response_model=ModelHealthResponse, summary="Model Integrity & Readiness Check")
def model_health_check() -> ModelHealthResponse:
    """
    Verifies that the model registry is loaded, SHA-256 hashes are verified,
    and all tri-target pipelines are resident in memory.
    Returns 503 Service Unavailable if model integrity fails.
    """
    service = ScreeningService.get_instance()
    info = service.get_model_health()
    if info.get("status") != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_INTEGRITY_FAILURE",
                "message": "Model integrity verification failed or models are not loaded.",
                "details": [info.get("error", "Unknown integrity failure")]
            }
        )
    return ModelHealthResponse(**info)
