"""API routes package."""

from src.api.routes.health import router as health_router
from src.api.routes.screening import router as screening_router
from src.api.routes.metadata import router as metadata_router

__all__ = ["health_router", "screening_router", "metadata_router"]
