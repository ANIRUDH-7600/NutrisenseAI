"""
NutriSense AI: Childhood Malnutrition Risk Intelligence API.
Scenario A — Community Pre-Screening API Application.

Production-ready FastAPI application providing non-invasive risk intelligence
for early childhood undernutrition (Stunting, Underweight, Wasting).
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    get_allowed_origins,
    NUTRISENSE_ENV
)
from src.api.services.screening_service import ScreeningService
from src.api.routes import health_router, screening_router, metadata_router

# Configure minimal operational logger (strict privacy: zero feature/body logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [op_log] %(message)s"
)
logger = logging.getLogger("nutrisense_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes and cryptographically verifies model pipelines at startup.
    Fails startup immediately if model artifacts or checksums are invalid.
    """
    logger.info("Initializing NutriSense AI Screening Service and verifying model registry...")
    service = ScreeningService.get_instance()
    try:
        service.initialize()
        logger.info("Model pipelines verified and resident in memory. Service ready.")
    except Exception as e:
        logger.critical(f"FATAL: Model initialization/integrity audit failed: {e}")
        raise e
    yield
    logger.info("Shutting down NutriSense AI Backend API.")


def create_app() -> FastAPI:
    """Factory function for FastAPI application instance."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # 1. CORS Configuration (Environment-Controlled, Never Wildcard in Production)
    allowed_origins = get_allowed_origins()
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"]
        )

    # 2. Operational Privacy-Preserving Middleware
    @app.middleware("http")
    async def operational_logging_middleware(request: Request, call_next: Callable) -> Response:
        """
        Logs strictly non-sensitive operational telemetry (method, route, status, duration).
        STRICT PRIVACY POLICY:
        - NEVER logs request bodies
        - NEVER logs demographic, maternal, or health attributes
        - NEVER logs response payload predictions
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        # Operational telemetry only
        logger.info(
            f"req_id={request_id} method={request.method} path={request.url.path} "
            f"status={response.status_code} duration_ms={duration_ms}"
        )
        return response

    # 3. Exception Handlers (Standardized JSON Errors, Zero Stack Traces Leaked)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Formats Pydantic validation errors into structured response without leaking internals."""
        formatted_details = []
        for err in exc.errors():
            loc = " -> ".join(str(loc_item) for loc_item in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            formatted_details.append(f"{loc}: {msg}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_INPUT",
                    "message": "Validation failed for screening request.",
                    "details": formatted_details
                }
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Standardizes HTTP exception responses."""
        if isinstance(exc.detail, dict):
            code = exc.detail.get("code", "HTTP_ERROR")
            message = exc.detail.get("message", "An HTTP error occurred.")
            details = exc.detail.get("details", [])
        else:
            code = "HTTP_ERROR"
            message = str(exc.detail)
            details = []

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                    "details": details
                }
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Safely catches unhandled exceptions, logs internally, and conceals stack traces from clients."""
        logger.error(f"Internal unhandled exception on {request.method} {request.url.path}: {type(exc).__name__}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "details": []
                }
            }
        )

    # 4. Include Routers
    app.include_router(health_router)
    app.include_router(screening_router)
    app.include_router(metadata_router)

    return app


app = create_app()
