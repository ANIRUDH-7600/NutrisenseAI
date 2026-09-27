"""
Configuration module for NutriSense AI Backend API.
Environment variables control execution mode, CORS, and model paths.
"""

import os
from typing import List

# Environment: 'development' or 'production'
NUTRISENSE_ENV: str = os.getenv("NUTRISENSE_ENV", "development").lower()

# Default development origins
DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173"
]

def get_allowed_origins() -> List[str]:
    """Retrieves allowed CORS origins from environment or safe defaults."""
    env_origins = os.getenv("NUTRISENSE_ALLOWED_ORIGINS")
    if env_origins:
        return [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    if NUTRISENSE_ENV == "development":
        return DEFAULT_DEV_ORIGINS
    return []

ALLOWED_ORIGINS: List[str] = get_allowed_origins()

# Registry path
MODEL_REGISTRY_PATH: str = os.getenv("MODEL_REGISTRY_PATH", "models/model_registry.json")

# Database Configuration (MongoDB)
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "nutrisense_ai")
MONGODB_ENABLED: bool = os.getenv("MONGODB_ENABLED", "true").lower() in ("true", "1", "yes")
MONGODB_SERVER_SELECTION_TIMEOUT_MS: int = int(os.getenv("MONGODB_TIMEOUT_MS", "2000"))

# API Metadata
API_TITLE: str = "NutriSense AI: Childhood Malnutrition Risk Intelligence API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Community pre-screening risk intelligence API for early childhood undernutrition "
    "(Stunting, Underweight, Wasting) under Scenario A (non-invasive, scale-free community triage). "
    "All outputs are statistical screening predictions and do not represent clinical diagnoses."
)
