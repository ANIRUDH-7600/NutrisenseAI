"""
Service layer for NutriSense AI Screening API (v2.0.0, 30 features).
Binds FastAPI routes to the Step-16/Step-22 inference pipeline and Step-17/Step-22 model registry.
"""

from typing import Dict, Any, Optional
import src.models.inference_pipeline as inference_pipeline
from src.models.model_registry import (
    load_registry_v2,
    verify_all_models_integrity_v2,
    load_all_registered_pipelines_v2,
    get_registered_thresholds_v2,
    DEFAULT_REGISTRY_PATH
)
from src.api.config import API_TITLE, MODEL_REGISTRY_PATH


class ScreeningService:
    """Singleton service maintaining validated in-memory models and registry state for v2."""
    
    _instance: Optional["ScreeningService"] = None

    def __init__(self, registry_path: str = MODEL_REGISTRY_PATH):
        self.registry_path = registry_path
        self.registry: Optional[Dict[str, Any]] = None
        self.pipelines: Optional[Dict[str, Any]] = None
        self.is_initialized: bool = False
        self.init_error: Optional[str] = None

    @classmethod
    def get_instance(cls, registry_path: str = MODEL_REGISTRY_PATH) -> "ScreeningService":
        if cls._instance is None:
            cls._instance = cls(registry_path=registry_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets singleton instance (used for testing failure modes)."""
        cls._instance = None

    def initialize(self) -> None:
        """
        Loads and cryptographically audits the v2 models at application startup.
        Fails loudly if hashes do not match, files are missing, or thresholds are altered.
        """
        try:
            self.registry = load_registry_v2(self.registry_path)
            verify_all_models_integrity_v2(registry=self.registry)
            self.pipelines = load_all_registered_pipelines_v2(registry=self.registry)
            # Cache directly in inference pipeline to guarantee zero repeated disk I/O
            inference_pipeline._CACHED_MODELS = self.pipelines
            self.is_initialized = True
            self.init_error = None
        except Exception as e:
            self.is_initialized = False
            self.init_error = str(e)
            raise

    def screen_child(self, child_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes screening inference on validated child data.
        Raises RuntimeError if service was not successfully initialized.
        """
        if not self.is_initialized or self.registry is None:
            raise RuntimeError(
                f"ScreeningService is not initialized or model integrity check failed: {self.init_error}"
            )

        # Delegate execution to v2 inference engine
        result = inference_pipeline.predict_child_screening(child_data)
        if not result.get("valid", False):
            raise ValueError(result.get("errors", ["Invalid screening input."]))

        return {
            "success": True,
            "model_version": self.registry["model_version"],
            "feature_schema_version": self.registry["feature_schema_version"],
            "predictions": result["predictions"]
        }

    def get_model_health(self) -> Dict[str, Any]:
        """Returns model diagnostic health status without exposing filesystem paths or secrets."""
        if not self.is_initialized or self.registry is None:
            return {
                "status": "unhealthy",
                "error": self.init_error or "Models not loaded",
                "model_integrity_verified": False
            }

        return {
            "status": "healthy",
            "model_version": self.registry["model_version"],
            "registry_version": self.registry.get("registry_version", "2.0.0"),
            "feature_schema_version": self.registry["feature_schema_version"],
            "model_family": self.registry["model_family"],
            "targets": list(self.registry["models"].keys()),
            "model_integrity_verified": True
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns safe public model and application metadata."""
        if not self.is_initialized or self.registry is None:
            raise RuntimeError("ScreeningService is not initialized.")

        return {
            "application_name": API_TITLE,
            "model_version": self.registry["model_version"],
            "feature_schema_version": self.registry["feature_schema_version"],
            "scenario": self.registry["scenario"],
            "model_family": self.registry["model_family"],
            "target_names": list(self.registry["models"].keys()),
            "threshold_values": self.registry["thresholds"],
            "disclaimer": (
                "NutriSense AI is an epidemiological and community pre-screening risk intelligence tool. "
                "Predictions indicate statistical probability of exceeding validation-derived operating thresholds. "
                "All outputs are non-invasive screening predictions and do NOT constitute clinical diagnoses. "
                "Any child flagged as screen_positive requires secondary triage and direct anthropometric evaluation."
            )
        }
