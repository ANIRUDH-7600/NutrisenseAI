"""
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System
Step 17 & Step 22: Model Packaging, Versioning & Registry.

This module provides deterministic loading, cryptographic integrity auditing,
threshold management, and schema verification for the active canonical:
- Scenario-A v1 (30 non-invasive features, model_version: nutrisense-scenario-a-v1.0.0, schema: scenario-a-30-v1)

FAIL-LOUDLY DESIGN:
- Zero silent fallbacks (no globbing, no latest-file fallback).
- Rejects missing artifacts, hash mismatches, invalid thresholds, or altered feature schemas.
- Never automatically overwrites or modifies model files.
"""

import os
import copy
import json
import hashlib
import joblib
from typing import Dict, Any, List, Optional

DEFAULT_REGISTRY_PATH = "models/model_registry.json"
DEFAULT_REGISTRY_V1_PATH = "models/v1/model_registry.json"

EXPECTED_MODEL_VERSION = "nutrisense-scenario-a-v1.0.0"
EXPECTED_FEATURE_SCHEMA_VERSION = "scenario-a-30-v1"
EXPECTED_SCENARIO = "A"
EXPECTED_MODEL_FAMILY = "LightGBM"
EXPECTED_TARGETS = ["stunting", "underweight", "wasting"]

LOCKED_APPROVED_THRESHOLDS = {
    "stunting": 0.36,
    "underweight": 0.30,
    "wasting": 0.17
}

REMOVED_SENSITIVE_FEATURES = [
    "household_head_female",
    "caste_category",
    "religion_category",
    "mother_education_level"
]

EXPECTED_30_FEATURES = [
    "child_age_months",
    "child_age_group",
    "child_sex_male",
    "birth_order",
    "is_multiple_birth",
    "is_firstborn",
    "preceding_birth_interval_months",
    "birth_size_ordinal",
    "birth_weight_kg",
    "birth_weight_missing",
    "delivery_place_type",
    "still_breastfeeding",
    "diarrhea_recent",
    "fever_recent",
    "cough_recent",
    "mother_age_years",
    "mother_age_first_birth",
    "mother_bmi",
    "mother_bmi_missing",
    "total_children_born",
    "anc_visits_count",
    "anc_visits_missing",
    "wealth_quintile",
    "is_rural",
    "drinking_water_type",
    "sanitation_facility_type",
    "has_electricity",
    "clean_cooking_fuel",
    "household_size",
    "state_id"
]

# Legacy feature alias for backward compatibility
EXPECTED_34_FEATURES = EXPECTED_30_FEATURES
EXPECTED_MODEL_VERSION_V2 = EXPECTED_MODEL_VERSION
EXPECTED_FEATURE_SCHEMA_VERSION_V2 = EXPECTED_FEATURE_SCHEMA_VERSION
LOCKED_APPROVED_THRESHOLDS_V2 = LOCKED_APPROVED_THRESHOLDS


def compute_sha256(filepath: str) -> str:
    """Computes the SHA-256 hexadecimal digest of a file on disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for hash calculation: {filepath}")
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def _validate_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the canonical active v1.0.0 30-feature model registry specification."""
    required_keys = [
        "registry_version",
        "model_version",
        "feature_schema_version",
        "scenario",
        "model_family",
        "training_configuration",
        "features",
        "thresholds",
        "models"
    ]
    for key in required_keys:
        if key not in registry:
            raise ValueError(f"Model registry is missing required key: '{key}'")

    if registry["model_version"] != EXPECTED_MODEL_VERSION:
        raise ValueError(
            f"Registry model_version mismatch! Expected '{EXPECTED_MODEL_VERSION}', got '{registry['model_version']}'"
        )

    if registry["feature_schema_version"] != EXPECTED_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Registry feature_schema_version mismatch! Expected '{EXPECTED_FEATURE_SCHEMA_VERSION}', got '{registry['feature_schema_version']}'"
        )

    if registry["scenario"] != EXPECTED_SCENARIO:
        raise ValueError(f"Registry scenario mismatch! Expected '{EXPECTED_SCENARIO}', got '{registry['scenario']}'")

    if registry["model_family"] != EXPECTED_MODEL_FAMILY:
        raise ValueError(
            f"Registry model_family mismatch! Expected '{EXPECTED_MODEL_FAMILY}', got '{registry['model_family']}'"
        )

    # Validate decision thresholds
    thresholds = registry["thresholds"]
    for target, expected_val in LOCKED_APPROVED_THRESHOLDS.items():
        if target not in thresholds:
            raise ValueError(f"Threshold for target '{target}' missing from registry.")
        actual_val = float(thresholds[target])
        if abs(actual_val - expected_val) > 1e-6:
            raise ValueError(
                f"Threshold mismatch for target '{target}'! Expected {expected_val}, got {actual_val}"
            )

    # Verify features list (exact order and presence)
    features = registry["features"]
    if len(features) != 30:
        raise ValueError(f"Registry features count mismatch! Expected 30 features, got {len(features)}")

    found_sensitive = set(features).intersection(set(REMOVED_SENSITIVE_FEATURES))
    if found_sensitive:
        raise ValueError(f"Fatal: Sensitive features present in active registry: {found_sensitive}")

    if features != EXPECTED_30_FEATURES:
        raise ValueError("Registry features do not strictly match the approved 30 Scenario-A features and order.")

    # Validate target models entries
    models = registry["models"]
    for target in EXPECTED_TARGETS:
        if target not in models:
            raise ValueError(f"Model metadata for target '{target}' missing from registry.")

        entry = models[target]
        required_entry_keys = ["target", "artifact", "sha256", "model_family", "threshold", "decision_rule"]
        for ek in required_entry_keys:
            if ek not in entry:
                raise ValueError(f"Model metadata entry for '{target}' missing key: '{ek}'")

        if entry["target"] != target:
            raise ValueError(f"Model target mismatch: entry says '{entry['target']}', expected '{target}'")

        if entry["model_family"] != EXPECTED_MODEL_FAMILY:
            raise ValueError(
                f"Model family mismatch for '{target}': expected '{EXPECTED_MODEL_FAMILY}', got '{entry['model_family']}'"
            )

    return registry


def load_registry(
    registry_path: str = DEFAULT_REGISTRY_PATH,
    version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads and rigorously validates the model registry specification.
    Fails loudly if any required metadata, version, threshold, or feature is altered.
    """
    if not os.path.exists(registry_path):
        if os.path.exists(DEFAULT_REGISTRY_V1_PATH):
            registry_path = DEFAULT_REGISTRY_V1_PATH
        else:
            raise FileNotFoundError(f"Model registry file not found: {registry_path}")

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model registry file {registry_path} is not valid JSON: {str(e)}")

    if "versions" in raw_data and EXPECTED_MODEL_VERSION in raw_data["versions"]:
        active_dict = copy.deepcopy(raw_data["versions"][EXPECTED_MODEL_VERSION])
        if "registry_version" not in active_dict:
            active_dict["registry_version"] = raw_data.get("registry_version", "1.0.0")
        return _validate_registry(active_dict)

    return _validate_registry(raw_data)


def load_registry_v2(registry_path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    """Backward compatibility loader for active 30-feature model registry."""
    return load_registry(registry_path=registry_path)


def verify_model_integrity(target: str, registry: Optional[Dict[str, Any]] = None) -> bool:
    """
    Verifies that the model artifact for the target exists and matches its registered SHA-256 hash.
    Fails loudly on mismatch.
    """
    if registry is None:
        registry = load_registry()

    if target not in registry["models"]:
        raise ValueError(f"Target '{target}' is not registered.")

    model_info = registry["models"][target]
    artifact_path = model_info.get("artifact")
    expected_sha = model_info.get("sha256")

    if not artifact_path or not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found for {target} at path: '{artifact_path}'.")

    actual_sha = compute_sha256(artifact_path)
    if actual_sha.lower() != expected_sha.lower():
        raise ValueError(
            f"Model integrity verification failed for {target}!\n"
            f"  Artifact: {artifact_path}\n"
            f"  Expected SHA-256: {expected_sha}\n"
            f"  Actual SHA-256:   {actual_sha}"
        )

    return True


def verify_all_models_integrity(registry: Optional[Dict[str, Any]] = None) -> bool:
    """Verifies cryptographic integrity across all registered target models."""
    if registry is None:
        registry = load_registry()
    for target in EXPECTED_TARGETS:
        verify_model_integrity(target, registry=registry)
    return True


def verify_model_integrity_v2(target: str, registry: Optional[Dict[str, Any]] = None) -> bool:
    """Backward compatibility verifier for active 30-feature models."""
    return verify_model_integrity(target, registry=registry)


def verify_all_models_integrity_v2(registry: Optional[Dict[str, Any]] = None) -> bool:
    """Backward compatibility verifier for all active 30-feature models."""
    return verify_all_models_integrity(registry=registry)


def load_registered_pipeline(target: str, registry: Optional[Dict[str, Any]] = None) -> Any:
    """
    Loads the verified model pipeline for a given target.
    Guarantees deterministic loading from the exact registered artifact.
    """
    if registry is None:
        registry = load_registry()

    verify_model_integrity(target, registry=registry)
    artifact_path = registry["models"][target]["artifact"]

    pipeline = joblib.load(artifact_path)
    if not hasattr(pipeline, "named_steps"):
        raise ValueError(f"Loaded artifact at {artifact_path} is not a valid scikit-learn Pipeline.")

    return pipeline


def load_all_registered_pipelines(registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Loads all verified model pipelines for Stunting, Underweight, and Wasting."""
    if registry is None:
        registry = load_registry()
    return {target: load_registered_pipeline(target, registry=registry) for target in EXPECTED_TARGETS}


def load_registered_pipeline_v2(target: str, registry: Optional[Dict[str, Any]] = None) -> Any:
    """Backward compatibility loader for active 30-feature pipeline."""
    return load_registered_pipeline(target, registry=registry)


def load_all_registered_pipelines_v2(registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backward compatibility loader for all active 30-feature pipelines."""
    return load_all_registered_pipelines(registry=registry)


def get_registered_threshold(target: str, registry: Optional[Dict[str, Any]] = None) -> float:
    """Retrieves the pre-specified locked decision threshold for a target."""
    if registry is None:
        registry = load_registry()
    if target not in registry["thresholds"]:
        raise ValueError(f"Target '{target}' not found in registry thresholds.")
    return float(registry["thresholds"][target])


def get_registered_thresholds(registry: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Retrieves all pre-specified locked decision thresholds."""
    if registry is None:
        registry = load_registry()
    return {target: float(thresh) for target, thresh in registry["thresholds"].items()}


def get_registered_thresholds_v2(registry: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Backward compatibility getter for active decision thresholds."""
    return get_registered_thresholds(registry=registry)


def get_feature_schema(registry: Optional[Dict[str, Any]] = None) -> List[str]:
    """Retrieves the list of 30 approved Scenario-A feature names."""
    if registry is None:
        registry = load_registry()
    return list(registry["features"])


def get_feature_schema_v2(registry: Optional[Dict[str, Any]] = None) -> List[str]:
    """Backward compatibility getter for feature schema."""
    return get_feature_schema(registry=registry)


def get_model_metadata(target: str, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retrieves detailed metadata for a registered model target."""
    if registry is None:
        registry = load_registry()
    if target not in registry["models"]:
        raise ValueError(f"Target '{target}' not found in registry models.")
    return dict(registry["models"][target])


def get_model_metadata_v2(target: str, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backward compatibility getter for model metadata."""
    return get_model_metadata(target, registry=registry)
