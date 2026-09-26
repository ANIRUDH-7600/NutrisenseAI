"""
Comprehensive API Test Suite for Step 18 & Step 22 (Major Step 8): NutriSense AI Backend API.
Tests endpoints, schemas, validation, security, privacy, and immutability for v2.0.0.
"""

import os
import io
import copy
import logging
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.screening_service import ScreeningService
from src.models.model_registry import (
    load_registry,
    LOCKED_APPROVED_THRESHOLDS,
    EXPECTED_MODEL_VERSION,
    EXPECTED_FEATURE_SCHEMA_VERSION
)
from tests.test_inference_pipeline import get_sample_valid_input

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745


class TestBackendAPI(unittest.TestCase):
    """Test suite covering all functional, architectural, security, and privacy requirements for v2."""

    @classmethod
    def setUpClass(cls):
        """Initializes test client with startup lifespan context."""
        cls.client_context = TestClient(app, raise_server_exceptions=False)
        cls.client = cls.client_context.__enter__()
        cls.valid_payload = get_sample_valid_input()

    @classmethod
    def tearDownClass(cls):
        """Exits lifespan context cleanly."""
        cls.client_context.__exit__(None, None, None)

    # -------------------------------------------------------------------------
    # 1. GET /health
    # -------------------------------------------------------------------------
    def test_01_health_endpoint(self):
        """1. GET /health returns 200 and {'status': 'healthy'} without loading DHS data."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, {"status": "healthy"})

    # -------------------------------------------------------------------------
    # 2. GET /api/v1/health/model
    # -------------------------------------------------------------------------
    def test_02_model_health_endpoint(self):
        """2. GET /api/v1/health/model returns integrity and readiness status for v2."""
        response = self.client.get("/api/v1/health/model")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model_version"], EXPECTED_MODEL_VERSION)
        self.assertEqual(data["feature_schema_version"], EXPECTED_FEATURE_SCHEMA_VERSION)
        self.assertTrue(data["model_integrity_verified"])
        self.assertIn("targets", data)
        self.assertEqual(set(data["targets"]), {"stunting", "underweight", "wasting"})

    # -------------------------------------------------------------------------
    # 3. GET /api/v1/metadata
    # -------------------------------------------------------------------------
    def test_03_metadata_endpoint(self):
        """3. GET /api/v1/metadata returns public metadata with disclaimers."""
        response = self.client.get("/api/v1/metadata")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("application_name", data)
        self.assertEqual(data["model_version"], EXPECTED_MODEL_VERSION)
        self.assertEqual(data["feature_schema_version"], EXPECTED_FEATURE_SCHEMA_VERSION)
        self.assertEqual(data["scenario"], "A")
        self.assertEqual(data["model_family"], "LightGBM")
        self.assertIn("disclaimer", data)
        self.assertIn("screening", data["disclaimer"].lower())
        self.assertEqual(data["threshold_values"], LOCKED_APPROVED_THRESHOLDS)

    # -------------------------------------------------------------------------
    # 4. POST /api/v1/screen with valid input
    # -------------------------------------------------------------------------
    def test_04_screen_valid_input(self):
        """4. POST /api/v1/screen evaluates screening risk successfully."""
        response = self.client.post("/api/v1/screen", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success", False))
        self.assertEqual(data.get("model_version"), EXPECTED_MODEL_VERSION)
        self.assertEqual(data.get("feature_schema_version"), EXPECTED_FEATURE_SCHEMA_VERSION)
        self.assertIn("predictions", data)
        for target in ["stunting", "underweight", "wasting"]:
            self.assertIn(target, data["predictions"])

    # -------------------------------------------------------------------------
    # 5. POST /api/v1/screen with missing field
    # -------------------------------------------------------------------------
    def test_05_screen_missing_required_field(self):
        """5. POST /api/v1/screen with missing required field returns 422."""
        payload = copy.deepcopy(self.valid_payload)
        del payload["child_age_months"]
        response = self.client.post("/api/v1/screen", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    # -------------------------------------------------------------------------
    # 6. POST /api/v1/screen with invalid type
    # -------------------------------------------------------------------------
    def test_06_screen_invalid_type(self):
        """6. POST /api/v1/screen with invalid non-numeric type returns 422."""
        payload = copy.deepcopy(self.valid_payload)
        payload["child_age_months"] = "not_a_number"
        response = self.client.post("/api/v1/screen", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data["success"])

    # -------------------------------------------------------------------------
    # 7. POST /api/v1/screen with invalid category
    # -------------------------------------------------------------------------
    def test_07_screen_invalid_category(self):
        """7. POST /api/v1/screen with invalid category returns 422."""
        payload = copy.deepcopy(self.valid_payload)
        payload["delivery_place_type"] = "Alien_Facility_Space"
        response = self.client.post("/api/v1/screen", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data["success"])

    # -------------------------------------------------------------------------
    # 8. POST /api/v1/screen with prohibited anthropometric/sensitive field
    # -------------------------------------------------------------------------
    def test_08_screen_prohibited_variable(self):
        """8. POST /api/v1/screen strictly rejects prohibited anthropometric and sensitive fields."""
        prohibited_test_vars = ["hw70", "hw71", "hw2", "hw3", "hw57", "stunting", "caste_category", "religion_category", "mother_education_level", "household_head_female"]
        for pvar in prohibited_test_vars:
            payload = copy.deepcopy(self.valid_payload)
            payload[pvar] = 1
            response = self.client.post("/api/v1/screen", json=payload)
            self.assertEqual(
                response.status_code,
                422,
                f"Failed to reject prohibited variable: {pvar}"
            )
            data = response.json()
            self.assertFalse(data["success"])

    # -------------------------------------------------------------------------
    # 9. Response schema validation
    # -------------------------------------------------------------------------
    def test_09_response_schema_structure(self):
        """9. Response adheres strictly to expected contract."""
        response = self.client.post("/api/v1/screen", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("success", data)
        self.assertIn("model_version", data)
        self.assertIn("feature_schema_version", data)
        self.assertIn("predictions", data)

        for target in ["stunting", "underweight", "wasting"]:
            pred = data["predictions"][target]
            self.assertIn("probability", pred)
            self.assertIn("threshold", pred)
            self.assertIn("screen_positive", pred)
            self.assertIsInstance(pred["screen_positive"], bool)

    # -------------------------------------------------------------------------
    # 10. Probabilities strictly in [0, 1]
    # -------------------------------------------------------------------------
    def test_10_probabilities_within_unit_interval(self):
        """10. Unrounded risk probabilities are strictly within [0.0, 1.0]."""
        response = self.client.post("/api/v1/screen", json=self.valid_payload)
        data = response.json()
        for target, pred in data["predictions"].items():
            prob = pred["probability"]
            self.assertGreaterEqual(prob, 0.0, f"{target} probability < 0")
            self.assertLessEqual(prob, 1.0, f"{target} probability > 1")

    # -------------------------------------------------------------------------
    # 11. Thresholds match registry
    # -------------------------------------------------------------------------
    def test_11_thresholds_match_registry(self):
        """11. Returned decision thresholds strictly match locked v2 registry values."""
        response = self.client.post("/api/v1/screen", json=self.valid_payload)
        data = response.json()
        for target, expected_thresh in LOCKED_APPROVED_THRESHOLDS.items():
            actual_thresh = data["predictions"][target]["threshold"]
            self.assertEqual(
                actual_thresh,
                expected_thresh,
                f"Threshold mismatch for {target}: expected {expected_thresh}, got {actual_thresh}"
            )

    # -------------------------------------------------------------------------
    # 12. Model version matches registry
    # -------------------------------------------------------------------------
    def test_12_model_version_matches_registry(self):
        """12. Returned model version matches registered model version."""
        response = self.client.post("/api/v1/screen", json=self.valid_payload)
        data = response.json()
        self.assertEqual(data["model_version"], EXPECTED_MODEL_VERSION)
        self.assertEqual(data["feature_schema_version"], EXPECTED_FEATURE_SCHEMA_VERSION)

    # -------------------------------------------------------------------------
    # 13. Invalid model integrity causes appropriate failure
    # -------------------------------------------------------------------------
    def test_13_integrity_failure_causes_503(self):
        """13. Service fails with 503 if model integrity fails or service is uninitialized."""
        service = ScreeningService.get_instance()
        prev_status = service.is_initialized
        try:
            service.is_initialized = False
            service.init_error = "Simulated integrity corruption"

            # Check model health returns 503
            resp_health = self.client.get("/api/v1/health/model")
            self.assertEqual(resp_health.status_code, 503)

            # Check screen returns 503
            resp_screen = self.client.post("/api/v1/screen", json=self.valid_payload)
            self.assertEqual(resp_screen.status_code, 503)
            self.assertEqual(resp_screen.json()["error"]["code"], "SERVICE_UNAVAILABLE")
        finally:
            service.is_initialized = prev_status
            service.init_error = None

    # -------------------------------------------------------------------------
    # 14. DHS raw file is never required by API
    # -------------------------------------------------------------------------
    def test_14_dhs_raw_file_never_required(self):
        """14. API functions cleanly even if DHS raw microdata file is temporarily inaccessible."""
        with patch("builtins.open", side_effect=open) as mock_open:
            response = self.client.post("/api/v1/screen", json=self.valid_payload)
            self.assertEqual(response.status_code, 200)

            # Verify no file open calls targeted IAKR7EFL.DTA
            for call_arg in mock_open.call_args_list:
                arg_str = str(call_arg)
                self.assertNotIn("IAKR7EFL.DTA", arg_str)

    # -------------------------------------------------------------------------
    # 15. Request data is not written to logs
    # -------------------------------------------------------------------------
    def test_15_request_data_not_written_to_logs(self):
        """15. Sensitive request payload is strictly excluded from application logs."""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("nutrisense_api")
        logger.addHandler(handler)

        try:
            test_payload = copy.deepcopy(self.valid_payload)
            test_payload["child_age_months"] = 43.21987  # Distinctive marker
            test_payload["mother_age_years"] = 33.77112  # Distinctive marker

            response = self.client.post("/api/v1/screen", json=test_payload)
            self.assertEqual(response.status_code, 200)

            logs = log_stream.getvalue()
            self.assertNotIn("43.21987", logs)
            self.assertNotIn("33.77112", logs)
            self.assertNotIn("mother_bmi", logs)
        finally:
            logger.removeHandler(handler)

    # -------------------------------------------------------------------------
    # 16. Repeated same request gives deterministic result
    # -------------------------------------------------------------------------
    def test_16_repeated_request_deterministic(self):
        """16. Repeated calls with the exact same payload yield identical probabilities."""
        resp1 = self.client.post("/api/v1/screen", json=self.valid_payload)
        resp2 = self.client.post("/api/v1/screen", json=self.valid_payload)

        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)

        preds1 = resp1.json()["predictions"]
        preds2 = resp2.json()["predictions"]

        for target in ["stunting", "underweight", "wasting"]:
            self.assertAlmostEqual(preds1[target]["probability"], preds2[target]["probability"], places=9)
            self.assertEqual(preds1[target]["screen_positive"], preds2[target]["screen_positive"])

    # -------------------------------------------------------------------------
    # 17. Extra / arbitrary fields are forbidden
    # -------------------------------------------------------------------------
    def test_17_arbitrary_extra_fields_rejected(self):
        """17. Request containing unapproved extra fields is rejected with 422."""
        payload = copy.deepcopy(self.valid_payload)
        payload["arbitrary_hack_field"] = "malicious_data"
        response = self.client.post("/api/v1/screen", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data["success"])

    # -------------------------------------------------------------------------
    # 18. Client cannot select model path or threshold
    # -------------------------------------------------------------------------
    def test_18_client_cannot_override_models_or_thresholds(self):
        """18. Client injection of model paths or thresholds is strictly forbidden."""
        payload = copy.deepcopy(self.valid_payload)
        payload["model_path"] = "custom_path.joblib"
        payload["threshold"] = 0.99
        response = self.client.post("/api/v1/screen", json=payload)
        self.assertEqual(response.status_code, 422)

    # -------------------------------------------------------------------------
    # 19. Stack traces never exposed to client
    # -------------------------------------------------------------------------
    def test_19_stack_traces_never_exposed(self):
        """19. Internal server errors return standardized JSON without tracebacks."""
        with patch("src.api.services.screening_service.ScreeningService.screen_child", side_effect=Exception("Secret internal bug")):
            response = self.client.post("/api/v1/screen", json=self.valid_payload)
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")
            self.assertNotIn("Traceback", str(data))
            self.assertNotIn("Secret internal bug", str(data))

    # -------------------------------------------------------------------------
    # 20. Filesystem paths not exposed in public endpoints
    # -------------------------------------------------------------------------
    def test_20_filesystem_paths_not_exposed(self):
        """20. Public metadata and health endpoints never disclose internal filesystem paths."""
        meta_resp = self.client.get("/api/v1/metadata")
        self.assertEqual(meta_resp.status_code, 200)
        meta_str = str(meta_resp.json())
        self.assertNotIn(".joblib", meta_str)
        self.assertNotIn("models/", meta_str)
        self.assertNotIn("C:\\", meta_str)
        self.assertNotIn("d:\\", meta_str)

        health_resp = self.client.get("/api/v1/health/model")
        self.assertEqual(health_resp.status_code, 200)
        health_str = str(health_resp.json())
        self.assertNotIn(".joblib", health_str)
        self.assertNotIn("models/", health_str)


if __name__ == "__main__":
    unittest.main()
