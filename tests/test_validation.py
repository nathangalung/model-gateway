import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.request import PredictionRequest
from app.services.dummy_models import MODEL_REGISTRY
from app.utils.timestamp import get_current_timestamp_ms

# Constants
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNPROCESSABLE_ENTITY = 422
PERFORMANCE_LIMIT = 10000
ENTITY_COUNT = 2


class TestValidation:
    """Request validation tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
        self.available_models = list(MODEL_REGISTRY.keys())

    @pytest.mark.parametrize(
        "models,should_succeed",
        [
            # Valid formats
            (["fraud_detection:v1"], True),
            (["credit_score:v1"], True),
            (["fraud_detection:v1", "credit_score:v1"], True),
            (["model_name:version"], True),
            # Invalid formats
            (["fraud:detection:v1"], False),
            (["fraud_detection"], False),
            ([":v1"], False),
            (["model:"], False),
            ([""], False),
            # Mixed formats
            (["fraud_detection:v1", "invalid:format:here"], False),
        ],
    )
    def test_model_format_validation(self, models, should_succeed):
        """Model format validation"""
        payload = {
            "models": models,
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": get_current_timestamp_ms(),
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        if should_succeed and models:
            data = response.json()
            result = data["results"][0]
            # Check model statuses skip entity
            for status in result["statuses"][1:]:
                assert "400 BAD_REQUEST" not in status
        elif not should_succeed and models:
            data = response.json()
            result = data["results"][0]
            # Check model statuses skip entity
            has_bad_request = any("400 BAD_REQUEST" in status for status in result["statuses"][1:])
            assert has_bad_request

    @pytest.mark.parametrize(
        "entities,expected_status",
        [
            # Valid entities
            ({"cust_no": ["X123456"]}, HTTP_OK),
            ({"cust_no": [123456]}, HTTP_OK),
            ({"cust_no": ["X123456", 1002]}, HTTP_OK),
            # Edge cases
            ({"cust_no": []}, HTTP_OK),
            ({}, HTTP_UNPROCESSABLE_ENTITY),  # FastAPI validation error
        ],
    )
    def test_entity_validation(self, entities, expected_status):
        """Entity validation"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": entities,
            "event_timestamp": get_current_timestamp_ms(),
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == expected_status

    def test_request_validation_coverage(self):
        """Request validation edge cases"""
        # Test models validation
        with pytest.raises(ValidationError):
            PredictionRequest(models="not_a_list", entities={"cust_no": ["X123456"]})

        # Test entities validation
        with pytest.raises(ValidationError):
            PredictionRequest(models=["fraud_detection:v1"], entities="not_a_dict")

        # Test validator methods
        with pytest.raises(TypeError):
            PredictionRequest.validate_models("not_a_list")

        with pytest.raises(TypeError):
            PredictionRequest.validate_entities("not_a_dict")
