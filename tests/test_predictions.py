import asyncio
import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import _raise_no_entities_error, app, predict
from app.models.request import PredictionRequest
from app.services.dummy_models import MODEL_REGISTRY
from app.utils.timestamp import get_current_timestamp_ms

# Constants
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500
PERFORMANCE_LIMIT = 60000
FAST_PERFORMANCE_LIMIT = 5000
ENTITY_COUNT = 2


class TestPredictions:
    """Prediction functionality tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
        self.available_models = list(MODEL_REGISTRY.keys())

    def test_prediction_consistency(self):
        """Prediction consistency"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": get_current_timestamp_ms(),
        }

        responses = []
        for _ in range(3):
            response = self.client.post("/predict", json=payload)
            assert response.status_code == HTTP_OK
            responses.append(response.json())

        # Check consistency excluding entity
        for i in range(1, len(responses)):
            first_values = responses[0]["results"][0]["values"][1:]
            current_values = responses[i]["results"][0]["values"][1:]
            assert current_values == first_values

    def test_timestamp_behavior(self):
        """Response always current time"""
        old_timestamp = 1640995200000  # Jan 1, 2022
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": old_timestamp,
        }

        before_request = get_current_timestamp_ms()
        response = self.client.post("/predict", json=payload)
        after_request = get_current_timestamp_ms()

        assert response.status_code == HTTP_OK
        data = response.json()
        response_timestamp = data["results"][0]["event_timestamp"][0]

        # Should be current time
        assert before_request <= response_timestamp <= after_request + 2000

    def test_error_handling(self):
        """Error handling paths"""
        with patch("app.main.model_service.batch_predict") as mock_batch:
            mock_batch.side_effect = Exception("Test exception")

            payload = {
                "models": ["fraud_detection:v1"],
                "entities": {"cust_no": ["X123456"]},
            }

            response = self.client.post("/predict", json=payload)
            assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
            assert "Internal server error" in response.json()["detail"]

    def test_no_entities_error(self):
        """No entities error handling"""
        with pytest.raises(HTTPException) as exc_info:
            _raise_no_entities_error()
        assert exc_info.value.status_code == HTTP_BAD_REQUEST

        # Test empty entities check
        request = PredictionRequest(
            models=["fraud_detection:v1"], entities={"cust_no": ["X123456"]}
        )
        request.entities = {}

        with pytest.raises(HTTPException):
            asyncio.run(predict(request))

    @pytest.mark.parametrize("num_models,num_entities", [(2, 3), (3, 5), (4, 8)])
    def test_batch_performance(self, num_models, num_entities):
        """Batch performance CI adjusted"""
        actual_models = min(num_models, len(self.available_models))
        models = self.available_models[:actual_models]
        entities = [f"entity_{i}" for i in range(num_entities)]

        payload = {
            "models": models,
            "entities": {"cust_no": entities},
            "event_timestamp": get_current_timestamp_ms(),
        }

        start_time = time.time()
        response = self.client.post("/predict", json=payload)
        end_time = time.time()

        assert response.status_code == HTTP_OK
        processing_time = (end_time - start_time) * 1000
        assert processing_time < PERFORMANCE_LIMIT

        # Check response structure
        data = response.json()
        for result in data["results"]:
            # Entity plus model predictions
            assert len(result["values"]) == actual_models + 1

    def test_performance_edge_cases(self):
        """Performance edge cases coverage"""
        # Test empty models list
        payload = {
            "models": [],
            "entities": {"cust_no": ["X123456"]},
        }

        start_time = time.time()
        response = self.client.post("/predict", json=payload)
        end_time = time.time()

        assert response.status_code == HTTP_OK
        processing_time = (end_time - start_time) * 1000
        assert processing_time < FAST_PERFORMANCE_LIMIT

        # Test empty entities
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": []},
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK
        assert len(response.json()["results"]) == 0
