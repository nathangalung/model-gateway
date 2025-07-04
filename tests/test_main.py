import asyncio
import queue
import threading
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
THREAD_COUNT = 5
EXPECTED_RESULT_COUNT = 2


class TestBasicFunctionality:
    """Basic functionality tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """Health endpoint"""
        response = self.client.get("/health")
        assert response.status_code == HTTP_OK


class TestPredictionEndpoint:
    """Prediction endpoint tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)

    @pytest.mark.parametrize(
        "request_payload,expected",
        [
            # Basic cases
            (
                {"models": ["fraud_detection:v1"], "entities": {"cust_no": ["X123456"]}},
                {"status_code": HTTP_OK, "results_count": 1},
            ),
            (
                {
                    "models": ["fraud_detection:v1", "credit_score:v1"],
                    "entities": {"cust_no": ["X123456"]},
                },
                {"status_code": HTTP_OK, "results_count": 1},
            ),
            # Empty models
            (
                {"models": [], "entities": {"cust_no": ["X123456"]}},
                {"status_code": HTTP_OK, "results_count": 1},
            ),
            # Empty entities list
            (
                {"models": ["fraud_detection:v1"], "entities": {"cust_no": []}},
                {"status_code": HTTP_OK, "results_count": 0},
            ),
            # Multiple entities
            (
                {"models": ["fraud_detection:v1"], "entities": {"cust_no": ["X123456", "1002"]}},
                {"status_code": HTTP_OK, "results_count": EXPECTED_RESULT_COUNT},
            ),
            # Invalid model format
            (
                {"models": ["invalid::model"], "entities": {"cust_no": ["X123456"]}},
                {"status_code": HTTP_OK, "results_count": 1},
            ),
        ],
    )
    def test_basic_prediction_cases(self, request_payload, expected):
        """Basic prediction cases"""
        response = self.client.post("/predict", json=request_payload)
        assert response.status_code == expected["status_code"]

        if response.status_code == HTTP_OK:
            data = response.json()
            assert len(data["results"]) == expected["results_count"]

    @pytest.mark.parametrize(
        "request_payload,should_error",
        [
            # Edge cases that should not error
            ({"models": ["fraud_detection:v1"], "entities": {"cust_no": [None]}}, False),
            ({"models": ["fraud_detection:v1"], "entities": {"cust_no": [123]}}, False),
            (
                {"models": ["fraud_detection:v1"], "entities": {"cust_no": ["missing_entity"]}},
                False,
            ),
            (
                {
                    "models": ["fraud_detection:v1"],
                    "entities": {"cust_no": [{"complex": "object"}]},
                },
                False,
            ),
            ({"models": ["nonexistent:v1"], "entities": {"cust_no": ["X123456"]}}, False),
        ],
    )
    def test_edge_cases(self, request_payload, should_error):
        """Edge case handling"""
        response = self.client.post("/predict", json=request_payload)
        if should_error:
            assert response.status_code >= HTTP_BAD_REQUEST
        else:
            assert response.status_code == HTTP_OK

    @pytest.mark.parametrize(
        "num_models,num_entities",
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 4),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 4),
        ],
    )
    def test_matrix_dimensions(self, num_models, num_entities):
        """Matrix dimension testing"""
        available_models = list(MODEL_REGISTRY.keys())
        actual_models = min(num_models, len(available_models))
        models = available_models[:actual_models] if actual_models > 0 else []
        entities = [f"entity_{i}" for i in range(num_entities)]

        payload = {
            "models": models,
            "entities": {"cust_no": entities},
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        data = response.json()
        assert len(data["results"]) == num_entities


class TestBusinessLogic:
    """Business logic tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)

    def test_model_prediction_consistency(self):
        """Prediction consistency"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]},
        }

        responses = []
        for _ in range(3):
            response = self.client.post("/predict", json=payload)
            responses.append(response.json())

        # Results should be consistent
        for i in range(1, len(responses)):
            assert responses[i]["results"][0]["values"] == responses[0]["results"][0]["values"]

    def test_invalid_models_mixed_with_valid(self):
        """Mixed valid/invalid models"""
        payload = {
            "models": ["fraud_detection:v1", "invalid::model", "credit_score:v1"],
            "entities": {"cust_no": ["X123456"]},
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        result = response.json()["results"][0]
        assert result["statuses"][0] == "200 OK"
        assert "400" in result["statuses"][1]
        assert result["statuses"][2] == "200 OK"


class TestPerformanceAndLoad:
    """Performance tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)

    @pytest.mark.parametrize("num_models,num_entities", [(3, 5), (8, 3), (10, 2)])
    def test_large_matrix_performance(self, num_models, num_entities):
        """Large matrix performance"""
        available_models = list(MODEL_REGISTRY.keys())
        # Repeat models to reach count
        models = (available_models * ((num_models // len(available_models)) + 1))[:num_models]
        entities = [f"entity_{i}" for i in range(num_entities)]

        payload = {
            "models": models,
            "entities": {"cust_no": entities},
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        data = response.json()
        assert len(data["results"]) == num_entities
        for result in data["results"]:
            assert len(result["values"]) == num_models


class TestModelGateway:
    """Comprehensive model tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)

    def test_no_entities_validation_coverage(self):
        """No entities error handling"""
        with pytest.raises(HTTPException):
            _raise_no_entities_error()

    def test_no_entities_coverage_main(self):
        """Empty entities check"""
        request = PredictionRequest(
            models=["fraud_detection:v1"], entities={"cust_no": ["X123456"]}
        )
        request.entities = {}

        with pytest.raises(HTTPException):
            asyncio.run(predict(request))

    def test_error_handling_coverage(self):
        """Error handling"""
        with patch("app.main.model_service.batch_predict") as mock_batch:
            mock_batch.side_effect = Exception("Test exception")

            payload = {
                "models": ["fraud_detection:v1"],
                "entities": {"cust_no": ["X123456"]},
            }

            response = self.client.post("/predict", json=payload)
            assert response.status_code == HTTP_INTERNAL_SERVER_ERROR


class TestIntegration:
    """Integration tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)
        self.available_models = list(MODEL_REGISTRY.keys())

    @pytest.mark.parametrize(
        "num_models,num_entities",
        [
            (1, 1),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 2),
            (4, 4),
            (1, 10),
            (4, 1),
            (0, 1),
            (1, 0),
            (0, 0),
        ],
    )
    def test_matrix_dimensions(self, num_models, num_entities):
        """Matrix dimensions"""
        actual_models = min(num_models, len(self.available_models))
        models = self.available_models[:actual_models] if actual_models > 0 else []
        entities = [f"entity_{i}" for i in range(num_entities)] if num_entities > 0 else []

        payload = {
            "models": models,
            "entities": {"cust_no": entities},
            "event_timestamp": get_current_timestamp_ms(),
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        data = response.json()
        assert data["metadata"]["models_name"] == models
        assert len(data["results"]) == num_entities

    def test_invalid_models_handling(self):
        """Invalid model handling"""
        payload = {
            "models": [
                "fraud_detection:v1",
                "invalid_model:v1",
                "credit_score:v1",
                "bad:format:model",
            ],
            "entities": {"cust_no": ["X123456"]},
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        result = response.json()["results"][0]
        assert result["statuses"][0] == "200 OK"
        assert result["statuses"][1] == "404 MODEL_NOT_FOUND"
        assert result["statuses"][2] == "200 OK"
        assert result["statuses"][3] == "400 BAD_REQUEST"

    def test_concurrent_requests(self):
        """Concurrent requests"""

        def make_request(result_queue):
            payload = {
                "models": ["fraud_detection:v1"],
                "entities": {"cust_no": ["X123456"]},
            }
            response = self.client.post("/predict", json=payload)
            result_queue.put(response.status_code == HTTP_OK)

        threads = []
        result_queue = queue.Queue()

        for _ in range(THREAD_COUNT):
            thread = threading.Thread(target=make_request, args=(result_queue,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == THREAD_COUNT
        assert all(results)

    def test_response_structure(self):
        """Response structure"""
        payload = {
            "models": ["fraud_detection:v1", "credit_score:v1"],
            "entities": {"cust_no": ["X123456", "1002"]},
        }

        response = self.client.post("/predict", json=payload)
        assert response.status_code == HTTP_OK

        data = response.json()
        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) == EXPECTED_RESULT_COUNT

        for result in data["results"]:
            assert "values" in result
            assert "statuses" in result
            assert "event_timestamp" in result
            assert len(result["values"]) == EXPECTED_RESULT_COUNT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
