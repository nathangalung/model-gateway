import json
import time
import pytest
from typing import List, Dict, Any
from fastapi.testclient import TestClient
from app.main import app
from app.services.dummy_models import MODEL_REGISTRY
from app.utils.timestamp import get_current_timestamp_ms

class TestModelGateway:
    """Model Gateway tests"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup client"""
        self.client = TestClient(app)
        self.available_models = list(MODEL_REGISTRY.keys())
    
    def test_health_endpoint(self):
        """Health check"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "ok"
    
    def test_models_endpoint(self):
        """Models listing"""
        response = self.client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert len(data["available_models"]) > 0
    
    # Model validation tests
    @pytest.mark.parametrize("models,should_succeed", [
        # Valid formats
        (["fraud_detection:v1"], True),
        (["credit_score:v1"], True),
        (["fraud_detection:v1", "credit_score:v1"], True),
        (["fraud_detection:v2", "credit_score:v2"], True),
        (["model_name:version"], True),
        (["a:b"], True),
        # Invalid: Multiple colons
        (["fraud:detection:v1"], False),
        (["model:with:multiple:colons"], False),
        (["a:b:c:d"], False),
        # Invalid: No colon
        (["fraud_detection"], False),
        (["no_delimiter"], False),
        (["modelname"], False),
        # Invalid: Empty parts
        ([":v1"], False),
        (["model:"], False),
        ([":"], False),
        ([""], False),
        # Valid but nonexistent
        (["invalid_model:v1"], True),
        (["nonexistent:v99"], True),
        # Mixed formats
        (["fraud_detection:v1", "invalid:format:here", "credit_score:v1"], False),
        (["valid:format", "no_colon", "another:valid"], False),
        # Multiple valid
        (["fraud_detection:v1", "credit_score:v1", "fraud_detection:v2"], True),
    ])
    def test_model_format_validation(self, models, should_succeed):
        """Model format validation"""
        payload = {
            "models": models,
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": get_current_timestamp_ms()
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        
        if should_succeed and models:
            data = response.json()
            result = data["results"][0]
            for status in result["statuses"]:
                assert "400 BAD_REQUEST" not in status
        elif not should_succeed and models:
            data = response.json()
            result = data["results"][0]
            has_bad_request = any("400 BAD_REQUEST" in status for status in result["statuses"])
            assert has_bad_request
    
    @pytest.mark.parametrize("models", [
        [],
        ["fraud_detection:v1"],
        ["fraud_detection:v1", "credit_score:v1"],
        ["fraud_detection:v1", "credit_score:v1", "fraud_detection:v2", "credit_score:v2"],
        ["model:v1", "another:v2", "third:v3", "fourth:v4", "fifth:v5"],
    ])
    def test_multiple_models(self, models):
        """Multiple model handling"""
        payload = {
            "models": models,
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": get_current_timestamp_ms()
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["metadata"]["models_name"] == models
        
        if models and data["results"]:
            result = data["results"][0]
            assert len(result["values"]) == len(models)
            assert len(result["statuses"]) == len(models)
            assert len(result["event_timestamp"]) == len(models)
    
    # Entity validation tests
    @pytest.mark.parametrize("entities,expected_status", [
        # String entities
        ({"cust_no": ["X123456"]}, 200),
        ({"cust_no": ["X123456", "1002", "1003"]}, 200),
        ({"cust_no": ["MISSING_ENTITY", "ANOTHER_MISSING"]}, 200),
        ({"cust_no": [""]}, 200),
        # Numeric entities
        ({"cust_no": [123456]}, 200),
        ({"cust_no": [1, 2, 3, 999999]}, 200),
        ({"cust_no": [0]}, 200),
        ({"cust_no": [-1, -999]}, 200),
        # Big int entities
        ({"cust_no": [9223372036854775807]}, 200),
        ({"cust_no": [-9223372036854775808]}, 200),
        ({"cust_no": [999999999999999999]}, 200),
        # Float entities
        ({"cust_no": [123.456, 789.012]}, 200),
        ({"cust_no": [0.0, -0.0, 3.14159]}, 200),
        # Mixed types
        ({"cust_no": ["X123456", 1002, "1003", 999]}, 200),
        ({"cust_no": [123, "ABC", 456.78, "XYZ999", True, False]}, 200),
        ({"cust_no": ["string", 42, 3.14, True, None]}, 200),
        # Boolean entities
        ({"cust_no": [True, False]}, 200),
        # Null entities
        ({"cust_no": [None, "valid", None]}, 200),
        # Object entities
        ({"cust_no": [{"id": "X123456", "type": "premium"}]}, 200),
        ({"cust_no": [{"id": 1002}, {"id": "X123456"}]}, 200),
        ({"cust_no": [{"customer": {"id": 123, "segment": "A"}}]}, 200),
        ({"cust_no": [{"nested": {"deeply": {"value": 42}}}]}, 200),
        # Array entities
        ({"cust_no": [["array", "of", "values"]]}, 200),
        ({"cust_no": [[1, 2, 3], ["a", "b", "c"]]}, 200),
        # Different entity keys
        ({"customer_id": ["X123456"]}, 200),
        ({"entity_ids": [1, 2, 3]}, 200),
        ({"ids": [{"customer": 123}, {"customer": 456}]}, 200),
        ({"account_numbers": ["ACC123", "ACC456"]}, 200),
        # Multiple entity types
        ({"cust_no": ["X123456"], "account_no": [123]}, 200),
        ({"customer_id": ["C1", "C2"], "product_id": [1, 2, 3]}, 200),
        # Special characters
        ({"cust_no": ["X@123456", "user#789", "test$entity"]}, 200),
        ({"cust_no": ["id%with&symbols", "unicode™entity", "emoji🚀id"]}, 200),
        # Long strings
        ({"cust_no": ["very_long_string_entity_id_that_could_cause_issues_in_processing"]}, 200),
        ({"cust_no": ["x" * 1000]}, 200),
        # Empty entities
        ({}, 400),
        ({"cust_no": []}, 200),
    ])
    def test_entity_validation(self, entities, expected_status):
        """Entity validation"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": entities,
            "event_timestamp": get_current_timestamp_ms()
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == expected_status
    
    # Timestamp tests
    def test_timestamp_optional(self):
        """Optional timestamp"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]}
        }
        
        before_request = get_current_timestamp_ms()
        response = self.client.post("/predict", json=payload)
        after_request = get_current_timestamp_ms()
        
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        if data["results"]:
            response_timestamp = data["results"][0]["event_timestamp"][0]
            assert before_request <= response_timestamp <= after_request + 1000
    
    def test_timestamp_provided_response_is_current(self):
        """Response always current"""
        old_timestamp = 1640995200000
        
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": old_timestamp
        }
        
        before_request = get_current_timestamp_ms()
        response = self.client.post("/predict", json=payload)
        after_request = get_current_timestamp_ms()
        
        assert response.status_code == 200
        
        data = response.json()
        response_timestamp = data["results"][0]["event_timestamp"][0]
        
        assert response_timestamp >= before_request
        assert response_timestamp <= after_request + 1000
        assert response_timestamp != old_timestamp
    
    @pytest.mark.parametrize("timestamp", [
        1751429485000,
        1640995200000,
        0,
        999999999999,
        -1,
        9999999999999,
        1234567890123,
    ])
    def test_timestamp_formats_response_always_current(self, timestamp):
        """Current time response"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": timestamp
        }
        
        before_request = get_current_timestamp_ms()
        response = self.client.post("/predict", json=payload)
        after_request = get_current_timestamp_ms()
        
        assert response.status_code == 200
        
        data = response.json()
        response_timestamp = data["results"][0]["event_timestamp"][0]
        
        assert before_request <= response_timestamp <= after_request + 1000
    
    # Matrix dimension tests
    @pytest.mark.parametrize("num_models,num_entities", [
        (1, 1), (1, 3), (2, 1), (2, 3), (3, 2), (4, 4),
        (3, 5), (5, 3), (1, 10), (4, 1), (0, 1), (1, 0), (0, 0),
    ])
    def test_matrix_dimensions(self, num_models, num_entities):
        """Matrix dimensions"""
        models = self.available_models[:num_models] if num_models > 0 else []
        entities = [f"entity_{i}" for i in range(num_entities)] if num_entities > 0 else []
        
        payload = {
            "models": models,
            "entities": {"cust_no": entities},
            "event_timestamp": get_current_timestamp_ms()
        }
        
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "metadata" in data
        assert "results" in data
        
        assert data["metadata"]["models_name"] == models
        assert len(data["results"]) == num_entities
        
        if num_entities > 0:
            for result in data["results"]:
                assert len(result["values"]) == num_models
                assert len(result["statuses"]) == num_models
                assert len(result["event_timestamp"]) == num_models
    
    # Business logic tests
    def test_prediction_consistency(self):
        """Prediction consistency"""
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": get_current_timestamp_ms()
        }
        
        responses = []
        for _ in range(3):
            response = self.client.post("/predict", json=payload)
            assert response.status_code == 200
            responses.append(response.json())
        
        for i in range(1, len(responses)):
            assert responses[i]["results"][0]["values"] == responses[0]["results"][0]["values"]
            assert responses[i]["results"][0]["statuses"] == responses[0]["results"][0]["statuses"]
    
    def test_invalid_models_handling(self):
        """Invalid model handling"""
        payload = {
            "models": ["fraud_detection:v1", "invalid_model:v1", "credit_score:v1", "bad:format:model"],
            "entities": {"cust_no": ["X123456"]},
            "event_timestamp": get_current_timestamp_ms()
        }
        
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        result = data["results"][0]
        
        assert len(result["values"]) == 4
        assert len(result["statuses"]) == 4
        
        assert result["statuses"][0] == "200 OK"
        assert result["statuses"][1] == "404 MODEL_NOT_FOUND"
        assert result["statuses"][2] == "200 OK"
        assert result["statuses"][3] == "400 BAD_REQUEST"
    
    # Performance tests
    @pytest.mark.parametrize("num_models,num_entities", [
        (2, 5), (3, 10), (4, 15), (2, 25), (5, 5),
    ])
    def test_batch_performance(self, num_models, num_entities):
        """Batch performance"""
        models = self.available_models[:num_models]
        entities = [f"entity_{i}" for i in range(num_entities)]
        
        payload = {
            "models": models,
            "entities": {"cust_no": entities},
            "event_timestamp": get_current_timestamp_ms()
        }
        
        start_time = time.time()
        response = self.client.post("/predict", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        
        processing_time = (end_time - start_time) * 1000
        assert processing_time < 10000
        
        data = response.json()
        assert len(data["results"]) == num_entities
        for result in data["results"]:
            assert len(result["values"]) == num_models
    
    # Edge cases
    def test_concurrent_requests(self):
        """Concurrent requests"""
        import threading
        import queue
        
        def make_request(result_queue):
            payload = {
                "models": ["fraud_detection:v1"],
                "entities": {"cust_no": ["X123456"]},
                "event_timestamp": get_current_timestamp_ms()
            }
            response = self.client.post("/predict", json=payload)
            result_queue.put(response.status_code == 200)
        
        threads = []
        result_queue = queue.Queue()
        
        for _ in range(5):
            thread = threading.Thread(target=make_request, args=(result_queue,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 5
        assert all(results)
    
    def test_large_payload(self):
        """Large payload"""
        large_entities = [f"entity_{i}" for i in range(100)]
        
        payload = {
            "models": ["fraud_detection:v1", "credit_score:v1"],
            "entities": {"cust_no": large_entities},
            "event_timestamp": get_current_timestamp_ms()
        }
        
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["results"]) == 100
        for result in data["results"]:
            assert len(result["values"]) == 2
    
    def test_response_structure_completeness(self):
        """Response structure"""
        payload = {
            "models": ["fraud_detection:v1", "credit_score:v1"],
            "entities": {"cust_no": ["X123456", "1002"]},
            "event_timestamp": get_current_timestamp_ms()
        }
        
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        
        assert "metadata" in data
        assert "results" in data
        
        metadata = data["metadata"]
        assert "models_name" in metadata
        assert metadata["models_name"] == ["fraud_detection:v1", "credit_score:v1"]
        
        results = data["results"]
        assert len(results) == 2
        
        for result in results:
            assert "values" in result
            assert "statuses" in result
            assert "event_timestamp" in result
            
            assert len(result["values"]) == 2
            assert len(result["statuses"]) == 2
            assert len(result["event_timestamp"]) == 2
            
            for value in result["values"]:
                assert value is None or isinstance(value, (int, float))
            
            for status in result["statuses"]:
                assert isinstance(status, str)
                assert any(code in status for code in ["200", "400", "404", "500"])
            
            for timestamp in result["event_timestamp"]:
                assert isinstance(timestamp, int)
                assert timestamp > 0
    
    def test_gmt_timezone_consistency(self):
        """GMT timezone"""
        import datetime
        
        payload = {
            "models": ["fraud_detection:v1"],
            "entities": {"cust_no": ["X123456"]}
        }
        
        utc_before = datetime.datetime.now(datetime.timezone.utc)
        response = self.client.post("/predict", json=payload)
        utc_after = datetime.datetime.now(datetime.timezone.utc)
        
        assert response.status_code == 200
        
        data = response.json()
        response_timestamp = data["results"][0]["event_timestamp"][0]
        
        response_dt = datetime.datetime.fromtimestamp(response_timestamp / 1000, tz=datetime.timezone.utc)
        
        assert utc_before <= response_dt <= utc_after

if __name__ == "__main__":
    pytest.main([__file__, "-v"])