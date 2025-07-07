import pytest

from app.models.request import PredictionRequest
from app.models.response import ModelResult, PredictionResponse, ResponseMetadata

# Test constants
EXPECTED_MODELS_COUNT = 2
EXPECTED_ENTITIES_COUNT = 2
EXPECTED_TIMESTAMP = 1751429485000
EXPECTED_RESPONSE_TIMESTAMP = 1751429485010
EXPECTED_VALUES_COUNT = 3
EXPECTED_METADATA_COUNT = 3


class TestRequestModelsExtended:
    """Request model validation tests"""

    def test_valid_request_with_all_fields(self):
        """Complete request validation"""
        request = PredictionRequest(
            models=["fraud_detection:v1", "credit_score:v1"],
            entities={"cust_no": ["X123456", "1002"]},
            event_timestamp=EXPECTED_TIMESTAMP,
        )
        assert len(request.models) == EXPECTED_MODELS_COUNT
        assert len(request.entities["cust_no"]) == EXPECTED_ENTITIES_COUNT
        assert request.event_timestamp == EXPECTED_TIMESTAMP

    def test_request_without_timestamp(self):
        """Optional timestamp field"""
        request = PredictionRequest(
            models=["fraud_detection:v1"],
            entities={"cust_no": ["X123456"]},
        )
        assert request.event_timestamp is None

    def test_models_validator_coverage(self):
        """Models field validation"""
        # Valid case
        valid_models = ["fraud_detection:v1"]
        assert PredictionRequest.validate_models(valid_models) == valid_models

        # Invalid type
        with pytest.raises(TypeError):
            PredictionRequest.validate_models("not_a_list")

    def test_entities_validator_coverage(self):
        """Entities field validation"""
        # Valid case
        valid_entities = {"cust_no": ["X123456"]}
        assert PredictionRequest.validate_entities(valid_entities) == valid_entities

        # Invalid type
        with pytest.raises(TypeError):
            PredictionRequest.validate_entities("not_a_dict")

        # Empty dict
        with pytest.raises(ValueError):
            PredictionRequest.validate_entities({})


class TestResponseModels:
    """Response model tests"""

    def test_model_result_creation(self):
        """Model result structure"""
        timestamp_list = [
            EXPECTED_RESPONSE_TIMESTAMP,
            EXPECTED_RESPONSE_TIMESTAMP,
            EXPECTED_RESPONSE_TIMESTAMP,
        ]
        result = ModelResult(
            values=["X123456", 0.75, 0.82],
            statuses=["200 OK", "200 OK", "200 OK"],
            event_timestamp=timestamp_list,
        )
        assert len(result.values) == EXPECTED_VALUES_COUNT
        assert len(result.statuses) == EXPECTED_VALUES_COUNT
        assert len(result.event_timestamp) == EXPECTED_VALUES_COUNT

    def test_response_metadata_creation(self):
        """Metadata structure validation"""
        metadata = ResponseMetadata(
            models_name=["cust_no", "fraud_detection:v1", "credit_score:v1"]
        )
        assert len(metadata.models_name) == EXPECTED_METADATA_COUNT
        assert metadata.models_name[0] == "cust_no"

    def test_prediction_response_creation(self):
        """Full response structure"""
        metadata = ResponseMetadata(models_name=["cust_no", "fraud_detection:v1"])
        result = ModelResult(
            values=["X123456", 0.75],
            statuses=["200 OK", "200 OK"],
            event_timestamp=[EXPECTED_RESPONSE_TIMESTAMP, EXPECTED_RESPONSE_TIMESTAMP],
        )
        response = PredictionResponse(metadata=metadata, results=[result])

        assert len(response.metadata.models_name) == EXPECTED_MODELS_COUNT
        assert len(response.results) == 1
        assert len(response.results[0].values) == EXPECTED_MODELS_COUNT
