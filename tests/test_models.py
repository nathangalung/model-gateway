import pytest
from pydantic import ValidationError

from app.models.request import PredictionRequest


class TestRequestModels:
    """Test request model validation"""

    def test_valid_request(self):
        """Valid request creation"""
        request = PredictionRequest(
            models=["fraud_detection:v1"],
            entities={"cust_no": ["X123456"]},
            event_timestamp=1751429485000,
        )
        assert request.models == ["fraud_detection:v1"]
        assert request.entities == {"cust_no": ["X123456"]}

    def test_invalid_models_type(self):
        """Invalid models type"""
        with pytest.raises(ValidationError):
            PredictionRequest(models="not_a_list", entities={"cust_no": ["X123456"]})

    def test_empty_entities(self):
        """Empty entities"""
        with pytest.raises(ValidationError):
            PredictionRequest(models=["fraud_detection:v1"], entities={})
