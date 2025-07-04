import pytest

from app.services.model_service import ModelService

# Constants
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500
PERFORMANCE_LIMIT = 10000
ENTITY_COUNT = 2


class TestModelService:
    """Test model service functionality"""

    @pytest.fixture
    def service(self):
        return ModelService()

    def test_get_features_for_entity(self, service):
        """Test feature retrieval"""
        # Known entity
        features = service._get_features("X123456")
        assert features is not None
        assert "amount" in features

        # Unknown entity
        features = service._get_features("unknown")
        assert features is None

    async def test_predict_model_success(self, service):
        """Test successful prediction"""
        features = {"amount": 100.0}
        prediction, status = await service.predict_single("fraud_detection:v1", features)
        assert prediction is not None
        assert status == "200 OK"

    async def test_predict_model_invalid(self, service):
        """Test invalid model prediction"""
        features = {"amount": 100.0}

        # Invalid model format
        prediction, status = await service.predict_single("invalid::model", features)
        assert prediction is None
        assert status == "400 BAD_REQUEST"

        # Model not found
        prediction, status = await service.predict_single("nonexistent:v1", features)
        assert prediction is None
        assert status == "404 MODEL_NOT_FOUND"

    async def test_batch_predict(self, service):
        """Test batch prediction"""
        models = ["fraud_detection:v1", "credit_score:v1"]
        entities = ["X123456", "1002"]

        results = await service.batch_predict(models, entities)
        assert len(results) == ENTITY_COUNT

        for result in results:
            assert len(result["values"]) == ENTITY_COUNT
            assert len(result["statuses"]) == ENTITY_COUNT
