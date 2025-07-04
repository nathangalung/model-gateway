import pytest
from fastapi.testclient import TestClient

from app.main import app

# Constants
HTTP_OK = 200


class TestHealth:
    """Health and models tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Health check"""
        response = self.client.get("/health")
        assert response.status_code == HTTP_OK
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "ok"

    def test_models_endpoint(self):
        """Models listing"""
        response = self.client.get("/models")
        assert response.status_code == HTTP_OK
        data = response.json()
        assert "available_models" in data
        assert len(data["available_models"]) > 0
