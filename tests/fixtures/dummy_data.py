import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def sample_request():
    """Sample valid request payload"""
    return {
        "models": ["fraud_detection:v1", "credit_score:v1"],
        "entities": {"cust_no": ["X123456", "1002"]}
    }