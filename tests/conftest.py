import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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