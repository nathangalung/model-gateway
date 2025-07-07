import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Add project root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)
