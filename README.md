# Model Gateway

A high-performance FastAPI service for batch ML model predictions with comprehensive testing and CI/CD pipeline.

## Quick Start

### Using Docker (Recommended)
```bash
# Start the application server
docker-compose up app

# Run all tests
docker-compose up test

# Development mode with hot reload
docker-compose up dev
```

### Local Development
```bash
# Install dependencies with uv
uv sync

# Start development server
uv run uvicorn app.main:app --reload --port 8000

# Run tests with coverage
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Code quality checks
uv run ruff check app/ tests/
```

## Project Overview

### Architecture
```
model_gateway/
├── app/                    # Main application
│   ├── main.py            # FastAPI endpoints
│   ├── models/            # Request/Response models
│   ├── services/          # Business logic & ML models
│   └── utils/             # Utility functions
├── tests/                 # Comprehensive test suite (98%+ coverage)
├── data/                  # Sample entity features
├── .github/workflows/     # CI/CD pipeline
└── docker-compose.yaml    # Container orchestration
```

### Key Features
- **Batch Processing**: Handle multiple models and entities in one request
- **Model Registry**: Centralized management of ML models
- **Deterministic Results**: Consistent predictions for testing
- **Health Monitoring**: Built-in health checks and model listing
- **Comprehensive Testing**: 98%+ test coverage with edge cases
- **CI/CD Pipeline**: Automated testing, linting, and Docker integration

## API Documentation

### Endpoints
| Method | Endpoint | Description | Example |
|--------|----------|-------------|---------|
| `GET` | `/health` | Health check | `{"status": "ok", "timestamp": 1751429485000}` |
| `GET` | `/models` | List available models | `{"available_models": ["fraud_detection:v1", ...]}` |
| `POST` | `/predict` | Batch predictions | See examples below |

### Available Models
- `fraud_detection:v1` - Basic fraud detection (requires: amount)
- `fraud_detection:v2` - Advanced fraud detection (requires: amount, merchant_category)
- `credit_score:v1` - Basic credit scoring (requires: income)
- `credit_score:v2` - Advanced credit scoring (requires: income, age)

## API Usage Examples

### Basic Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["fraud_detection:v1"],
    "entities": {"cust_no": ["X123456"]},
    "event_timestamp": 1751429485000
  }'
```

### Multiple Models and Entities
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["fraud_detection:v1", "credit_score:v1"],
    "entities": {"cust_no": ["X123456", "1002", "1003"]},
    "event_timestamp": 1751429485000
  }'
```

### Request Format
```json
{
  "models": ["fraud_detection:v1", "credit_score:v1"],
  "entities": {"cust_no": ["X123456", "1002"]},
  "event_timestamp": 1751429485000
}
```

### Response Format
```json
{
  "metadata": {
    "models_name": ["fraud_detection:v1", "credit_score:v1"]
  },
  "results": [
    {
      "values": [0.75, 0.82],
      "statuses": ["200 OK", "200 OK"],
      "event_timestamp": [1751429485010, 1751429485010]
    },
    {
      "values": [0.23, 0.91],
      "statuses": ["200 OK", "200 OK"],
      "event_timestamp": [1751429485010, 1751429485010]
    }
  ]
}
```

## Testing

### Test Structure
```
tests/
├── test_health.py         # Health & models endpoints
├── test_main.py          # Integration & comprehensive tests
├── test_predictions.py   # Prediction functionality
├── test_validation.py    # Input validation
├── test_services.py      # Service layer tests
├── test_models.py        # Pydantic model tests
├── test_coverage.py      # Edge cases & coverage
└── conftest.py           # Test configuration
```

### Running Tests
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test categories
uv run pytest tests/test_health.py -v          # Health endpoints
uv run pytest tests/test_predictions.py -v    # Prediction logic
uv run pytest tests/test_validation.py -v     # Input validation

# Generate HTML coverage report
uv run pytest tests/ --cov=app --cov-report=html
```

### Test Coverage: 98%+
- **Unit Tests**: Individual components
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Load and response time
- **Edge Cases**: Error handling and boundaries
- **Validation Tests**: Request/response validation

## Docker Usage

### Build and Run
```bash
# Build the image
docker build -t model-gateway .

# Run the container
docker run -p 8000:8000 model-gateway

# Use docker-compose for full setup
docker compose up app
```

### Container Features
- Health checks for orchestration
- Multi-stage build with uv package manager
- Optimized for production deployment
- CORS middleware enabled

## Development

### Code Quality Tools
```bash
# Linting and formatting
uv run ruff check app/ tests/           # Fast linting
uv run black app/ tests/                # Code formatting
uv run isort app/ tests/                # Import sorting

# Type checking
uv run mypy app/                        # Static type checking
```

### Project Configuration
- **Python**: 3.13+
- **Package Manager**: uv (faster than pip)
- **Framework**: FastAPI + Uvicorn
- **Testing**: pytest with comprehensive fixtures
- **Linting**: ruff + black + isort
- **CI/CD**: GitHub Actions

### Environment Setup
1. **Install uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Clone repository**: `git clone <repo-url>`
3. **Install dependencies**: `uv sync`
4. **Start development**: `uv run uvicorn app.main:app --reload`

## Sample Data

### Entity Features (data/dummy_features.json)
```json
{
  "X123456": {
    "amount": 1500.50,
    "merchant_category": "grocery",
    "income": 75000,
    "age": 35,
    "credit_history": 7
  },
  "1002": {
    "amount": 250.00,
    "merchant_category": "restaurant", 
    "income": 45000,
    "age": 28,
    "credit_history": 3
  }
}
```

## CI/CD Pipeline

### GitHub Actions Workflow
- **Test Job**: Runs all tests with coverage reporting
- **Lint Job**: Code quality checks (ruff, black, isort)
- **Docker Test**: Build and test Docker image
- **Integration Test**: End-to-end testing with Docker Compose

### Pipeline Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` branch
- Automated testing on every commit

## Performance

### Monitoring
```bash
# Health check
curl http://localhost:8000/health

# Available models
curl http://localhost:8000/models

# Docker health check
docker ps --format "table {{.Names}}\t{{.Status}}"
```