# Model Gateway

Simple FastAPI gateway for model predictions.

## Usage

### Start server:
```bash
docker-compose up app
```

### Run tests:
```bash
docker-compose up test
```

## API

### Prediction
```
POST /predict
```

### Health
```
GET /health
```

### Models
```
GET /models
```

## Request Format
```json
{
  "models": ["fraud_detection:v1"],
  "entities": {"cust_no": ["X123456"]},
  "event_timestamp": 1751429485000
}
```

## Response Format
```json
{
  "metadata": {"models_name": ["fraud_detection:v1"]},
  "results": [{
    "values": [0.75],
    "statuses": ["200 OK"],
    "event_timestamp": [1751429485010]
  }]
}
```