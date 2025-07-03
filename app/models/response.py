from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class ModelResult(BaseModel):
    values: List[Optional[float]] = Field(..., description="Model prediction values")
    statuses: List[str] = Field(..., description="HTTP status codes with descriptions (200 OK, 400 BAD_REQUEST, 404 MODEL_NOT_FOUND, 500 SERVER_ERROR)")
    event_timestamp: List[int] = Field(..., description="Timestamp for each prediction")

class ResponseMetadata(BaseModel):
    models_name: List[str] = Field(..., description="List of model names from request")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metadata": {
                    "models_name": ["fraud_detection:v1", "credit_score:v1"]
                },
                "results": [
                    {
                        "values": [0.72, 0.85],
                        "statuses": ["200 OK", "200 OK"],
                        "event_timestamp": [1751429485000, 1751429485000]
                    }
                ]
            }
        }
    )
    
    metadata: ResponseMetadata
    results: List[ModelResult]