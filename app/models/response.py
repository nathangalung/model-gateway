from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    """Individual entity result"""

    values: list[str | float | None] = Field(..., description="Entity ID and prediction values")
    statuses: list[str] = Field(..., description="Status codes for entity and models")
    event_timestamp: list[int] = Field(..., description="Event timestamps")


class ResponseMetadata(BaseModel):
    """Response metadata"""

    models_name: list[str] = Field(..., description="Entity key and model names used")


class PredictionResponse(BaseModel):
    """Batch prediction response"""

    metadata: ResponseMetadata
    results: list[ModelResult]
