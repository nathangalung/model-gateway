from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    """Individual entity result"""

    values: list[float | None] = Field(..., description="Prediction values")
    statuses: list[str] = Field(..., description="Status codes")
    event_timestamp: list[int] = Field(..., description="Event timestamps")


class ResponseMetadata(BaseModel):
    """Response metadata"""

    models_name: list[str] = Field(..., description="Model names used")


class PredictionResponse(BaseModel):
    """Batch prediction response"""

    metadata: ResponseMetadata
    results: list[ModelResult]
