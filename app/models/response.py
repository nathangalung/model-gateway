from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    """Individual result for entity prediction"""

    values: list[float | None] = Field(..., description="List of prediction values")
    statuses: list[str] = Field(
        ...,
        description="Status codes with descriptions "
        "(200 OK, 400 BAD_REQUEST, 404 MODEL_NOT_FOUND, 500 SERVER_ERROR)",
    )
    event_timestamp: list[int] = Field(..., description="Event timestamp for each prediction")


class ResponseMetadata(BaseModel):
    """Response metadata"""

    models_name: list[str] = Field(..., description="List of model names used in prediction")


class PredictionResponse(BaseModel):
    """Response model for batch prediction"""

    metadata: ResponseMetadata
    results: list[ModelResult]
