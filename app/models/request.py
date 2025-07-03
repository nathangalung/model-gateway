from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request model for batch prediction"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "models": ["fraud_detection:v1", "credit_score:v1"],
                "entities": {"cust_no": ["X123456", "1002"]},
                "event_timestamp": 1751429485000,
            }
        }
    }

    models: list[str] = Field(
        ..., description="List of model names with versions in format 'model_name:version'"
    )
    entities: dict[str, list[Any]] = Field(..., description="Entity mapping with IDs")
    event_timestamp: int | None = Field(None, description="Event timestamp in milliseconds GMT+0")

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        if not isinstance(v, list):
            raise TypeError("models must be a list")

        # Allow empty list and any model format - validation happens at service level
        return v

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, v):
        if not isinstance(v, dict):
            raise TypeError("entities must be a dictionary")
        if not v:
            raise ValueError("entities cannot be empty")  # noqa: EM101
        return v
