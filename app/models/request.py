from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional
import re

class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": ["fraud_detection:v1", "credit_score:v1"],
                "entities": {"cust_no": ["X123456", "1002"]},
                "event_timestamp": 1751429485000
            }
        }
    )
    
    models: List[str] = Field(..., description="List of model names with versions in format 'model_name:version'")
    entities: Dict[str, List[Any]] = Field(..., description="Entity mapping with IDs")
    event_timestamp: Optional[int] = Field(None, description="Event timestamp in milliseconds GMT+0")
    
    @field_validator('models')
    @classmethod
    def validate_models(cls, v):
        if not isinstance(v, list):
            raise ValueError("models must be a list")
        
        # Allow empty list and any model format - validation happens at service level
        # This allows the API to accept requests and handle errors gracefully
        return v
    
    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v):
        if not isinstance(v, dict):
            raise ValueError("entities must be a dictionary")
        if not v:
            raise ValueError("entities cannot be empty")
        return v