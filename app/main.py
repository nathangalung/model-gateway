from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models.request import PredictionRequest
from .models.response import ModelResult, PredictionResponse, ResponseMetadata
from .services.dummy_models import MODEL_REGISTRY
from .services.model_service import ModelService
from .utils.timestamp import get_current_timestamp_ms

app = FastAPI(title="Model Gateway", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"])
model_service = ModelService()


def _raise_no_entities_error() -> None:
    """Raise no entities error"""
    raise HTTPException(status_code=400, detail="No entities provided")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "ok", "timestamp": get_current_timestamp_ms()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    try:
        if not request.entities:
            _raise_no_entities_error()

        entity_key = next(iter(request.entities))
        entity_ids = request.entities[entity_key]

        # Build metadata with entity key and models
        metadata_models = [entity_key] + request.models

        # Handle empty entity IDs
        if not entity_ids:
            return PredictionResponse(
                metadata=ResponseMetadata(models_name=metadata_models), results=[]
            )

        # Handle empty models
        if not request.models:
            results = []
            current_timestamp = get_current_timestamp_ms()
            for entity_id in entity_ids:
                results.append(
                    ModelResult(
                        values=[str(entity_id)],
                        statuses=["200 OK"],
                        event_timestamp=[current_timestamp],
                    )
                )
            return PredictionResponse(
                metadata=ResponseMetadata(models_name=[entity_key]), results=results
            )

        # Batch prediction
        prediction_results = await model_service.batch_predict(request.models, entity_ids)
        current_timestamp = get_current_timestamp_ms()

        # Format results with entity ID first
        formatted_results = []
        for entity_result in prediction_results:
            entity_id = str(entity_result["entity_id"])
            values = [entity_id] + entity_result["values"]
            statuses = ["200 OK"] + entity_result["statuses"]
            timestamps = [current_timestamp] * len(values)

            formatted_results.append(
                ModelResult(
                    values=values,
                    statuses=statuses,
                    event_timestamp=timestamps,
                )
            )

        return PredictionResponse(
            metadata=ResponseMetadata(models_name=metadata_models), results=formatted_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@app.get("/models")
async def list_models():
    """List available models"""
    return {"available_models": list(MODEL_REGISTRY.keys())}
