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

        # Handle empty cases
        if not entity_ids:
            return PredictionResponse(
                metadata=ResponseMetadata(models_name=request.models), results=[]
            )

        if not request.models:
            results = []
            for _ in entity_ids:
                results.append(ModelResult(values=[], statuses=[], event_timestamp=[]))
            return PredictionResponse(metadata=ResponseMetadata(models_name=[]), results=results)

        # Batch prediction
        prediction_results = await model_service.batch_predict(request.models, entity_ids)

        # Always current timestamp
        current_timestamp = get_current_timestamp_ms()

        # Format results
        formatted_results = []
        for entity_result in prediction_results:
            timestamps = [current_timestamp] * len(request.models)
            formatted_results.append(
                ModelResult(
                    values=entity_result["values"],
                    statuses=entity_result["statuses"],
                    event_timestamp=timestamps,
                )
            )

        return PredictionResponse(
            metadata=ResponseMetadata(models_name=request.models), results=formatted_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@app.get("/models")
async def list_models():
    """List available models"""
    return {"available_models": list(MODEL_REGISTRY.keys())}
