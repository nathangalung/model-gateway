import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .dummy_models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model predictions"""

    def __init__(self):
        """Initialize model service with dummy features"""
        self.dummy_features = self._load_dummy_features()

    def _load_dummy_features(self) -> dict[str, dict[str, Any]]:
        """Load dummy features from JSON file"""
        try:
            data_path = Path(__file__).parent.parent.parent / "data" / "dummy_features.json"
            if data_path.exists():
                return json.loads(data_path.read_text())
            return self._default_features()
        except (json.JSONDecodeError, OSError):
            return self._default_features()

    def _default_features(self) -> dict[str, dict[str, Any]]:
        """Default dummy features"""
        return {
            "X123456": {
                "amount": 1500.50,
                "merchant_category": "grocery",
                "income": 75000,
                "age": 35,
            },
            "1002": {
                "amount": 250.00,
                "merchant_category": "restaurant",
                "income": 45000,
                "age": 28,
            },
            "1003": {
                "amount": 3500.75,
                "merchant_category": "electronics",
                "income": 95000,
                "age": 42,
            },
        }

    def _get_features(self, entity_id: str | int) -> dict[str, Any] | None:
        """Get features for entity ID"""
        return self.dummy_features.get(str(entity_id))

    async def predict_single(
        self, model_name: str, features: dict[str, Any] | None
    ) -> tuple[float | None, str]:
        """Single model prediction"""
        await asyncio.sleep(0.005)

        # Validate model format: exactly one colon
        if model_name.count(":") != 1 or not all(part.strip() for part in model_name.split(":")):
            return None, "400 BAD_REQUEST"

        model = MODEL_REGISTRY.get(model_name)
        if not model:
            return None, "404 MODEL_NOT_FOUND"

        if not features:
            return None, "400 BAD_REQUEST"

        try:
            prediction = model.predict(features)
            if prediction is not None:
                return prediction, "200 OK"
        except Exception:
            logger.exception("Model prediction failed")

        return None, "500 SERVER_ERROR"

    async def batch_predict(self, models: list[str], entity_ids: list[Any]) -> list[dict[str, Any]]:
        """Batch prediction"""
        if not models:
            return [
                {"entity_id": entity_id, "values": [], "statuses": []} for entity_id in entity_ids
            ]

        if not entity_ids:
            return []

        results = []
        for entity_id in entity_ids:
            features = self._get_features(entity_id)

            # Predict for all models for this entity
            predictions = []
            statuses = []

            for model_name in models:
                prediction, status = await self.predict_single(model_name, features)
                predictions.append(prediction)
                statuses.append(status)

            results.append({"entity_id": entity_id, "values": predictions, "statuses": statuses})

        return results
