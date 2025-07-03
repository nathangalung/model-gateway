import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class ModelService:
    """Service for model predictions"""
    
    def __init__(self):
        self.dummy_features = self._load_features()
    
    def _load_features(self) -> Dict[str, Any]:
        """Load dummy features"""
        try:
            data_path = Path(__file__).parent.parent.parent / "data" / "dummy_features.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    return json.loads(f.read())
            return self._default_features()
        except Exception:
            return self._default_features()
    
    def _default_features(self) -> Dict[str, Any]:
        """Default dummy features"""
        return {
            "X123456": {"amount": 1500.50, "merchant_category": "grocery", "income": 75000, "age": 35},
            "1002": {"amount": 250.00, "merchant_category": "restaurant", "income": 45000, "age": 28},
            "1003": {"amount": 3500.75, "merchant_category": "electronics", "income": 95000, "age": 42}
        }
    
    def _validate_format(self, model_name: str) -> bool:
        """Validate model format"""
        if not isinstance(model_name, str) or model_name.count(':') != 1:
            return False
        model_part, version_part = model_name.split(':')
        return bool(model_part.strip() and version_part.strip())
    
    async def get_entity_features(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity features"""
        await asyncio.sleep(0.001)
        return self.dummy_features.get(str(entity_id))
    
    async def predict_single(self, model_name: str, features: Optional[Dict[str, Any]]) -> Tuple[Optional[float], str]:
        """Single model prediction"""
        await asyncio.sleep(0.005)
        
        if not self._validate_format(model_name):
            return None, "400 BAD_REQUEST"
        
        from .dummy_models import MODEL_REGISTRY
        
        model = MODEL_REGISTRY.get(model_name)
        if not model:
            return None, "404 MODEL_NOT_FOUND"
        
        if not features:
            return None, "400 BAD_REQUEST"
        
        try:
            prediction = model.predict(features)
            if prediction is not None:
                return prediction, "200 OK"
            else:
                return None, "400 BAD_REQUEST"
        except Exception:
            return None, "500 SERVER_ERROR"
    
    async def batch_predict(self, models: List[str], entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction"""
        if not models:
            return [{"entity_id": entity_id, "values": [], "statuses": []} for entity_id in entity_ids]
        
        if not entity_ids:
            return []
        
        # Get all entity features
        entity_features = {}
        for entity_id in entity_ids:
            features = await self.get_entity_features(str(entity_id))
            entity_features[entity_id] = features
        
        # Create prediction tasks
        tasks = []
        for entity_id in entity_ids:
            for model_name in models:
                task = self.predict_single(model_name, entity_features[entity_id])
                tasks.append(task)
        
        # Execute predictions
        results = await asyncio.gather(*tasks)
        
        # Reshape to matrix
        num_models = len(models)
        matrix_results = []
        
        for i, entity_id in enumerate(entity_ids):
            start_idx = i * num_models
            end_idx = start_idx + num_models
            entity_results = results[start_idx:end_idx]
            
            values = [result[0] for result in entity_results]
            statuses = [result[1] for result in entity_results]
            
            matrix_results.append({
                "entity_id": entity_id,
                "values": values,
                "statuses": statuses
            })
        
        return matrix_results