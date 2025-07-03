import hashlib
import random
from typing import Any


class DummyModel:
    """Base class for dummy models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Use model name as seed for reproducible results
        # Note: Using MD5 for non-cryptographic deterministic hashing only
        self.seed = int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)  # noqa: S324

    def predict(self, features: dict[str, Any]) -> float | None:
        """Override in subclasses"""
        raise NotImplementedError


class FraudDetectionV1(DummyModel):
    def predict(self, features: dict[str, Any]) -> float | None:
        if not features or "amount" not in features:
            return None

        amount = features.get("amount")
        if amount is None:
            return None

        # Seed random with model name + amount for reproducibility
        # Note: Using random for deterministic dummy data generation only
        random.seed(self.seed + int(amount))
        return round(random.uniform(0.1, 0.95), 2)  # noqa: S311


class FraudDetectionV2(DummyModel):
    def predict(self, features: dict[str, Any]) -> float | None:
        if not features or "amount" not in features or "merchant_category" not in features:
            return None

        amount = features.get("amount")
        merchant_category = features.get("merchant_category")

        if amount is None or merchant_category is None:
            return None

        random.seed(self.seed + int(amount) + len(str(merchant_category)))
        return round(random.uniform(0.15, 0.92), 2)  # noqa: S311


class CreditScoreV1(DummyModel):
    def predict(self, features: dict[str, Any]) -> float | None:
        if not features or "income" not in features:
            return None

        income = features.get("income")
        if income is None:
            return None

        random.seed(self.seed + int(income))
        return round(random.uniform(0.3, 0.98), 2)  # noqa: S311


class CreditScoreV2(DummyModel):
    def predict(self, features: dict[str, Any]) -> float | None:
        if not features or "income" not in features or "age" not in features:
            return None

        income = features.get("income")
        age = features.get("age")

        if income is None or age is None:
            return None

        random.seed(self.seed + int(income) + int(age))
        return round(random.uniform(0.25, 0.99), 2)  # noqa: S311


# Model registry
MODEL_REGISTRY = {
    "fraud_detection:v1": FraudDetectionV1("fraud_detection:v1"),
    "fraud_detection:v2": FraudDetectionV2("fraud_detection:v2"),
    "credit_score:v1": CreditScoreV1("credit_score:v1"),
    "credit_score:v2": CreditScoreV2("credit_score:v2"),
}
