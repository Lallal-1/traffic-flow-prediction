from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def build_models(random_state: int = 42) -> dict[str, object]:
    """Build a single Linear Regression model."""
    return {
        "LinearRegression": LinearRegression(),
    }


def train_and_predict(
    models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Train model and make predictions."""
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        predictions[name] = model.predict(X_test)
        print(f"✓ {name} training completed")

    return fitted, predictions
