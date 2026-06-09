from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor


ModelMap = Mapping[str, object]


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=random_state),
        "SVM": MultiOutputRegressor(
            LinearSVR(C=1.0, epsilon=0.1, random_state=random_state, max_iter=5000)
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_and_predict(
    models: ModelMap,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        predictions[name] = model.predict(X_test)

    return fitted, predictions
