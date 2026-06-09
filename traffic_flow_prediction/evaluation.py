from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    if np.isnan(mape):
        mape = float("nan")

    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}


def cross_validated_rmse(model: object, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        clone(model),
        X,
        y,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    return float(-scores.mean())


def evaluate_predictions(
    fitted_models: dict[str, object],
    predictions: dict[str, np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = 3,
) -> pd.DataFrame:
    rows = []
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_test, y_pred)
        metrics["CV_RMSE"] = cross_validated_rmse(fitted_models[name], X_train, y_train, cv=cv)
        metrics["Model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df[["Model", "MAE", "RMSE", "MAPE", "R2", "CV_RMSE"]].sort_values("RMSE")
