from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    location_idx: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    x = y_true[:, location_idx]
    y = y_pred[:, location_idx]
    ax.scatter(x, y, alpha=0.4, s=18)
    min_v, max_v = float(np.min(x)), float(np.max(x))
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2)
    ax.set_title(f"Actual vs Predicted (Location {location_idx})")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    _save(fig, path)


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    melted = metrics_df.melt(id_vars=["Model"], value_vars=["MAE", "RMSE", "MAPE", "R2", "CV_RMSE"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="Model", y="value", hue="variable", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_time_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    location_idx: int = 0,
    max_points: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    count = min(max_points, y_true.shape[0])
    idx = np.arange(count)
    ax.plot(idx, y_true[:count, location_idx], label="Actual", lw=2)
    ax.plot(idx, y_pred[:count, location_idx], label="Predicted", lw=2)
    ax.set_title(f"Traffic Flow Time Series (Location {location_idx})")
    ax.set_xlabel("Quarter-hour Index")
    ax.set_ylabel("Traffic Flow")
    ax.legend()
    _save(fig, path)


def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path_hist: Path,
    path_scatter: Path,
    location_idx: int = 0,
) -> None:
    residuals = y_true[:, location_idx] - y_pred[:, location_idx]

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title(f"Residual Distribution (Location {location_idx})")
    ax1.set_xlabel("Residual")
    _save(fig1, path_hist)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(y_pred[:, location_idx], residuals, alpha=0.4, s=16)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_title(f"Residuals vs Predictions (Location {location_idx})")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    _save(fig2, path_scatter)


def plot_feature_importance(
    feature_importance: np.ndarray,
    path: Path,
    top_k: int = 20,
) -> None:
    top_k = min(top_k, feature_importance.shape[0])
    sorted_idx = np.argsort(feature_importance)[::-1][:top_k]
    values = feature_importance[sorted_idx]
    labels = [f"PC{i + 1}" for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=values, y=labels, ax=ax)
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    _save(fig, path)
