from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import load_traffic_dataset
from .evaluation import evaluate_predictions
from .models import build_models, train_and_predict
from .preprocessing import preprocess_features
from .visualization import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residual_analysis,
    plot_time_series,
)


def run_pipeline(mat_path: str, output_dir: str = "outputs", pca_components: int = 64, cv: int = 3) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_traffic_dataset(mat_path)
    X_train, X_test, _ = preprocess_features(
        dataset.X_train,
        dataset.X_test,
        adjacency=dataset.adjacency,
        pca_components=pca_components,
    )

    models = build_models()
    fitted_models, predictions = train_and_predict(models, X_train, dataset.y_train, X_test)
    metrics_df = evaluate_predictions(fitted_models, predictions, X_train, dataset.y_train, dataset.y_test, cv=cv)

    best_model_name = metrics_df.iloc[0]["Model"]
    best_pred = predictions[best_model_name]

    metrics_df.to_csv(out_dir / "model_performance_report.csv", index=False)
    metrics_df.to_json(out_dir / "model_performance_report.json", orient="records", force_ascii=False, indent=2)

    plot_actual_vs_predicted(dataset.y_test, best_pred, out_dir / "actual_vs_predicted.png")
    plot_model_comparison(metrics_df, out_dir / "model_comparison.png")
    plot_time_series(dataset.y_test, best_pred, out_dir / "time_series_prediction.png")
    plot_residual_analysis(
        dataset.y_test,
        best_pred,
        out_dir / "residual_distribution.png",
        out_dir / "residuals_vs_predictions.png",
    )

    rf = fitted_models["RandomForest"]
    if hasattr(rf, "feature_importances_"):
        plot_feature_importance(rf.feature_importances_, out_dir / "feature_importance.png")

    return metrics_df
