from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import load_traffic_dataset
from .evaluation import evaluate_predictions
from .models import build_models, train_and_predict
from .preprocessing import preprocess_features
from .visualization import (
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_residual_analysis,
    plot_time_series,
)


def run_pipeline(
    mat_path: str, 
    output_dir: str = "outputs", 
    pca_components: int = 64, 
    cv: int = 3
) -> pd.DataFrame:
    """
    Run the complete traffic flow prediction pipeline using Linear Regression.
    
    Parameters
    ----------
    mat_path : str
        Path to the .mat dataset file
    output_dir : str
        Directory to save outputs
    pca_components : int
        Number of PCA components (default: 64)
    cv : int
        Cross-validation folds (default: 3)
        
    Returns
    -------
    pd.DataFrame
        Model performance metrics
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Traffic Flow Prediction Pipeline (Linear Regression)")
    print("="*60 + "\n")
    
    # Load data
    print("1. Loading traffic dataset...")
    dataset = load_traffic_dataset(mat_path)
    
    # Preprocess features
    print("\n2. Preprocessing features...")
    X_train, X_test, _ = preprocess_features(
        dataset.X_train,
        dataset.X_test,
        adjacency=dataset.adjacency,
        pca_components=pca_components,
    )
    print(f"   ✓ X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Build and train models
    print("\n3. Building and training model...")
    models = build_models()
    fitted_models, predictions = train_and_predict(
        models, X_train, dataset.y_train, X_test
    )
    
    # Evaluate
    print("\n4. Evaluating model performance...")
    metrics_df = evaluate_predictions(
        fitted_models, predictions, X_train, 
        dataset.y_train, dataset.y_test, cv=cv
    )
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    best_model_name = metrics_df.iloc[0]["Model"]
    best_pred = predictions[best_model_name]
    
    plot_actual_vs_predicted(
        dataset.y_test, best_pred, 
        out_dir / "actual_vs_predicted.png"
    )
    plot_model_comparison(
        metrics_df, 
        out_dir / "model_comparison.png"
    )
    plot_time_series(
        dataset.y_test, best_pred, 
        out_dir / "time_series_prediction.png"
    )
    plot_residual_analysis(
        dataset.y_test,
        best_pred,
        out_dir / "residual_distribution.png",
        out_dir / "residuals_vs_predictions.png",
    )
    
    # Save reports
    print("\n6. Saving performance reports...")
    metrics_df.to_csv(out_dir / "model_performance_report.csv", index=False)
    metrics_df.to_json(
        out_dir / "model_performance_report.json", 
        orient="records", 
        force_ascii=False, 
        indent=2
    )
    print(f"   ✓ Reports saved to {out_dir}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60 + "\n")
    
    return metrics_df
