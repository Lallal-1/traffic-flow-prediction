from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


def evaluate_predictions(
    fitted_models: dict[str, object],
    predictions: dict[str, np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = 3,
) -> pd.DataFrame:
    """Evaluate model predictions with multiple metrics."""
    
    results = []
    
    for model_name, model in fitted_models.items():
        y_pred = predictions[model_name]
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        # MAPE with handling for zero values
        mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) * 100
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Cross-validation RMSE
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())
        
        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "R²": r2,
            "CV RMSE": cv_rmse,
            "CV Std": cv_std,
        })
    
    df = pd.DataFrame(results)
    # Sort by RMSE (best first)
    df = df.sort_values("RMSE").reset_index(drop=True)
    return df
