from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Flatten for visualization
    y_test_flat = y_test.ravel()
    y_pred_flat = y_pred.ravel()
    
    ax.scatter(y_test_flat, y_pred_flat, alpha=0.5, s=20)
    
    # Add perfect prediction line
    min_val = min(y_test_flat.min(), y_pred_flat.min())
    max_val = max(y_test_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Actual vs Predicted Traffic Flow (Linear Regression)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Plot model performance metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Linear Regression Performance Metrics', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    metrics = ['MAE', 'RMSE', 'MAPE (%)', 'R²', 'CV RMSE', 'CV Std']
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_df)))
    
    for idx, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            value = metrics_df.iloc[0][metric]
            axes[idx].bar([0], [value], color=colors[0], width=0.5)
            axes[idx].set_ylabel(metric, fontsize=11, fontweight='bold')
            axes[idx].set_xticks([])
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].text(0, value, f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_time_series(y_test: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot time series prediction for first location."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use first location time series (36 locations, plot first one)
    n_samples = y_test.shape[0]
    time_steps = np.arange(min(200, n_samples))  # Show first 200 time steps
    
    ax.plot(time_steps, y_test[time_steps, 0], 'b-', linewidth=2, label='Actual', alpha=0.7)
    ax.plot(time_steps, y_pred[time_steps, 0], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Traffic Flow', fontsize=12)
    ax.set_title('Time Series Prediction (Location 1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_residual_analysis(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path_dist: Path,
    output_path_scatter: Path,
) -> None:
    """Plot residual distribution and residuals vs predictions."""
    residuals = y_test - y_pred
    residuals_flat = residuals.ravel()
    y_pred_flat = y_pred.ravel()
    
    # Plot 1: Residual Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(residuals_flat, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(residuals_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals_flat.mean():.4f}')
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path_dist, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path_dist}")
    
    # Plot 2: Residuals vs Predictions
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(y_pred_flat, residuals_flat, alpha=0.5, s=20)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path_scatter, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path_scatter}")
