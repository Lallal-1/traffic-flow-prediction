from __future__ import annotations

import argparse

from traffic_flow_prediction.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic Flow Prediction with traditional ML models")
    parser.add_argument("--mat-path", required=True, help="Path to .mat dataset file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save reports and plots")
    parser.add_argument("--pca-components", type=int, default=64, help="Number of PCA components")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_df = run_pipeline(
        mat_path=args.mat_path,
        output_dir=args.output_dir,
        pca_components=args.pca_components,
        cv=args.cv,
    )
    print("\nModel Performance Report:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
