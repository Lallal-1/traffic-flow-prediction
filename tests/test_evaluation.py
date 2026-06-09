import unittest

import numpy as np

from traffic_flow_prediction.evaluation import compute_metrics


class TestEvaluation(unittest.TestCase):
    def test_compute_metrics_handles_zero_targets(self) -> None:
        y_true = np.array([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
        y_pred = np.array([[0.0, 1.2], [1.5, 0.0], [2.8, 4.2]])

        metrics = compute_metrics(y_true, y_pred)

        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("R2", metrics)
        self.assertTrue(np.isfinite(metrics["MAE"]))
        self.assertTrue(np.isfinite(metrics["RMSE"]))
        self.assertTrue(np.isfinite(metrics["R2"]))


if __name__ == "__main__":
    unittest.main()
