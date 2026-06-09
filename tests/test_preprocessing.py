import unittest

import numpy as np

from traffic_flow_prediction.preprocessing import preprocess_features


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_shapes_and_finite_values(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(20, 36 * 48))
        X_test = rng.normal(size=(10, 36 * 48))
        X_train[0, 0] = np.nan
        X_train[1, 1] = 1e9
        adjacency = np.eye(36)

        Xt, Xv, _ = preprocess_features(X_train, X_test, adjacency=adjacency, pca_components=16)

        self.assertEqual(Xt.shape, (20, 16))
        self.assertEqual(Xv.shape, (10, 16))
        self.assertTrue(np.isfinite(Xt).all())
        self.assertTrue(np.isfinite(Xv).all())


if __name__ == "__main__":
    unittest.main()
