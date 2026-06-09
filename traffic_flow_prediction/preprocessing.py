from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class IQRClipper:
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "IQRClipper":
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise ValueError("IQRClipper must be fitted before transform")
        return np.clip(X, self.lower_, self.upper_)


class FeaturePreprocessor:
    def __init__(self, pca_components: int = 64):
        self.imputer = SimpleImputer(strategy="median")
        self.clipper = IQRClipper()
        self.scaler = StandardScaler()
        self.pca_components = pca_components
        self.pca: PCA | None = None

    @staticmethod
    def _augment(X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        n_samples = X.shape[0]
        base = X.reshape(n_samples, 36, 48)
        loc_mean = base.mean(axis=2)
        loc_std = base.std(axis=2)
        feat_mean = base.mean(axis=1)

        parts = [X, loc_mean, loc_std, feat_mean]
        if adjacency is not None and adjacency.shape == (36, 36):
            degree = adjacency.sum(axis=1)
            degree = np.where(degree == 0, 1.0, degree)
            normalized_adj = adjacency / degree[:, None]
            neighbor_context = loc_mean @ normalized_adj.T
            parts.append(neighbor_context)

        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        X_aug = self._augment(X, adjacency)
        X_imp = self.imputer.fit_transform(X_aug)
        X_clip = self.clipper.fit(X_imp).transform(X_imp)
        X_scaled = self.scaler.fit_transform(X_clip)

        n_components = min(self.pca_components, X_scaled.shape[1], X_scaled.shape[0])
        self.pca = PCA(n_components=n_components, random_state=42)
        return self.pca.fit_transform(X_scaled)

    def transform(self, X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        if self.pca is None:
            raise ValueError("FeaturePreprocessor must be fitted before transform")
        X_aug = self._augment(X, adjacency)
        X_imp = self.imputer.transform(X_aug)
        X_clip = self.clipper.transform(X_imp)
        X_scaled = self.scaler.transform(X_clip)
        return self.pca.transform(X_scaled)


def preprocess_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    adjacency: np.ndarray | None = None,
    pca_components: int = 64,
) -> tuple[np.ndarray, np.ndarray, FeaturePreprocessor]:
    preprocessor = FeaturePreprocessor(pca_components=pca_components)
    X_train_processed = preprocessor.fit_transform(X_train, adjacency=adjacency)
    X_test_processed = preprocessor.transform(X_test, adjacency=adjacency)
    return X_train_processed, X_test_processed, preprocessor
