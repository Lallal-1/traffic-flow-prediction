from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class IQRClipper:
    """Clip outliers using Interquartile Range (IQR) method."""
    
    def __init__(self, factor: float = 1.5):
        """
        Initialize IQR Clipper.
        
        Parameters
        ----------
        factor : float
            Multiplier for IQR (default: 1.5 for standard boxplot)
        """
        self.factor = factor
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "IQRClipper":
        """Fit the clipper by calculating Q1, Q3 and IQR."""
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values to lower and upper bounds."""
        if self.lower_ is None or self.upper_ is None:
            raise ValueError("IQRClipper must be fitted before transform")
        return np.clip(X, self.lower_, self.upper_)


class FeaturePreprocessor:
    """Complete feature preprocessing pipeline."""
    
    def __init__(self, pca_components: int = 64):
        """
        Initialize feature preprocessor.
        
        Parameters
        ----------
        pca_components : int
            Number of PCA components to retain
        """
        self.imputer = SimpleImputer(strategy="median")
        self.clipper = IQRClipper()
        self.scaler = StandardScaler()
        self.pca_components = pca_components
        self.pca: PCA | None = None

    @staticmethod
    def _augment(X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        """
        Augment features with spatial statistics.
        
        Features added:
        1. Location-wise mean (36 features)
        2. Location-wise std (36 features)
        3. Global feature mean (36 features)
        4. Neighbor context from adjacency matrix (36 features)
        """
        n_samples = X.shape[0]
        base = X.reshape(n_samples, 36, 48)
        
        # Location statistics
        loc_mean = base.mean(axis=2)  # (n_samples, 36)
        loc_std = base.std(axis=2)    # (n_samples, 36)
        
        # Global statistics
        feat_mean = base.mean(axis=1)  # (n_samples, 48)
        
        parts = [X, loc_mean, loc_std, feat_mean]
        
        # Spatial neighbor context
        if adjacency is not None and adjacency.shape == (36, 36):
            degree = adjacency.sum(axis=1)
            degree = np.where(degree == 0, 1.0, degree)
            normalized_adj = adjacency / degree[:, None]
            neighbor_context = loc_mean @ normalized_adj.T
            parts.append(neighbor_context)
        
        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform features."""
        print("   Feature augmentation...")
        X_aug = self._augment(X, adjacency)
        print(f"     • After augmentation: {X_aug.shape}")
        
        print("   Missing value imputation (median strategy)...")
        X_imp = self.imputer.fit_transform(X_aug)
        print(f"     • After imputation: {X_imp.shape}")
        
        print("   Outlier clipping (IQR method, factor=1.5)...")
        X_clip = self.clipper.fit(X_imp).transform(X_imp)
        print(f"     • After clipping: {X_clip.shape}")
        
        print("   Standardization (zero mean, unit variance)...")
        X_scaled = self.scaler.fit_transform(X_clip)
        print(f"     • After scaling: {X_scaled.shape}")
        
        print(f"   PCA dimensionality reduction ({self.pca_components} components)...")
        n_components = min(self.pca_components, X_scaled.shape[1], X_scaled.shape[0])
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"     • After PCA: {X_pca.shape}")
        print(f"     • Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return X_pca

    def transform(self, X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        """Transform features using fitted preprocessor."""
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
    """
    Complete preprocessing pipeline for training and test features.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    adjacency : np.ndarray, optional
        Adjacency matrix for spatial context
    pca_components : int
        Number of PCA components
        
    Returns
    -------
    tuple
        Processed (X_train, X_test, preprocessor)
    """
    preprocessor = FeaturePreprocessor(pca_components=pca_components)
    
    print(f"  Training set: {X_train.shape}")
    X_train_processed = preprocessor.fit_transform(X_train, adjacency=adjacency)
    
    print(f"  Test set: {X_test.shape}")
    X_test_processed = preprocessor.transform(X_test, adjacency=adjacency)
    
    return X_train_processed, X_test_processed, preprocessor
