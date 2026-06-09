from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse


@dataclass
class TrafficDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    adjacency: np.ndarray


def _convert_to_array(data):
    """Convert sparse arrays or other types to numpy array."""
    if issparse(data):
        return data.toarray()
    return np.asarray(data)


def _stack_time_steps(raw: np.ndarray) -> np.ndarray:
    """
    Stack time steps from raw traffic data.
    
    Handles:
    - Object arrays (MATLAB cell arrays)
    - 3D arrays (time_steps × spatial × features)
    - 2D arrays (direct use)
    """
    # Convert sparse arrays to dense
    raw = _convert_to_array(raw)
    
    if raw.dtype == object:
        # Handle object arrays (MATLAB cell arrays)
        flattened = []
        for step in raw.ravel():
            step_array = _convert_to_array(step)
            step_array = np.asarray(step_array, dtype=float)
            flattened.append(step_array.reshape(-1))
        return np.vstack(flattened)
    
    # Handle numeric arrays
    array = np.asarray(raw, dtype=float)
    
    if array.ndim == 3:
        if array.shape[0] == 1:
            # (1, spatial, features) -> (spatial, features)
            return array[0].reshape(array.shape[1], -1)
        else:
            # (time_steps, spatial, features) -> (time_steps, spatial*features)
            return array.reshape(array.shape[0], -1)
    
    if array.ndim == 2:
        return array
    
    raise ValueError(f"Unsupported traffic feature shape: {array.shape}")


def _normalize_targets(raw: np.ndarray) -> np.ndarray:
    """
    Normalize target data to (n_samples, n_locations) format.
    
    Handles both (n_locations, n_samples) and (n_samples, n_locations) formats.
    """
    # Convert sparse arrays to dense
    raw = _convert_to_array(raw)
    
    y = np.asarray(raw, dtype=float)
    
    if y.ndim != 2:
        raise ValueError(f"Target must be 2D, got shape {y.shape}")
    
    # If shape is (36, n_samples), transpose to (n_samples, 36)
    if y.shape[0] == 36:
        return y.T
    
    return y


def load_traffic_dataset(mat_path: str | Path) -> TrafficDataset:
    """
    Load traffic dataset from .mat file.
    
    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file containing traffic data
        
    Returns
    -------
    TrafficDataset
        Dataset with train/test splits and adjacency matrix
        
    Raises
    ------
    KeyError
        If required keys are missing from .mat file
    """
    print(f"Loading traffic data from {mat_path}...")
    data = loadmat(mat_path)
    
    required = ["tra_X_tr", "tra_X_te", "tra_Y_tr", "tra_Y_te", "tra_adj_mat"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing keys in .mat file: {missing}")
    
    print("Processing training features...")
    X_train = _stack_time_steps(data["tra_X_tr"])
    
    print("Processing test features...")
    X_test = _stack_time_steps(data["tra_X_te"])
    
    print("Processing training targets...")
    y_train = _normalize_targets(data["tra_Y_tr"])
    
    print("Processing test targets...")
    y_test = _normalize_targets(data["tra_Y_te"])
    
    print("Processing adjacency matrix...")
    adj = _convert_to_array(data["tra_adj_mat"])
    adjacency = np.asarray(adj, dtype=float)
    
    print(f"✓ Data loaded successfully!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Adjacency matrix shape: {adjacency.shape}")
    
    return TrafficDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        adjacency=adjacency,
    )
