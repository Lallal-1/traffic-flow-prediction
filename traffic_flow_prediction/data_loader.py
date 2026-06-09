from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class TrafficDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    adjacency: np.ndarray


def _stack_time_steps(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == object:
        flattened = [np.asarray(step, dtype=float).reshape(-1) for step in raw.ravel()]
        return np.vstack(flattened)

    array = np.asarray(raw, dtype=float)
    if array.ndim == 3 and array.shape[0] != 1:
        return array.reshape(array.shape[0], -1)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0].reshape(array.shape[1], -1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported traffic feature shape: {array.shape}")


def _normalize_targets(raw: np.ndarray) -> np.ndarray:
    y = np.asarray(raw, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"Target must be 2D, got shape {y.shape}")
    return y.T if y.shape[0] == 36 else y


def load_traffic_dataset(mat_path: str | Path) -> TrafficDataset:
    data = loadmat(mat_path)
    required = ["tra_X_tr", "tra_X_te", "tra_Y_tr", "tra_Y_te", "tra_adj_mat"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing keys in .mat file: {missing}")

    return TrafficDataset(
        X_train=_stack_time_steps(data["tra_X_tr"]),
        X_test=_stack_time_steps(data["tra_X_te"]),
        y_train=_normalize_targets(data["tra_Y_tr"]),
        y_test=_normalize_targets(data["tra_Y_te"]),
        adjacency=np.asarray(data["tra_adj_mat"], dtype=float),
    )
