"""Appearance-only tools: thumbnails of the ink map and a no-learning nearest-neighbour baseline.

The 1-NN baseline copies the label of the most similar training image (cosine on 32x32 thumbnails). It knows nothing
about tremor or strokes: if it already does well, the labels can be read from the overall look of the images, and any
network that beats it has to be compared against THAT bar, not against 0.5.
"""
from __future__ import annotations

import numpy as np

from .cv import labels


def thumbnails(X: np.ndarray, size: int = 32) -> np.ndarray:
    """Average-pool to size x size, flatten, remove the mean and scale to unit length."""
    n, h, w = X.shape
    f = h // size
    t = X[:, :size * f, :size * f].astype(np.float32).reshape(n, size, f, size, f).mean((2, 4))
    t = t.reshape(n, -1)
    t -= t.mean(1, keepdims=True)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)


def knn_fit_predict(X: np.ndarray, size: int = 32):
    """`fit_predict(train_df, test_df)` (column "idx" points into X).

    Score = best similarity to a parkinson training image minus best similarity to a healthy one (>0 -> parkinson).
    """
    T = thumbnails(X, size)

    def fit_predict(train_df, test_df):
        tr = train_df["idx"].to_numpy()
        pos = labels(train_df) == 1
        sim = T[test_df["idx"].to_numpy()] @ T[tr].T
        score = sim[:, pos].max(1) - sim[:, ~pos].max(1)
        return (score > 0).astype(int), score

    return fit_predict
