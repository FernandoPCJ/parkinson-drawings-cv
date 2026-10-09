import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.parkinson_cv.appearance import knn_fit_predict, thumbnails
from src.parkinson_cv.metrics import auc_roc

ROOT = Path(__file__).resolve().parents[1]


def test_thumbnails_are_unit_length_and_zero_mean():
    X = np.random.default_rng(0).integers(0, 255, size=(5, 64, 64)).astype(np.uint8)
    T = thumbnails(X)
    assert T.shape == (5, 1024)
    np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(T.mean(1), 0.0, atol=1e-5)


def test_one_nn_separates_classes_that_look_different_and_is_chance_otherwise():
    rng = np.random.default_rng(1)
    n = 80
    y = np.array([1, 0] * (n // 2))
    noise = rng.integers(0, 20, size=(n, 64, 64))
    X = noise.astype(np.uint8)
    X[y == 1, 10:20, :] += 150                      # parkinson images have a bright band: easy to tell apart
    df = pd.DataFrame({"diagnosis": np.where(y == 1, "parkinson", "healthy"), "idx": np.arange(n)})
    _, s = knn_fit_predict(X)(df.iloc[:60], df.iloc[60:])
    assert auc_roc(y[60:], s) > 0.95
    Xn = noise.astype(np.uint8)                     # same labels, but nothing to see
    _, s2 = knn_fit_predict(Xn)(df.iloc[:60], df.iloc[60:])
    assert 0.2 < auc_roc(y[60:], s2) < 0.8


def test_check_cnn_neighbors_still_exposes_thumbnails():
    spec = importlib.util.spec_from_file_location("c", ROOT / "scripts" / "check_cnn_neighbors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.thumbnails is thumbnails
