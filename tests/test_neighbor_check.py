import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "check_cnn_neighbors", Path(__file__).resolve().parents[1] / "scripts" / "check_cnn_neighbors.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_twin_in_other_fold_is_found_and_same_fold_is_ignored():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, size=(6, 64, 64)).astype(np.uint8)
    X[3] = X[0]                                   # exact twin of image 0
    fold = np.array([0, 0, 1, 1, 2, 2])           # 0 and 3 are in different folds
    sim, j = _mod.nearest_other_fold(_mod.thumbnails(X), fold)
    assert j[0] == 3 and j[3] == 0
    assert sim[0] > 0.999
    assert sim[1] < 0.9                           # no twin for image 1


def test_twin_in_same_fold_does_not_count():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 255, size=(4, 64, 64)).astype(np.uint8)
    X[1] = X[0]
    fold = np.array([0, 0, 1, 1])                 # the twins sit together in fold 0
    sim, j = _mod.nearest_other_fold(_mod.thumbnails(X), fold)
    assert sim[0] < 0.9 and j[0] in (2, 3)
