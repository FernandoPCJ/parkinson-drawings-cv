import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.parkinson_cv.features import ALL_FEATURES
from src.parkinson_cv.groupcv import cluster_bootstrap_auc, grouped_folds, leave_one_group_out
from src.parkinson_cv.appearance import knn_fit_predict
from src.parkinson_cv.metrics import auc_roc

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("run_cluster_cv", ROOT / "scripts" / "run_cluster_cv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prototypes(n_proto=40, copies=6, size=64, seed=0):
    """Random 'drawings' with RANDOM labels, each repeated as rotated / mirrored copies on a noisy background.

    There is nothing to learn across drawings: any score above chance can only come from recognising a copy.
    """
    rng = np.random.default_rng(seed)
    X, y, cl = [], [], []
    for p in range(n_proto):
        base = np.zeros((size, size), dtype=np.uint8)
        for _ in range(5):
            y0, x0 = rng.integers(0, size - 24, 2)
            h, w = rng.integers(4, 24, 2)
            base[y0:y0 + h, x0:x0 + w] = 200
        lab = int(rng.integers(0, 2))
        for c in range(copies):
            img = base if c < 2 else np.rot90(base, c - 1)         # two same-orientation copies, then rotated ones
            img = img[:, ::-1] if c >= 4 else img                  # and mirrored ones
            img = np.where(img == 0, rng.integers(0, 12, img.shape), img).astype(np.uint8)
            X.append(img); y.append(lab); cl.append(p)
    return np.stack(X), np.array(y), np.array(cl)


def test_grouped_folds_never_split_a_cluster_and_use_every_fold():
    _, y, cl = _prototypes()
    folds = grouped_folds(y, cl, k=5)
    assert set(folds) == set(range(5))
    for c in np.unique(cl):
        assert len(set(folds[cl == c])) == 1


def test_one_nn_wins_with_leaky_folds_and_falls_to_chance_with_cluster_folds():
    X, y, cl = _prototypes()
    df = pd.DataFrame({"diagnosis": np.where(y == 1, "parkinson", "healthy"), "idx": np.arange(len(y))})
    leaky = np.arange(len(y)) % 5                              # copies spread over the folds
    _, _, s_leaky = leave_one_group_out(df, leaky, knn_fit_predict(_bin(X)))
    _, _, s_clean = leave_one_group_out(df, grouped_folds(y, cl, 5), knn_fit_predict(_bin(X)))
    assert auc_roc(y, s_leaky) > 0.8                       # copies on both sides: looks like a good model
    assert 0.3 < auc_roc(y, s_clean) < 0.7


def _bin(X):
    from src.parkinson_cv.ablate import binarize
    return binarize(X)


def test_cluster_bootstrap_interval_is_wider_than_the_image_bootstrap_when_images_are_copies():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(30), 8)
    y = np.repeat(rng.integers(0, 2, 30), 8)
    s = np.repeat(rng.random(30), 8) + y * 0.0
    _, lo, hi = cluster_bootstrap_auc(y, s, cl, n_boot=200)
    _, lo_i, hi_i = cluster_bootstrap_auc(y, s, np.arange(len(y)), n_boot=200)
    assert (hi - lo) > (hi_i - lo_i)


def test_script_reports_the_one_nn_near_chance_on_copies_with_random_labels(tmp_path):
    mod = _script()
    X, y, cl = _prototypes(n_proto=60, copies=6)
    n = len(y)
    dx = np.where(y == 1, "parkinson", "healthy")
    meta = {"diagnosis": np.concatenate([dx, dx]), "drawing_type": np.array(["spiral"] * n + ["wave"] * n),
            "class_name": np.concatenate([[f"{d}_spiral" for d in dx], [f"{d}_wave" for d in dx]]),
            "name_number": np.concatenate([np.arange(n), np.arange(n)])}
    np.savez(tmp_path / "cache.npz", X=np.concatenate([X, X]), **meta)
    rng = np.random.default_rng(3)
    feats = pd.DataFrame(rng.normal(size=(2 * n, len(ALL_FEATURES))), columns=ALL_FEATURES)
    pd.concat([pd.DataFrame(meta), feats], axis=1).to_csv(tmp_path / "strokes.csv", index=False)
    assert mod.main(tmp_path / "cache.npz", tmp_path / "strokes.csv", tmp_path / "out", n_boot=30, skip_cnn=True) == 0
    text = (tmp_path / "out" / "cluster_cv.md").read_text(encoding="utf-8")
    assert "60 distinct drawings" in text and "thumbnail 1-NN" in text and "Network skipped" in text
