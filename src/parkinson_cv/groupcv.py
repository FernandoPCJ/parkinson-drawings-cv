"""Leave-one-group-out validation: the test images come from a group the model never saw.

The blocked CV of `cv.py` keeps neighbouring file numbers together, but images with the same kind of
background still appear on both sides of every split. Here a whole group (for example the lowest
third of an image-background measure) is held out, so the model must work on a background regime it was
not trained on. Groups are only a stand-in for "a different data source": the dataset has no source label.
"""
from __future__ import annotations

import numpy as np

from .cv import labels


def tercile_groups(values) -> np.ndarray:
    """0 / 1 / 2 = lowest / middle / highest third of `values` (about equal sizes)."""
    values = np.asarray(values, dtype=float)
    return np.digitize(values, np.quantile(values, [1 / 3, 2 / 3]))


def leave_one_group_out(df, groups, fit_predict):
    """Every group is the test set once; the model is trained on the other groups.

    fit_predict(train_df, test_df) -> (predicted_label, score_for_parkinson), as in `cv.oof_predictions`.
    Returns y, preds, scores, all aligned with the rows of `df`.
    """
    df = df.reset_index(drop=True)
    groups = np.asarray(groups)
    if len(groups) != len(df):
        raise ValueError("groups must have one entry per row of df")
    preds = np.zeros(len(df), dtype=int)
    scores = np.zeros(len(df), dtype=float)
    for g in np.unique(groups):
        test = groups == g
        p, s = fit_predict(df[~test], df[test])
        preds[test] = p
        scores[test] = s
    return labels(df), preds, scores


def grouped_folds(y, groups, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id (0..k-1) for every row, keeping every group (cluster of copies) inside ONE fold.

    Stratified by label as far as the group sizes allow (sklearn StratifiedGroupKFold, shuffled with `seed`).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups=np.asarray(groups))):
        folds[test] = f
    return folds


def cluster_bootstrap_auc(y, scores, groups, n_boot: int = 500, seed: int = 0):
    """AUC-ROC with a 95% CI that resamples whole groups (clusters of copies), not single images.

    Images of one cluster are copies of the same drawing, so they are not independent: resampling them one by one
    would make the interval far too narrow. Returns (point, low, high).
    """
    from .metrics import auc_roc

    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    point = auc_roc(y, scores)
    ids, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(ids))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[g] for g in pick])
        if len(set(y[idx])) == 2:
            vals.append(auc_roc(y[idx], scores[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)
