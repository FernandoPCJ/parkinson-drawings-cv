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
