"""Cross-validation loop that returns out-of-fold predictions.

"Out-of-fold" means every image is predicted by a model that never saw it (nor, thanks to the
blocked folds and the deduplication, its neighbours or copies). The loop is repeated with
shifted block boundaries (`offsets`) to see how much the result depends on where the cuts fall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import DEFAULT_STRATA, blocked_stratified_folds

DEFAULT_OFFSETS = (0.0, 0.2, 0.4, 0.6, 0.8)  # in units of one block


def labels(df: pd.DataFrame) -> np.ndarray:
    """1 = parkinson, 0 = healthy."""
    return (df["diagnosis"] == "parkinson").to_numpy().astype(int)


def oof_predictions(df: pd.DataFrame, fit_predict, k: int = 5, offsets=DEFAULT_OFFSETS,
                    strata=DEFAULT_STRATA):
    """Run blocked CV for every offset.

    fit_predict(train_df, test_df) -> (predicted_label[0/1], score_for_parkinson)
    Returns y (n,), preds (n_offsets, n), scores (n_offsets, n).
    """
    df = df.reset_index(drop=True)
    y = labels(df)
    preds = np.zeros((len(offsets), len(df)), dtype=int)
    scores = np.zeros((len(offsets), len(df)), dtype=float)
    for r, off in enumerate(offsets):
        fold = blocked_stratified_folds(df, k=k, strata=strata, offset=off).to_numpy()
        for f in range(k):
            test = fold == f
            p, s = fit_predict(df[~test], df[test])
            preds[r, test] = p
            scores[r, test] = s
    return y, preds, scores
