"""Baselines. Rung 1 of the ladder: always predict the majority class of the training folds.

Any real model must beat this floor, and the floor shows why accuracy alone misleads: it can
look decent (the majority share) while sensitivity or specificity is 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def majority_fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Predict the most common training label for every test image.

    The score is the training prevalence of parkinson, the same for every image, so the model
    has no ranking ability: AUC-ROC is about 0.5 and AUC-PR about the prevalence (not exactly,
    because the prior differs slightly between training folds).
    """
    prior = float((train_df["diagnosis"] == "parkinson").mean())
    label = int(prior >= 0.5)
    n = len(test_df)
    return np.full(n, label, dtype=int), np.full(n, prior, dtype=float)
