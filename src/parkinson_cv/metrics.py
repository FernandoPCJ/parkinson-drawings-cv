"""Metrics and bootstrap confidence intervals, written with numpy only.

Convention: y is 0/1 with 1 = parkinson (the "positive" class).
  sensitivity = share of parkinson drawings correctly flagged  (recall of the positive class)
  specificity = share of healthy drawings correctly cleared
Each function is short on purpose so every line can be explained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_counts(y, pred) -> tuple[int, int, int, int]:
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tn, fp, fn, tp


def accuracy(y, pred) -> float:
    tn, fp, fn, tp = confusion_counts(y, pred)
    return (tp + tn) / (tp + tn + fp + fn)


def sensitivity(y, pred) -> float:
    _, _, fn, tp = confusion_counts(y, pred)
    return tp / (tp + fn) if (tp + fn) else float("nan")


def specificity(y, pred) -> float:
    tn, fp, _, _ = confusion_counts(y, pred)
    return tn / (tn + fp) if (tn + fp) else float("nan")


def balanced_accuracy(y, pred) -> float:
    """Mean of sensitivity and specificity: 0.5 = no better than a coin, 1.0 = perfect.

    Unlike plain accuracy it does not reward predicting the majority class.
    """
    return float(np.nanmean([sensitivity(y, pred), specificity(y, pred)]))


def auc_roc(y, score) -> float:
    """Probability that a random parkinson drawing scores higher than a random healthy one.

    Rank formula (Mann-Whitney): ties get the average rank, so a constant score gives 0.5.
    """
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, score) -> float:
    """Area under the precision-recall curve (AUC-PR), step-wise.

    A model that cannot rank (constant score) gets the prevalence of the positive class,
    which is the floor to compare against, not 0.5.
    """
    y = np.asarray(y)
    p = int((y == 1).sum())
    if p == 0:
        return float("nan")
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    ys, ss = y[order], score[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]  # evaluate only after a whole tie group
    tp, fp = tp[last_of_tie], fp[last_of_tie]
    precision = tp / (tp + fp)
    recall = tp / p
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def bootstrap_ci(metric, y, preds, scores, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Point estimate and percentile CI of a metric, averaged over CV repeats.

    metric(y, pred, score) -> float.  preds/scores have shape (n_repeats, n_images).
    Each bootstrap draw resamples images (with replacement, separately inside each class so
    both classes are always present) and averages the metric over the repeats.

    Caveat: images are treated as independent. If neighbours come from the same person the
    true uncertainty is larger than this interval says.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    def value(idx):
        vals = [metric(y[idx], preds[r][idx], scores[r][idx]) for r in range(preds.shape[0])]
        return float(np.nanmean(vals))

    point = value(np.arange(len(y)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws[b] = value(idx)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
