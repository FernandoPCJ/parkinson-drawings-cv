import numpy as np
import pandas as pd
import pytest

from src.parkinson_cv.baselines import majority_fit_predict
from src.parkinson_cv.cv import oof_predictions
from src.parkinson_cv.metrics import (
    accuracy, auc_roc, average_precision, bootstrap_ci, confusion_counts,
    sensitivity, specificity)


def test_confusion_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 0, 1, 0])
    assert confusion_counts(y, pred) == (3, 1, 1, 2)  # tn, fp, fn, tp
    assert sensitivity(y, pred) == pytest.approx(2 / 3)
    assert specificity(y, pred) == pytest.approx(3 / 4)
    assert accuracy(y, pred) == pytest.approx(5 / 7)


def test_auc_roc_known_cases():
    y = np.array([1, 0, 1, 0])
    assert auc_roc(y, [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)   # 3 of 4 pairs ordered
    assert auc_roc(y, [0.9, 0.1, 0.8, 0.2]) == pytest.approx(1.0)
    assert auc_roc(y, [0.1, 0.9, 0.2, 0.8]) == pytest.approx(0.0)
    assert auc_roc(y, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)    # ties -> 0.5


def test_average_precision_known_cases():
    y = np.array([1, 0, 1, 0])
    assert average_precision(y, [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert average_precision(y, [0.9, 0.1, 0.8, 0.2]) == pytest.approx(1.0)
    assert average_precision(y, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # = prevalence


def _df():
    rows, num = [], 0
    for diag, dtype, n in (("healthy", "spiral", 60), ("parkinson", "spiral", 80),
                           ("healthy", "wave", 50), ("parkinson", "wave", 70)):
        for _ in range(n):
            num += 1
            rows.append({"diagnosis": diag, "drawing_type": dtype, "name_number": num})
    return pd.DataFrame(rows)


def test_oof_covers_every_image_once_per_repeat():
    df = _df()
    y, preds, scores = oof_predictions(df, majority_fit_predict, k=5, offsets=(0.0, 0.4))
    assert preds.shape == (2, len(df)) and scores.shape == (2, len(df))
    assert set(np.unique(preds)) <= {0, 1}


def test_majority_baseline_has_no_ranking_ability():
    df = _df()
    y, preds, scores = oof_predictions(df, majority_fit_predict, k=5, offsets=(0.0,))
    prevalence = y.mean()
    assert (preds[0] == 1).all()                      # parkinson is the majority
    assert sensitivity(y, preds[0]) == 1.0
    assert specificity(y, preds[0]) == 0.0
    assert accuracy(y, preds[0]) == pytest.approx(prevalence)
    # the constant score differs slightly between folds (each fold's training prevalence),
    # so the pooled values are only approximately 0.5 and the prevalence
    assert auc_roc(y, scores[0]) == pytest.approx(0.5, abs=0.03)
    assert average_precision(y, scores[0]) == pytest.approx(prevalence, abs=0.03)


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100, int), np.zeros(100, int)]
    score = np.r_[rng.normal(1, 1, 100), rng.normal(0, 1, 100)]
    preds = (score > 0.5).astype(int)[None, :]
    scores = score[None, :]
    a = bootstrap_ci(lambda y_, p_, s_: auc_roc(y_, s_), y, preds, scores, n_boot=200, seed=1)
    b = bootstrap_ci(lambda y_, p_, s_: auc_roc(y_, s_), y, preds, scores, n_boot=200, seed=1)
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi
    assert 0.6 < point < 0.95
