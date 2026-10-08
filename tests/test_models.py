import numpy as np
import pandas as pd

from src.parkinson_cv.cv import oof_predictions
from src.parkinson_cv.metrics import auc_roc, balanced_accuracy
from src.parkinson_cv.models import logistic_fit_predict


def _df(signal: float, n_per=120, seed=0):
    rng = np.random.default_rng(seed)
    rows, num = [], 0
    for diag in ("healthy", "parkinson"):
        for dtype in ("spiral", "wave"):
            for _ in range(n_per):
                num += 1
                shift = signal if diag == "parkinson" else 0.0
                rows.append({"diagnosis": diag, "drawing_type": dtype, "name_number": num,
                             "f1": rng.normal(shift, 1), "f2": rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_informative_feature_beats_chance():
    y, preds, scores = oof_predictions(_df(signal=2.0), logistic_fit_predict(["f1", "f2"]),
                                       k=5, offsets=(0.0,))
    assert auc_roc(y, scores[0]) > 0.85
    assert balanced_accuracy(y, preds[0]) > 0.75


def test_noise_feature_stays_near_chance():
    y, preds, scores = oof_predictions(_df(signal=0.0), logistic_fit_predict(["f1", "f2"]),
                                       k=5, offsets=(0.0,))
    assert 0.40 < auc_roc(y, scores[0]) < 0.60
    assert 0.40 < balanced_accuracy(y, preds[0]) < 0.60


def test_balanced_accuracy_of_the_majority_answer_is_half():
    y = np.array([1, 1, 1, 0])
    assert balanced_accuracy(y, np.ones(4, int)) == 0.5
