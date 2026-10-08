"""Shared evaluation: run blocked CV for a model and format metrics with bootstrap CIs."""
from __future__ import annotations

from .cv import oof_predictions
from .metrics import (accuracy, auc_roc, average_precision, balanced_accuracy, bootstrap_ci,
                      sensitivity, specificity)

# name -> metric(y, pred, score). Some use the hard prediction, some the score.
METRICS = {
    "accuracy": lambda y, p, s: accuracy(y, p),
    "balanced acc": lambda y, p, s: balanced_accuracy(y, p),
    "sensitivity": lambda y, p, s: sensitivity(y, p),
    "specificity": lambda y, p, s: specificity(y, p),
    "AUC-ROC": lambda y, p, s: auc_roc(y, s),
    "AUC-PR": lambda y, p, s: average_precision(y, s),
}


def evaluate(df, fit_predict, n_boot: int = 1000, k: int = 5):
    """Return ({metric: 'value [lo-hi]'}, prevalence) for one subset of drawings."""
    y, preds, scores = oof_predictions(df, fit_predict, k=k)
    out = {}
    for name, fn in METRICS.items():
        pt, lo, hi = bootstrap_ci(fn, y, preds, scores, n_boot=n_boot)
        out[name] = f"{pt:.3f} [{lo:.3f}-{hi:.3f}]"
    return out, float(y.mean())


def table_header() -> list[str]:
    cols = ["subset", "n", "prevalence"] + list(METRICS)
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def table_row(name: str, n: int, prevalence: float, res: dict) -> str:
    return "| " + " | ".join([name, str(n), f"{prevalence:.3f}"] + [res[m] for m in METRICS]) + " |"
