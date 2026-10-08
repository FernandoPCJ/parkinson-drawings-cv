"""Experiment tracking: read our markdown reports and log them to MLflow (history backfill).

The experiments were run before MLflow was added, so instead of re-running hours of training we
parse the numbers already written in reports/*.md ("0.869 [0.851-0.883]" cells) and log them.
`parse_report` needs no MLflow; `log_run` imports it lazily.
"""
from __future__ import annotations

import re

CELL = re.compile(r"^\s*(-?[0-9]*\.?[0-9]+)\s*(?:\[\s*(-?[0-9.]+)\s*-\s*(-?[0-9.]+)\s*\])?\s*$")


def metric_name(header: str) -> str:
    """'AUC-ROC' -> 'auc_roc', 'balanced acc' -> 'balanced_acc' (safe MLflow metric key)."""
    return re.sub(r"[^0-9a-z_]+", "_", header.lower()).strip("_")


def parse_report(text: str) -> list[dict]:
    """Rows of every markdown table: {section, subset, metrics: {header: (value, lo, hi)}}.

    lo/hi are None for plain numbers (n, prevalence). `section` is the closest heading above.
    """
    rows, section, header = [], "", None
    for line in text.splitlines():
        if line.startswith("#"):
            section, header = line.lstrip("# ").strip(), None
            continue
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):  # |---|---|
            continue
        if header is None:
            header = cells
            continue
        metrics = {}
        for h, c in zip(header[1:], cells[1:]):
            m = CELL.match(c)
            if m:
                value, lo, hi = m.groups()
                metrics[h] = (float(value), float(lo) if lo else None, float(hi) if hi else None)
        rows.append({"section": section, "subset": cells[0], "metrics": metrics})
    return rows


def flatten(metrics: dict) -> dict:
    """{'AUC-ROC': (0.87, 0.85, 0.88)} -> {'auc_roc': 0.87, 'auc_roc_ci_lo': 0.85, 'auc_roc_ci_hi': 0.88}."""
    out = {}
    for h, (v, lo, hi) in metrics.items():
        k = metric_name(h)
        out[k] = v
        if lo is not None:
            out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = lo, hi
    return out


def log_run(experiment: str, run_name: str, params=None, metrics=None, tags=None, artifacts=(),
            tracking_uri: str | None = None):
    """One MLflow run. Raises ImportError with a clear message if mlflow is not installed."""
    try:
        import mlflow
    except ImportError as e:  # pragma: no cover
        raise ImportError("mlflow is not installed: pip install mlflow") from e
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in (params or {}).items()})
        mlflow.set_tags({k: str(v) for k, v in (tags or {}).items()})
        for k, v in (metrics or {}).items():
            mlflow.log_metric(k, float(v))
        for a in artifacts:
            mlflow.log_artifact(str(a))
