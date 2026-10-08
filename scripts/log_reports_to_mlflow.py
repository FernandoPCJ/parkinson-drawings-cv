"""Log the results already written in reports/*.md to MLflow (no retraining).

Usage:
    python scripts/log_reports_to_mlflow.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db        # then open http://127.0.0.1:5000

Each table row (one per subset and feature set) becomes one run in experiment 'parkinson-drawings',
with its metrics and 95% bootstrap CI bounds (auc_roc, auc_roc_ci_lo, auc_roc_ci_hi, ...).
Run it once after the experiments; running it again adds duplicates, so delete mlflow.db first
if you want to start clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.tracking import flatten, log_run, parse_report  # noqa: E402

PATTERNS = ("baseline_*.md", "cnn_small*.md")


def main() -> int:
    files = sorted(f for pat in PATTERNS for f in (ROOT / "reports").glob(pat))
    if not files:
        print("no reports found in reports/")
        return 1
    uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    n = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        setup = next((ln for ln in text.splitlines() if ln.startswith(("Blocked", "Input"))), "")
        for r in parse_report(text):
            metrics = flatten(r["metrics"])
            if not metrics:
                continue
            log_run("parkinson-drawings", f"{f.stem} | {r['section']} | {r['subset']}",
                    params={"report": f.name, "subset": r["subset"], "section": r["section"]},
                    metrics=metrics, tags={"setup": setup[:250]}, artifacts=[f], tracking_uri=uri)
            n += 1
        print(f"logged {f.name}")
    print(f"{n} runs logged to {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
