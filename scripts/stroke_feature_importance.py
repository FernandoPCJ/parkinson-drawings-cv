"""Which stroke features separate healthy from parkinson? (interpretation, not evaluation)

Usage:
    python -X utf8 scripts/stroke_feature_importance.py [stroke_features.csv]

For each drawing type it prints, per feature:
  * direction   : is the feature higher or lower in parkinson drawings?
  * single AUC  : how well that ONE feature separates the classes (0.5 = not at all),
                  folded so it is always >= 0.5
  * LR coef     : standardized logistic-regression coefficient when all features are used
                  together (sign = direction, size = weight). Correlated features share
                  weight, so read the single AUC first and the coefficient second.
Numbers are computed on all images of the type (no CV): they describe the data, they are not
a performance estimate. Writes reports/stroke_feature_importance.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.features import ALL_FEATURES  # noqa: E402
from src.parkinson_cv.metrics import auc_roc  # noqa: E402


def main(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
    lines = ["# Stroke feature importance (descriptive)", "",
             "single AUC: 0.5 = no separation. coef: standardized logistic regression, all features.", ""]
    for dtype in ("spiral", "wave"):
        g = df[df["drawing_type"] == dtype]
        y = (g["diagnosis"] == "parkinson").to_numpy().astype(int)
        X = g[ALL_FEATURES].to_numpy(float)
        coefs = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            StandardScaler().fit_transform(X), y).coef_[0]
        rows = []
        for f, c in zip(ALL_FEATURES, coefs):
            a = auc_roc(y, g[f].to_numpy(float))
            rows.append((f, "higher" if a >= 0.5 else "lower", max(a, 1 - a), c))
        rows.sort(key=lambda r: -r[2])
        lines += [f"## {dtype} (n={len(g)})", "",
                  "| feature | in parkinson | single AUC | LR coef |", "|---|---|---|---|"]
        lines += [f"| {f} | {d} | {a:.3f} | {c:+.2f} |" for f, d, a, c in rows] + [""]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports" / "stroke_feature_importance.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    default = ROOT / "reports" / "stroke_features.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
