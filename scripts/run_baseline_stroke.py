"""Rung 2b: logistic regression on hand-crafted stroke features.

Usage:
    python -X utf8 scripts/run_baseline_stroke.py [stroke_features.csv]

Reads reports/stroke_features.csv (from extract_stroke_features.py), writes
reports/baseline_stroke.md. Same protocol as the earlier rungs, so numbers are comparable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.evaluate import evaluate, table_header, table_row  # noqa: E402
from src.parkinson_cv.features import ALL_FEATURES, CONTRAST_NOISE, GEOMETRY  # noqa: E402
from src.parkinson_cv.models import logistic_fit_predict  # noqa: E402

FEATURE_SETS = {
    "geometry only (no paper/contrast numbers)": GEOMETRY,
    "geometry + contrast + paper noise": ALL_FEATURES,
    "contrast + paper noise only (side-channel check)": CONTRAST_NOISE,
}


def main(csv_path: Path, n_boot: int = 500, k: int = 5, tag: str = "") -> int:
    df = pd.read_csv(csv_path)
    n_nan = int(df[ALL_FEATURES].isna().any(axis=1).sum())
    # Empty/odd drawings: fill with the column median (inside CV this slightly leaks the
    # median; with so few rows it does not matter, and the count is reported).
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
    df["is_wave"] = (df["drawing_type"] == "wave").astype(float)
    subsets = {"spiral": df[df["drawing_type"] == "spiral"],
               "wave": df[df["drawing_type"] == "wave"],
               "pooled": df}
    lines = ["# Baseline 2b: stroke-feature logistic regression", "",
             f"Blocked {k}-fold CV, 5 repeats. Value [95% bootstrap CI]. Rows with NaN features "
             f"filled with the median: {n_nan}.", "",
             "Bars to beat: floor (balanced acc 0.5, AUC 0.5) and metadata baseline 2a.", ""]
    for set_name, feats in FEATURE_SETS.items():
        lines += [f"## {set_name}", "", f"features: {feats}", ""] + table_header()
        for name, sub in subsets.items():
            res, prev = evaluate(sub, logistic_fit_predict(feats + ["is_wave"]), n_boot=n_boot, k=k)
            lines.append(table_row(name, len(sub), prev, res))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports" / f"baseline_stroke{tag}.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", type=Path, default=ROOT / "reports" / "stroke_features.csv")
    ap.add_argument("--k", type=int, default=5, help="number of CV folds (2 = coarser blocks)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _k2")
    a = ap.parse_args()
    sys.exit(main(a.csv, k=a.k, tag=a.tag))
