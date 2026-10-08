"""Rung 2a: logistic regression on image-level numbers that have nothing to do with the hand.

Usage:
    python -X utf8 scripts/run_baseline_metadata.py [image_stats.csv]

Reads reports/image_stats.csv (from audit_image_stats.py), writes reports/baseline_metadata.md.

This is a SHORTCUT detector, not a real model: paper brightness, file size, share of ink and
whether the file has an alpha channel say little or nothing about Parkinson's. If this beats
the majority floor clearly, the dataset leaks the label through those side channels and any
image model must be compared against this number, not just against the floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.evaluate import evaluate, table_header, table_row  # noqa: E402
from src.parkinson_cv.models import logistic_fit_predict  # noqa: E402

FEATURE_SETS = {
    "alpha channel only": ["is_rgba"],
    "brightness + ink + file size": ["mean_gray", "median_gray", "p5_gray", "ink_frac", "filesize"],
    "all of the above": ["is_rgba", "mean_gray", "median_gray", "p5_gray", "ink_frac", "filesize"],
}


def main(stats_csv: Path, n_boot: int = 500) -> int:
    df = pd.read_csv(stats_csv)
    df["is_rgba"] = (df["mode"] == "RGBA").astype(float)
    df["is_wave"] = (df["drawing_type"] == "wave").astype(float)  # lets the pooled model separate types
    subsets = {"spiral": df[df["drawing_type"] == "spiral"],
               "wave": df[df["drawing_type"] == "wave"],
               "pooled": df}
    lines = ["# Baseline 2a: metadata-only logistic regression (shortcut check)", "",
             "Blocked 5-fold CV, 5 repeats. Value [95% bootstrap CI]. Floor = baseline 1 "
             "(balanced accuracy 0.5, AUC-ROC 0.5).", ""]
    for set_name, feats in FEATURE_SETS.items():
        lines += [f"## {set_name}", "", f"features: {feats}", ""] + table_header()
        for name, sub in subsets.items():
            res, prev = evaluate(sub, logistic_fit_predict(feats + ["is_wave"]), n_boot=n_boot)
            lines.append(table_row(name, len(sub), prev, res))
        lines.append("")
    lines += ["Reading: balanced acc / AUC-ROC clearly above 0.5 mean the label leaks through "
              "these side channels. Any image model has to beat THESE numbers to show it "
              "learned something about the drawing.", ""]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "baseline_metadata.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    default = ROOT / "reports" / "image_stats.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
