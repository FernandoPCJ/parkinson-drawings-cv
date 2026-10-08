"""Rung 1: majority-class baseline under the blocked CV protocol.

Usage:
    python -X utf8 scripts/run_baseline_majority.py [clean_manifest.csv]

Reads data/processed/yolo_full_manifest_clean.csv and writes reports/baseline_majority.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.baselines import majority_fit_predict  # noqa: E402
from src.parkinson_cv.evaluate import evaluate, table_header, table_row  # noqa: E402
from src.parkinson_cv.splits import load_clean  # noqa: E402


def main(manifest: Path) -> int:
    df = load_clean(manifest)
    subsets = {"spiral": df[df["drawing_type"] == "spiral"],
               "wave": df[df["drawing_type"] == "wave"],
               "pooled": df}
    lines = ["# Baseline 1: majority class (blocked 5-fold CV, 5 repeats)", "",
             "Values: point estimate [95% bootstrap CI]. Images treated as independent.", ""]
    lines += table_header()
    for name, sub in subsets.items():
        res, prev = evaluate(sub, majority_fit_predict)
        lines.append(table_row(name, len(sub), prev, res))
    lines += ["", "Reading: this is the floor. Balanced accuracy 0.5 = coin flip. A model must beat "
              "it in sensitivity AND specificity, not just accuracy.", ""]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "baseline_majority.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    default = ROOT / "data" / "processed" / "yolo_full_manifest_clean.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
