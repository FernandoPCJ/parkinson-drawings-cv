"""Compute stroke features for every usable drawing.

Usage:
    python -X utf8 scripts/extract_stroke_features.py [clean_manifest.csv]

Reads data/processed/yolo_full_manifest_clean.csv, writes reports/stroke_features.csv and
prints the median of every feature per class so you can eyeball which ones differ.
Needs numpy, pandas, Pillow. Takes a few minutes (3,221 images at full resolution).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.features import ALL_FEATURES, load_gray, stroke_features  # noqa: E402
from src.parkinson_cv.splits import load_clean  # noqa: E402


def main(manifest: Path) -> int:
    df = load_clean(manifest)
    print(f"{len(df)} images; computing features...")
    rows = []
    for i, p in enumerate(df["path"], 1):
        rows.append(stroke_features(load_gray(p)))
        if i % 500 == 0:
            print(f"  {i}/{len(df)}")
    feats = pd.DataFrame(rows)
    out = pd.concat([df[["path", "class_name", "diagnosis", "drawing_type", "name_number"]]
                     .reset_index(drop=True), feats], axis=1)
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports" / "stroke_features.csv", index=False)

    nan_rows = int(feats.isna().any(axis=1).sum())
    print(f"\nrows with at least one NaN feature: {nan_rows} of {len(out)}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    med = out.groupby("class_name")[ALL_FEATURES].median().T
    print("\nMedian of each feature per class:\n")
    print(med.round(4).to_string())
    print("\nSaved reports/stroke_features.csv")
    return 0


if __name__ == "__main__":
    default = ROOT / "data" / "processed" / "yolo_full_manifest_clean.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
