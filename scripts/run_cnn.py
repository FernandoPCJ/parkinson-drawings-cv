"""Rung 3: small CNN, same blocked CV and metrics as the baselines.

Usage (Colab with GPU, or CPU for a smoke test):
    python scripts/run_cnn.py data/processed/cnn_cache_128.npz            # full run
    python scripts/run_cnn.py data/processed/cnn_cache_128.npz --epochs 2 --repeats 1   # smoke test

Writes reports/cnn_small.md and reports/cnn_oof_<type>.npy (out-of-fold scores, reusable later
for Grad-CAM picks or for comparing errors with baseline 2b).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.cnn import cnn_fit_predict  # noqa: E402
from src.parkinson_cv.cv import DEFAULT_OFFSETS, oof_predictions  # noqa: E402
from src.parkinson_cv.evaluate import METRICS, table_header, table_row  # noqa: E402
from src.parkinson_cv.metrics import bootstrap_ci  # noqa: E402

BARS = {"spiral": 0.738, "wave": 0.748}  # AUC-ROC of rung 2b (best feature set)


def main(cache: Path, epochs: int, repeats: int, width: int, n_boot: int, k: int = 5,
         shuffle_labels: bool = False, tag: str = "", crop: int = 0) -> int:
    z = np.load(cache)
    X = z["X"]
    if crop > 0:  # diagnostic: remove an outer frame of `crop` pixels (paper/border shortcut test)
        X = np.ascontiguousarray(X[:, crop:-crop, crop:-crop])
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"],
                       "class_name": z["class_name"], "name_number": z["name_number"],
                       "idx": np.arange(len(X))})
    offsets = DEFAULT_OFFSETS[:repeats]
    extra = f"; {k} folds" + ("; LABELS SHUFFLED (control, expect AUC ~0.5)" if shuffle_labels else "") \
        + (f"; outer {crop}px frame removed" if crop else "")
    lines = ["# Rung 3: small CNN from scratch", "",
             f"Input {X.shape[1]}x{X.shape[2]} paper-relative ink map; width={width}; "
             f"{epochs} fixed epochs (no early stopping); light augmentation; "
             f"blocked CV, {repeats} repeat(s){extra}. Value [95% bootstrap CI].", "",
             "Bar to beat: rung 2b AUC-ROC, spiral 0.738 and wave 0.748.", ""] + table_header()
    for dtype in ("spiral", "wave"):
        sub = df[df["drawing_type"] == dtype].reset_index(drop=True)
        if shuffle_labels:  # control: a pipeline without leakage must drop to chance
            sub["diagnosis"] = np.random.default_rng(0).permutation(sub["diagnosis"].to_numpy())
        t0 = time.time()
        y, preds, scores = oof_predictions(sub, cnn_fit_predict(X, epochs=epochs, width=width),
                                           offsets=offsets, k=k)
        res = {}
        for name, fn in METRICS.items():
            pt, lo, hi = bootstrap_ci(fn, y, preds, scores, n_boot=n_boot)
            res[name] = f"{pt:.3f} [{lo:.3f}-{hi:.3f}]"
        lines.append(table_row(dtype, len(sub), float(y.mean()), res))
        np.save(ROOT / "reports" / f"cnn_oof_{dtype}{tag}.npy", scores)
        print(f"{dtype}: AUC {res['AUC-ROC']} (bar {BARS[dtype]}) in {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"cnn_small{tag}.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5, choices=range(1, 6))
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--k", type=int, default=5, help="number of CV folds (2 = coarser blocks)")
    ap.add_argument("--shuffle-labels", action="store_true", help="control run with shuffled labels")
    ap.add_argument("--crop", type=int, default=0, help="remove this many pixels from every border")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _k2")
    a = ap.parse_args()
    sys.exit(main(a.cache, a.epochs, a.repeats, a.width, a.n_boot, a.k, a.shuffle_labels, a.tag, a.crop))
