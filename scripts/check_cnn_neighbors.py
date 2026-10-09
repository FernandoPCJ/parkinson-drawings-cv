"""Leakage audit of the CNN result: does it win mostly on test images that have a near-twin in train?

Usage (needs numpy, pandas; NO torch):
    python -X utf8 scripts/check_cnn_neighbors.py data/processed/cnn_cache_128.npz

Reads the out-of-fold scores saved by run_cnn.py (reports/cnn_oof_<type>.npy) and rebuilds the same
folds (offset 0). For every image it finds the most similar image that sat in a DIFFERENT fold,
i.e. one the model could have trained on. Then it reports, by similarity tercile:
  * CNN AUC            -> if AUC is much higher where a near-twin exists, the gain is leakage-like;
  * 1-nearest-neighbour accuracy (copy the label of the most similar training image, no learning).
Similarity is cosine on 32x32 thumbnails of the ink map. Writes reports/cnn_leakage_neighbors.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.appearance import thumbnails  # noqa: E402,F401
from src.parkinson_cv.metrics import auc_roc  # noqa: E402
from src.parkinson_cv.splits import blocked_stratified_folds  # noqa: E402


def nearest_other_fold(T: np.ndarray, fold: np.ndarray):
    """For every row: (max cosine similarity, index) over rows that are in a different fold."""
    sim = T @ T.T
    sim[fold[:, None] == fold[None, :]] = -2.0  # same fold (incl. itself) is not allowed
    j = sim.argmax(1)
    return sim[np.arange(len(T)), j], j


def main(cache: Path) -> int:
    z = np.load(cache)
    X = z["X"]
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"],
                       "class_name": z["class_name"], "name_number": z["name_number"]})
    lines = ["# CNN leakage audit: performance vs similarity to the closest training image", ""]
    for dtype in ("spiral", "wave"):
        mask = (df["drawing_type"] == dtype).to_numpy()
        sub = df[mask].reset_index(drop=True)
        score = np.load(ROOT / "reports" / f"cnn_oof_{dtype}.npy")[0]
        fold = blocked_stratified_folds(sub, k=5, offset=0.0).to_numpy()
        y = (sub["diagnosis"] == "parkinson").to_numpy().astype(int)
        sim, j = nearest_other_fold(thumbnails(X[mask]), fold)
        nn_pred = y[j]
        q = np.quantile(sim, [1 / 3, 2 / 3])
        bins = np.digitize(sim, q)
        lines += [f"## {dtype} (n={len(sub)})", "",
                  f"similarity to closest other-fold image: median {np.median(sim):.3f}, "
                  f"p90 {np.quantile(sim, .9):.3f}, max {sim.max():.3f}; "
                  f"share >= 0.95: {(sim >= .95).mean():.1%}", "",
                  "| similarity tercile | range | n | CNN AUC | 1-NN accuracy |", "|---|---|---|---|---|"]
        for b, name in enumerate(["low", "mid", "high"]):
            m = bins == b
            auc = auc_roc(y[m], score[m]) if len(set(y[m])) == 2 else float("nan")
            lines.append(f"| {name} | {sim[m].min():.3f}-{sim[m].max():.3f} | {m.sum()} | "
                         f"{auc:.3f} | {(nn_pred[m] == y[m]).mean():.3f} |")
        lines += ["", f"overall: CNN AUC {auc_roc(y, score):.3f}; 1-NN accuracy {(nn_pred == y).mean():.3f}", ""]
    lines += ["Reading: if AUC and 1-NN accuracy rise sharply from the low to the high tercile, the model",
              "profits from near-twins across folds (leakage). If AUC is similar in all terciles, the gain",
              "does not come from twins."]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports" / "cnn_leakage_neighbors.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
