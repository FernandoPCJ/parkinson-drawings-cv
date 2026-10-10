"""Find near-duplicate drawings ("twins") in the CNN cache, after removing the background.

Usage (numpy, pandas, scipy only; no torch):
    python -X utf8 scripts/find_twins.py data/processed/cnn_cache_128.npz

Why: the 1-NN baseline (copy the label of the most similar image, no learning) beat the networks in the background
hold-out, which is only possible if the images that were held out have near-copies in training. On the raw ink maps
the copies look different because their BACKGROUND differs, so earlier audits missed them. Here every image is
binarised first (ink or paper), reduced to a 32x32 thumbnail and compared with all 8 rotations / mirror images.

For thresholds 0.80 / 0.90 / 0.95 of similarity it reports, per drawing type:
  * how many images have at least one twin and how large the twin clusters (connected components) are;
  * among twin PAIRS: how often both have the same label, how often they sit in DIFFERENT background groups
    (terciles of frame_mean, as in run_group_holdout.py) and in DIFFERENT blocked-CV folds.
It also saves the cluster id of every cache row (data/processed/twin_clusters.npz) for a cluster-grouped evaluation.
Writes reports/twins.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.ablate import apply_ablation  # noqa: E402
from src.parkinson_cv.appearance import thumbnails  # noqa: E402
from src.parkinson_cv.background import background_features  # noqa: E402
from src.parkinson_cv.groupcv import tercile_groups  # noqa: E402
from src.parkinson_cv.splits import blocked_stratified_folds  # noqa: E402

THRESHOLDS = (0.80, 0.90, 0.95)


def dihedral_similarity(T: np.ndarray, size: int = 32) -> np.ndarray:
    """(n, n) matrix: for each pair the best cosine similarity over the 8 rotations / mirrors of the second image."""
    t = T.reshape(len(T), size, size)
    best = np.full((len(T), len(T)), -1.0, dtype=np.float32)
    for k in range(4):
        r = np.rot90(t, k, axes=(1, 2))
        for variant in (r, r[:, :, ::-1]):
            np.maximum(best, T @ variant.reshape(len(T), -1).T, out=best)
    np.fill_diagonal(best, -1.0)
    return best


def clusters(sim: np.ndarray, tau: float) -> np.ndarray:
    """Connected components of the graph "similarity >= tau" (every image gets a component id)."""
    ii, jj = np.nonzero(np.triu(sim >= tau, 1))
    g = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=sim.shape)
    return connected_components(g, directed=False)[1]


def pair_stats(sim, tau, y, group, fold):
    ii, jj = np.nonzero(np.triu(sim >= tau, 1))
    if len(ii) == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return (len(ii), float((y[ii] == y[jj]).mean()), float((group[ii] != group[jj]).mean()),
            float((fold[ii] != fold[jj]).mean()))


def main(cache: Path, out_dir: Path | None = None, save_tau: float = 0.90, out_clusters: Path | None = None,
         on: str = "binary") -> int:
    out_dir = out_dir or ROOT / "reports"
    z = np.load(cache)
    X = z["X"]
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"],
                       "class_name": z["class_name"], "name_number": z["name_number"]})
    cluster_id = np.full(len(df), -1, dtype=int)
    next_id = 0
    lines = ["# Near-duplicate ('twin') audit on binarised drawings", "",
             f"Similarity: cosine on 32x32 thumbnails of the {'binarised and cropped (position and size removed)' if on == 'binary_crop' else 'binarised'} ink map, best of 8 rotations / mirrors. "
             "Groups = terciles of frame_mean (as in the background hold-out); folds = blocked 5-fold (offset 0).", ""]
    for dtype in ("spiral", "wave"):
        mask = (df["drawing_type"] == dtype).to_numpy()
        sub = df[mask].reset_index(drop=True)
        y = (sub["diagnosis"] == "parkinson").to_numpy().astype(int)
        group = tercile_groups(background_features(X[mask])[0])
        fold = blocked_stratified_folds(sub, k=5, offset=0.0).to_numpy()
        sim = dihedral_similarity(thumbnails(apply_ablation(X[mask], on)))
        lines += [f"## {dtype} (n={len(sub)})", "",
                  f"best similarity to any other image: median {np.median(sim.max(1)):.3f}, "
                  f"p10 {np.quantile(sim.max(1), .1):.3f}", "",
                  "| threshold | images with a twin | clusters of size >=2 | largest cluster | label-pure clusters "
                  "| twin pairs | same label | different bg group | different CV fold |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tau in THRESHOLDS:
            comp = clusters(sim, tau)
            sizes = np.bincount(comp)
            big = np.flatnonzero(sizes >= 2)
            in_big = np.isin(comp, big)
            pure = np.mean([len(set(y[comp == c])) == 1 for c in big]) if len(big) else float("nan")
            n_pairs, same_lab, diff_grp, diff_fold = pair_stats(sim, tau, y, group, fold)
            lines.append(f"| {tau:.2f} | {in_big.mean():.1%} | {len(big)} | {sizes.max()} | {pure:.1%} | {n_pairs} "
                         f"| {same_lab:.1%} | {diff_grp:.1%} | {diff_fold:.1%} |")
            if abs(tau - save_tau) < 1e-9:
                cluster_id[mask] = comp + next_id
                next_id += int(comp.max()) + 1
        lines.append("")
    lines += ["Reading: 'different bg group' close to 100% among twin pairs means the background hold-out puts copies on "
              "both sides of the split (so it is easier than it looks). 'different CV fold' high means the blocked CV "
              "does too. 'same label' near 100% means the copies carry the label with them.",
              f"Cluster ids at threshold {save_tau:.2f} were saved for a cluster-grouped evaluation."]
    text = "\n".join(lines)
    print(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("twins.md" if on == "binary" else f"twins_{on}.md")).write_text(text, encoding="utf-8")
    np.savez(out_clusters or ROOT / "data" / "processed" / ("twin_clusters.npz" if on == "binary" else f"twin_clusters_{on}.npz"),
             cluster=cluster_id,
             tau=save_tau, class_name=z["class_name"], name_number=z["name_number"])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--on", choices=["binary", "binary_crop"], default="binary",
                    help="what the copy detector sees; binary_crop also removes position and size")
    ap.add_argument("--save-tau", type=float, default=0.90, help="threshold whose clusters are saved")
    a = ap.parse_args()
    sys.exit(main(a.cache, save_tau=a.save_tau, on=a.on))
