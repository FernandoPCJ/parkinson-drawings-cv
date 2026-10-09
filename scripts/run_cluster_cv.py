"""Cross-validation in which copies of the same drawing NEVER sit on both sides of a split.

Usage:
    python -X utf8 scripts/run_cluster_cv.py data/processed/cnn_cache_128.npz --skip-cnn          # seconds, no torch
    python -X utf8 scripts/run_cluster_cv.py data/processed/cnn_cache_128.npz --model resnet18

Why: scripts/find_twins.py showed that ~90% of the images have near-copies (rotated / mirrored versions of the same
drawing, about 8 per drawing) and that the blocked CV puts about 85% of those pairs in different folds. A
no-learning 1-NN beat every network because of it. Here the copies are found on the binarised drawings
(threshold --tau), grouped into clusters, and whole clusters are assigned to folds. The confidence intervals resample
clusters, not images, so they reflect the real number of distinct drawings.

The 1-NN row is the leakage detector: with clusters kept together it should fall to about 0.5. If it does not, there
is still some structure shared between folds that this script does not remove.
Writes reports/cluster_cv<_model><_ablate>.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hold = _load("run_group_holdout")
_twins = _load("find_twins")

from src.parkinson_cv.ablate import apply_ablation, binarize  # noqa: E402
from src.parkinson_cv.appearance import thumbnails  # noqa: E402
from src.parkinson_cv.groupcv import cluster_bootstrap_auc, grouped_folds, leave_one_group_out  # noqa: E402


def main(cache: Path, stroke_csv: Path, out_dir: Path, tau: float = 0.90, k: int = 5, model: str = "cnn",
         epochs: int | None = None, width: int = 16, seeds: int = 1, n_boot: int = 500, skip_cnn: bool = False,
         ablate: str = "none", seed: int = 0) -> int:
    if epochs is None:
        epochs = 20 if model == "cnn" else 8
    df, X0 = _hold.load_table(cache, stroke_csv)
    X = apply_ablation(X0, ablate)                       # what the networks / 1-NN see
    suffix = (("" if model == "cnn" else f"_{model.replace('-', '_')}") + ("" if ablate == "none" else f"_{ablate}")
              + ("" if abs(tau - 0.90) < 1e-9 else f"_tau{int(round(tau * 100)):03d}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Cluster-grouped cross-validation (copies never split across folds)", "",
             f"Copies: cosine >= {tau:.2f} on 32x32 thumbnails of the binarised drawing, best of 8 rotations / mirrors, "
             f"joined into clusters. {k}-fold, whole clusters per fold, stratified by label, seed {seed}. "
             f"AUC-ROC of the pooled out-of-fold scores, [95% CI resampling CLUSTERS], {n_boot} resamples. "
             f"Input: {ablate}." + (" Network skipped." if skip_cnn else f" Network: {model}, {epochs} fixed epochs."), ""]
    for dtype in ("spiral", "wave"):
        mask = (df["drawing_type"] == dtype).to_numpy()
        sub = df[mask].reset_index(drop=True)
        y = (sub["diagnosis"] == "parkinson").to_numpy().astype(int)
        sim = _twins.dihedral_similarity(thumbnails(binarize(X0[mask])))
        cluster = _twins.clusters(sim, tau)
        sizes = np.bincount(cluster)
        folds = grouped_folds(y, cluster, k=k, seed=seed)
        lines += [f"## {dtype}", "",
                  f"{len(sub)} images = {len(sizes)} distinct drawings (clusters); {int((sizes == 1).sum())} have no copy; "
                  f"largest cluster {sizes.max()}; parkinson rate {y.mean():.2f}. "
                  f"Fold sizes: {np.bincount(folds).tolist()}.", "",
                  "| model | AUC-ROC [95% CI over clusters] |", "|---|---|"]
        for name, fns in _hold.models_for(X, epochs, width, seeds, skip_cnn, model).items():
            t0 = time.time()
            runs = [leave_one_group_out(sub, folds, fn) for fn in fns]
            scores = np.mean([r[2] for r in runs], axis=0)
            pt, lo, hi = cluster_bootstrap_auc(y, scores, cluster, n_boot=n_boot, seed=seed)
            lines.append(f"| {name} | {pt:.3f} [{lo:.3f}-{hi:.3f}] |")
            print(f"{dtype} / {name}: AUC {pt:.3f} [{lo:.3f}-{hi:.3f}] ({time.time() - t0:.0f}s)")
        lines.append("")
    lines += ["Reading:",
              "- Compare with the blocked CV and the background hold-out: those numbers were measured with copies on both "
              "sides of the split. The drop is the part that was memorisation of seen drawings.",
              "- The 'thumbnail 1-NN' row should be close to 0.5 here. If it is not, copies below the threshold or other "
              "shared structure remain; try a lower --tau.",
              "- Intervals resample distinct drawings, so they are wider than the old ones on purpose.",
              "- Still no subject IDs: even this does not show generalisation to new patients."]
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / f"cluster_cv{suffix}.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--stroke-csv", type=Path, default=ROOT / "reports" / "stroke_features.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--tau", type=float, default=0.90, help="similarity threshold that defines a copy")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", choices=["cnn", "resnet18", "resnet18-scratch"], default="cnn")
    ap.add_argument("--ablate", choices=_hold.ABLATIONS, default="none")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="seed of the fold assignment")
    ap.add_argument("--skip-cnn", action="store_true", help="only the non-network rows (fast, no torch)")
    a = ap.parse_args()
    sys.exit(main(a.cache, a.stroke_csv, a.out_dir, a.tau, a.k, a.model, a.epochs, a.width, a.seeds, a.n_boot,
                  a.skip_cnn, a.ablate, a.seed))
