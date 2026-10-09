"""Leakage audit of a leave-one-background-group-out result: does a model win on images with a near-twin in training?

Usage (numpy + pandas only, NO torch; run AFTER run_group_holdout.py has saved its score files):
    python -X utf8 scripts/check_group_neighbors.py data/processed/cnn_cache_128.npz --by frame_mean

For every image it finds the most similar image in a DIFFERENT group, i.e. one that sat in the training
set when this image was held out. Similarity is cosine on 32x32 thumbnails of the ink map, taken as the
maximum over the 8 rotations / mirror images, so a twin that was rotated or flipped is also found
(the upstream data contains augmented copies). Then, for every saved score file of the hold-out
(reports/group_holdout_<by>*_<type>_*.npy), it reports AUC inside similarity terciles and the top 5%.

How to read it: a model that profits from twins has a much higher AUC where similarity is high.
A model that does not has a similar AUC in every row. 1-NN accuracy (copy the label of the most similar
training image, no learning) shows how far plain copying goes. Writes reports/group_neighbors_<by>.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.background import background_features  # noqa: E402
from src.parkinson_cv.groupcv import tercile_groups  # noqa: E402
from src.parkinson_cv.metrics import auc_roc  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_cnn_neighbors", ROOT / "scripts" / "check_cnn_neighbors.py")
_nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nb)
thumbnails = _nb.thumbnails


def dihedral(T: np.ndarray, size: int = 32):
    """The 8 rotations / mirror images of every thumbnail (each still unit length, zero mean)."""
    t = T.reshape(len(T), size, size)
    for k in range(4):
        r = np.rot90(t, k, axes=(1, 2))
        yield r.reshape(len(T), -1)
        yield r[:, :, ::-1].reshape(len(T), -1)


def nearest_other_group(T: np.ndarray, groups: np.ndarray, transforms: bool = True):
    """For every row: (max similarity, index) over rows of a different group; optionally over 8 transforms."""
    best = np.full(len(T), -2.0)
    idx = np.zeros(len(T), dtype=int)
    others = dihedral(T) if transforms else [T]
    blocked = groups[:, None] == groups[None, :]
    for Tk in others:
        sim = T @ Tk.T
        sim[blocked] = -2.0
        j = sim.argmax(1)
        s = sim[np.arange(len(T)), j]
        better = s > best
        best[better], idx[better] = s[better], j[better]
    return best, idx


def labelled_auc(y, s):
    return auc_roc(y, s) if len(set(y)) == 2 else float("nan")


def main(cache: Path, by: str = "frame_mean", out_dir: Path | None = None, transforms: bool = True) -> int:
    out_dir = out_dir or ROOT / "reports"
    z = np.load(cache)
    X = z["X"]
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"]})
    df["value"] = background_features(X)[0 if by == "frame_mean" else 1]
    lines = [f"# Group hold-out leakage audit (groups by {by})", "",
             "Similarity = cosine on 32x32 ink-map thumbnails, best of " + ("8 rotations/mirrors" if transforms else "1 orientation")
             + ", against images of the OTHER groups (the training set of that hold-out).", ""]
    for dtype in ("spiral", "wave"):
        mask = (df["drawing_type"] == dtype).to_numpy()
        sub = df[mask].reset_index(drop=True)
        groups = tercile_groups(sub["value"].to_numpy())
        y = (sub["diagnosis"] == "parkinson").to_numpy().astype(int)
        sim, j = nearest_other_group(thumbnails(X[mask]), groups, transforms)
        nn_ok = (y[j] == y)
        q = np.quantile(sim, [1 / 3, 2 / 3])
        bins = np.digitize(sim, q)
        top = sim >= np.quantile(sim, 0.95)
        lines += [f"## {dtype} (n={len(sub)})", "",
                  f"similarity to the closest training image: median {np.median(sim):.3f}, p90 {np.quantile(sim, .9):.3f}, "
                  f"max {sim.max():.3f}; share >= 0.95: {(sim >= .95).mean():.1%}; share >= 0.90: {(sim >= .90).mean():.1%}", ""]
        files = sorted(out_dir.glob(f"group_holdout_{by}*_{dtype}_*.npy"))
        header = "| model | low | mid | high | top 5% | all |\n|---|---|---|---|---|---|"
        lines += ["AUC-ROC by similarity tercile (low = least similar to anything in training). `all` pools the "
                  "three hold-outs into one AUC, so it is not the same number as the mean of the three in-group AUCs:",
                  "", header]
        seen = set()
        for f in files:
            score = np.load(f)[0]
            if len(score) != len(sub):
                continue
            label = f.stem.split(f"_{dtype}_", 1)[1]
            if label in seen:       # baselines are saved again by every run; deterministic, so identical
                continue
            seen.add(label)
            cells = [f"{labelled_auc(y[bins == b], score[bins == b]):.3f}" for b in range(3)]
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {labelled_auc(y[top], score[top]):.3f} | {labelled_auc(y, score):.3f} |")
        lines += ["", "| similarity tercile | range | n | 1-NN accuracy (copy the neighbour's label) |", "|---|---|---|---|"]
        for b, name in enumerate(["low", "mid", "high"]):
            m = bins == b
            lines.append(f"| {name} | {sim[m].min():.3f}-{sim[m].max():.3f} | {int(m.sum())} | {nn_ok[m].mean():.3f} |")
        lines.append(f"| top 5% | >= {sim[top].min():.3f} | {int(top.sum())} | {nn_ok[top].mean():.3f} |")
        lines.append("")
    lines += ["Reading: a model that profits from twins has a much higher AUC in `high`/`top 5%` than in `low`.",
              "If the pre-trained network is high everywhere (also in `low`), twins do not explain it and another",
              "cause must be looked for (a shortcut shared by all groups, such as the acquisition source)."]
    text = "\n".join(lines)
    print(text)
    (out_dir / f"group_neighbors_{by}.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--by", choices=["speckle", "frame_mean"], default="frame_mean")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-transforms", action="store_true", help="only the original orientation")
    a = ap.parse_args()
    sys.exit(main(a.cache, a.by, a.out_dir, not a.no_transforms))
