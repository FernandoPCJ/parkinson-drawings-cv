"""Does the result survive a NEW background regime? Leave-one-background-group-out.

Usage:
    python -X utf8 scripts/run_group_holdout.py data/processed/cnn_cache_128.npz [--by speckle|frame_mean]
    python -X utf8 scripts/run_group_holdout.py data/processed/cnn_cache_128.npz --skip-cnn   # seconds, no torch
    python -X utf8 scripts/run_group_holdout.py data/processed/cnn_cache_128.npz --model resnet18   # transfer
    python -X utf8 scripts/run_group_holdout.py data/processed/cnn_cache_128.npz --model resnet18-scratch   # control
    python -X utf8 scripts/run_group_holdout.py data/processed/cnn_cache_128.npz --model resnet18 --ablate binary_crop

Motivation: the dataset is a compilation of several sources and the background style (grey frame, paper
grain) differs in class balance between groups. The blocked CV still puts every background style on both
sides of each split. Here the drawings of each drawing type are cut into three groups by a background
number (lowest / middle / highest third); each group is the test set once and the models are trained on the
other two. Models compared on the SAME splits:
  * background only: logistic regression on (frame_mean, speckle), the shortcut itself;
  * stroke features: logistic regression, geometry only, and geometry + contrast + paper noise;
  * the small CNN (same recipe as run_cnn.py), unless --skip-cnn.
Only the AUC INSIDE each held-out group is reported (with a bootstrap CI): it asks whether the ranking learned
on other backgrounds still works on this one, and it cannot be won by knowing the group's class balance.
The lowest and highest thirds are extrapolation (outside the training range); the middle third is interpolation.
Groups are a crude stand-in for "a new source": the dataset has no source label.

Needs reports/stroke_features.csv (extract_stroke_features.py) and the CNN cache (build_cnn_cache.py).
Writes reports/group_holdout_<by>.md. With the CNN: 3 groups x 2 types x seeds trainings (about 10-15 min on CPU).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.ablate import ABLATIONS, apply_ablation  # noqa: E402
from src.parkinson_cv.appearance import knn_fit_predict  # noqa: E402
from src.parkinson_cv.background import background_features  # noqa: E402
from src.parkinson_cv.features import ALL_FEATURES, GEOMETRY  # noqa: E402
from src.parkinson_cv.groupcv import leave_one_group_out, tercile_groups  # noqa: E402
from src.parkinson_cv.metrics import auc_roc, bootstrap_ci  # noqa: E402
from src.parkinson_cv.models import logistic_fit_predict  # noqa: E402

GROUP_NAMES = ["low", "mid", "high"]
CNN_REFERENCE = {"spiral": 0.869, "wave": 0.804}  # blocked-CV AUC-ROC, 5 repeats (README)


def load_table(cache: Path, stroke_csv: Path):
    """Cache rows + background numbers + stroke features, in cache order. Returns (df, X)."""
    z = np.load(cache)
    X = z["X"]
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"],
                       "class_name": z["class_name"], "name_number": z["name_number"],
                       "idx": np.arange(len(X))})
    df["frame_mean"], df["speckle"] = background_features(X)
    strokes = pd.read_csv(stroke_csv)[["class_name", "name_number"] + ALL_FEATURES]
    df = df.merge(strokes, on=["class_name", "name_number"], how="left", validate="one_to_one")
    if df[ALL_FEATURES].isna().all(axis=1).any():
        raise SystemExit("some drawings of the cache have no row in the stroke feature file; "
                         "rebuild both from the same clean manifest")
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
    return df, X


def models_for(X, epochs: int, width: int, seeds: int, skip_cnn: bool, model: str = "cnn") -> dict:
    """name -> list of fit_predict functions (one per repeat; deterministic models have one).

    model: "cnn" (small CNN from scratch), "resnet18" (ImageNet pre-trained, fine-tuned) or
    "resnet18-scratch" (same architecture, random weights: the control for transfer).
    """
    out = {
        "background only (frame_mean, speckle)": [logistic_fit_predict(["frame_mean", "speckle"])],
        "stroke geometry only": [logistic_fit_predict(GEOMETRY)],
        "stroke geometry + contrast + paper noise": [logistic_fit_predict(ALL_FEATURES)],
        "thumbnail 1-NN (appearance only, no learning)": [knn_fit_predict(X)],
    }
    if skip_cnn:
        return out
    if model == "cnn":
        from src.parkinson_cv.cnn import cnn_fit_predict
        out["small CNN"] = [cnn_fit_predict(X, epochs=epochs, width=width, seed=s) for s in range(seeds)]
    else:
        from src.parkinson_cv.transfer import resnet_fit_predict
        pre = model == "resnet18"
        label = "ResNet18 (ImageNet, fine-tuned)" if pre else "ResNet18 (random init)"
        out[label] = [resnet_fit_predict(X, epochs=epochs, seed=s, pretrained=pre) for s in range(seeds)]
    return out


def ci_text(y, preds, scores, n_boot: int):
    pt, lo, hi = bootstrap_ci(lambda yy, p, s: auc_roc(yy, s), y, preds, scores, n_boot=n_boot)
    return pt, f"{pt:.3f} [{lo:.3f}-{hi:.3f}]"


def main(cache: Path, stroke_csv: Path, out_dir: Path, by: str = "speckle", epochs: int | None = None,
         width: int = 16, seeds: int = 1, n_boot: int = 500, skip_cnn: bool = False,
         model: str = "cnn", ablate: str = "none") -> int:
    if epochs is None:
        epochs = 20 if model == "cnn" else 8
    suffix = ("" if model == "cnn" else f"_{model.replace('-', '_')}") + ("" if ablate == "none" else f"_{ablate}")
    df, X = load_table(cache, stroke_csv)      # groups and stroke features come from the ORIGINAL images
    X = apply_ablation(X, ablate)              # networks and 1-NN see the ablated ones
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Leave-one-background-group-out (groups by {by})", "",
             f"Each drawing type is cut into three groups by terciles of `{by}` (low / mid / high); each group is "
             f"the test set once, training uses the other two. AUC-ROC INSIDE each held-out group, "
             f"value [95% bootstrap CI over images, averaged over {seeds if not skip_cnn else 1} repeat(s)]. "
             + f"Input: {ablate}. "
             + (f"Network ({model}): {epochs} fixed epochs" + (f", width {width}." if model == "cnn" else ".")
                if not skip_cnn else "Network skipped."), "",
             "Low and high are extrapolation, mid is interpolation. Groups are a crude stand-in for a new data "
             "source. Images are treated as independent, so the intervals are optimistic.", ""]
    for dtype in ("spiral", "wave"):
        sub = df[df["drawing_type"] == dtype].reset_index(drop=True)
        groups = tercile_groups(sub[by].to_numpy())
        y = (sub["diagnosis"] == "parkinson").to_numpy().astype(int)
        lines += [f"## {dtype} (n={len(sub)})", "", "| group | n | parkinson rate | " + by + " range |",
                  "|---|---|---|---|"]
        for g, name in enumerate(GROUP_NAMES):
            m = groups == g
            v = sub[by].to_numpy()[m]
            lines.append(f"| {name} | {int(m.sum())} | {y[m].mean():.2f} | {v.min():.3f}-{v.max():.3f} |")
        lines += ["", "| model | " + " | ".join(f"held out: {n}" for n in GROUP_NAMES) + " | mean of the 3 |",
                  "|---|" + "---|" * (len(GROUP_NAMES) + 1)]
        for name, fns in models_for(X, epochs, width, seeds, skip_cnn, model).items():
            t0 = time.time()
            runs = [leave_one_group_out(sub, groups, fn) for fn in fns]
            preds = np.stack([r[1] for r in runs])
            scores = np.stack([r[2] for r in runs])
            cells, points = [], []
            for g in range(len(GROUP_NAMES)):
                m = groups == g
                pt, txt = ci_text(y[m], preds[:, m], scores[:, m], n_boot)
                cells.append(txt)
                points.append(pt)
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.mean(points):.3f} |")
            slug = re.sub(r"\W+", "_", name).strip("_")[:24]
            np.save(out_dir / f"group_holdout_{by}{suffix}_{dtype}_{slug}.npy", scores)
            print(f"{dtype} / {name}: mean AUC {np.mean(points):.3f} ({time.time() - t0:.0f}s)")
        if not skip_cnn:
            lines += ["", f"Reference, small CNN with blocked CV (every background on both sides): "
                          f"AUC-ROC {CNN_REFERENCE[dtype]:.3f}. Stroke references: reports/baseline_stroke.md."]
            if model != "cnn":
                lines += ["", "Compare with the small CNN under the same hold-out: "
                              f"reports/group_holdout_{by}.md."]
        lines.append("")
    lines += ["Reading:",
              "- Compare the CNN row with its blocked-CV reference. A small drop means the ranking it learned still "
              "works on a background regime it never saw; a large drop toward the 'background only' row means it "
              "leaned on the background.",
              "- 'background only' is the shortcut. Inside a group the background number varies little, so near 0.5 "
              "(or below) is expected; a value clearly above 0.5 would mean the shortcut carries signal even inside "
              "a group.",
              "- Stroke-feature rows are the non-deep-learning reference under the same shift.",
              "- Only three groups and two drawing types: differences of a few points are within the noise."]
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / f"group_holdout_{by}{suffix}.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--stroke-csv", type=Path, default=ROOT / "reports" / "stroke_features.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--by", choices=["speckle", "frame_mean"], default="speckle")
    ap.add_argument("--epochs", type=int, default=None, help="default 20 for cnn, 8 for resnet18")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1, help="CNN repeats with different seeds")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--model", choices=["cnn", "resnet18", "resnet18-scratch"], default="cnn",
                    help="network to compare (resnet18 needs torchvision and downloads weights once)")
    ap.add_argument("--ablate", choices=ABLATIONS, default="none",
                    help="binary: grey levels removed; binary_crop: also position and size removed (groups unchanged)")
    ap.add_argument("--skip-cnn", action="store_true", help="only background and stroke baselines (fast, no torch)")
    a = ap.parse_args()
    sys.exit(main(a.cache, a.stroke_csv, a.out_dir, a.by, a.epochs, a.width, a.seeds, a.n_boot, a.skip_cnn, a.model, a.ablate))
