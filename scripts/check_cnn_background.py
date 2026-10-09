"""Does the CNN follow the DRAWING or the BACKGROUND STYLE? (numpy only, no torch)

Usage:
    python -X utf8 scripts/check_cnn_background.py data/processed/cnn_cache_128.npz

Motivation: in the Grad-CAM pictures, errors seem to follow the background: images with a grey
shaded frame (vignette) are mostly answered "healthy", images with plain speckled white paper mostly
"parkinson", whatever the true label. This script measures that instead of trusting an impression.

Two background numbers per image, taken from the paper-relative ink map (strokes excluded as far as
possible by looking at the frame, or at faint values only):
  * frame_mean : mean ink-map value in the outer 10% frame. Grey vignette -> high, clean paper -> ~0.
  * speckle    : share of pixels with faint values (1..10) in the central 60% area (paper grain plus
                 anti-aliased stroke edges, so it is a rough number).
For each drawing type and each number, images are split in terciles and we print:
  true parkinson rate | mean CNN score | CNN AUC inside the tercile.
Reading:
  - If the true parkinson rate barely changes across terciles but the mean CNN score changes a lot,
    the CNN follows the background, not the label  -> shortcut.
  - If the CNN AUC INSIDE each tercile stays high, it separates classes even among images with the
    same kind of background  -> it is using the drawing.
Needs reports/cnn_oof_<type>.npy from run_cnn.py (first repeat). Writes reports/cnn_background_audit.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.background import background_features  # noqa: E402,F401  (also used by tests)
from src.parkinson_cv.metrics import auc_roc  # noqa: E402


def safe_auc(y, s):
    return auc_roc(y, s) if len(set(y)) == 2 else float("nan")


def main(cache: Path) -> int:
    z = np.load(cache)
    X, dx, dt = z["X"], z["diagnosis"], z["drawing_type"]
    fm, sp = background_features(X)
    lines = ["# Background audit of the CNN", ""]
    for dtype in ("spiral", "wave"):
        m = dt == dtype
        y = (dx[m] == "parkinson").astype(int)
        score = np.load(ROOT / "reports" / f"cnn_oof_{dtype}.npy")[0]
        lines += [f"## {dtype} (n={m.sum()}); CNN AUC overall {auc_roc(y, score):.3f}", ""]
        for name, feat in (("frame_mean (grey frame / vignette)", fm[m]),
                           ("speckle (faint paper grain)", sp[m])):
            a = auc_roc(y, feat)
            lines += [f"### {name}: as a lone classifier AUC {max(a, 1 - a):.3f} "
                      f"({'higher' if a >= .5 else 'lower'} in parkinson); "
                      f"correlation with CNN score {np.corrcoef(feat, score)[0, 1]:+.2f}", "",
                      "| tercile | feature range | n | true parkinson rate | mean CNN score | CNN AUC inside |",
                      "|---|---|---|---|---|---|"]
            q = np.quantile(feat, [1 / 3, 2 / 3])
            bins = np.digitize(feat, q)
            for b_, label in enumerate(["low", "mid", "high"]):
                t = bins == b_
                lines.append(f"| {label} | {feat[t].min():.3f}-{feat[t].max():.3f} | {t.sum()} | "
                             f"{y[t].mean():.2f} | {score[t].mean():.2f} | {safe_auc(y[t], score[t]):.3f} |")
            lines.append("")
    lines += ["Reading: compare the 'true parkinson rate' column with 'mean CNN score'. If the score swings",
              "across terciles while the true rate does not, the CNN follows the background. The last column",
              "says whether it still separates the classes among images with a similar background."]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports" / "cnn_background_audit.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
