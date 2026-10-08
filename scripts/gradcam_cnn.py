"""Grad-CAM audit: does the CNN look at the stroke, or at something else?

Usage:
    python scripts/gradcam_cnn.py data/processed/cnn_cache_128.npz [spiral|wave] [--epochs 20]

Trains ONE model (blocked CV, offset 0) on 4 folds, then explains its answers on the 5th, held-out
fold. Takes about as long as one fold of run_cnn.py (a couple of minutes on CPU).
Writes:
  reports/gradcam_<type>.png : rows = true positive / true negative / false positive / false negative
                               (red = where the evidence comes from; ink is dark)
  reports/gradcam_<type>.md  : numbers (see below)

Numbers, over all held-out images:
  * ink enrichment   = (share of heat that falls on ink, widened by 3 px) / (share of the image that
                       is ink, widened by 3 px). 1.0 = heat ignores the stroke; >1 = heat prefers the stroke.
  * border share     = share of heat in the outer 10% frame of the image. Pure paper, so a high value
                       would point to a scan/border shortcut.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.cnn import grad_cam, predict_proba, train_cnn  # noqa: E402
from src.parkinson_cv.cv import labels  # noqa: E402
from src.parkinson_cv.splits import blocked_stratified_folds  # noqa: E402

INK_MIN = 15      # ink-map value (0-255) above which a pixel counts as stroke
WIDEN = 3         # pixels added around the stroke before measuring overlap
BORDER = 0.10     # outer frame share of the image


def ink_masks(X: np.ndarray) -> np.ndarray:
    t = torch.from_numpy((X >= INK_MIN).astype(np.float32)).unsqueeze(1)
    k = 2 * WIDEN + 1
    return (F.max_pool2d(t, k, stride=1, padding=WIDEN)[:, 0] > 0).numpy()


def heat_stats(cam: np.ndarray, ink: np.ndarray) -> tuple[float, float]:
    """(ink enrichment, border share) for one image."""
    total = cam.sum() + 1e-8
    on_ink = float(cam[ink].sum() / total)
    area = float(ink.mean()) + 1e-8
    h, w = cam.shape
    b = int(round(BORDER * h))
    inner = np.zeros_like(ink)
    inner[b:h - b, b:w - b] = True
    return on_ink / area, float(cam[~inner].sum() / total)


def overlay(ink_img: np.ndarray, cam: np.ndarray, scale: int = 2) -> Image.Image:
    """White paper, dark stroke, red heat on top."""
    gray = 255 - np.clip(ink_img.astype(np.float32) * 3, 0, 255)
    rgb = np.stack([gray] * 3, -1)
    a = (cam[..., None] * 0.65)
    rgb = rgb * (1 - a) + np.array([255, 0, 0], np.float32) * a
    im = Image.fromarray(rgb.astype(np.uint8))
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main(cache: Path, dtype: str, epochs: int, n_show: int) -> int:
    z = np.load(cache)
    X = z["X"]
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"],
                       "name_number": z["name_number"], "idx": np.arange(len(X))})
    sub = df[df["drawing_type"] == dtype].reset_index(drop=True)
    fold = blocked_stratified_folds(sub, k=5, offset=0.0).to_numpy()
    train, test = sub[fold != 0], sub[fold == 0].reset_index(drop=True)
    print(f"{dtype}: training on {len(train)} images, explaining {len(test)} held-out images")
    model, dev = train_cnn(X, train, epochs=epochs)

    y = labels(test)
    proba = predict_proba(model, X, test["idx"].to_numpy(), dev)
    pred = (proba >= 0.5).astype(int)
    masks = ink_masks(X[test["idx"].to_numpy()])
    cams, enrich, border = [], [], []
    for r, row in enumerate(test["idx"].to_numpy()):
        x = torch.from_numpy(X[row]).float().div(255)[None, None].to(dev)
        cam, _ = grad_cam(model, x)
        cams.append(cam)
        e, b = heat_stats(cam, masks[r])
        enrich.append(e)
        border.append(b)
    enrich, border = np.array(enrich), np.array(border)

    groups = {"true positive": (y == 1) & (pred == 1), "true negative": (y == 0) & (pred == 0),
              "false positive": (y == 0) & (pred == 1), "false negative": (y == 1) & (pred == 0)}
    lines = [f"# Grad-CAM audit: {dtype}", "",
             f"One model, trained on folds 1-4, explained on held-out fold 0 ({len(test)} images). "
             f"Held-out accuracy {np.mean(y == pred):.3f}. Ink = pixel >= {INK_MIN} widened by {WIDEN} px.", "",
             "| group | n | ink enrichment (1 = ignores stroke) | heat in outer 10% frame |", "|---|---|---|---|"]
    rng = np.random.default_rng(0)
    cell, label_h = 2 * X.shape[1], 14
    sheet = Image.new("RGB", (cell * n_show, (cell + label_h) * len(groups)), "white")
    draw = ImageDraw.Draw(sheet)
    for g, (name, m) in enumerate(groups.items()):
        ids = np.flatnonzero(m)
        if len(ids):
            lines.append(f"| {name} | {len(ids)} | {enrich[ids].mean():.2f} | {border[ids].mean():.1%} |")
        else:
            lines.append(f"| {name} | 0 | - | - |")
        draw.text((2, g * (cell + label_h) + 1), f"{name} (n={len(ids)})", fill="black")
        for c, i in enumerate(rng.permutation(ids)[:n_show]):
            sheet.paste(overlay(X[test["idx"].iloc[i]], cams[i]),
                        (c * cell, g * (cell + label_h) + label_h))
    lines += ["", f"all images: ink enrichment {enrich.mean():.2f}, heat in frame {border.mean():.1%} "
                  f"(uniform heat would put {1 - (1 - 2 * BORDER) ** 2:.0%} in the frame)", "",
              "Reading: enrichment near 1.0 or a frame share above the uniform value means the model is not",
              "focused on the stroke. Enrichment clearly above 1 with a low frame share means it uses the stroke.",
              "Look at the PNG as well: numbers do not show WHICH part of the drawing is used."]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"gradcam_{dtype}.md").write_text(text, encoding="utf-8")
    sheet.save(ROOT / "reports" / f"gradcam_{dtype}.png")
    print(f"saved reports/gradcam_{dtype}.png")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("dtype", nargs="?", default="spiral", choices=["spiral", "wave"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--show", type=int, default=6, help="examples per group in the PNG")
    a = ap.parse_args()
    sys.exit(main(a.cache, a.dtype, a.epochs, a.show))
