"""Hand-crafted stroke features from a grayscale drawing (numpy only).

Idea: separate "ink" from "paper", then describe the ink with a few numbers a person could
check by eye: how much ink, how thick, how jagged, where it sits, how often it crosses a line.

Ink = pixels more than INK_DELTA gray levels darker than the paper (the median gray of the
image). This is the same rule used in scripts/audit_image_stats.py, so numbers are comparable.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

INK_DELTA = 40       # how much darker than the paper a pixel must be to count as ink
MAX_LAYERS = 10      # erosion steps used to estimate thickness
MIN_INK_PIXELS = 50  # below this the drawing is considered empty -> features are NaN

GEOMETRY = ["ink_frac", "thickness_idx", "roughness", "bbox_w", "bbox_h", "cx", "cy",
            "sx", "sy", "r_mean", "r_std", "cross_h", "cross_v"]
CONTRAST_NOISE = ["ink_contrast", "paper_noise"]
ALL_FEATURES = GEOMETRY + CONTRAST_NOISE


def load_gray(path: str) -> np.ndarray:
    """Open an image and return a 2-D uint8 grayscale array (alpha, if any, is ignored)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def ink_mask(gray: np.ndarray, delta: int = INK_DELTA) -> tuple[np.ndarray, float]:
    paper = float(np.median(gray))
    return gray < paper - delta, paper


def erode(mask: np.ndarray) -> np.ndarray:
    """One step of binary erosion with a 4-neighbour cross: a pixel survives only if it and
    its up/down/left/right neighbours are all ink. Thin lines vanish after few steps."""
    p = np.pad(mask, 1)
    return p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def stroke_features(gray: np.ndarray) -> dict[str, float]:
    h, w = gray.shape
    m, paper = ink_mask(gray)
    area = int(m.sum())
    if area < MIN_INK_PIXELS:
        return {k: float("nan") for k in ALL_FEATURES}

    # --- thickness: how many erosion layers the ink survives, averaged over all ink pixels.
    # Sum of the remaining areas / initial area. Thicker strokes survive more steps.
    cur, layers = m, 0
    for _ in range(MAX_LAYERS):
        cur = erode(cur)
        n = int(cur.sum())
        if n == 0:
            break
        layers += n
    thickness_idx = 1.0 + layers / area

    # --- roughness: boundary pixels per ink pixel, corrected for thickness. For a straight
    # line it stays near a constant; wobbling or jagged lines expose more boundary.
    boundary = int((m & ~erode(m)).sum())
    roughness = boundary * thickness_idx / area

    # --- where the ink is and how it is spread
    ys, xs = np.nonzero(m)
    cx_px, cy_px = xs.mean(), ys.mean()
    r = np.hypot(xs - cx_px, ys - cy_px) / (0.5 * w)

    # --- how many times a horizontal / vertical scan line enters ink (stroke crossings)
    cross_h = float((m[:, 1:] & ~m[:, :-1]).sum()) / h
    cross_v = float((m[1:, :] & ~m[:-1, :]).sum()) / w

    # --- contrast and paper noise (pixels at least 2 px away from any ink)
    far = erode(erode(~m))
    noise = float(gray[far].std()) if int(far.sum()) > 100 else float("nan")

    return {
        "ink_frac": area / (h * w),
        "thickness_idx": thickness_idx,
        "roughness": roughness,
        "bbox_w": (xs.max() - xs.min() + 1) / w,
        "bbox_h": (ys.max() - ys.min() + 1) / h,
        "cx": cx_px / w,
        "cy": cy_px / h,
        "sx": xs.std() / w,
        "sy": ys.std() / h,
        "r_mean": float(r.mean()),
        "r_std": float(r.std()),
        "cross_h": cross_h,
        "cross_v": cross_v,
        "ink_contrast": paper - float(gray[m].mean()),
        "paper_noise": noise,
    }
