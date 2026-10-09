"""Input ablations for the ink maps: remove one kind of information and see if a model still works.

  binarize      grey levels / texture / vignette removed: every pixel is ink (255) or paper (0)
  crop_to_ink   position and size removed: the ink's bounding box is centred on a square canvas and rescaled

If a score survives `binary_crop` the signal is in the SHAPE of the stroke. If it collapses, it was in intensity,
texture, position or size, which a drawing test should not depend on.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def binarize(X: np.ndarray, rel: float = 0.35, floor: int = 32) -> np.ndarray:
    """Ink = pixels above max(rel * brightest pixel of the image, floor). Returns uint8 0 / 255."""
    X = np.asarray(X)
    peak = X.reshape(len(X), -1).max(1).astype(np.float32)
    thr = np.maximum(rel * peak, float(floor))
    return np.where(X > thr[:, None, None], 255, 0).astype(np.uint8)


def crop_to_ink(X: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """Bounding box of the non-zero pixels, padded to a square with `margin`, resized to the input size.

    Aspect ratio is kept (a wave stays wide inside the square). Images without ink stay empty.
    """
    n, h, w = X.shape
    out = np.zeros_like(X)
    for i in range(n):
        ys, xs = np.nonzero(X[i])
        if len(ys) == 0:
            continue
        sub = X[i, ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        side = int(round(max(sub.shape) * (1 + 2 * margin)))
        canvas = np.zeros((side, side), dtype=X.dtype)
        y0, x0 = (side - sub.shape[0]) // 2, (side - sub.shape[1]) // 2
        canvas[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
        out[i] = np.asarray(Image.fromarray(canvas).resize((w, h), Image.BILINEAR))
    return out


ABLATIONS = ("none", "binary", "binary_crop")


def apply_ablation(X: np.ndarray, name: str) -> np.ndarray:
    if name == "none":
        return X
    if name == "binary":
        return binarize(X)
    if name == "binary_crop":
        return crop_to_ink(binarize(X))
    raise ValueError(f"unknown ablation {name!r}; choose from {ABLATIONS}")
