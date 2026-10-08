"""Preprocessing for hand-drawn spiral/wave images.

Design choices (each one is explainable in an interview):
- Grayscale: the pen stroke is the signal, colour is not.
- Resize *with padding* instead of stretching: stretching a non-square image
  would distort the curvature of the stroke, which is exactly what we want to measure.
- Pad with the background colour of the image itself (median of the border pixels),
  so padding does not create an artificial edge the model could latch onto.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SIZE = 224


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image file as a 2-D uint8 array (H, W) in grayscale."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def estimate_background(img: np.ndarray) -> int:
    """Median of the 1-pixel border: a robust guess of the paper colour."""
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    return int(np.median(border))


def resize_with_padding(
    img: np.ndarray, size: int = DEFAULT_SIZE, pad_value: int | None = None
) -> np.ndarray:
    """Resize so the longest side equals `size`, keep aspect ratio, pad to a square.

    Returns a (size, size) uint8 array. The original image is centred.
    """
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {img.shape}")
    h, w = img.shape
    if h == 0 or w == 0:
        raise ValueError("empty image")
    if pad_value is None:
        pad_value = estimate_background(img)

    scale = size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)

    canvas = np.full((size, size), pad_value, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = np.asarray(resized)
    return canvas


def preprocess(source: str | Path | np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Full pipeline: load (if a path) -> grayscale -> resize+pad -> float32 in [0, 1]."""
    img = load_grayscale(source) if isinstance(source, (str, Path)) else source
    out = resize_with_padding(img, size=size)
    return out.astype(np.float32) / 255.0
