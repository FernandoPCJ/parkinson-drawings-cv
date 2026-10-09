"""Two crude numbers that describe the BACKGROUND of a drawing (paper, scan, vignette), not the stroke.

Used by the shortcut audit (`scripts/check_cnn_background.py`) and by the background-group hold-out
(`scripts/run_group_holdout.py`). Both work on the paper-relative ink maps of the CNN cache.
"""
from __future__ import annotations

import numpy as np


def background_features(X: np.ndarray, frame: float = 0.10, inner: float = 0.20):
    """(frame_mean, speckle) for every image of X (n, H, W) uint8.

    frame_mean: mean ink-map value in the outer `frame` share of the image (grey vignette -> high,
                clean paper -> ~0).
    speckle:    share of pixels with faint values (1..10) in the central area (paper grain plus
                anti-aliased stroke edges, so it is a rough number).
    """
    n, h, w = X.shape
    b = max(1, int(round(frame * h)))
    mask = np.zeros((h, w), bool)
    mask[:b] = mask[-b:] = True
    mask[:, :b] = mask[:, -b:] = True
    Xf = X.astype(np.float32)
    frame_mean = Xf[:, mask].mean(1)
    c = int(inner * h)
    centre = Xf[:, c:h - c, c:w - c].reshape(n, -1)
    speckle = ((centre >= 1) & (centre <= 10)).mean(1)
    return frame_mean, speckle
