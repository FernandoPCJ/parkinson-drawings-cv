"""Paper-relative ink map: the exact preprocessing used for training AND for inference."""
from __future__ import annotations

import numpy as np
from PIL import Image


def ink_image(gray: np.ndarray, size: int = 128) -> np.ndarray:
    """Paper-relative ink map as uint8 (0 = paper, larger = darker ink), resized by averaging.

    ink = clip(median(gray) - gray, 0, 255): the paper level becomes 0, so overall paper brightness
    is removed. Then the image is shrunk to size x size with area averaging (BOX filter).
    NOTE: the dataset images are drawings already cropped to their bounding box; an arbitrary photo
    with a lot of margin will be squeezed differently and is out of distribution.
    """
    ink = np.clip(float(np.median(gray)) - gray.astype(np.float32), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(ink).resize((size, size), Image.BOX), dtype=np.uint8)


def to_model_input(ink: np.ndarray) -> np.ndarray:
    """uint8 ink map (H, W) -> float32 array (1, 1, H, W) in 0..1, the network input."""
    return (ink.astype(np.float32) / 255.0)[None, None]
