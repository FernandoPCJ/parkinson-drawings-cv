"""Build a small, portable image cache for CNN training (runs on YOUR computer, no torch needed).

Usage:
    python -X utf8 scripts/build_cnn_cache.py [clean_manifest.csv] [size]

For every usable drawing it:
  1. opens the image in grayscale (alpha ignored);
  2. subtracts the image from its own paper level:  ink = clip(median - gray, 0, 255).
     Paper becomes 0 and ink becomes positive. This removes paper brightness and most of the
     paper noise, the two side channels found in baseline 2a, so the CNN has less to cheat with;
  3. shrinks 512x512 -> size x size by area averaging (default 128, an exact 4x4 average).
Writes data/processed/cnn_cache_<size>.npz (about 50 MB at 128). Upload that single file to
Colab: the Colab run then needs neither the Kaggle dataset nor the manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.features import load_gray  # noqa: E402
from src.parkinson_cv.splits import load_clean  # noqa: E402


def ink_image(gray: np.ndarray, size: int) -> np.ndarray:
    """Paper-relative ink map as uint8 (0 = paper, larger = darker ink), resized by averaging."""
    ink = np.clip(float(np.median(gray)) - gray.astype(np.float32), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(ink).resize((size, size), Image.BOX), dtype=np.uint8)


def main(manifest: Path, size: int) -> int:
    df = load_clean(manifest)
    print(f"{len(df)} images -> {size}x{size}")
    X = np.zeros((len(df), size, size), dtype=np.uint8)
    for i, p in enumerate(df["path"]):
        X[i] = ink_image(load_gray(p), size)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)}")
    out = ROOT / "data" / "processed" / f"cnn_cache_{size}.npz"
    np.savez_compressed(
        out, X=X,
        diagnosis=df["diagnosis"].to_numpy(str), drawing_type=df["drawing_type"].to_numpy(str),
        class_name=df["class_name"].to_numpy(str), name_number=df["name_number"].to_numpy(int))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    default = ROOT / "data" / "processed" / "yolo_full_manifest_clean.csv"
    m = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    sys.exit(main(m, int(sys.argv[2]) if len(sys.argv) > 2 else 128))
