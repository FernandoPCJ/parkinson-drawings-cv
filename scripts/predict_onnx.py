"""Score one drawing with an exported ONNX model (needs onnxruntime, numpy, Pillow; NO torch).

Usage:
    python scripts/predict_onnx.py <image.png> <spiral|wave> [--model models/cnn_spiral.onnx]

The image should look like the dataset images: a drawing already cropped to its bounding box.
The score is the model's probability for the 'parkinson' class on this research dataset. It is a
research prototype and NOT a diagnosis or a medical device.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.inference import DISCLAIMER, load_session, score  # noqa: E402


def main(image: Path, dtype: str, model: Path | None, size: int = 128) -> int:
    if not image.is_file():
        print(f"image not found: {image}  (pass the real path of a drawing, in quotes if it has spaces)")
        return 1
    model = model or ROOT / "models" / f"cnn_{dtype}.onnx"
    if not model.is_file():
        print(f"model not found: {model}  (run scripts/train_final.py first)")
        return 1
    res = score(load_session(model), image, size)
    print(f"{image.name}: model score for 'parkinson' class = {res['prob']:.3f} (logit {res['logit']:+.2f})")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("dtype", choices=["spiral", "wave"])
    ap.add_argument("--model", type=Path, default=None)
    a = ap.parse_args()
    sys.exit(main(a.image, a.dtype, a.model))
