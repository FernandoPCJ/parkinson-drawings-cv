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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.features import load_gray  # noqa: E402
from src.parkinson_cv.inkmap import ink_image, to_model_input  # noqa: E402


def main(image: Path, dtype: str, model: Path | None, size: int = 128) -> int:
    import onnxruntime as ort
    if not image.is_file():
        print(f"image not found: {image}  (pass the real path of a drawing, in quotes if it has spaces)")
        return 1
    model = model or ROOT / "models" / f"cnn_{dtype}.onnx"
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    x = to_model_input(ink_image(load_gray(str(image)), size))
    logit = float(sess.run(None, {"ink_map": x})[0][0])
    prob = 1.0 / (1.0 + np.exp(-logit))
    print(f"{image.name}: model score for 'parkinson' class = {prob:.3f} (logit {logit:+.2f})")
    print("Research prototype only: not a medical device, no diagnostic meaning.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("dtype", choices=["spiral", "wave"])
    ap.add_argument("--model", type=Path, default=None)
    a = ap.parse_args()
    sys.exit(main(a.image, a.dtype, a.model))
