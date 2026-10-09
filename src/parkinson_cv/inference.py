"""Scoring one drawing with an exported ONNX model. No torch needed (numpy, Pillow, onnxruntime).

Shared by `scripts/predict_onnx.py` (command line) and `scripts/demo_gradio.py` (web demo), so both
use exactly the same preprocessing as training (`inkmap.py`).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .inkmap import ink_image, to_model_input

DISCLAIMER = ("Research prototype only. Not a medical device and not a diagnosis. The score is the output "
              "of a small model trained on a public research dataset; it says nothing about a real person.")


def load_session(path):
    """ONNX Runtime session on CPU for one exported model."""
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def to_gray(image) -> np.ndarray:
    """File path, PIL image or numpy array (H, W), (H, W, 3) or (H, W, 4) -> 2-D uint8 grayscale.

    Alpha is ignored, like in `features.load_gray`, so a transparent PNG is read the same way it was
    when the training cache was built.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return np.asarray(im.convert("L"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return np.asarray(Image.fromarray(arr[:, :, :3].astype(np.uint8)).convert("L"), dtype=np.uint8)
    raise ValueError(f"unsupported image shape {arr.shape}")


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def score(session, image, size: int = 128) -> dict:
    """Model output for one drawing: {'logit', 'prob'}; prob is the score for the 'parkinson' class."""
    x = to_model_input(ink_image(to_gray(image), size))
    logit = float(session.run(None, {"ink_map": x})[0][0])
    return {"logit": logit, "prob": sigmoid(logit)}


def describe(prob: float) -> str:
    """Plain-language reading of the score, deliberately cautious (the threshold 0.5 was never tuned)."""
    side = "closer to the 'parkinson' examples" if prob >= 0.5 else "closer to the 'healthy' examples"
    return (f"The model score is {prob:.2f}: this drawing looks {side} of the research dataset. "
            "The 0.5 cut-off was not tuned and the model was not validated on new patients.")
