"""Small web demo: upload a spiral or wave drawing and see the model score.

Usage:
    pip install gradio
    python scripts/demo_gradio.py            # then open http://127.0.0.1:7860
    python scripts/demo_gradio.py --share    # temporary public link (your images leave your computer)

Needs models/cnn_spiral.onnx and models/cnn_wave.onnx (made by scripts/train_final.py).
Runs on CPU with ONNX Runtime only, no PyTorch. Research prototype: NOT a medical device, no diagnosis.
Do not upload images of real people's drawings that you are not allowed to share.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.inference import DISCLAIMER, describe, load_session, score  # noqa: E402

TYPES = ("spiral", "wave")


def load_models(models_dir: Path) -> dict:
    """One ONNX session per drawing type; stops with a clear message if a file is missing."""
    missing = [str(models_dir / f"cnn_{t}.onnx") for t in TYPES if not (models_dir / f"cnn_{t}.onnx").is_file()]
    if missing:
        raise SystemExit("Missing model file(s):\n  " + "\n  ".join(missing)
                         + "\nRun `python scripts/train_final.py <cache>` first (see README).")
    return {t: load_session(models_dir / f"cnn_{t}.onnx") for t in TYPES}


def make_predict(sessions: dict):
    def predict(image, drawing_type):
        if image is None:
            return "Upload a drawing first.", None
        res = score(sessions[drawing_type], image)
        return describe(res["prob"]), {"parkinson-like": res["prob"], "healthy-like": 1 - res["prob"]}
    return predict


def build_app(sessions: dict):
    import gradio as gr
    with gr.Blocks(title="Parkinson drawings: research demo") as app:
        gr.Markdown("# Spiral / wave drawings: research demo\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Drawing (cropped to the stroke, like the dataset images)", type="numpy")
                kind = gr.Radio(TYPES, value="spiral", label="Drawing type (a separate model per type)")
                run = gr.Button("Score")
            with gr.Column():
                text = gr.Textbox(label="Reading", lines=4)
                label = gr.Label(label="Model score (not a probability of disease)")
        run.click(make_predict(sessions), [image, kind], [text, label])
        gr.Markdown("Cross-validated AUC-ROC of these models on the research data: spiral 0.87, wave 0.80. "
                    "No test on unseen patients was possible (the dataset has no subject identifiers).")
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = ap.parse_args()
    build_app(load_models(a.models_dir)).launch(share=a.share)
