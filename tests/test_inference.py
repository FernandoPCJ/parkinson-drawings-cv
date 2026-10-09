import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.parkinson_cv.inference import describe, score, sigmoid, to_gray
from src.parkinson_cv.inkmap import ink_image, to_model_input

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    """Stands in for an onnxruntime session: records the input and returns a fixed logit."""

    def __init__(self, logit=0.0):
        self.logit, self.seen = logit, None

    def run(self, _outputs, feeds):
        self.seen = feeds["ink_map"]
        return [np.array([self.logit], dtype=np.float32)]


def _drawing(n=200):
    rng = np.random.default_rng(0)
    img = np.full((n, n), 230, np.uint8)
    rows, cols = slice(n // 5, 4 * n // 5), slice(9 * n // 20, 11 * n // 20)  # a dark vertical stroke
    img[rows, cols] = 20 + rng.integers(0, 30, img[rows, cols].shape)
    return img


def test_to_gray_gives_the_same_array_for_path_pil_and_numpy(tmp_path):
    g = _drawing()
    p = tmp_path / "d.png"
    Image.fromarray(g).save(p)
    a, b, c = to_gray(p), to_gray(str(p)), to_gray(Image.open(p))
    assert a.dtype == np.uint8 and a.shape == g.shape
    assert (a == b).all() and (a == c).all() and (a == g).all()
    assert (to_gray(g) == g).all()


def test_to_gray_ignores_the_alpha_channel():
    g = _drawing(50)
    rgb = np.stack([g, g, g], axis=2)
    opaque = np.concatenate([rgb, np.full((50, 50, 1), 255, np.uint8)], axis=2)
    faded = np.concatenate([rgb, np.full((50, 50, 1), 10, np.uint8)], axis=2)
    assert (to_gray(opaque) == to_gray(faded)).all()
    assert abs(int(to_gray(rgb)[0, 0]) - int(g[0, 0])) <= 1


def test_to_gray_rejects_odd_shapes():
    with pytest.raises(ValueError):
        to_gray(np.zeros((5, 5, 2), np.uint8))


def test_score_uses_the_training_preprocessing_and_applies_a_sigmoid():
    g = _drawing()
    sess = FakeSession(logit=1.5)
    res = score(sess, g)
    expected = to_model_input(ink_image(g, 128))
    assert sess.seen.dtype == np.float32 and sess.seen.shape == (1, 1, 128, 128)
    assert np.array_equal(sess.seen, expected)          # same function as training and predict_onnx
    assert res["logit"] == pytest.approx(1.5) and res["prob"] == pytest.approx(sigmoid(1.5))
    assert 0.0 < res["prob"] < 1.0


def test_score_does_not_depend_on_overall_paper_brightness():
    g = _drawing()
    brighter = np.clip(g.astype(int) + 20, 0, 255).astype(np.uint8)
    a, b = FakeSession(), FakeSession()
    score(a, g)
    score(b, brighter)
    assert np.abs(a.seen - b.seen).max() < 0.02          # paper level is subtracted


def test_describe_is_cautious_in_both_directions():
    for p in (0.9, 0.1):
        text = describe(p)
        assert f"{p:.2f}" in text and "not tuned" in text and "not validated" in text
    assert "parkinson" in describe(0.9) and "healthy" in describe(0.1)


def _demo_module():
    spec = importlib.util.spec_from_file_location("demo_gradio", ROOT / "scripts" / "demo_gradio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                          # gradio is imported only inside build_app
    return mod


def test_demo_stops_with_a_clear_message_when_models_are_missing(tmp_path):
    demo = _demo_module()
    with pytest.raises(SystemExit) as e:
        demo.load_models(tmp_path)
    assert "cnn_spiral.onnx" in str(e.value) and "train_final" in str(e.value)


def test_demo_predict_returns_text_and_two_scores_that_sum_to_one():
    demo = _demo_module()
    predict = demo.make_predict({"spiral": FakeSession(2.0), "wave": FakeSession(-2.0)})
    text, scores = predict(_drawing(), "spiral")
    assert scores["parkinson-like"] > 0.5 and scores["parkinson-like"] + scores["healthy-like"] == pytest.approx(1)
    assert "closer to the 'parkinson'" in text
    text, scores = predict(_drawing(), "wave")
    assert scores["parkinson-like"] < 0.5 and "healthy" in text
    assert predict(None, "spiral") == ("Upload a drawing first.", None)


def test_real_onnx_model_gives_the_same_score_as_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from src.parkinson_cv.cnn import SmallCNN
    from src.parkinson_cv.export import export_onnx
    from src.parkinson_cv.inference import load_session

    torch.manual_seed(0)
    model = SmallCNN(width=4).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, size=64)
    g = _drawing()
    res = score(load_session(path), g, size=64)
    with torch.no_grad():
        ref = float(model(torch.from_numpy(to_model_input(ink_image(g, 64)))))
    assert res["logit"] == pytest.approx(ref, abs=1e-4)
