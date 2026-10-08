import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from src.parkinson_cv.cnn import SmallCNN  # noqa: E402
from src.parkinson_cv.export import check_equivalence, export_onnx, latency_ms  # noqa: E402


def test_onnx_matches_pytorch_and_accepts_any_batch_size(tmp_path):
    torch.manual_seed(0)
    model = SmallCNN(width=4).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, size=64)
    x = np.random.default_rng(0).random((5, 1, 64, 64)).astype(np.float32)
    res = check_equivalence(model, path, x)
    assert res["ok"] and res["same_decisions"] and res["n"] == 5
    assert check_equivalence(model, path, x[:1])["ok"]  # dynamic batch axis


def test_latency_reports_positive_times():
    r = latency_ms(lambda: sum(range(1000)), runs=20, warmup=2)
    assert 0 < r["median_ms"] <= r["p95_ms"]
