"""ONNX export of the CNN, numerical-equivalence check and CPU latency."""
from __future__ import annotations

import time

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, path, size: int = 128, opset: int = 17) -> None:
    """Export `model` (1-channel input, one logit out) with a dynamic batch axis."""
    model = model.cpu().eval()
    dummy = torch.zeros(1, 1, size, size)
    kwargs = dict(input_names=["ink_map"], output_names=["logit"],
                  dynamic_axes={"ink_map": {0: "batch"}, "logit": {0: "batch"}},
                  opset_version=opset)
    try:  # new torch versions default to the dynamo exporter; the classic one is simpler and enough here
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, str(path), **kwargs)


def onnx_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def check_equivalence(model: torch.nn.Module, path, x: np.ndarray, atol: float = 1e-4) -> dict:
    """Compare PyTorch and ONNX Runtime logits on x (float32, shape (n, 1, H, W))."""
    model = model.cpu().eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = onnx_session(path).run(None, {"ink_map": x})[0]
    diff = float(np.abs(ref - out).max())
    return {"max_abs_diff": diff, "same_decisions": bool(((ref > 0) == (out > 0)).all()),
            "ok": diff <= atol, "n": int(len(x))}


def latency_ms(fn, runs: int = 200, warmup: int = 20) -> dict:
    """Median and 95th percentile wall time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}
