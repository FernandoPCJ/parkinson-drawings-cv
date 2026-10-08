"""Train the FINAL models (all drawings of one type), export to ONNX, verify, time, and track.

Usage:
    python scripts/train_final.py data/processed/cnn_cache_128.npz [--epochs 20] [--mlflow]

For spiral and wave it:
  1. trains one SmallCNN on ALL usable drawings (same recipe as the CV runs: 20 epochs, seed 0);
  2. saves models/cnn_<type>.pt and exports models/cnn_<type>.onnx (dynamic batch);
  3. checks PyTorch vs ONNX Runtime logits on 64 real drawings + a blank page (max abs difference);
  4. measures single-image CPU latency of both (median / p95 over 200 runs);
  5. writes reports/export.md and, with --mlflow, logs everything to MLflow (sqlite:///mlflow.db).
These models have NO held-out evaluation: their expected quality is the cross-validated number in
README (AUC 0.87 spiral / 0.80 wave), not something measured here. Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.cnn import train_cnn  # noqa: E402
from src.parkinson_cv.export import check_equivalence, export_onnx, latency_ms, onnx_session  # noqa: E402


def main(cache: Path, epochs: int, width: int, use_mlflow: bool) -> int:
    z = np.load(cache)
    X = z["X"]
    df = pd.DataFrame({"diagnosis": z["diagnosis"], "drawing_type": z["drawing_type"],
                       "idx": np.arange(len(X))})
    (ROOT / "models").mkdir(exist_ok=True)
    lines = ["# Export report", "",
             "Final models trained on all usable drawings of each type. No held-out evaluation here: "
             "expected quality is the cross-validated result in the README.", "",
             "| type | n train | ONNX size (KB) | max abs logit diff | same decisions | torch median / p95 (ms) | "
             "ONNX median / p95 (ms) |", "|---|---|---|---|---|---|---|"]
    for dtype in ("spiral", "wave"):
        sub = df[df["drawing_type"] == dtype].reset_index(drop=True)
        print(f"{dtype}: training on {len(sub)} drawings, {epochs} epochs ...")
        model, _ = train_cnn(X, sub, epochs=epochs, width=width)
        model = model.cpu().eval()
        pt, onnx_path = ROOT / "models" / f"cnn_{dtype}.pt", ROOT / "models" / f"cnn_{dtype}.onnx"
        torch.save(model.state_dict(), pt)
        export_onnx(model, onnx_path, size=X.shape[1])

        rows = np.random.default_rng(0).choice(sub["idx"].to_numpy(), size=min(64, len(sub)), replace=False)
        x = np.concatenate([(X[rows].astype(np.float32) / 255.0)[:, None],
                            np.zeros((1, 1, X.shape[1], X.shape[2]), np.float32)])  # + a blank page
        eq = check_equivalence(model, onnx_path, x)
        one = x[:1]
        sess = onnx_session(onnx_path)
        with torch.no_grad():
            lt = latency_ms(lambda: model(torch.from_numpy(one)))
        lo = latency_ms(lambda: sess.run(None, {"ink_map": one}))
        size_kb = onnx_path.stat().st_size / 1024
        print(f"  equivalence: max abs diff {eq['max_abs_diff']:.2e}, same decisions {eq['same_decisions']}; "
              f"latency torch {lt['median_ms']:.2f} ms, onnx {lo['median_ms']:.2f} ms")
        if not eq["ok"]:
            print("  WARNING: ONNX output differs from PyTorch by more than 1e-4")
        lines.append(f"| {dtype} | {len(sub)} | {size_kb:.0f} | {eq['max_abs_diff']:.2e} | {eq['same_decisions']} | "
                     f"{lt['median_ms']:.2f} / {lt['p95_ms']:.2f} | {lo['median_ms']:.2f} / {lo['p95_ms']:.2f} |")
        if use_mlflow:
            from src.parkinson_cv.tracking import log_run
            log_run("parkinson-final-models", f"final_{dtype}",
                    params={"drawing_type": dtype, "epochs": epochs, "width": width, "n_train": len(sub),
                            "input": f"{X.shape[1]}x{X.shape[2]}", "seed": 0},
                    metrics={"onnx_max_abs_diff": eq["max_abs_diff"], "onnx_kb": size_kb,
                             "latency_torch_median_ms": lt["median_ms"], "latency_onnx_median_ms": lo["median_ms"]},
                    tags={"note": "no held-out eval; see CV runs"}, artifacts=[onnx_path],
                    tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "export.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--mlflow", action="store_true", help="also log to MLflow")
    a = ap.parse_args()
    sys.exit(main(a.cache, a.epochs, a.width, a.mlflow))
