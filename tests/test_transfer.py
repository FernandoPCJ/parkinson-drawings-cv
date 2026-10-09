import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.parkinson_cv.transfer import InkResNet, resnet_fit_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / ("tests/test_groupcv.py" if name == "tg" else f"scripts/{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(n=24, size=32):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 14, size=(n, size, size)).astype(np.uint8)
    df = pd.DataFrame({"diagnosis": ["parkinson", "healthy"] * (n // 2), "idx": np.arange(n)})
    return X, df


def test_model_takes_one_channel_and_returns_one_logit_per_image():
    model = InkResNet(pretrained=False).eval()
    out = model(torch.rand(3, 1, 32, 32))
    assert out.shape == (3,)


def test_fit_predict_returns_labels_and_probabilities_in_range():
    X, df = _data()
    preds, proba = resnet_fit_predict(X, epochs=1, batch=8, pretrained=False)(df.iloc[:16], df.iloc[16:])
    assert preds.shape == proba.shape == (8,)
    assert ((proba >= 0) & (proba <= 1)).all() and set(preds) <= {0, 1}


def test_same_seed_gives_same_probabilities():
    X, df = _data()
    f = lambda: resnet_fit_predict(X, epochs=1, batch=8, seed=3, pretrained=False)(df.iloc[:16], df.iloc[16:])[1]
    np.testing.assert_allclose(f(), f(), atol=1e-5)


def test_holdout_script_writes_a_separate_report_for_the_resnet(tmp_path):
    mod = _load("run_group_holdout")
    cache, strokes = _load("tg")._fake_inputs(tmp_path, n=60)
    assert mod.main(cache, strokes, tmp_path / "out", by="speckle", epochs=1, n_boot=20,
                    model="resnet18-scratch") == 0
    text = (tmp_path / "out" / "group_holdout_speckle_resnet18_scratch.md").read_text(encoding="utf-8")
    assert "ResNet18 (random init)" in text and "small CNN" in text  # reference line mentions the small CNN
    assert not (tmp_path / "out" / "group_holdout_speckle.md").exists()  # does not overwrite the CNN report
