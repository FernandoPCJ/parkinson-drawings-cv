import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.parkinson_cv.cnn import SmallCNN, cnn_fit_predict, random_affine  # noqa: E402


def test_model_outputs_one_logit_per_image():
    out = SmallCNN()(torch.zeros(5, 1, 64, 64))
    assert out.shape == (5,)


def test_augmentation_keeps_shape_and_range():
    x = torch.rand(4, 1, 32, 32)
    y = random_affine(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1


def test_cnn_learns_an_easy_synthetic_task():
    """Class 1 = a bright blob, class 0 = empty paper. A working pipeline must separate them."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.integers(0, 10, size=(n, 32, 32)).astype(np.uint8)
    y = np.array([0, 1] * (n // 2))
    X[y == 1, 8:24, 8:24] = 200
    df = pd.DataFrame({"diagnosis": np.where(y == 1, "parkinson", "healthy"), "idx": np.arange(n)})
    fit_predict = cnn_fit_predict(X, epochs=8, batch=16, width=4, device="cpu")
    pred, proba = fit_predict(df.iloc[:60], df.iloc[60:])
    assert proba.shape == (20,) and set(np.unique(pred)) <= {0, 1}
    assert (pred == y[60:]).mean() > 0.9


def test_gradcam_returns_a_normalised_map_for_a_single_image():
    from src.parkinson_cv.cnn import grad_cam
    model = SmallCNN(width=4).eval()
    cam, p = grad_cam(model, torch.rand(1, 1, 64, 64))
    assert cam.shape == (64, 64)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    assert 0.0 <= p <= 1.0
