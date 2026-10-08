import numpy as np
import pytest
from PIL import Image

from src.parkinson_cv.preprocess import (
    estimate_background,
    load_grayscale,
    preprocess,
    resize_with_padding,
)


def _white_with_stroke(h, w):
    img = np.full((h, w), 255, dtype=np.uint8)
    img[h // 2, :] = 0  # a horizontal "stroke"
    return img


def test_output_is_square_with_requested_size():
    out = resize_with_padding(_white_with_stroke(100, 300), size=64)
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_aspect_ratio_is_preserved_not_stretched():
    # 100x300 image -> content must occupy 64 wide x ~21 tall, rest is padding
    out = resize_with_padding(_white_with_stroke(100, 300), size=64, pad_value=128)
    rows_with_padding = np.all(out == 128, axis=1)
    content_rows = int((~rows_with_padding).sum())
    assert 19 <= content_rows <= 23  # 64 * 100/300 ~= 21


def test_padding_uses_background_colour():
    img = np.full((50, 100), 200, dtype=np.uint8)
    out = resize_with_padding(img, size=32)
    assert out[0, 0] == 200 and out[-1, -1] == 200


def test_estimate_background_picks_border_median():
    img = np.full((10, 10), 255, dtype=np.uint8)
    img[5, 5] = 0
    assert estimate_background(img) == 255


def test_preprocess_range_dtype_and_determinism():
    img = _white_with_stroke(80, 80)
    a, b = preprocess(img, size=32), preprocess(img, size=32)
    assert a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0
    np.testing.assert_array_equal(a, b)


def test_preprocess_accepts_file_path_and_rgb(tmp_path):
    rgb = np.stack([_white_with_stroke(60, 90)] * 3, axis=-1)
    p = tmp_path / "x.png"
    Image.fromarray(rgb).save(p)
    assert load_grayscale(p).ndim == 2
    assert preprocess(p, size=48).shape == (48, 48)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        resize_with_padding(np.zeros((4, 4, 3), dtype=np.uint8))
