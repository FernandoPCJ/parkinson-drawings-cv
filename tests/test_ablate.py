import numpy as np
import pytest

from src.parkinson_cv.ablate import apply_ablation, binarize, crop_to_ink


def _img(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def test_binarize_keeps_ink_and_removes_grey_vignette_and_grain():
    a = _img()
    a[:, :4] = 20                    # grey frame (vignette)
    a[10:50, 30] = 200               # a stroke
    a[20, 10] = 5                    # paper grain
    b = binarize(a[None])[0]
    assert set(np.unique(b)) == {0, 255}
    assert (b[10:50, 30] == 255).all() and b[:, :4].sum() == 0 and b[20, 10] == 0


def test_binarize_of_an_image_without_ink_is_empty():
    a = _img()
    a[:] = 12
    assert binarize(a[None]).sum() == 0


def test_crop_removes_position_and_size():
    small = _img(); small[4:28, 4:28] = 255          # small square in a corner
    big = _img(); big[10:58, 12:60] = 255            # larger square somewhere else
    a, b = crop_to_ink(np.stack([small, big]))
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 8   # same shape -> (nearly) same picture
    assert a[32, 32] == 255 and a[0, 0] == 0                  # centred, with a margin


def test_crop_keeps_the_aspect_ratio():
    wave = _img(); wave[28:36, 4:60] = 255           # wide and flat
    out = crop_to_ink(wave[None])[0]
    rows, cols = np.nonzero(out > 128)
    assert (cols.max() - cols.min()) > 3 * (rows.max() - rows.min())


def test_crop_leaves_empty_images_empty_and_keeps_shape_and_dtype():
    X = np.zeros((2, 64, 48), dtype=np.uint8)
    out = crop_to_ink(X)
    assert out.shape == X.shape and out.dtype == np.uint8 and out.sum() == 0


def test_apply_ablation_names():
    X = np.random.default_rng(0).integers(0, 255, size=(3, 32, 32)).astype(np.uint8)
    assert apply_ablation(X, "none") is X
    assert set(np.unique(apply_ablation(X, "binary"))) <= {0, 255}
    assert apply_ablation(X, "binary_crop").shape == X.shape
    with pytest.raises(ValueError):
        apply_ablation(X, "blur")
