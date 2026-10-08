import numpy as np
import pytest

from src.parkinson_cv.features import ALL_FEATURES, erode, ink_mask, stroke_features


def _line(thickness, length=300, size=200, wiggle=0, seed=0):
    """White paper (230) with one dark horizontal line (60) across the image."""
    g = np.full((size, size), 230, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    y0 = size // 2
    for x in range(10, min(size - 10, 10 + length)):
        y = y0 + (int(rng.integers(-wiggle, wiggle + 1)) if wiggle else 0)
        g[y - thickness // 2: y + thickness // 2 + 1, x] = 60
    return g


def test_ink_mask_uses_paper_median():
    g = _line(3)
    m, paper = ink_mask(g)
    assert paper == 230
    assert 0 < m.sum() < 0.1 * g.size


def test_erode_shrinks_and_removes_thin_lines():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert erode(m).sum() == 8 * 8            # lost one pixel on each side
    thin = np.zeros((20, 20), bool)
    thin[10, 2:18] = True
    assert erode(thin).sum() == 0             # a 1-px line disappears


def test_all_features_present_and_finite_for_a_normal_drawing():
    f = stroke_features(_line(5))
    assert set(f) == set(ALL_FEATURES)
    assert all(np.isfinite(v) for v in f.values() if v == v)  # no inf
    assert not np.isnan(f["ink_frac"])


def test_thicker_line_has_larger_thickness_and_more_ink():
    thin, thick = stroke_features(_line(3)), stroke_features(_line(9))
    assert thick["thickness_idx"] > thin["thickness_idx"]
    assert thick["ink_frac"] > thin["ink_frac"]


def test_jagged_line_is_rougher_than_a_straight_one():
    straight = stroke_features(_line(5, wiggle=0))
    jagged = stroke_features(_line(5, wiggle=4, seed=1))
    assert jagged["roughness"] > straight["roughness"]


def test_position_and_extent():
    f = stroke_features(_line(5, length=100, size=200))
    assert f["bbox_w"] == pytest.approx(100 / 200, abs=0.02)
    assert f["cy"] == pytest.approx(0.5, abs=0.02)
    assert f["cx"] < 0.5                         # the line sits in the left half


def test_empty_page_gives_nan():
    f = stroke_features(np.full((100, 100), 220, dtype=np.uint8))
    assert all(np.isnan(v) for v in f.values())
