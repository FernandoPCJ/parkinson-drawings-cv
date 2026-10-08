import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "build_cnn_cache", Path(__file__).resolve().parents[1] / "scripts" / "build_cnn_cache.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _page(paper):
    g = np.full((512, 512), paper, dtype=np.uint8)
    g[200:204, 50:450] = 40  # a dark stroke
    return g


def test_paper_brightness_is_removed():
    bright, dim = _mod.ink_image(_page(240), 128), _mod.ink_image(_page(200), 128)
    assert bright[:20, :20].max() == 0 and dim[:20, :20].max() == 0  # paper is exactly 0
    assert bright.shape == (128, 128) and bright.max() > 0           # stroke survives


def test_stroke_is_kept_after_shrinking():
    out = _mod.ink_image(_page(230), 128)
    assert out[48:52, 20:100].max() > 100
