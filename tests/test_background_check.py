import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "check_cnn_background", Path(__file__).resolve().parents[1] / "scripts" / "check_cnn_background.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_grey_frame_raises_frame_mean_but_not_speckle():
    clean = np.zeros((1, 100, 100), np.uint8)
    shaded = clean.copy()
    shaded[:, :10, :] = shaded[:, -10:, :] = 40
    shaded[:, :, :10] = shaded[:, :, -10:] = 40
    fm, sp = _mod.background_features(np.concatenate([clean, shaded]))
    assert fm[1] > 30 and fm[0] == 0
    assert sp[0] == sp[1] == 0          # the centre is untouched in both


def test_speckle_counts_faint_pixels_in_the_centre_only():
    img = np.zeros((1, 100, 100), np.uint8)
    img[:, 20:80, 20:80] = 5            # faint grain over the whole centre area (60x60 px)
    img[:, :5, :] = 200                 # strong ink in the frame must not count as speckle
    fm, sp = _mod.background_features(img)
    assert sp[0] > 0.9 and fm[0] > 0
