import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_group_neighbors", ROOT / "scripts" / "check_group_neighbors.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _images(n=6):
    return np.random.default_rng(0).integers(0, 255, size=(n, 64, 64)).astype(np.uint8)


def test_a_mirrored_twin_in_another_group_is_found_only_with_transforms():
    X = _images()
    X[3] = X[0][:, ::-1]                          # mirror image of image 0
    groups = np.array([0, 0, 1, 1, 2, 2])
    T = _mod.thumbnails(X)
    sim_t, j_t = _mod.nearest_other_group(T, groups, transforms=True)
    sim_p, _ = _mod.nearest_other_group(T, groups, transforms=False)
    assert j_t[0] == 3 and sim_t[0] > 0.999
    assert sim_p[0] < 0.9


def test_a_rotated_twin_is_found():
    X = _images()
    X[4] = np.rot90(X[1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    sim, j = _mod.nearest_other_group(_mod.thumbnails(X), groups)
    assert j[1] == 4 and sim[1] > 0.999


def test_a_twin_in_the_same_group_does_not_count():
    X = _images(4)
    X[1] = X[0]
    groups = np.array([0, 0, 1, 1])
    sim, j = _mod.nearest_other_group(_mod.thumbnails(X), groups)
    assert sim[0] < 0.9 and j[0] in (2, 3)


def test_script_writes_a_report_from_saved_scores(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    dx = np.array(["parkinson", "healthy"] * (n // 2))
    X = rng.integers(0, 14, size=(2 * n, 32, 32)).astype(np.uint8)
    np.savez(tmp_path / "cache.npz", X=X, diagnosis=np.concatenate([dx, dx]),
             drawing_type=np.array(["spiral"] * n + ["wave"] * n))
    for dtype in ("spiral", "wave"):
        np.save(tmp_path / f"group_holdout_frame_mean_{dtype}_small_CNN.npy", rng.random((1, n)))
    assert _mod.main(tmp_path / "cache.npz", "frame_mean", tmp_path) == 0
    text = (tmp_path / "group_neighbors_frame_mean.md").read_text(encoding="utf-8")
    assert "## spiral" in text and "small_CNN" in text and "top 5%" in text
