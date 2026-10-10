import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("find_twins", ROOT / "scripts" / "find_twins.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _strokes(n, seed=0):
    """Random 'drawings': a few bright random rectangles on an empty page."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 64, 64), dtype=np.uint8)
    for i in range(n):
        for _ in range(4):
            y0, x0 = rng.integers(0, 40, 2)
            h, w = rng.integers(4, 24, 2)
            X[i, y0:y0 + h, x0:x0 + w] = 200
    return X


def test_a_twin_with_a_different_background_and_orientation_is_found_after_binarising():
    from src.parkinson_cv.ablate import binarize
    from src.parkinson_cv.appearance import thumbnails

    X = _strokes(6)
    twin = np.rot90(X[0]).copy()
    twin = np.where(twin == 0, 18, twin)          # same drawing, rotated, with a grey background
    X[3] = twin
    sim = _mod.dihedral_similarity(thumbnails(binarize(X)))
    assert sim[0, 3] > 0.95 and sim[3, 0] > 0.95
    assert sim[0, 1] < 0.9


def test_clusters_join_chains_of_twins():
    sim = np.full((5, 5), 0.1, dtype=np.float32)
    for a, b in [(0, 1), (1, 2)]:
        sim[a, b] = sim[b, a] = 0.99
    comp = _mod.clusters(sim, 0.9)
    assert comp[0] == comp[1] == comp[2] and comp[3] != comp[0] and comp[4] != comp[3]


def test_script_writes_report_and_cluster_ids(tmp_path):
    n = 30
    X = np.concatenate([_strokes(n, 1), _strokes(n, 2)])
    X[5] = X[4]                                  # a twin inside the spirals
    dx = np.array(["parkinson", "healthy"] * n)
    np.savez(tmp_path / "cache.npz", X=X, diagnosis=dx, drawing_type=np.array(["spiral"] * n + ["wave"] * n),
             class_name=np.array([f"{d}_{t}" for d, t in zip(dx, ["spiral"] * n + ["wave"] * n)]),
             name_number=np.concatenate([np.arange(n), np.arange(n)]))
    assert _mod.main(tmp_path / "cache.npz", tmp_path, 0.90, tmp_path / "clusters.npz") == 0
    text = (tmp_path / "twins.md").read_text(encoding="utf-8")
    assert "## spiral" in text and "different bg group" in text
    c = np.load(tmp_path / "clusters.npz")["cluster"]
    assert c[4] == c[5] and (c >= 0).all() and len(c) == 2 * n


def test_a_shifted_and_rescaled_copy_is_only_found_when_the_detector_crops():
    from src.parkinson_cv.ablate import apply_ablation
    from src.parkinson_cv.appearance import thumbnails

    X = _strokes(5)
    small = np.zeros((64, 64), dtype=np.uint8)           # same drawing, half the size, in a corner
    small[:32, :32] = X[0][::2, ::2]
    X[3] = small
    plain = _mod.dihedral_similarity(thumbnails(apply_ablation(X, "binary")))
    cropped = _mod.dihedral_similarity(thumbnails(apply_ablation(X, "binary_crop")))
    assert plain[0, 3] < 0.9 and cropped[0, 3] > 0.9
