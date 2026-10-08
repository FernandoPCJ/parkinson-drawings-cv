import numpy as np
from PIL import Image

from src.parkinson_cv.data import (
    find_cross_split_duplicates,
    guess_subject_ids,
    list_images,
)


def _save(path, seed, size=40):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8)).save(path)


def _build(tmp_path):
    root = tmp_path / "drawings"
    _save(root / "spiral/training/healthy/a.png", 1)
    _save(root / "spiral/training/parkinson/b.png", 2)
    _save(root / "spiral/testing/healthy/c.png", 3)
    _save(root / "spiral/testing/parkinson/d.png", 4)
    return root


def test_list_images_indexes_everything(tmp_path):
    df = list_images(_build(tmp_path))
    assert len(df) == 4
    assert set(df["split"]) == {"training", "testing"}
    assert set(df["label"]) == {"healthy", "parkinson"}


def test_no_duplicates_when_images_differ(tmp_path):
    df = list_images(_build(tmp_path))
    assert find_cross_split_duplicates(df).empty


def test_exact_duplicate_across_splits_is_detected(tmp_path):
    root = _build(tmp_path)
    (root / "spiral/testing/healthy/copy_of_a.png").write_bytes(
        (root / "spiral/training/healthy/a.png").read_bytes()
    )
    dups = find_cross_split_duplicates(list_images(root))
    assert (dups["kind"] == "exact").any()


def test_guess_subject_ids():
    import pandas as pd

    ids = guess_subject_ids(pd.Series(["P12_spiral.png", "subject-7.png", "V01HE01.png"]))
    assert ids.iloc[0] == "12"
    assert ids.iloc[1] == "7"
    assert ids.isna().iloc[2]
