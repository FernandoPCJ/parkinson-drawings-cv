import numpy as np
import pandas as pd
import pytest

from src.parkinson_cv.splits import blocked_stratified_folds, load_clean


def _df():
    rows = []
    num = 0
    for diag, n_sp, n_wv in (("healthy", 53, 41), ("parkinson", 67, 59)):
        for dtype, n in (("spiral", n_sp), ("wave", n_wv)):
            for _ in range(n):
                num += 1
                rows.append({"diagnosis": diag, "drawing_type": dtype, "name_number": num})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffled


def test_every_row_gets_a_valid_fold():
    f = blocked_stratified_folds(_df(), k=5)
    assert f.between(0, 4).all()


def test_folds_are_balanced_inside_each_stratum():
    df = _df()
    df["fold"] = blocked_stratified_folds(df, k=5)
    for _, g in df.groupby(["diagnosis", "drawing_type"]):
        sizes = g["fold"].value_counts()
        assert len(sizes) == 5
        assert sizes.max() - sizes.min() <= 1


def test_blocks_are_contiguous_when_sorted_by_number():
    df = _df()
    df["fold"] = blocked_stratified_folds(df, k=5)
    for _, g in df.groupby(["diagnosis", "drawing_type"]):
        folds_by_number = g.sort_values("name_number")["fold"].to_numpy()
        assert (np.diff(folds_by_number) >= 0).all()  # never goes back to an earlier fold


def test_offset_moves_the_boundaries_but_keeps_all_folds():
    df = _df()
    a = blocked_stratified_folds(df, k=5, offset=0.0)
    b = blocked_stratified_folds(df, k=5, offset=0.4)
    assert not a.equals(b)
    assert set(b.unique()) == {0, 1, 2, 3, 4}


def test_invalid_arguments():
    with pytest.raises(ValueError):
        blocked_stratified_folds(_df(), k=1)
    with pytest.raises(ValueError):
        blocked_stratified_folds(_df(), k=5, offset=5.0)


def test_load_clean_keeps_only_usable_rows(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"split_clean": ["train", "val", "dup", "drop"], "x": [1, 2, 3, 4]}).to_csv(p, index=False)
    out = load_clean(p)
    assert list(out["x"]) == [1, 2]
