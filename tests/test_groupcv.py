import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.parkinson_cv.features import ALL_FEATURES
from src.parkinson_cv.groupcv import leave_one_group_out, tercile_groups
from src.parkinson_cv.metrics import auc_roc
from src.parkinson_cv.models import logistic_fit_predict

ROOT = Path(__file__).resolve().parents[1]


def test_tercile_groups_are_three_equal_ordered_thirds():
    v = np.random.default_rng(0).permutation(300).astype(float)
    g = tercile_groups(v)
    assert set(g) == {0, 1, 2} and [int((g == k).sum()) for k in range(3)] == [100, 100, 100]
    assert v[g == 0].max() < v[g == 1].min() and v[g == 1].max() < v[g == 2].min()


def test_every_group_is_tested_by_a_model_that_never_saw_it():
    df = pd.DataFrame({"diagnosis": ["parkinson", "healthy"] * 30, "g": np.repeat([0, 1, 2], 20)})
    seen = []

    def fit_predict(train, test):
        seen.append((set(train["g"]), set(test["g"])))
        return np.zeros(len(test), int), np.full(len(test), 0.5)

    y, preds, scores = leave_one_group_out(df, df["g"].to_numpy(), fit_predict)
    assert len(seen) == 3 and all(len(test) == 1 and not (test & train) for train, test in seen)
    assert y.shape == preds.shape == scores.shape == (60,)


def test_groups_must_match_rows():
    df = pd.DataFrame({"diagnosis": ["healthy", "parkinson"]})
    with pytest.raises(ValueError):
        leave_one_group_out(df, [0], lambda a, b: (np.zeros(len(b), int), np.zeros(len(b))))


def _grouped(sign_by_group, n_per_group=200, seed=0):
    """One feature x; inside group g the label rises with x when sign is +1 and falls when it is -1."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, sign in enumerate(sign_by_group):
        y = rng.integers(0, 2, n_per_group)
        x = sign * (y - 0.5) * 2 + rng.normal(0, 0.5, n_per_group)
        rows.append(pd.DataFrame({"diagnosis": np.where(y == 1, "parkinson", "healthy"), "x": x, "g": g}))
    return pd.concat(rows, ignore_index=True)


def _auc_per_group(df):
    y, _, s = leave_one_group_out(df, df["g"].to_numpy(), logistic_fit_predict(["x"]))
    return [auc_roc(y[df["g"] == g], s[df["g"] == g]) for g in range(3)]


def test_a_signal_that_is_the_same_everywhere_survives_the_hold_out():
    assert min(_auc_per_group(_grouped([1, 1, 1]))) > 0.9


def test_a_signal_that_flips_between_groups_is_exposed_by_the_hold_out():
    aucs = _auc_per_group(_grouped([-1, 1, 1]))
    assert aucs[0] < 0.2        # trained where x rises with the label, tested where it falls


def _script():
    spec = importlib.util.spec_from_file_location("run_group_holdout", ROOT / "scripts" / "run_group_holdout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_inputs(tmp_path, n=90):
    rng = np.random.default_rng(1)
    rows = {"diagnosis": [], "drawing_type": [], "class_name": [], "name_number": []}
    for dtype in ("spiral", "wave"):
        for i in range(n):
            dx = "parkinson" if i % 2 else "healthy"
            rows["diagnosis"].append(dx)
            rows["drawing_type"].append(dtype)
            rows["class_name"].append(f"{dx}_{dtype}")
            rows["name_number"].append(i)
    meta = pd.DataFrame(rows)
    X = rng.integers(0, 14, size=(len(meta), 32, 32)).astype(np.uint8)
    np.savez(tmp_path / "cache.npz", X=X, diagnosis=meta["diagnosis"].to_numpy(str),
             drawing_type=meta["drawing_type"].to_numpy(str), class_name=meta["class_name"].to_numpy(str),
             name_number=meta["name_number"].to_numpy(int))
    feats = pd.DataFrame(rng.normal(size=(len(meta), len(ALL_FEATURES))), columns=ALL_FEATURES)
    pd.concat([meta, feats], axis=1).to_csv(tmp_path / "strokes.csv", index=False)
    return tmp_path / "cache.npz", tmp_path / "strokes.csv"


def test_script_writes_a_report_with_all_baselines_without_torch(tmp_path):
    mod = _script()
    cache, strokes = _fake_inputs(tmp_path)
    assert mod.main(cache, strokes, tmp_path / "out", by="speckle", n_boot=20, skip_cnn=True) == 0
    text = (tmp_path / "out" / "group_holdout_speckle.md").read_text(encoding="utf-8")
    assert "## spiral" in text and "## wave" in text
    for name in ("background only", "stroke geometry only", "stroke geometry + contrast + paper noise"):
        assert name in text
    assert "small CNN" not in text and "held out: low" in text and "mean of the 3" in text


def test_script_with_cnn_adds_a_cnn_row_and_its_reference(tmp_path):
    pytest.importorskip("torch")
    mod = _script()
    cache, strokes = _fake_inputs(tmp_path, n=60)
    assert mod.main(cache, strokes, tmp_path / "out", by="frame_mean", epochs=1, width=4, n_boot=20) == 0
    text = (tmp_path / "out" / "group_holdout_frame_mean.md").read_text(encoding="utf-8")
    assert "small CNN" in text and "blocked CV" in text


def test_script_runs_with_an_ablated_input_and_includes_the_one_nn_row(tmp_path):
    mod = _script()
    cache, strokes = _fake_inputs(tmp_path)
    assert mod.main(cache, strokes, tmp_path / "out", by="frame_mean", n_boot=20, skip_cnn=True,
                    ablate="binary_crop") == 0
    text = (tmp_path / "out" / "group_holdout_frame_mean_binary_crop.md").read_text(encoding="utf-8")
    assert "Input: binary_crop" in text and "thumbnail 1-NN" in text
    assert not (tmp_path / "out" / "group_holdout_frame_mean.md").exists()
