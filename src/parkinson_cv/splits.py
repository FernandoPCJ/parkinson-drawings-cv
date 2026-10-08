"""Leakage-aware splits for the cleaned manifest.

Why "blocked" folds?
  The dataset has no subject codes, so a subject-level split cannot be built or proven.
  File numbers inside each class are sequential and the authors cut train/val as contiguous
  blocks, so neighbouring numbers may come from the same person or session. Cutting each
  class into contiguous blocks keeps neighbours together, which is the closest proxy for a
  grouped split we have. It is a proxy, not a proof (see docs/problem_statement.md).

Exact and near-identical copies are removed beforehand by scripts/build_clean_manifest.py,
so the same drawing can never sit in two folds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STRATA = ("diagnosis", "drawing_type")


def load_clean(path: str | Path) -> pd.DataFrame:
    """Rows usable for modelling: one image per drawing, contradictory labels dropped."""
    df = pd.read_csv(path)
    return df[df["split_clean"].isin(["train", "val"])].reset_index(drop=True)


def blocked_stratified_folds(
    df: pd.DataFrame,
    k: int = 5,
    strata: tuple[str, ...] = DEFAULT_STRATA,
    order_col: str = "name_number",
    offset: float = 0.0,
) -> pd.Series:
    """Assign each row to one of k folds.

    Inside every stratum (e.g. parkinson+spiral) rows are sorted by `order_col` and cut into
    k contiguous blocks of near-equal size (sizes differ by at most 1), so every fold keeps
    the class mix and no block is interleaved with another.

    `offset` (0 <= offset < k) rotates where the cuts fall, to repeat the CV with different
    boundaries: offset=0 gives clean blocks; other offsets wrap one block around the ends.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if not 0 <= offset < k:
        raise ValueError("offset must be in [0, k)")
    fold = pd.Series(-1, index=df.index, dtype=int)
    for _, g in df.groupby(list(strata)):
        g = g.sort_values(order_col, kind="stable")
        n = len(g)
        pos = (np.arange(n) * k / n + offset) % k
        fold.loc[g.index] = np.floor(pos).astype(int)
    return fold
