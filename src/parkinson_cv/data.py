"""Dataset indexing and leakage checks.

Expected layout (Parkinson's Drawings, Kaggle):
    <root>/<category>/<split>/<class>/<file>
    category in {spiral, wave}, split in {training, testing}
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(root: str | Path) -> pd.DataFrame:
    """Index every image under `root` into a DataFrame.

    Columns: category, split, label, filename, path.
    """
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        rel = path.relative_to(root).parts
        if len(rel) != 4:  # category / split / class / file
            continue
        category, split, label, filename = rel
        rows.append(
            {
                "category": category,
                "split": split,
                "label": label,
                "filename": filename,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows, columns=["category", "split", "label", "filename", "path"])


def file_sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def average_hash(path: str | Path, hash_size: int = 8) -> int:
    """Tiny perceptual hash (aHash). Equal/near-equal hashes flag near-duplicates."""
    with Image.open(path) as im:
        small = np.asarray(
            im.convert("L").resize((hash_size, hash_size), Image.LANCZOS), dtype=np.float32
        )
    bits = (small > small.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_cross_split_duplicates(df: pd.DataFrame, max_hamming: int = 0) -> pd.DataFrame:
    """Pairs (train, test) of the same category that look identical.

    max_hamming=0 -> identical aHash (very strict near-duplicate check);
    exact byte-identical files are always reported with kind='exact'.
    """
    out = []
    for category, sub in df.groupby("category"):
        train = sub[sub["split"] == "training"]
        test = sub[sub["split"] == "testing"]
        if train.empty or test.empty:
            continue
        train_sha = {r.path: file_sha1(r.path) for r in train.itertuples()}
        test_sha = {r.path: file_sha1(r.path) for r in test.itertuples()}
        train_ah = {r.path: average_hash(r.path) for r in train.itertuples()}
        test_ah = {r.path: average_hash(r.path) for r in test.itertuples()}
        for tp, ts in train_sha.items():
            for qp, qs in test_sha.items():
                if ts == qs:
                    out.append({"category": category, "train": tp, "test": qp, "kind": "exact"})
                elif hamming(train_ah[tp], test_ah[qp]) <= max_hamming:
                    out.append({"category": category, "train": tp, "test": qp, "kind": "near"})
    return pd.DataFrame(out, columns=["category", "train", "test", "kind"])


_SUBJECT_PATTERNS = [
    re.compile(r"(?:^|[_\-\s])(?:p|pt|pac|paciente|subj|subject|sujeito)[_\-]?(\d+)", re.I),
    re.compile(r"^(\d+)[_\-]"),
]


def guess_subject_ids(filenames: pd.Series) -> pd.Series:
    """Try to pull a subject code out of file names. NaN when nothing matches."""

    def _one(name: str):
        stem = Path(name).stem
        for pat in _SUBJECT_PATTERNS:
            m = pat.search(stem)
            if m:
                return m.group(1)
        return np.nan

    return filenames.map(_one)
