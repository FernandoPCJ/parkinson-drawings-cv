"""Group identical / near-identical drawings and build a cleaned manifest.

Usage:
    python -X utf8 scripts/build_clean_manifest.py [manifest.csv]

Input : data/processed/yolo_full_manifest.csv   (from build_manifest.py)
Output: data/processed/yolo_full_manifest_clean.csv
        reports/clean_summary.md
Needs numpy + Pillow.

Two images are linked (same group) when, within the same drawing type, either
  * their files are byte-identical (SHA1), or
  * thumbnail correlation >= 0.99 AND, at 256x256, mean abs difference <= 3
    and <= 1% of pixels differ by more than 40 gray levels.
Groups are the connected components of those links.

New columns:
  group_id, group_size
  group_conflict  True if the group contains BOTH diagnoses (same drawing, opposite labels)
  role            rep = the one image kept per group, dup = extra copy (not used)
  split_clean     train | val | drop | dup
                  - conflict groups -> drop (label cannot be trusted)
                  - group with any member in the authors' train -> train
                  - otherwise the authors' split (val)
                  so no drawing can sit on both sides.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 48
CORR_MIN = 0.99
MAD_MAX = 3.0
DIFFPX_MAX = 0.01


def thumb_signature(path: str) -> np.ndarray:
    with Image.open(path) as im:
        small = im.convert("L").resize((SIZE, SIZE), Image.BOX)
    v = (255.0 - np.asarray(small, dtype=np.float32)).flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def gray256(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((256, 256), Image.BOX), dtype=np.uint8)


def sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main(manifest: Path) -> int:
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    print(f"{n} images: hashing and computing thumbnails (a few minutes)...")

    shas, sigs = [], []
    for i, r in enumerate(rows, 1):
        shas.append(sha1(r["path"]))
        sigs.append(thumb_signature(r["path"]))
        if i % 500 == 0:
            print(f"  {i}/{n}")
    X = np.stack(sigs)

    uf = UnionFind(n)
    by_sha = defaultdict(list)
    for i, s in enumerate(shas):
        by_sha[(rows[i]["drawing_type"], s)].append(i)
    exact_links = 0
    for idxs in by_sha.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
            exact_links += 1

    cache: dict[int, np.ndarray] = {}

    def g(i: int) -> np.ndarray:
        if i not in cache:
            cache[i] = gray256(rows[i]["path"])
        return cache[i]

    near_links = 0
    for dtype in sorted({r["drawing_type"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["drawing_type"] == dtype]
        S = X[idx] @ X[idx].T
        a, b = np.where(np.triu(S >= CORR_MIN, k=1))
        for ia, ib in zip(a, b):
            i, j = idx[ia], idx[ib]
            if uf.find(i) == uf.find(j):
                continue
            d = np.abs(g(i).astype(np.int16) - g(j).astype(np.int16))
            if d.mean() <= MAD_MAX and (d > 40).mean() <= DIFFPX_MAX:
                uf.union(i, j)
                near_links += 1

    members = defaultdict(list)
    for i in range(n):
        members[uf.find(i)].append(i)

    out_rows = [dict(r) for r in rows]
    stats = Counter()
    for gid, idxs in members.items():
        diags = {rows[i]["diagnosis"] for i in idxs}
        splits = {rows[i]["split"] for i in idxs}
        conflict = len(diags) > 1
        rep = min(idxs, key=lambda i: int(rows[i]["name_number"]))
        if conflict:
            sc = "drop"
        elif "train" in splits:
            sc = "train"
        else:
            sc = next(iter(splits))
        for i in idxs:
            o = out_rows[i]
            o["group_id"] = gid
            o["group_size"] = len(idxs)
            o["group_conflict"] = conflict
            o["role"] = "rep" if i == rep else "dup"
            o["split_clean"] = "dup" if (i != rep and not conflict) else sc
        stats["groups"] += 1
        if len(idxs) > 1:
            stats["groups_multi"] += 1
            stats["images_in_multi"] += len(idxs)
        if len(splits) > 1:
            stats["groups_cross_split"] += 1
        if conflict:
            stats["conflict_groups"] += 1
            stats["conflict_images"] += len(idxs)

    val_total = sum(1 for r in rows if r["split"] == "val")
    val_with_twin = sum(
        1 for i, r in enumerate(rows)
        if r["split"] == "val" and any(rows[j]["split"] == "train" for j in members[uf.find(i)])
    )

    out_csv = ROOT / "data" / "processed" / "yolo_full_manifest_clean.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    L = ["# Clean manifest summary", "",
         f"Images in manifest: {n}",
         f"Links found: {exact_links} byte-identical, {near_links} near-identical (pixel-verified)",
         f"Distinct drawings (groups): **{stats['groups']}**",
         f"Groups with more than one file: {stats['groups_multi']} "
         f"({stats['images_in_multi']} files)",
         f"Groups that span authors' train AND val: {stats['groups_cross_split']}",
         f"Authors' val images with a twin in train: {val_with_twin} of {val_total} "
         f"({100 * val_with_twin / val_total:.1f}%)",
         f"Groups with contradictory diagnosis (dropped): {stats['conflict_groups']} "
         f"({stats['conflict_images']} files)", ""]
    L += ["## After cleaning (one image per drawing, conflicts dropped)", ""]
    c = Counter((o["split_clean"], o["class_name"]) for o in out_rows
                if o["split_clean"] in ("train", "val"))
    for (s, cn), k in sorted(c.items()):
        L.append(f"- {s} / {cn}: {k}")
    tot = Counter(o["split_clean"] for o in out_rows)
    L += ["", f"Totals by split_clean: {dict(tot)}", ""]
    for s in ("train", "val"):
        sub = [o for o in out_rows if o["split_clean"] == s]
        if sub:
            pk = sum(1 for o in sub if o["diagnosis"] == "parkinson")
            L.append(f"- {s}: {len(sub)} images, {100 * pk / len(sub):.1f}% parkinson")
    L.append("")
    text = "\n".join(L)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "clean_summary.md").write_text(text, encoding="utf-8")
    print(f"Saved {out_csv}")
    return 0


if __name__ == "__main__":
    default = ROOT / "data" / "processed" / "yolo_full_manifest.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
