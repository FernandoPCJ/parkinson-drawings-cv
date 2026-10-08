"""Contact sheets of near-duplicate pairs, built from reports/near_duplicates_pairs.csv.

Usage:
    python -X utf8 scripts/show_pairs.py

Writes (left = val, right = train; caption strip shows correlation and both labels):
    reports/near_dup_conflicts.png  pairs whose DIAGNOSIS differs (healthy vs parkinson)
    reports/near_dup_same_label.png pairs with the same diagnosis, spread across the list
Also prints a breakdown of the conflicts by drawing type and label combination.
Needs Pillow and numpy.
"""
from __future__ import annotations

import csv
import hashlib
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TILE = 220
CAP = 22


def sheet(pairs: list[dict], out: Path) -> None:
    img = Image.new("RGB", (TILE * 2, (TILE + CAP) * len(pairs)), "white")
    dr = ImageDraw.Draw(img)
    for i, p in enumerate(pairs):
        y = i * (TILE + CAP)
        dr.text((4, y + 4), f"corr {float(p['correlation']):.4f} | val: {p['val_label']} | "
                            f"train: {p['train_label']}", fill="black")
        for j, key in enumerate(("val_path", "train_path")):
            with Image.open(p[key]) as im:
                t = im.convert("L")
                t.thumbnail((TILE, TILE))
                img.paste(t.convert("RGB"), (j * TILE, y + CAP))
    img.save(out)


def pair_stats(p: dict) -> dict:
    """Objective check: same bytes? same size? how different are the pixels at 256x256?"""
    a, b = Path(p["val_path"]), Path(p["train_path"])
    same_bytes = hashlib.sha1(a.read_bytes()).hexdigest() == hashlib.sha1(b.read_bytes()).hexdigest()
    with Image.open(a) as ia, Image.open(b) as ib:
        same_size = ia.size == ib.size
        ga = np.asarray(ia.convert("L").resize((256, 256), Image.BOX), dtype=np.float32)
        gb = np.asarray(ib.convert("L").resize((256, 256), Image.BOX), dtype=np.float32)
    d = np.abs(ga - gb)
    return {"same_bytes": same_bytes, "same_size": same_size,
            "mad": float(d.mean()), "diff_px": float((d > 40).mean())}


def summarize(pairs: list[dict], title: str) -> None:
    groups = defaultdict(list)
    for p in pairs:
        st = pair_stats(p)
        dtype = p["val_label"].split()[1]
        kind = "conflict" if p["val_label"].split()[0] != p["train_label"].split()[0] else "same-label"
        groups[(dtype, kind)].append(st)
    print(f"\n{title}")
    print("(mad = mean abs pixel difference 0-255 at 256x256; diff_px = share of pixels differing by >40)")
    for (dtype, kind), sts in sorted(groups.items()):
        print(f"  {dtype} / {kind}: n={len(sts)} | identical bytes={sum(s['same_bytes'] for s in sts)} "
              f"| same size={sum(s['same_size'] for s in sts)} "
              f"| median mad={statistics.median(s['mad'] for s in sts):.2f} "
              f"| median diff_px={100 * statistics.median(s['diff_px'] for s in sts):.2f}%")


def main() -> int:
    src = ROOT / "reports" / "near_duplicates_pairs.csv"
    if not src.exists():
        print(f"Run check_near_duplicates.py first ({src} missing)")
        return 1
    with src.open(encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    diag = lambda s: s.split()[0]
    conflicts = [p for p in pairs if diag(p["val_label"]) != diag(p["train_label"])]
    same = [p for p in pairs if diag(p["val_label"]) == diag(p["train_label"])]

    print(f"pairs >= 0.98: {len(pairs)} | conflicting diagnosis: {len(conflicts)} | same: {len(same)}")
    print("\nConflicts by label combination (val vs train):")
    for (a, b), n in Counter((p["val_label"], p["train_label"]) for p in conflicts).most_common():
        print(f"  {a}  vs  {b}: {n}")
    print("\nSame-label pairs by label:")
    for lab, n in Counter(p["val_label"] for p in same).most_common():
        print(f"  {lab}: {n}")

    summarize(pairs, "Objective pair check (all pairs with correlation >= 0.98):")

    random.seed(0)
    out_dir = ROOT / "reports"
    if conflicts:
        sheet(random.sample(conflicts, min(6, len(conflicts))), out_dir / "near_dup_conflicts.png")
    if same:
        step = max(1, len(same) // 6)
        sheet(same[::step][:6], out_dir / "near_dup_same_label.png")
    print("\nSaved reports/near_dup_conflicts.png and reports/near_dup_same_label.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
