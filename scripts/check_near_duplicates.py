"""Near-duplicate check between train and val ORIGINALS (needs numpy + Pillow).

Usage:
    python -X utf8 scripts/check_near_duplicates.py [manifest.csv]

Method: each image -> grayscale, inverted (stroke = bright), area-averaged to 48x48,
z-scored. Similarity of two images = correlation of those vectors (1.0 = same drawing,
also robust to small brightness changes). Compared only within the same drawing type
(spiral vs spiral, wave vs wave).

Writes:
    reports/near_duplicates.md        summary
    reports/near_duplicates_pairs.csv pairs with correlation >= 0.98 (max 1000 rows)
    reports/near_dup_examples.png     top pairs side by side (val | train) to eyeball
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 48


def signature(path: str) -> np.ndarray:
    with Image.open(path) as im:
        g = im.convert("L")
        small = g.resize((SIZE, SIZE), Image.BOX)
    v = 255.0 - np.asarray(small, dtype=np.float32)
    v = v.flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def main(manifest: Path) -> int:
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} images; computing signatures (a minute or two)...")
    sigs = []
    for i, r in enumerate(rows, 1):
        sigs.append(signature(r["path"]))
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}")
    X = np.stack(sigs)

    lines = ["# Near-duplicate report (train vs val originals)", "",
             "Similarity = correlation of 48x48 inverted thumbnails (1.0 = identical).",
             "Calibration: centred spirals of DIFFERENT people already correlate around the",
             "'median best-match' value below, so only >= 0.99 is treated as a near-duplicate;",
             "0.95-0.99 needs a visual check in near_dup_examples.png.", ""]
    pairs = []
    for dtype in sorted({r["drawing_type"] for r in rows}):
        tr = [i for i, r in enumerate(rows) if r["drawing_type"] == dtype and r["split"] == "train"]
        va = [i for i, r in enumerate(rows) if r["drawing_type"] == dtype and r["split"] == "val"]
        S = X[va] @ X[tr].T
        best = S.max(axis=1)
        arg = S.argmax(axis=1)
        lines += [f"## {dtype}: {len(va)} val images vs {len(tr)} train images", ""]
        for thr in (0.995, 0.99, 0.98, 0.95):
            k = int((best >= thr).sum())
            lines.append(f"- val images with a train image at correlation >= {thr}: {k} "
                         f"({100 * k / len(va):.1f}%)")
        lines.append(f"- median best-match correlation: {np.median(best):.3f}")
        lines.append("")
        for j, vi in enumerate(va):
            if best[j] >= 0.98:
                ti = tr[arg[j]]
                pairs.append((float(best[j]), vi, ti))

        # duplicates inside train (same type)
        St = X[tr] @ X[tr].T
        np.fill_diagonal(St, 0)
        iu = np.triu_indices_from(St, k=1)
        dup_in_train = int((St[iu] >= 0.99).sum())
        lines += [f"- pairs inside train with correlation >= 0.99: {dup_in_train}", ""]

    # label disagreement among near-identical cross-split pairs
    disagree = sum(1 for c, vi, ti in pairs
                   if rows[vi]["diagnosis"] != rows[ti]["diagnosis"])
    lines += ["## Cross-split pairs >= 0.98", "",
              f"- total: {len(pairs)}",
              f"- with DIFFERENT diagnosis label (possible label noise): {disagree}", ""]

    pairs.sort(reverse=True)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "near_duplicates_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["correlation", "val_path", "train_path", "val_label", "train_label"])
        for c, vi, ti in pairs[:1000]:
            w.writerow([f"{c:.4f}", rows[vi]["path"], rows[ti]["path"],
                        rows[vi]["class_name"], rows[ti]["class_name"]])

    # contact sheet: top 6 pairs + 2 mid-range pairs
    pick = pairs[:6]
    if pick:
        tile = 200
        sheet = Image.new("L", (tile * 2, tile * len(pick)), 255)
        for r_i, (c, vi, ti) in enumerate(pick):
            for c_i, idx in enumerate((vi, ti)):
                with Image.open(rows[idx]["path"]) as im:
                    t = im.convert("L")
                    t.thumbnail((tile, tile))
                    sheet.paste(t, (c_i * tile, r_i * tile))
        sheet.save(out_dir / "near_dup_examples.png")
        lines += ["Top pairs saved to reports/near_dup_examples.png (left = val, right = train).", ""]

    text = "\n".join(lines)
    print(text)
    (out_dir / "near_duplicates.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    default = ROOT / "data" / "processed" / "yolo_full_manifest.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
