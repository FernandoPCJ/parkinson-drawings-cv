"""Build a manifest of the ORIGINAL images of the Kaggle parkinson-yolo-dataset.

Only the 'YOLODatasetFull' folder is used (the 'YOLODatasetFull_Augmented' folder mixes
augmented copies of validation images into train and must not be used for evaluation).

Usage:
    python -X utf8 scripts/build_manifest.py "<dataset_root>"
Writes:
    data/processed/yolo_full_manifest.csv   (no image is copied)
    reports/manifest_summary.md
Pure standard library.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = {"train": "train", "valid": "val", "val": "val", "validation": "val",
          "test": "test"}
AUG_RE = re.compile(r"_aug\d+$")
NUM_RE = re.compile(r"^(?P<prefix>.*?)_?(?P<num>\d+)$")


def read_names(root: Path) -> list[str]:
    for y in root.rglob("*.y*ml"):
        m = re.search(r"names:\s*\[(.*?)\]", y.read_text(errors="replace"), re.S)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    return []


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def main(dataset_root: Path, folder: str = "YOLODatasetFull") -> int:
    base = dataset_root / folder
    if not base.exists():
        print(f"Folder not found: {base}")
        return 1
    names = read_names(base)
    rows, problems = [], Counter()
    for p in sorted(base.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        rel = p.relative_to(base).parts
        split = next((SPLITS[x.lower()] for x in rel[:-1] if x.lower() in SPLITS), "?")
        if AUG_RE.search(p.stem):
            problems["augmented_name_inside_Full"] += 1
            continue
        lf = label_path(p)
        if not lf.exists():
            problems["missing_label"] += 1
            continue
        first = lf.read_text(errors="replace").split()
        cid = int(float(first[0]))
        cname = names[cid] if cid < len(names) else str(cid)
        diagnosis, _, dtype = cname.partition(" ")
        m = NUM_RE.match(p.stem)
        prefix, num = (m.group("prefix"), int(m.group("num"))) if m else (p.stem, -1)
        if prefix and prefix != diagnosis:
            problems["name_prefix_disagrees_with_label"] += 1
        rows.append({"path": str(p), "split": split, "class_id": cid, "class_name": cname,
                     "diagnosis": diagnosis, "drawing_type": dtype,
                     "name_prefix": prefix, "name_number": num})

    out_csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "yolo_full_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Manifest summary (YOLODatasetFull, originals only)", "",
             f"Images: {len(rows)}", ""]
    lines += ["## Per split / class", ""]
    c = Counter((r["split"], r["class_name"]) for r in rows)
    lines += [f"- {s} / {n}: {k}" for (s, n), k in sorted(c.items())] + [""]

    lines += ["## Number range in file names, per split / class",
              "(density = count / (max-min+1); ~1.0 means a contiguous block of numbers)", ""]
    g = defaultdict(list)
    for r in rows:
        g[(r["split"], r["class_name"])].append(r["name_number"])
    for key, nums in sorted(g.items()):
        lo, hi = min(nums), max(nums)
        lines.append(f"- {key[0]} / {key[1]}: min {lo}, max {hi}, count {len(nums)}, "
                     f"density {len(nums) / (hi - lo + 1):.2f}")
    lines.append("")

    # do number ranges of train and val overlap within the same class?
    lines += ["## Do train and val number ranges interleave? (same class)", ""]
    for cls in sorted({r["class_name"] for r in rows}):
        tr = sorted(n for (s, k), v in g.items() if s == "train" and k == cls for n in v)
        va = sorted(n for (s, k), v in g.items() if s == "val" and k == cls for n in v)
        if tr and va:
            inter = sum(1 for n in va if tr[0] <= n <= tr[-1])
            lines.append(f"- {cls}: {inter} of {len(va)} val numbers fall inside the train range "
                         f"[{tr[0]}, {tr[-1]}]")
    lines += ["", "## Data-quality flags", ""]
    lines += [f"- {k}: {v}" for k, v in problems.items()] or ["- none"]
    lines.append("")

    text = "\n".join(lines)
    print(text)
    rep = Path(__file__).resolve().parents[1] / "reports" / "manifest_summary.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(text, encoding="utf-8")
    print(f"Saved {out_csv} and {rep}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
