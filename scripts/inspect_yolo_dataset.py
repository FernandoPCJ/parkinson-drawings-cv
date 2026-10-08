"""Inspect the Kaggle 'parkinson-yolo-dataset' (v2: understands '_augN' names).

Usage:
    python -X utf8 scripts/inspect_yolo_dataset.py <dataset_root>
    python -X utf8 scripts/inspect_yolo_dataset.py            # downloads via kagglehub

Pure standard library. Saves reports/yolo_dataset_report.md.

What it reports, per top-level folder and overall:
  * images per split, split into ORIGINALS and AUGMENTED copies
    (augmented = name ends in one or more '_aug<number>', or Roboflow '.rf.<hash>')
  * how many validation/test images have a sibling (same original, or an augmented
    copy of it) in the training split  -> direct leakage
  * class balance (from label files + names in dataset.yaml)
  * whether boxes cover the whole image (=> 'detection' format on a classification task)
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "valid": "val", "val": "val",
                 "validation": "val", "test": "test", "testing": "test"}
AUG_RE = re.compile(r"^(?P<base>.+?)(?:_aug\d+)+$")
RF_RE = re.compile(r"^(?P<base>.*?)(?:_(?:jpg|jpeg|png|bmp))?\.rf\.[0-9a-fA-F]{8,}$")


def parse_stem(stem: str) -> tuple[str, bool]:
    """(original name, is_augmented_copy)."""
    for rx in (AUG_RE, RF_RE):
        m = rx.match(stem)
        if m:
            return m.group("base"), True
    return stem, False


def split_of(rel_parts: tuple[str, ...]) -> str:
    for part in rel_parts[:-1]:
        if part.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.lower()]
    return "?"


def read_names(yaml_path: Path) -> list[str]:
    text = yaml_path.read_text(errors="replace")
    m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if not m:
        return []
    return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def contamination(records: list[dict]) -> list[str]:
    """Lines describing how much of val/test shares an original with train."""
    train_bases = {r["base"] for r in records if r["split"] == "train"}
    lines = []
    for sp in ("val", "test"):
        sub = [r for r in records if r["split"] == sp]
        if not sub:
            continue
        hit = [r for r in sub if r["base"] in train_bases]
        pct = 100 * len(hit) / len(sub)
        lines.append(f"- {sp}: {len(hit)} of {len(sub)} images ({pct:.1f}%) have the same "
                     f"original (or an augmented copy of it) in train")
    return lines


def main(root: Path) -> int:
    root = root.resolve()
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not imgs:
        print(f"No images under {root}")
        return 1

    records = []
    for p in imgs:
        rel = p.relative_to(root).parts
        base, aug = parse_stem(p.stem)
        records.append({"path": p, "top": rel[0] if len(rel) > 1 else ".",
                        "split": split_of(rel), "base": base, "aug": aug})

    out = ["# YOLO dataset report (v2)", "", f"Root: `{root}`", ""]

    # ---- counts per top folder / split, originals vs augmented
    out += ["## Images per top-level folder and split", ""]
    cnt = Counter((r["top"], r["split"], "augmented" if r["aug"] else "original") for r in records)
    for key in sorted(cnt):
        out.append(f"- {key[0]} / {key[1]} / {key[2]}: {cnt[key]}")
    out.append("")
    n_orig = len({r["base"] for r in records})
    out += [f"Distinct original images (after removing _augN suffixes): **{n_orig}**",
            f"Total image files: {len(records)}", ""]

    # ---- leakage
    out += ["## Leakage: validation/test images that have a sibling in train", ""]
    out += ["Overall (all folders together):", ""] + contamination(records) + [""]
    for top in sorted({r["top"] for r in records}):
        sub = [r for r in records if r["top"] == top]
        out += [f"Inside `{top}` only:", ""] + (contamination(sub) or ["- no val/test split"]) + [""]

    # same original present as ORIGINAL in both splits (pure duplicates across the split)
    by_base: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if not r["aug"]:
            by_base[r["base"]].add(r["split"])
    dup_orig = [b for b, s in by_base.items() if len(s) > 1]
    out += [f"Originals that appear (unaugmented) in more than one split: {len(dup_orig)}", ""]

    # ---- labels
    yaml_names: list[str] = []
    for y in root.rglob("*.y*ml"):
        yaml_names = read_names(y) or yaml_names
        if yaml_names:
            break
    cls_count: Counter = Counter()
    wh = []
    missing = 0
    for r in records:
        lf = label_for(r["path"])
        if not lf.exists():
            missing += 1
            continue
        for line in lf.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                cls_count[yaml_names[cid] if cid < len(yaml_names) else str(cid)] += 1
                wh.append((float(parts[3]), float(parts[4])))
    out += ["## Labels", "", f"- images without a label file: {missing}"]
    out += [f"- class {k}: {v}" for k, v in sorted(cls_count.items())]
    if wh:
        full = sum(1 for w, h in wh if w * h >= 0.9)
        out += [f"- boxes covering >= 90% of the image: {full} of {len(wh)} "
                f"({100 * full / len(wh):.1f}%)  "
                "-> if this is ~100%, the 'detection' format is really whole-image classification",
                f"- mean box width/height (normalised): {sum(w for w, _ in wh) / len(wh):.2f} / "
                f"{sum(h for _, h in wh) / len(wh):.2f}"]
    out.append("")

    out += ["## Sample names", ""]
    for top in sorted({r["top"] for r in records}):
        for sp in ("train", "val", "test"):
            names = sorted(r["path"].name for r in records if r["top"] == top and r["split"] == sp)[:5]
            if names:
                out.append(f"- {top} / {sp}: " + ", ".join(names))
    out.append("")

    text = "\n".join(out)
    print(text)
    rep = Path(__file__).resolve().parents[1] / "reports" / "yolo_dataset_report.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(text, encoding="utf-8")
    print(f"\nSaved to {rep}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        import kagglehub

        folder = Path(kagglehub.dataset_download("cornelioac/parkinson-yolo-dataset"))
        print("Path to dataset files:", folder)
    sys.exit(main(folder))
