"""Image-level audit: what do the pictures look like, and can trivial metadata give away the label?

Usage:
    python -X utf8 scripts/audit_image_stats.py [clean_manifest.csv]

Reads data/processed/yolo_full_manifest_clean.csv (usable drawings only) and writes
    reports/image_stats.csv            one row per image (size, file size, brightness, ink share)
    reports/image_stats_summary.md     per-class summary + single-feature AUC
    reports/contact_sheet.png          5 random images of each of the 4 classes
Needs numpy, pandas, Pillow.

Why: if a number that has nothing to do with the hand (image width, file size, paper
brightness) separates healthy from parkinson, a model can "cheat" with it. A single-feature
AUC far from 0.5 is a red flag; it is computed here before any model is trained.
"""
from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parkinson_cv.metrics import auc_roc  # noqa: E402
from src.parkinson_cv.splits import load_clean  # noqa: E402

FEATURES = ["width", "height", "filesize", "mean_gray", "median_gray", "p5_gray", "ink_frac"]


def image_stats(path: str) -> dict:
    p = Path(path)
    with Image.open(p) as im:
        w, h = im.size
        mode = im.mode
        g = np.asarray(im.convert("L"), dtype=np.uint8)
    med = float(np.median(g))
    return {
        "width": w, "height": h, "mode": mode, "filesize": p.stat().st_size,
        "mean_gray": float(g.mean()), "median_gray": med,
        "p5_gray": float(np.percentile(g, 5)),
        "ink_frac": float((g < med - 40).mean()),  # share of clearly darker-than-paper pixels
    }


def contact_sheet(df, out: Path, per_class: int = 5, tile: int = 200, cap: int = 20) -> None:
    random.seed(0)
    classes = sorted(df["class_name"].unique())
    sheet = Image.new("RGB", (tile * per_class, (tile + cap) * len(classes)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, cname in enumerate(classes):
        paths = df.loc[df["class_name"] == cname, "path"].tolist()
        pick = random.sample(paths, min(per_class, len(paths)))
        y0 = r * (tile + cap)
        dr.text((4, y0 + 4), cname, fill="black")
        for c, p in enumerate(pick):
            with Image.open(p) as im:
                t = im.convert("L")
                t.thumbnail((tile, tile))
                sheet.paste(t.convert("RGB"), (c * tile, y0 + cap))
    sheet.save(out)


def main(manifest: Path) -> int:
    df = load_clean(manifest)
    print(f"{len(df)} images; reading each file (a minute or two)...")
    recs = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        recs.append(image_stats(row.path))
        if i % 500 == 0:
            print(f"  {i}/{len(df)}")
    for k in recs[0]:
        df[k] = [r[k] for r in recs]

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    keep = ["path", "class_name", "diagnosis", "drawing_type", "name_number"] + list(recs[0])
    df[keep].to_csv(out_dir / "image_stats.csv", index=False)

    L = ["# Image audit", "", f"Images: {len(df)}", "", "## Per class", ""]
    for cname, g in df.groupby("class_name"):
        sizes = Counter(zip(g["width"], g["height"]))
        (top_size, top_n), = sizes.most_common(1)
        L += [f"### {cname} (n={len(g)})", "",
              f"- distinct (width, height): {len(sizes)}; most common {top_size} "
              f"({100 * top_n / len(g):.0f}%)",
              f"- width min/median/max: {g['width'].min()} / {int(g['width'].median())} / {g['width'].max()}",
              f"- height min/median/max: {g['height'].min()} / {int(g['height'].median())} / {g['height'].max()}",
              f"- color modes: {dict(Counter(g['mode']))}",
              f"- file size median: {int(g['filesize'].median())} bytes",
              f"- paper brightness (median gray) median: {g['median_gray'].median():.1f}",
              f"- darkest 5% gray level, median: {g['p5_gray'].median():.1f}",
              f"- ink share (pixels >40 darker than paper), median: {100 * g['ink_frac'].median():.2f}%", ""]

    L += ["## Can metadata alone tell the classes apart?", "",
          "Single-feature AUC-ROC for parkinson vs healthy, folded so 0.5 = no information "
          "(max(AUC, 1-AUC)). Values near 0.5 are good news; values >= 0.65 are shortcut "
          "candidates.", "",
          "| drawing | " + " | ".join(FEATURES) + " |", "|" + "---|" * (len(FEATURES) + 1)]
    flags = []
    for dtype, g in df.groupby("drawing_type"):
        y = (g["diagnosis"] == "parkinson").to_numpy().astype(int)
        cells = []
        for f in FEATURES:
            a = auc_roc(y, g[f].to_numpy(dtype=float))
            a = max(a, 1 - a)
            cells.append(f"{a:.3f}")
            if a >= 0.65:
                flags.append(f"{dtype}/{f}={a:.3f}")
        L.append(f"| {dtype} | " + " | ".join(cells) + " |")
    L += ["", "Shortcut candidates (>= 0.65): " + (", ".join(flags) if flags else "none"), ""]

    text = "\n".join(L)
    print(text)
    (out_dir / "image_stats_summary.md").write_text(text, encoding="utf-8")
    contact_sheet(df, out_dir / "contact_sheet.png")
    print("Saved reports/image_stats.csv, image_stats_summary.md and contact_sheet.png")
    return 0


if __name__ == "__main__":
    default = ROOT / "data" / "processed" / "yolo_full_manifest_clean.csv"
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
