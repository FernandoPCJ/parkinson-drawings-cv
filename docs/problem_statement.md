# Problem statement

> Research/engineering project. **Not a medical device; no diagnostic claims.**
> Data source, license and attribution: `docs/DATA.md`.

## Research question

Can a small vision model separate hand-drawn **spirals** and **waves** of people with
Parkinson's disease from those of healthy people better than (a) a majority-class guess and
(b) a simple model on hand-crafted stroke features, under an evaluation that does not leak
information between training and test?

Secondary: which drawing type (spiral, wave, or both together) carries more signal?

## Data

- Dataset: *Parkinson YOLO Dataset* (Kaggle, `cornelioac/parkinson-yolo-dataset`).
  Only the folder `YOLODatasetFull` (original images) is used. The folder
  `YOLODatasetFull_Augmented` is **not** used (see audit below).
- Source, authors and **license: CC BY 4.0** as declared by the Kaggle publisher; the data are a compilation of four upstream sources (see `docs/DATA.md`), whose own licenses were not checked. Per-subject provenance of the images is
  undocumented: `dataset.yaml` points to a personal folder of the uploader and there are no
  subject codes. `data/raw/` is not versioned; the repo only holds download instructions.
- Labels: one whole-image box per file (100% of boxes cover >= 90% of the image), i.e. this is
  whole-image classification stored in YOLO format. Four classes: healthy/parkinson x
  spiral/wave. Object detection (YOLO) is out of scope.

## Data audit (reproducible with the scripts in `scripts/`)

| Finding | Number |
|---|---|
| Image files in the whole download | 15,518 |
| Distinct originals (after removing `_augN` copies) | 3,732 |
| Authors' split in `_Augmented`: val images with a copy or augmented copy in train | 57.6% (inside that folder) |
| Authors' split in `YOLODatasetFull`: val images with a twin in train | 263 of 747 (35.2%) |
| ...of which byte-identical files | spirals 100, waves 76; the rest are near-identical |
| Groups of identical drawings with **contradictory labels** (healthy and parkinson) | 61 groups, 126 files |
| Distinct drawings after deduplication | 3,282 |
| Usable drawings (conflicts dropped) | **3,221** |

Usable drawings per class: healthy spiral 843, healthy wave 588, parkinson spiral 996,
parkinson wave 794.

The authors' own split is not usable as a test set: after cleaning, its validation part has
443 images, 69.3% parkinson (train: 53.4%), with only 55 healthy waves and 81 healthy spirals.

## Label and task

- Label: `healthy` (0) vs `parkinson` (1). Spiral and wave models are trained separately,
  plus a pooled comparison.
- Task: binary image classification.

## Evaluation protocol

- **Primary:** 5-fold *blocked* stratified cross-validation over the 3,221 deduplicated
  drawings (`src/parkinson_cv/splits.py`). Inside each class, files are sorted by number and cut
  into contiguous blocks; repeated with rotated cut offsets. Rationale: file numbers are
  sequential, so neighbours may share a person or session.
- **Secondary (sanity check only):** the authors' validation part after cleaning (443 images),
  reported with its caveats.
- Metrics: **sensitivity** (recall of Parkinson) with **specificity**, AUC-ROC, AUC-PR,
  confusion matrix; every number with a 95% bootstrap CI and compared against the previous
  rung: majority class -> simple features + logistic regression/SVM -> small CNN -> transfer
  learning (ResNet18 / EfficientNet-B0). Accuracy alone is never a headline result.

## Leakage and limitations (to be repeated in the README)

1. **No subject codes.** A subject-level split cannot be proven. Blocked folds are a proxy.
   Results are likely optimistic and must not be described as generalising to new patients.
2. Deduplication is conservative (byte-identical, or thumbnail correlation >= 0.99 plus a
   pixel check). Rotated, cropped or otherwise transformed copies, and different drawings by
   the same person, are not detected.
3. The data are a compilation of four upstream sources with different acquisition conditions (one of them named as augmented data); the publisher declares CC BY 4.0, but upstream licenses were not checked, overlap between sources was not checked, and there is no per-subject provenance (see `docs/DATA.md`).
4. Small effective size after cleaning; confidence intervals will be wide, especially for
   healthy waves.
5. A suspiciously high score is a trigger to look for leakage first.
6. Possible shortcut learning (paper, scan or background artefacts): to be inspected with
   Grad-CAM on correct and wrong predictions. A metadata-only baseline (brightness, ink share,
   file size, alpha channel) reaches AUC 0.60 (spiral) / 0.59 (wave), so a weak side channel exists.
7. Images seem to be cropped to the drawing's bounding box and resized to 512x512 (ink spans
   ~98% of the frame in all classes): absolute size and aspect ratio are lost, so micrographia
   cannot be measured.
8. Background style (grey vignette frame vs plain speckled paper) is not neutral: the paper-grain groups
   differ strongly in class balance (parkinson rate 0.41/0.77/0.45 on spirals), which hints at different
   data sources. The CNN still separates the classes inside each group (AUC 0.79-0.95).

## Bars to beat (blocked CV, AUC-ROC)

| | spiral | wave |
|---|---|---|
| majority class | 0.50 | 0.50 |
| metadata only | 0.60 | 0.59 |
| stroke features + logistic regression | 0.74 | 0.75 |
| small CNN from scratch (5 repeats) | 0.87 | 0.80 |

The small CNN beats the stroke baseline; audits of that result (shuffled labels, coarse blocks,
near-twins, border crop, background groups, Grad-CAM) are summarised in the README. A transfer-learning
model is only interesting if it clearly beats the CNN row.

## Out of scope

Clinical claims, deployment in care settings, identifiable data, large models, paid services.

## Definition of done for Week 1

- [x] Dataset chosen, audited and cleaned (`scripts/`, `reports/`)
- [x] Evaluation protocol defined (`splits.py` + tests)
- [x] Preprocessing pipeline with tests
- [x] Source and license copied from the Kaggle page (`docs/DATA.md`)
- [ ] `01_exploracao` notebook adapted to the cleaned manifest and run end to end
- [ ] `pytest` green locally and in CI
