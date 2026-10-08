# parkinson-drawings-cv

Research/engineering project: classifying Parkinson's vs. healthy hand-drawn spirals and
waves with a baseline ladder, leakage-aware evaluation, Grad-CAM and ONNX export.

> **Not a medical device.** Educational/portfolio project. It makes no diagnostic claims.

## Status
Work in progress, MVP target: early November 2026. Done: data audit and cleaning, evaluation
protocol, baseline ladder up to a small CNN, and a leakage/shortcut audit of the CNN. Pending:
transfer learning, ONNX export with a numerical-equivalence test and CPU latency, MLflow tracking,
optional demo, source/license of the data, exploration notebook on the cleaned data.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data audit (Week 1)
The Kaggle *Parkinson YOLO Dataset* ships a train/val split that leaks: 35% of the validation
originals have an identical or near-identical twin in train, and 61 groups of identical
drawings carry both labels. After deduplication and dropping contradictory labels, **3,221
distinct drawings** remain (843 healthy spirals, 588 healthy waves, 996 parkinson spirals,
794 parkinson waves). Details and numbers: `docs/problem_statement.md`.

```bash
python scripts/inspect_yolo_dataset.py <dataset_root>   # layout, _augN leakage
python scripts/build_manifest.py <dataset_root>         # manifest of originals, number ranges
python scripts/check_near_duplicates.py                 # train vs val near-duplicates
python scripts/show_pairs.py                            # objective pair check + contact sheets
python scripts/build_clean_manifest.py                  # groups, conflicts, clean split
python scripts/audit_image_stats.py                     # image sizes, brightness, side-channel check
python scripts/run_baseline_majority.py                 # rung 1: majority-class floor
python scripts/run_baseline_metadata.py                 # rung 2a: metadata-only shortcut check
python scripts/extract_stroke_features.py               # stroke features per drawing
python scripts/run_baseline_stroke.py                   # rung 2b: stroke-feature logistic regression
python scripts/stroke_feature_importance.py             # which features separate the classes
python scripts/build_cnn_cache.py                        # 128px ink maps -> data/processed/cnn_cache_128.npz
python scripts/run_cnn.py data/processed/cnn_cache_128.npz   # rung 3: small CNN (~1.5 h on CPU for 5 repeats)
# audits of the CNN result (run_cnn.py first; each writes reports/*.md)
python scripts/run_cnn.py data/processed/cnn_cache_128.npz --repeats 1 --shuffle-labels --tag _shuffle
python scripts/run_cnn.py data/processed/cnn_cache_128.npz --repeats 1 --k 2 --tag _k2
python scripts/run_cnn.py data/processed/cnn_cache_128.npz --repeats 1 --crop 13 --tag _crop13
python scripts/run_baseline_stroke.py --k 2 --tag _k2   # same coarse blocks for the baseline
python scripts/check_cnn_neighbors.py data/processed/cnn_cache_128.npz
python scripts/check_cnn_background.py data/processed/cnn_cache_128.npz
python scripts/gradcam_cnn.py data/processed/cnn_cache_128.npz spiral   # also: wave
pytest -q
```

## Evaluation protocol
Blocked stratified 5-fold CV over the deduplicated drawings (`src/parkinson_cv/splits.py`),
spiral and wave modelled separately, sensitivity/specificity/AUC with bootstrap CIs.

## Results so far
Blocked 5-fold CV (5 repeats) over the deduplicated drawings. Balanced accuracy / AUC-ROC;
95% bootstrap CIs are in `reports/*.md` (images are treated as independent, so the CIs are
optimistic).

| Rung | Model | Spiral | Wave | Pooled |
|---|---|---|---|---|
| 1 | Majority class (floor) | 0.500 / 0.499 | 0.500 / 0.499 | 0.500 / 0.499 |
| 2a | Logistic regression on metadata only (paper brightness, ink share, file size, alpha channel) | 0.587 / 0.601 | 0.555 / 0.588 | 0.590 / 0.625 |
| 2b | Logistic regression on stroke geometry (ink, thickness, crossings, spread) | 0.670 / 0.738 | 0.638 / 0.697 | 0.621 / 0.658 |
| 2b+ | ...plus ink contrast and paper noise | 0.673 / 0.741 | 0.664 / 0.748 | 0.613 / 0.673 |
| 3 | Small CNN from scratch (128 px paper-relative ink map, 20 epochs, light augmentation) | 0.747 / **0.869** | 0.711 / **0.804** | not run |

CNN, 95% CI of AUC-ROC: spiral 0.869 [0.851-0.883], wave 0.804 [0.783-0.827]; sensitivity/specificity
0.87/0.62 (spiral) and 0.84/0.58 (wave) at a fixed 0.5 threshold that was not tuned.
Reading: rung 2a uses nothing about the hand, yet it clearly beats the floor, so the data has a
weak side channel. Stroke features (2b) beat 2a by a wide margin (spiral AUC 0.74 vs 0.60), and
contrast + paper noise alone stay near chance, so the signal comes from the drawing. The
bar for image models is therefore about AUC 0.74 (spiral) and 0.75 (wave). Separate models per
drawing type beat one pooled model. The small CNN beats the best stroke-feature model by about
0.13 AUC on spirals and 0.06 on waves, with non-overlapping CIs; the five CV repeats agree with a
single repeat (spiral 0.867, wave 0.809), so the result does not depend on where the blocks are cut.

### Checks on the CNN result (is 0.87 real?)
A jump from 0.74 to 0.87 deserved suspicion, so it was tested before being trusted. Numbers are AUC-ROC
(spiral / wave), CNN trained with 1 repeat unless stated.

| Check | Result | What it says |
|---|---|---|
| Labels shuffled | 0.458 / 0.507 | Pipeline does not leak the label; chance level. |
| Coarse blocks (2 folds, half the training data) | CNN 0.811 / 0.796 vs stroke model 0.733 / 0.750 | CNN still wins, but loses more than the stroke model (0.056 on spirals), so part of the drop may be proximity in file numbering; not separable from the smaller training set. |
| Near-twins in other folds | AUC by similarity tercile low/mid/high: 0.865/0.834/0.899 (spiral), 0.878/0.817/0.806 (wave); only 1-2% of images have a neighbour with cosine similarity >= 0.95 | Gain does not come from near-duplicates (coarse 32 px measure; rotated or shifted copies would not be caught). |
| Outer 10% frame removed | 0.851 / 0.813 | Not dependent on the image border. |
| Background style (grey frame, paper grain) | CNN AUC inside every background tercile: 0.79-0.95 | Separates classes among images with similar backgrounds, so it is not only reading the background. |
| Grad-CAM (one held-out fold) | heat on stroke vs image: enrichment 1.22 (spiral), 0.97 (wave); 27% of the heat in the outer frame vs 36% if uniform | Spiral: mild preference for the stroke. Wave: no clear preference. The 8x8 map is coarse, so this is only an indication. |

Two things the checks did find. First, the background is not neutral: on waves the mean CNN
score falls from 0.82 to 0.48 across grey-frame terciles although the true parkinson rate stays
0.52-0.66, which produces false positives on clean white paper; this lowers rather than inflates the
pooled AUC (within-group AUC is higher than pooled). Second, the paper grain groups differ strongly in
class balance (parkinson rate 0.41 / 0.77 / 0.45 on spirals, 0.53 / 0.71 / 0.48 on waves), which is
compatible with different data sources, so a metadata-style signal exists in the data.

## Limitations & ethics
- No subject identifiers: a subject-level split cannot be proven; blocked folds are only a
  proxy, so results are probably optimistic and say nothing about unseen patients.
- Deduplication catches byte-identical and near-identical files, not rotated/cropped copies.
- Data provenance and license are unclear (see `docs/problem_statement.md`).
- Small effective sample size; wide confidence intervals, especially for healthy waves.
- The drawings appear to be cropped to their bounding box and resized to 512x512 by the dataset
  authors (the ink spans ~98% of the frame in every class), so absolute size and aspect ratio,
  e.g. micrographia, cannot be measured.
- Shortcut learning was checked (table above) but cannot be excluded: the background groups differ
  in class balance, and the checks use crude, three-level background measures. Grad-CAM on waves
  does not show a clear focus on the stroke.
- The CNN's operating threshold (0.5) was not tuned; balanced accuracy is for reference, AUC is
  the main metric. The CNN was only run per drawing type, not pooled.
- Confidence intervals treat images as independent and the models were trained once per fold.
- Not a diagnostic tool and not clinically validated.
