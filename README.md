# parkinson-drawings-cv

Research/engineering project: classifying Parkinson's vs. healthy hand-drawn spirals and
waves with a baseline ladder, leakage-aware evaluation, Grad-CAM and ONNX export.

> **Not a medical device.** Educational/portfolio project. It makes no diagnostic claims.


> **Correction (2026-10-09): near-duplicate images leaked across the folds of the evaluation in this README, and the label may be readable from who or where a drawing came from.**
> An audit (`scripts/find_twins.py`, `reports/twins.md`) found that about 90% of the images have rotated or mirrored
> copies of the same drawing, roughly 8 per drawing: the 1,839 spirals are about 345-412 distinct drawings and the 1,382
> waves about 175-191. The blocked CV placed about 85% of those copy pairs in different folds. A baseline with no learning
> (copy the label of the most similar training image) reached AUC 0.94-0.96 in the background hold-out, which exposed it.
> Re-evaluated with copies kept together (`scripts/run_cluster_cv.py`; copies found on binarised and cropped drawings, so
> shifted and rescaled ones are linked too; intervals resample distinct drawings): stroke features 0.74 (spiral) / 0.71
> (wave); small CNN 0.84 / 0.80, close to its blocked-CV values (0.87 / 0.80); ImageNet-pretrained ResNet18 0.97 / 0.96.
> But the no-learning baseline, after cropping, reaches 0.88 / 0.93. A result that simple suggests the label can be read
> from the overall look of a drawing, which can reflect the disease but equally the person (the sources may contain
> several drawings per person) or the data source. The dataset has no subject or source IDs, so these cannot be
> separated, and none of the numbers should be read as accuracy on new patients.
## Status
Work in progress, MVP target: early November 2026. Done: data audit and cleaning, evaluation
protocol, baseline ladder up to a small CNN, a leakage/shortcut audit of the CNN, ONNX export with a
numerical-equivalence test and CPU latency, and MLflow tracking, plus a Gradio demo. A near-duplicate audit and a copy-grouped evaluation were added (see the correction above); transfer learning (ResNet18) was run and is reported with its caveats. Pending: exploration notebook on the cleaned data.

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
# near-duplicate audit and copy-grouped evaluation (supersedes the blocked-CV numbers)
python scripts/find_twins.py data/processed/cnn_cache_128.npz --on binary_crop
python scripts/run_cluster_cv.py data/processed/cnn_cache_128.npz --cluster-on binary_crop --skip-cnn
python scripts/run_cluster_cv.py data/processed/cnn_cache_128.npz --cluster-on binary_crop --model resnet18   # needs torchvision; downloads weights once
python scripts/run_group_holdout.py data/processed/cnn_cache_128.npz --by frame_mean   # earlier hold-out; it leaks copies
# final models, ONNX export and tracking (see "Export and tracking" below)
python scripts/train_final.py data/processed/cnn_cache_128.npz --mlflow
python scripts/predict_onnx.py <drawing.png> spiral
python scripts/log_reports_to_mlflow.py
pytest -q
```

## Evaluation protocol
Headline: copy-grouped, label-stratified 5-fold CV (`scripts/run_cluster_cv.py`), where copies of the same drawing never sit on both sides of a split; AUC-ROC of the pooled out-of-fold scores with 95% CIs that resample distinct drawings. Spiral and wave are modelled separately. The earlier blocked CV (`src/parkinson_cv/splits.py`) is kept as a superseded result below.

## Results so far

Headline evaluation: copy-grouped 5-fold CV (`scripts/run_cluster_cv.py`). Copies of the same drawing (rotated, mirrored,
shifted or rescaled) are found on binarised and cropped drawings and kept inside one fold; the 1,839 spirals are about
345 distinct drawings and the 1,382 waves about 191. AUC-ROC of the pooled out-of-fold scores, [95% CI resampling
distinct drawings, not images]. Spiral and wave are modelled separately. No subject IDs exist (see the correction at the top).

| Model | Spiral | Wave |
|---|---|---|
| Background numbers only (frame brightness, paper grain) | 0.519 [0.450-0.587] | 0.552 [0.458-0.635] |
| Logistic regression, stroke geometry | 0.742 [0.678-0.808] | 0.665 [0.582-0.739] |
| ...plus ink contrast and paper noise | 0.743 [0.678-0.807] | 0.710 [0.635-0.783] |
| 1-NN on thumbnails, no learning (normal image) | 0.819 [0.760-0.878] | 0.854 [0.802-0.898] |
| 1-NN on thumbnails, no learning (binarised and cropped) | 0.875 [0.822-0.920] | 0.926 [0.885-0.958] |
| Small CNN from scratch (128 px ink map, 20 epochs)* | 0.842 [0.781-0.891] | 0.796 [0.727-0.858] |
| ResNet18 pre-trained on ImageNet, fine-tuned (8 epochs) | 0.969 [0.943-0.990] | 0.961 [0.932-0.985] |

\* The small CNN was run with copies found on the uncropped binarised drawings (412 spiral / 175 wave clusters); the
other rows use the cropped detector (345 / 191). Fold assignments differ, so differences of about 0.03 are noise.

How to read it:
- The background numbers alone stay near chance, so the grey frame and paper grain are not what separates the classes here.
- Stroke features reach 0.74 / 0.71. This is the hand-built reference.
- **A baseline that learns nothing (copy the label of the most similar training drawing) reaches 0.88 / 0.93 once
  position and size are removed.** It beats the hand-built features. The overall look of a drawing therefore carries the
  label across distinct drawings. That can be the disease, but it can equally be the person (the sources may contain
  several drawings per person) or the data source, and the dataset has no IDs to tell them apart.
- The small CNN (0.84 / 0.80) is not clearly better than that baseline, and on waves it is worse.
- The pre-trained ResNet18 (0.97 / 0.96) is far above everything. Lowering the copy threshold to 0.70 gives
  0.938 / 0.935 and grouping copies after cropping gives 0.969 / 0.961, so near-copies do not explain it. It is not
  explained, and it is not reported as a performance estimate.

What this can and cannot show. It shows that images of distinct drawings can be sorted by class well above chance using
overall appearance, and that simple models and networks agree on the direction. It cannot show that a model works on a
new patient: without subject or source IDs, drawings by the same person or from the same source can sit on both sides of
every split, and nothing here separates that from a disease signal.

### Earlier results (blocked CV, superseded)

The first evaluation used blocked 5-fold CV (`src/parkinson_cv/splits.py`, 5 repeats). It splits file-number blocks, but
about 85% of the copy pairs ended up in different folds, so its numbers are optimistic. Balanced accuracy / AUC-ROC:

| Rung | Model | Spiral | Wave | Pooled |
|---|---|---|---|---|
| 1 | Majority class (floor) | 0.500 / 0.499 | 0.500 / 0.499 | 0.500 / 0.499 |
| 2a | Logistic regression on metadata only (paper brightness, ink share, file size, alpha channel) | 0.587 / 0.601 | 0.555 / 0.588 | 0.590 / 0.625 |
| 2b | Logistic regression on stroke geometry (ink, thickness, crossings, spread) | 0.670 / 0.738 | 0.638 / 0.697 | 0.621 / 0.658 |
| 2b+ | ...plus ink contrast and paper noise | 0.673 / 0.741 | 0.664 / 0.748 | 0.613 / 0.673 |
| 3 | Small CNN from scratch (128 px paper-relative ink map, 20 epochs, light augmentation) | 0.747 / 0.869 | 0.711 / 0.804 | not run |

Rung 2a uses nothing about the hand, yet it clearly beats the floor, so the data has a weak side channel. Contrast and
paper noise alone stay near chance. Separate models per drawing type beat one pooled model. The CNN row (AUC 0.869 and
0.804) turned out close to the copy-grouped values above (0.842 and 0.796), so the small CNN did not depend much on copies.

### Checks run on the blocked-CV result, and what they missed

The jump from 0.74 to 0.87 deserved suspicion, so it was tested before being trusted. Numbers are AUC-ROC (spiral / wave).
Several checks passed and the result still turned out to be affected by copies, because the one check aimed at copies used a
measure too coarse to see them.

| Check | Result | What it says |
|---|---|---|
| Labels shuffled | 0.458 / 0.507 | Pipeline does not leak the label; chance level. |
| Coarse blocks (2 folds, half the training data) | CNN 0.811 / 0.796 vs stroke model 0.733 / 0.750 | CNN still wins, but loses more than the stroke model. |
| Near-twins in other folds, 32 px thumbnails of the raw ink map | AUC by similarity tercile 0.865/0.834/0.899 (spiral), 0.878/0.817/0.806 (wave); 1-2% of images with cosine >= 0.95 | **Missed the copies.** The background differs between copies, which hides them in raw thumbnails. On binarised drawings about 90% of the images have a copy. |
| Outer 10% frame removed | 0.851 / 0.813 | Not dependent on the image border. |
| Background style (grey frame, paper grain) | CNN AUC inside every background tercile: 0.79-0.95 | Separates classes among images with similar backgrounds. |
| Grad-CAM (one held-out fold) | enrichment on the stroke 1.22 (spiral), 0.97 (wave) | Spiral: mild preference for the stroke. Wave: no clear preference. The 8x8 map is coarse. |
| Copies grouped (`run_cluster_cv.py`) | CNN 0.842 / 0.796 | The small CNN barely depended on copies. |
| Copy threshold lowered to 0.80 and 0.70 | 1-NN 0.746 / 0.819 and 0.740 / 0.809 (from 0.785 / 0.837 at 0.90); ResNet18 0.938 / 0.935 at 0.70 | No collapse; not explained by near-copies. |
| Copies found after cropping | 1-NN 0.875 / 0.926 (cropped input); ResNet18 0.969 / 0.961 | Shifted or rescaled copies do not explain it either. |

Two things the early checks did find. First, the background is not neutral: on waves the mean CNN score falls from 0.82 to
0.48 across grey-frame terciles although the true parkinson rate stays 0.52-0.66, which produces false positives on clean
white paper; this lowers rather than inflates the pooled AUC. Second, the paper grain groups differ strongly in class
balance (parkinson rate 0.41 / 0.77 / 0.45 on spirals, 0.53 / 0.71 / 0.48 on waves), which is compatible with different
data sources, so a metadata-style signal exists in the data.

## Export and tracking
`scripts/train_final.py` trains one final CNN per drawing type on all usable drawings, exports it to
ONNX (dynamic batch) and checks that ONNX Runtime reproduces the PyTorch logits (max absolute difference,
same decisions on 64 real drawings plus a blank page). It also times single-image CPU inference for both
runtimes; results go to `reports/export.md`. These final models have no held-out evaluation: their expected
quality is the cross-validated result above. `scripts/predict_onnx.py` scores one image with ONNX Runtime
only (no PyTorch needed). Measured on the author's CPU: ONNX file 239 KB per model; max absolute
logit difference to PyTorch 3.8e-06 with identical decisions on 65 test inputs; single-image latency
(median of 200 runs) 0.27 ms with ONNX Runtime vs 1.7-1.9 ms with PyTorch. The preprocessing (`src/parkinson_cv/inkmap.py`) is shared by training and inference.

Experiments are tracked with MLflow. `scripts/log_reports_to_mlflow.py` logs the numbers already written in
`reports/*.md` (metrics and CI bounds, one run per table row) without re-training; browse them with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` (if port 5000 is blocked on Windows, add `--port 5001`).

## Demo
`scripts/demo_gradio.py` is a small web app: upload a spiral or wave drawing, pick the type, and read the
model score. It runs the exported ONNX models on CPU (no PyTorch) with the same preprocessing as training
(`src/parkinson_cv/inference.py`). It needs `models/cnn_spiral.onnx` and `models/cnn_wave.onnx` from
`scripts/train_final.py`.

```bash
pip install gradio
python scripts/demo_gradio.py     # then open http://127.0.0.1:7860
```

Research prototype: not a medical device, no diagnosis. Use images that look like the dataset (cropped to the
stroke) and do not upload drawings of real people without permission. Each model only knows its own drawing type: a spiral scored with the wave model (or the reverse) gives a meaningless but often very confident score, so pick the type carefully.

## Data and attribution
The images are **not** included in this repository (`data/` is not versioned). They come from the Kaggle
*Parkinson Yolo Dataset* by CornelioAC (https://www.kaggle.com/datasets/cornelioac/parkinson-yolo-dataset),
declared by the publisher under CC BY 4.0. The publisher states that the data were compiled from other
sources (listed in `docs/DATA.md`). The code in this repository is under the MIT license (`LICENSE`).

## Held-out background groups (flawed experiment, kept for the record)

`scripts/run_group_holdout.py` cuts each drawing type into three groups by terciles of a background measure, holds one
group out and trains on the other two. The idea was a harder test: a new background style at test time. It was not harder:
copies of the same drawing sit in different groups in 10-16% of the twin pairs, and with about 8 copies per drawing almost
every held-out image had a copy in training. What exposed it was a baseline that learns nothing. AUC-ROC inside the
held-out groups, mean of the three, groups by `frame_mean`:

| Model | Spiral | Wave |
|---|---|---|
| Stroke features (best of 2) | 0.735 | 0.748 |
| Small CNN | 0.852 | 0.803 |
| ResNet18 pre-trained, normal input | 0.985 | 0.944 |
| ResNet18 pre-trained, binarised input | 0.903 | 0.861 |
| 1-NN on thumbnails, no learning (binarised input) | 0.938 | 0.964 |

A model that copies the label of its nearest training image beat the small CNN and the binarised ResNet. These numbers
were measured with copies on both sides of the split and are not performance estimates (see `reports/twins.md` and the
copy-grouped results above).

## Limitations & ethics
- No subject identifiers: a subject-level split cannot be proven; blocked folds are only a
  proxy, so results are probably optimistic and say nothing about unseen patients.
- Deduplication catches byte-identical and near-identical files only. About 90% of the remaining images still have rotated or mirrored copies of another image (roughly 8 per drawing, about 345 distinct spirals and 191 distinct waves). The headline evaluation keeps copies together, but copies below the detector's threshold may remain.
- The data are a compilation of four upstream sources with different acquisition conditions. The publisher declares CC BY 4.0, but the upstream licenses and the overlap between sources were not checked (see `docs/DATA.md`).
- Small effective sample size; wide confidence intervals, especially for healthy waves.
- The drawings appear to be cropped to their bounding box and resized to 512x512 by the dataset
  authors (the ink spans ~98% of the frame in every class), so absolute size and aspect ratio,
  e.g. micrographia, cannot be measured.
- Shortcut learning cannot be excluded: a nearest-neighbour baseline that learns nothing reaches 0.88 / 0.93 after cropping, so the label is readable from the overall look of a drawing, which may reflect the disease, the person or the data source; the dataset has no IDs to separate them. Grad-CAM on waves does not show a clear focus on the stroke.
  in class balance, and the checks use crude, three-level background measures. Grad-CAM on waves
  does not show a clear focus on the stroke.
- The CNN's operating threshold (0.5) was not tuned; balanced accuracy is for reference, AUC is
  the main metric. The CNN was only run per drawing type, not pooled.
- Intervals in the headline table resample distinct drawings; older tables in this README treat images as independent and are optimistic. Models were trained once per fold with a single seed.
- Not a diagnostic tool and not clinically validated.

![MLflow comparison of the spiral baseline ladder](docs/img/mlflow_compare.png)

Spiral drawings, blocked 5-fold CV, point estimates (AUC-ROC, balanced accuracy): majority class, metadata, stroke features and the small CNN. Confidence intervals are in the results table above.

