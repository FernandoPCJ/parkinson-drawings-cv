# Cluster-grouped cross-validation (copies never split across folds)

Copies: cosine >= 0.90 on 32x32 thumbnails of the binarised drawing, best of 8 rotations / mirrors, joined into clusters. 5-fold, whole clusters per fold, stratified by label, seed 0. AUC-ROC of the pooled out-of-fold scores, [95% CI resampling CLUSTERS], 500 resamples. Input: binary_crop. Network skipped.

## spiral

1839 images = 412 distinct drawings (clusters); 214 have no copy; largest cluster 16; parkinson rate 0.54. Fold sizes: [369, 369, 367, 367, 367].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.542 [0.475-0.605] |
| stroke geometry only | 0.742 [0.672-0.798] |
| stroke geometry + contrast + paper noise | 0.743 [0.675-0.799] |
| thumbnail 1-NN (appearance only, no learning) | 0.860 [0.815-0.907] |

## wave

1382 images = 175 distinct drawings (clusters); 8 have no copy; largest cluster 16; parkinson rate 0.57. Fold sizes: [281, 280, 274, 274, 273].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.564 [0.476-0.652] |
| stroke geometry only | 0.651 [0.582-0.733] |
| stroke geometry + contrast + paper noise | 0.704 [0.637-0.781] |
| thumbnail 1-NN (appearance only, no learning) | 0.876 [0.826-0.921] |

Reading:
- Compare with the blocked CV and the background hold-out: those numbers were measured with copies on both sides of the split. The drop is the part that was memorisation of seen drawings.
- The 'thumbnail 1-NN' row should be close to 0.5 here. If it is not, copies below the threshold or other shared structure remain; try a lower --tau.
- Intervals resample distinct drawings, so they are wider than the old ones on purpose.
- Still no subject IDs: even this does not show generalisation to new patients.