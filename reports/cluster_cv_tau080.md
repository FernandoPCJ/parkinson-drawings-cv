# Cluster-grouped cross-validation (copies never split across folds)

Copies: cosine >= 0.80 on 32x32 thumbnails of the binarised drawing, best of 8 rotations / mirrors, joined into clusters. 5-fold, whole clusters per fold, stratified by label, seed 0. AUC-ROC of the pooled out-of-fold scores, [95% CI resampling CLUSTERS], 500 resamples. Input: none. Network skipped.

## spiral

1839 images = 228 distinct drawings (clusters); 67 have no copy; largest cluster 173; parkinson rate 0.54. Fold sizes: [368, 368, 368, 368, 367].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.509 [0.425-0.580] |
| stroke geometry only | 0.699 [0.623-0.782] |
| stroke geometry + contrast + paper noise | 0.692 [0.620-0.772] |
| thumbnail 1-NN (appearance only, no learning) | 0.746 [0.682-0.813] |

## wave

1382 images = 152 distinct drawings (clusters); 3 have no copy; largest cluster 41; parkinson rate 0.57. Fold sizes: [278, 276, 275, 281, 272].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.562 [0.471-0.654] |
| stroke geometry only | 0.705 [0.620-0.777] |
| stroke geometry + contrast + paper noise | 0.745 [0.663-0.814] |
| thumbnail 1-NN (appearance only, no learning) | 0.819 [0.758-0.876] |

Reading:
- Compare with the blocked CV and the background hold-out: those numbers were measured with copies on both sides of the split. The drop is the part that was memorisation of seen drawings.
- The 'thumbnail 1-NN' row should be close to 0.5 here. If it is not, copies below the threshold or other shared structure remain; try a lower --tau.
- Intervals resample distinct drawings, so they are wider than the old ones on purpose.
- Still no subject IDs: even this does not show generalisation to new patients.