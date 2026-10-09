# Cluster-grouped cross-validation (copies never split across folds)

Copies: cosine >= 0.70 on 32x32 thumbnails of the binarised drawing, best of 8 rotations / mirrors, joined into clusters. 5-fold, whole clusters per fold, stratified by label, seed 0. AUC-ROC of the pooled out-of-fold scores, [95% CI resampling CLUSTERS], 500 resamples. Input: none. Network: resnet18, 8 fixed epochs.

## spiral

1839 images = 134 distinct drawings (clusters); 17 have no copy; largest cluster 239; parkinson rate 0.54. Fold sizes: [554, 321, 320, 321, 323].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.516 [0.400-0.593] |
| stroke geometry only | 0.656 [0.585-0.762] |
| stroke geometry + contrast + paper noise | 0.662 [0.593-0.760] |
| thumbnail 1-NN (appearance only, no learning) | 0.740 [0.678-0.809] |
| ResNet18 (ImageNet, fine-tuned) | 0.938 [0.861-0.988] |

## wave

1382 images = 122 distinct drawings (clusters); 1 have no copy; largest cluster 137; parkinson rate 0.57. Fold sizes: [277, 276, 277, 271, 281].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.540 [0.453-0.634] |
| stroke geometry only | 0.639 [0.547-0.735] |
| stroke geometry + contrast + paper noise | 0.702 [0.611-0.787] |
| thumbnail 1-NN (appearance only, no learning) | 0.809 [0.737-0.877] |
| ResNet18 (ImageNet, fine-tuned) | 0.935 [0.884-0.972] |

Reading:
- Compare with the blocked CV and the background hold-out: those numbers were measured with copies on both sides of the split. The drop is the part that was memorisation of seen drawings.
- The 'thumbnail 1-NN' row should be close to 0.5 here. If it is not, copies below the threshold or other shared structure remain; try a lower --tau.
- Intervals resample distinct drawings, so they are wider than the old ones on purpose.
- Still no subject IDs: even this does not show generalisation to new patients.