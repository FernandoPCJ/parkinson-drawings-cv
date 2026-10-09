# Cluster-grouped cross-validation (copies never split across folds)

Copies: cosine >= 0.90 on 32x32 thumbnails of the binarised and CROPPED (position and size removed) drawing, best of 8 rotations / mirrors, joined into clusters. 5-fold, whole clusters per fold, stratified by label, seed 0. AUC-ROC of the pooled out-of-fold scores, [95% CI resampling CLUSTERS], 500 resamples. Input: none. Network: resnet18, 8 fixed epochs.

## spiral

1839 images = 345 distinct drawings (clusters); 154 have no copy; largest cluster 82; parkinson rate 0.54. Fold sizes: [368, 368, 367, 368, 368].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.519 [0.450-0.587] |
| stroke geometry only | 0.742 [0.678-0.808] |
| stroke geometry + contrast + paper noise | 0.743 [0.678-0.807] |
| thumbnail 1-NN (appearance only, no learning) | 0.819 [0.760-0.878] |
| ResNet18 (ImageNet, fine-tuned) | 0.969 [0.943-0.990] |

## wave

1382 images = 191 distinct drawings (clusters); 12 have no copy; largest cluster 16; parkinson rate 0.57. Fold sizes: [277, 276, 276, 277, 276].

| model | AUC-ROC [95% CI over clusters] |
|---|---|
| background only (frame_mean, speckle) | 0.552 [0.458-0.635] |
| stroke geometry only | 0.665 [0.582-0.739] |
| stroke geometry + contrast + paper noise | 0.710 [0.635-0.783] |
| thumbnail 1-NN (appearance only, no learning) | 0.854 [0.802-0.898] |
| ResNet18 (ImageNet, fine-tuned) | 0.961 [0.932-0.985] |

Reading:
- Compare with the blocked CV and the background hold-out: those numbers were measured with copies on both sides of the split. The drop is the part that was memorisation of seen drawings.
- The 'thumbnail 1-NN' row should be close to 0.5 here. If it is not, copies below the threshold or other shared structure remain; try a lower --tau.
- Intervals resample distinct drawings, so they are wider than the old ones on purpose.
- Still no subject IDs: even this does not show generalisation to new patients.