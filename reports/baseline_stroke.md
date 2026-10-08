# Baseline 2b: stroke-feature logistic regression

Blocked 5-fold CV, 5 repeats. Value [95% bootstrap CI]. Rows with NaN features filled with the median: 0.

Bars to beat: floor (balanced acc 0.5, AUC 0.5) and metadata baseline 2a.

## geometry only (no paper/contrast numbers)

features: ['ink_frac', 'thickness_idx', 'roughness', 'bbox_w', 'bbox_h', 'cx', 'cy', 'sx', 'sy', 'r_mean', 'r_std', 'cross_h', 'cross_v']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.666 [0.646-0.686] | 0.670 [0.650-0.690] | 0.629 [0.603-0.657] | 0.711 [0.683-0.736] | 0.738 [0.717-0.760] | 0.805 [0.788-0.821] |
| wave | 1382 | 0.575 | 0.640 [0.616-0.663] | 0.638 [0.613-0.662] | 0.652 [0.619-0.681] | 0.624 [0.590-0.660] | 0.697 [0.668-0.725] | 0.756 [0.729-0.786] |
| pooled | 3221 | 0.556 | 0.619 [0.604-0.635] | 0.621 [0.606-0.637] | 0.603 [0.582-0.625] | 0.639 [0.616-0.662] | 0.658 [0.639-0.677] | 0.704 [0.685-0.723] |

## geometry + contrast + paper noise

features: ['ink_frac', 'thickness_idx', 'roughness', 'bbox_w', 'bbox_h', 'cx', 'cy', 'sx', 'sy', 'r_mean', 'r_std', 'cross_h', 'cross_v', 'ink_contrast', 'paper_noise']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.670 [0.649-0.689] | 0.673 [0.653-0.693] | 0.632 [0.604-0.660] | 0.715 [0.688-0.742] | 0.741 [0.719-0.762] | 0.802 [0.784-0.818] |
| wave | 1382 | 0.575 | 0.665 [0.640-0.688] | 0.664 [0.640-0.689] | 0.665 [0.634-0.696] | 0.664 [0.625-0.699] | 0.748 [0.722-0.773] | 0.803 [0.783-0.825] |
| pooled | 3221 | 0.556 | 0.612 [0.596-0.629] | 0.613 [0.598-0.630] | 0.607 [0.585-0.628] | 0.619 [0.594-0.642] | 0.673 [0.654-0.691] | 0.731 [0.714-0.746] |

## contrast + paper noise only (side-channel check)

features: ['ink_contrast', 'paper_noise']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.561 [0.542-0.580] | 0.575 [0.555-0.593] | 0.408 [0.383-0.436] | 0.742 [0.716-0.768] | 0.556 [0.530-0.580] | 0.589 [0.567-0.615] |
| wave | 1382 | 0.575 | 0.504 [0.483-0.526] | 0.506 [0.485-0.528] | 0.495 [0.468-0.523] | 0.517 [0.484-0.550] | 0.514 [0.487-0.541] | 0.573 [0.554-0.598] |
| pooled | 3221 | 0.556 | 0.545 [0.529-0.560] | 0.553 [0.537-0.568] | 0.478 [0.457-0.500] | 0.629 [0.606-0.651] | 0.562 [0.543-0.579] | 0.596 [0.581-0.614] |
