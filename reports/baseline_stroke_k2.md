# Baseline 2b: stroke-feature logistic regression

Blocked 2-fold CV, 5 repeats. Value [95% bootstrap CI]. Rows with NaN features filled with the median: 0.

Bars to beat: floor (balanced acc 0.5, AUC 0.5) and metadata baseline 2a.

## geometry only (no paper/contrast numbers)

features: ['ink_frac', 'thickness_idx', 'roughness', 'bbox_w', 'bbox_h', 'cx', 'cy', 'sx', 'sy', 'r_mean', 'r_std', 'cross_h', 'cross_v']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.665 [0.649-0.684] | 0.670 [0.653-0.688] | 0.610 [0.587-0.637] | 0.730 [0.705-0.755] | 0.730 [0.708-0.749] | 0.794 [0.776-0.810] |
| wave | 1382 | 0.575 | 0.639 [0.615-0.661] | 0.641 [0.616-0.663] | 0.630 [0.598-0.657] | 0.652 [0.620-0.687] | 0.699 [0.670-0.725] | 0.757 [0.731-0.785] |
| pooled | 3221 | 0.556 | 0.623 [0.608-0.638] | 0.626 [0.612-0.641] | 0.596 [0.576-0.618] | 0.656 [0.635-0.677] | 0.663 [0.645-0.681] | 0.703 [0.684-0.722] |

## geometry + contrast + paper noise

features: ['ink_frac', 'thickness_idx', 'roughness', 'bbox_w', 'bbox_h', 'cx', 'cy', 'sx', 'sy', 'r_mean', 'r_std', 'cross_h', 'cross_v', 'ink_contrast', 'paper_noise']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.671 [0.653-0.690] | 0.676 [0.658-0.695] | 0.616 [0.593-0.642] | 0.736 [0.712-0.761] | 0.733 [0.711-0.753] | 0.793 [0.775-0.811] |
| wave | 1382 | 0.575 | 0.670 [0.648-0.693] | 0.671 [0.648-0.695] | 0.666 [0.638-0.695] | 0.676 [0.640-0.711] | 0.750 [0.725-0.774] | 0.805 [0.785-0.825] |
| pooled | 3221 | 0.556 | 0.610 [0.594-0.625] | 0.612 [0.597-0.627] | 0.591 [0.570-0.611] | 0.634 [0.611-0.655] | 0.674 [0.654-0.692] | 0.729 [0.712-0.745] |

## contrast + paper noise only (side-channel check)

features: ['ink_contrast', 'paper_noise']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.524 [0.507-0.539] | 0.537 [0.521-0.553] | 0.371 [0.350-0.393] | 0.704 [0.682-0.727] | 0.546 [0.525-0.566] | 0.572 [0.555-0.592] |
| wave | 1382 | 0.575 | 0.509 [0.493-0.526] | 0.510 [0.494-0.527] | 0.503 [0.479-0.525] | 0.517 [0.495-0.544] | 0.516 [0.494-0.542] | 0.580 [0.563-0.604] |
| pooled | 3221 | 0.556 | 0.529 [0.516-0.541] | 0.537 [0.524-0.549] | 0.464 [0.447-0.483] | 0.610 [0.590-0.631] | 0.554 [0.537-0.570] | 0.590 [0.577-0.607] |
