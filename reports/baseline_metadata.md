# Baseline 2a: metadata-only logistic regression (shortcut check)

Blocked 5-fold CV, 5 repeats. Value [95% bootstrap CI]. Floor = baseline 1 (balanced accuracy 0.5, AUC-ROC 0.5).

## alpha channel only

features: ['is_rgba']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.566 [0.559-0.573] | 0.526 [0.519-0.534] | 1.000 [1.000-1.000] | 0.052 [0.038-0.068] | 0.514 [0.491-0.537] | 0.548 [0.535-0.561] |
| wave | 1382 | 0.575 | 0.600 [0.592-0.608] | 0.530 [0.520-0.540] | 1.000 [1.000-1.000] | 0.060 [0.041-0.079] | 0.519 [0.491-0.549] | 0.583 [0.566-0.602] |
| pooled | 3221 | 0.556 | 0.543 [0.529-0.558] | 0.522 [0.508-0.537] | 0.711 [0.692-0.730] | 0.334 [0.312-0.356] | 0.538 [0.519-0.559] | 0.577 [0.563-0.593] |

## brightness + ink + file size

features: ['mean_gray', 'median_gray', 'p5_gray', 'ink_frac', 'filesize']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.576 [0.557-0.597] | 0.587 [0.568-0.608] | 0.456 [0.431-0.482] | 0.717 [0.691-0.743] | 0.591 [0.565-0.617] | 0.682 [0.661-0.704] |
| wave | 1382 | 0.575 | 0.533 [0.511-0.558] | 0.543 [0.522-0.569] | 0.474 [0.441-0.506] | 0.613 [0.578-0.651] | 0.559 [0.532-0.589] | 0.651 [0.625-0.679] |
| pooled | 3221 | 0.556 | 0.580 [0.565-0.595] | 0.589 [0.573-0.604] | 0.508 [0.486-0.530] | 0.670 [0.647-0.692] | 0.601 [0.583-0.619] | 0.690 [0.673-0.707] |

## all of the above

features: ['is_rgba', 'mean_gray', 'median_gray', 'p5_gray', 'ink_frac', 'filesize']

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.580 [0.560-0.601] | 0.587 [0.568-0.608] | 0.504 [0.479-0.530] | 0.670 [0.644-0.698] | 0.601 [0.575-0.626] | 0.684 [0.665-0.706] |
| wave | 1382 | 0.575 | 0.555 [0.532-0.579] | 0.555 [0.532-0.578] | 0.553 [0.521-0.586] | 0.557 [0.522-0.591] | 0.588 [0.561-0.616] | 0.659 [0.634-0.687] |
| pooled | 3221 | 0.556 | 0.587 [0.571-0.601] | 0.590 [0.575-0.604] | 0.560 [0.539-0.580] | 0.621 [0.596-0.644] | 0.625 [0.606-0.641] | 0.699 [0.682-0.715] |

Reading: balanced acc / AUC-ROC clearly above 0.5 mean the label leaks through these side channels. Any image model has to beat THESE numbers to show it learned something about the drawing.
