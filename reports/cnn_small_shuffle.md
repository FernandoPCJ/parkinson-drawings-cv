# Rung 3: small CNN from scratch

Input 128x128 paper-relative ink map; width=16; 20 fixed epochs (no early stopping); light augmentation; blocked CV, 1 repeat(s); 5 folds; LABELS SHUFFLED (control, expect AUC ~0.5). Value [95% bootstrap CI].

Bar to beat: rung 2b AUC-ROC, spiral 0.738 and wave 0.748.

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.470 [0.449-0.493] | 0.466 [0.445-0.489] | 0.514 [0.484-0.546] | 0.419 [0.387-0.450] | 0.458 [0.433-0.485] | 0.506 [0.488-0.529] |
| wave | 1382 | 0.575 | 0.527 [0.504-0.552] | 0.509 [0.485-0.535] | 0.634 [0.604-0.668] | 0.384 [0.346-0.425] | 0.507 [0.477-0.536] | 0.587 [0.562-0.613] |