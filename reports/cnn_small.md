# Rung 3: small CNN from scratch

Input 128x128 paper-relative ink map; width=16; 20 fixed epochs (no early stopping); light augmentation; blocked 5-fold CV, 1 repeat(s). Value [95% bootstrap CI].

Bar to beat: rung 2b AUC-ROC, spiral 0.738 and wave 0.748.

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.749 [0.729-0.769] | 0.740 [0.719-0.759] | 0.855 [0.832-0.877] | 0.624 [0.591-0.654] | 0.867 [0.849-0.882] | 0.901 [0.888-0.913] |
| wave | 1382 | 0.575 | 0.730 [0.710-0.753] | 0.711 [0.689-0.735] | 0.841 [0.817-0.866] | 0.580 [0.546-0.622] | 0.809 [0.788-0.832] | 0.867 [0.852-0.883] |