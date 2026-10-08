# Rung 3: small CNN from scratch

Input 128x128 paper-relative ink map; width=16; 20 fixed epochs (no early stopping); light augmentation; blocked CV, 5 repeat(s); 5 folds. Value [95% bootstrap CI].

Bar to beat: rung 2b AUC-ROC, spiral 0.738 and wave 0.748.

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.757 [0.739-0.773] | 0.747 [0.728-0.763] | 0.873 [0.854-0.892] | 0.621 [0.592-0.648] | 0.869 [0.851-0.883] | 0.899 [0.884-0.911] |
| wave | 1382 | 0.575 | 0.731 [0.709-0.752] | 0.711 [0.688-0.733] | 0.843 [0.818-0.866] | 0.579 [0.546-0.618] | 0.804 [0.783-0.827] | 0.863 [0.848-0.879] |