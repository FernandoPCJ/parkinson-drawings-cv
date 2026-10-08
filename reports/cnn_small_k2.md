# Rung 3: small CNN from scratch

Input 128x128 paper-relative ink map; width=16; 20 fixed epochs (no early stopping); light augmentation; blocked CV, 1 repeat(s); 2 folds. Value [95% bootstrap CI].

Bar to beat: rung 2b AUC-ROC, spiral 0.738 and wave 0.748.

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.723 [0.701-0.743] | 0.716 [0.695-0.738] | 0.791 [0.766-0.819] | 0.642 [0.607-0.673] | 0.811 [0.792-0.830] | 0.856 [0.839-0.871] |
| wave | 1382 | 0.575 | 0.720 [0.698-0.744] | 0.705 [0.681-0.729] | 0.806 [0.780-0.831] | 0.604 [0.564-0.642] | 0.796 [0.772-0.819] | 0.855 [0.838-0.871] |