# Rung 3: small CNN from scratch

Input 102x102 paper-relative ink map; width=16; 20 fixed epochs (no early stopping); light augmentation; blocked CV, 1 repeat(s); 5 folds; outer 13px frame removed. Value [95% bootstrap CI].

Bar to beat: rung 2b AUC-ROC, spiral 0.738 and wave 0.748.

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.701 [0.682-0.718] | 0.680 [0.660-0.697] | 0.931 [0.914-0.946] | 0.429 [0.394-0.459] | 0.851 [0.833-0.867] | 0.885 [0.866-0.899] |
| wave | 1382 | 0.575 | 0.727 [0.705-0.753] | 0.715 [0.693-0.739] | 0.796 [0.766-0.821] | 0.634 [0.597-0.673] | 0.813 [0.793-0.836] | 0.878 [0.864-0.892] |