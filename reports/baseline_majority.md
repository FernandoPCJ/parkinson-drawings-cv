# Baseline 1: majority class (blocked 5-fold CV, 5 repeats)

Values: point estimate [95% bootstrap CI]. Images treated as independent.

| subset | n | prevalence | accuracy | balanced acc | sensitivity | specificity | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| spiral | 1839 | 0.542 | 0.542 [0.542-0.542] | 0.500 [0.500-0.500] | 1.000 [1.000-1.000] | 0.000 [0.000-0.000] | 0.499 [0.492-0.506] | 0.541 [0.537-0.545] |
| wave | 1382 | 0.575 | 0.575 [0.575-0.575] | 0.500 [0.500-0.500] | 1.000 [1.000-1.000] | 0.000 [0.000-0.000] | 0.499 [0.490-0.509] | 0.574 [0.569-0.579] |
| pooled | 3221 | 0.556 | 0.556 [0.556-0.556] | 0.500 [0.500-0.500] | 1.000 [1.000-1.000] | 0.000 [0.000-0.000] | 0.499 [0.494-0.504] | 0.555 [0.552-0.559] |

Reading: this is the floor. Balanced accuracy 0.5 = coin flip. A model must beat it in sensitivity AND specificity, not just accuracy.
