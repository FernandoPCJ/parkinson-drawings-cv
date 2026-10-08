# CNN leakage audit: performance vs similarity to the closest training image

## spiral (n=1839)

similarity to closest other-fold image: median 0.517, p90 0.901, max 0.988; share >= 0.95: 1.0%

| similarity tercile | range | n | CNN AUC | 1-NN accuracy |
|---|---|---|---|---|
| low | 0.184-0.356 | 613 | 0.865 | 0.628 |
| mid | 0.356-0.771 | 613 | 0.834 | 0.750 |
| high | 0.772-0.988 | 613 | 0.899 | 0.670 |

overall: CNN AUC 0.867; 1-NN accuracy 0.683

## wave (n=1382)

similarity to closest other-fold image: median 0.659, p90 0.904, max 0.996; share >= 0.95: 2.0%

| similarity tercile | range | n | CNN AUC | 1-NN accuracy |
|---|---|---|---|---|
| low | 0.178-0.397 | 461 | 0.878 | 0.670 |
| mid | 0.397-0.794 | 460 | 0.817 | 0.702 |
| high | 0.794-0.996 | 461 | 0.806 | 0.570 |

overall: CNN AUC 0.809; 1-NN accuracy 0.648

Reading: if AUC and 1-NN accuracy rise sharply from the low to the high tercile, the model
profits from near-twins across folds (leakage). If AUC is similar in all terciles, the gain
does not come from twins.