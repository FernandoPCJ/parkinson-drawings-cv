# Background audit of the CNN

## spiral (n=1839); CNN AUC overall 0.867

### frame_mean (grey frame / vignette): as a lone classifier AUC 0.535 (higher in parkinson); correlation with CNN score -0.00

| tercile | feature range | n | true parkinson rate | mean CNN score | CNN AUC inside |
|---|---|---|---|---|---|
| low | 3.990-9.510 | 613 | 0.49 | 0.64 | 0.952 |
| mid | 9.633-14.289 | 612 | 0.55 | 0.57 | 0.812 |
| high | 14.290-28.824 | 614 | 0.58 | 0.63 | 0.846 |

### speckle (faint paper grain): as a lone classifier AUC 0.517 (higher in parkinson); correlation with CNN score +0.11

| tercile | feature range | n | true parkinson rate | mean CNN score | CNN AUC inside |
|---|---|---|---|---|---|
| low | 0.033-0.075 | 613 | 0.41 | 0.51 | 0.834 |
| mid | 0.076-0.631 | 612 | 0.77 | 0.75 | 0.870 |
| high | 0.632-0.827 | 614 | 0.45 | 0.59 | 0.864 |

## wave (n=1382); CNN AUC overall 0.809

### frame_mean (grey frame / vignette): as a lone classifier AUC 0.522 (lower in parkinson); correlation with CNN score -0.46

| tercile | feature range | n | true parkinson rate | mean CNN score | CNN AUC inside |
|---|---|---|---|---|---|
| low | 3.375-6.889 | 460 | 0.54 | 0.82 | 0.852 |
| mid | 6.895-13.463 | 461 | 0.66 | 0.68 | 0.792 |
| high | 13.464-17.037 | 461 | 0.52 | 0.48 | 0.895 |

### speckle (faint paper grain): as a lone classifier AUC 0.527 (lower in parkinson); correlation with CNN score +0.44

| tercile | feature range | n | true parkinson rate | mean CNN score | CNN AUC inside |
|---|---|---|---|---|---|
| low | 0.035-0.091 | 458 | 0.53 | 0.45 | 0.847 |
| mid | 0.091-0.731 | 463 | 0.71 | 0.78 | 0.822 |
| high | 0.732-0.893 | 461 | 0.48 | 0.75 | 0.849 |

Reading: compare the 'true parkinson rate' column with 'mean CNN score'. If the score swings
across terciles while the true rate does not, the CNN follows the background. The last column
says whether it still separates the classes among images with a similar background.