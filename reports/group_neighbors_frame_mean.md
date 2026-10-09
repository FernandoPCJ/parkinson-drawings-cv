# Group hold-out leakage audit (groups by frame_mean)

Similarity = cosine on 32x32 ink-map thumbnails, best of 8 rotations/mirrors, against images of the OTHER groups (the training set of that hold-out).

## spiral (n=1839)

similarity to the closest training image: median 0.761, p90 0.890, max 1.000; share >= 0.95: 4.8%; share >= 0.90: 8.1%

AUC-ROC by similarity tercile (low = least similar to anything in training). `all` pools the three hold-outs into one AUC, so it is not the same number as the mean of the three in-group AUCs:

| model | low | mid | high | top 5% | all |
|---|---|---|---|---|---|
| background_only_frame_me | 0.341 | 0.501 | 0.595 | 0.683 | 0.486 |
| ResNet18_random_init | 0.797 | 0.834 | 0.806 | 0.953 | 0.789 |
| stroke_geometry_contrast | 0.807 | 0.669 | 0.642 | 0.734 | 0.685 |
| stroke_geometry_only | 0.815 | 0.694 | 0.650 | 0.781 | 0.704 |
| ResNet18_ImageNet_fine_t | 0.988 | 0.984 | 0.972 | 0.995 | 0.979 |
| small_CNN | 0.931 | 0.777 | 0.819 | 0.978 | 0.804 |

| similarity tercile | range | n | 1-NN accuracy (copy the neighbour's label) |
|---|---|---|---|
| low | 0.219-0.566 | 613 | 0.868 |
| mid | 0.591-0.821 | 613 | 0.856 |
| high | 0.821-1.000 | 613 | 0.850 |
| top 5% | >= 0.942 | 92 | 0.978 |

## wave (n=1382)

similarity to the closest training image: median 0.731, p90 0.942, max 1.000; share >= 0.95: 7.5%; share >= 0.90: 16.3%

AUC-ROC by similarity tercile (low = least similar to anything in training). `all` pools the three hold-outs into one AUC, so it is not the same number as the mean of the three in-group AUCs:

| model | low | mid | high | top 5% | all |
|---|---|---|---|---|---|
| background_only_frame_me | 0.614 | 0.602 | 0.415 | 0.570 | 0.498 |
| ResNet18_random_init | 0.835 | 0.817 | 0.737 | 0.836 | 0.776 |
| stroke_geometry_contrast | 0.738 | 0.637 | 0.643 | 0.836 | 0.678 |
| stroke_geometry_only | 0.734 | 0.714 | 0.620 | 0.796 | 0.655 |
| ResNet18_ImageNet_fine_t | 0.871 | 0.982 | 0.976 | 0.975 | 0.936 |
| small_CNN | 0.822 | 0.870 | 0.798 | 0.947 | 0.765 |

| similarity tercile | range | n | 1-NN accuracy (copy the neighbour's label) |
|---|---|---|---|
| low | 0.196-0.531 | 461 | 0.950 |
| mid | 0.531-0.818 | 460 | 0.780 |
| high | 0.818-1.000 | 461 | 0.690 |
| top 5% | >= 1.000 | 70 | 1.000 |

Reading: a model that profits from twins has a much higher AUC in `high`/`top 5%` than in `low`.
If the pre-trained network is high everywhere (also in `low`), twins do not explain it and another
cause must be looked for (a shortcut shared by all groups, such as the acquisition source).