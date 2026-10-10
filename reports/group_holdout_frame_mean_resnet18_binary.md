# Leave-one-background-group-out (groups by frame_mean)

Each drawing type is cut into three groups by terciles of `frame_mean` (low / mid / high); each group is the test set once, training uses the other two. AUC-ROC INSIDE each held-out group, value [95% bootstrap CI over images, averaged over 1 repeat(s)]. Input: binary. Network (resnet18): 8 fixed epochs.

Low and high are extrapolation, mid is interpolation. Groups are a crude stand-in for a new data source. Images are treated as independent, so the intervals are optimistic.

## spiral (n=1839)

| group | n | parkinson rate | frame_mean range |
|---|---|---|---|
| low | 613 | 0.49 | 3.990-9.510 |
| mid | 612 | 0.55 | 9.633-14.289 |
| high | 614 | 0.58 | 14.290-28.824 |

| model | held out: low | held out: mid | held out: high | mean of the 3 |
|---|---|---|---|---|
| background only (frame_mean, speckle) | 0.378 [0.331-0.419] | 0.570 [0.528-0.612] | 0.616 [0.566-0.661] | 0.521 |
| stroke geometry only | 0.822 [0.789-0.852] | 0.639 [0.593-0.683] | 0.745 [0.706-0.780] | 0.735 |
| stroke geometry + contrast + paper noise | 0.806 [0.770-0.837] | 0.639 [0.591-0.686] | 0.704 [0.662-0.744] | 0.716 |
| thumbnail 1-NN (appearance only, no learning) | 0.997 [0.994-0.999] | 0.874 [0.850-0.899] | 0.941 [0.923-0.955] | 0.938 |
| ResNet18 (ImageNet, fine-tuned) | 0.938 [0.916-0.958] | 0.887 [0.860-0.911] | 0.882 [0.857-0.905] | 0.903 |

Reference, small CNN with blocked CV (every background on both sides): AUC-ROC 0.869. Stroke references: reports/baseline_stroke.md.

Compare with the small CNN under the same hold-out: reports/group_holdout_frame_mean.md.

## wave (n=1382)

| group | n | parkinson rate | frame_mean range |
|---|---|---|---|
| low | 460 | 0.54 | 3.375-6.889 |
| mid | 461 | 0.66 | 6.895-13.463 |
| high | 461 | 0.52 | 13.464-17.037 |

| model | held out: low | held out: mid | held out: high | mean of the 3 |
|---|---|---|---|---|
| background only (frame_mean, speckle) | 0.613 [0.559-0.661] | 0.608 [0.550-0.656] | 0.471 [0.424-0.524] | 0.564 |
| stroke geometry only | 0.793 [0.750-0.833] | 0.627 [0.575-0.677] | 0.811 [0.774-0.845] | 0.743 |
| stroke geometry + contrast + paper noise | 0.783 [0.737-0.824] | 0.700 [0.652-0.748] | 0.761 [0.722-0.799] | 0.748 |
| thumbnail 1-NN (appearance only, no learning) | 1.000 [1.000-1.000] | 0.937 [0.917-0.956] | 0.955 [0.939-0.970] | 0.964 |
| ResNet18 (ImageNet, fine-tuned) | 0.847 [0.817-0.880] | 0.831 [0.792-0.868] | 0.905 [0.879-0.928] | 0.861 |

Reference, small CNN with blocked CV (every background on both sides): AUC-ROC 0.804. Stroke references: reports/baseline_stroke.md.

Compare with the small CNN under the same hold-out: reports/group_holdout_frame_mean.md.

Reading:
- Compare the CNN row with its blocked-CV reference. A small drop means the ranking it learned still works on a background regime it never saw; a large drop toward the 'background only' row means it leaned on the background.
- 'background only' is the shortcut. Inside a group the background number varies little, so near 0.5 (or below) is expected; a value clearly above 0.5 would mean the shortcut carries signal even inside a group.
- Stroke-feature rows are the non-deep-learning reference under the same shift.
- Only three groups and two drawing types: differences of a few points are within the noise.