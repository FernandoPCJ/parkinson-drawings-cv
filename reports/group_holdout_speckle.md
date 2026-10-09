# Leave-one-background-group-out (groups by speckle)

Each drawing type is cut into three groups by terciles of `speckle` (low / mid / high); each group is the test set once, training uses the other two. AUC-ROC INSIDE each held-out group, value [95% bootstrap CI over images, averaged over 1 repeat(s)]. CNN: 20 fixed epochs, width 16.

Low and high are extrapolation, mid is interpolation. Groups are a crude stand-in for a new data source. Images are treated as independent, so the intervals are optimistic.

## spiral (n=1839)

| group | n | parkinson rate | speckle range |
|---|---|---|---|
| low | 613 | 0.41 | 0.033-0.075 |
| mid | 612 | 0.77 | 0.076-0.631 |
| high | 614 | 0.45 | 0.632-0.827 |

| model | held out: low | held out: mid | held out: high | mean of the 3 |
|---|---|---|---|---|
| background only (frame_mean, speckle) | 0.561 [0.520-0.604] | 0.526 [0.471-0.582] | 0.480 [0.432-0.523] | 0.522 |
| stroke geometry only | 0.587 [0.539-0.636] | 0.715 [0.671-0.756] | 0.795 [0.756-0.830] | 0.699 |
| stroke geometry + contrast + paper noise | 0.619 [0.575-0.666] | 0.731 [0.689-0.770] | 0.791 [0.752-0.826] | 0.714 |
| small CNN | 0.813 [0.777-0.849] | 0.871 [0.840-0.897] | 0.802 [0.760-0.836] | 0.828 |

Reference, same CNN recipe with blocked CV (every background on both sides): AUC-ROC 0.869. Stroke references: reports/baseline_stroke.md.

## wave (n=1382)

| group | n | parkinson rate | speckle range |
|---|---|---|---|
| low | 458 | 0.53 | 0.035-0.091 |
| mid | 463 | 0.71 | 0.091-0.731 |
| high | 461 | 0.48 | 0.732-0.893 |

| model | held out: low | held out: mid | held out: high | mean of the 3 |
|---|---|---|---|---|
| background only (frame_mean, speckle) | 0.568 [0.516-0.621] | 0.583 [0.521-0.641] | 0.530 [0.478-0.585] | 0.560 |
| stroke geometry only | 0.740 [0.688-0.785] | 0.581 [0.506-0.649] | 0.759 [0.715-0.797] | 0.693 |
| stroke geometry + contrast + paper noise | 0.713 [0.661-0.761] | 0.617 [0.542-0.686] | 0.722 [0.676-0.763] | 0.684 |
| small CNN | 0.742 [0.695-0.780] | 0.791 [0.746-0.837] | 0.827 [0.793-0.861] | 0.787 |

Reference, same CNN recipe with blocked CV (every background on both sides): AUC-ROC 0.804. Stroke references: reports/baseline_stroke.md.

Reading:
- Compare the CNN row with its blocked-CV reference. A small drop means the ranking it learned still works on a background regime it never saw; a large drop toward the 'background only' row means it leaned on the background.
- 'background only' is the shortcut. Inside a group the background number varies little, so near 0.5 (or below) is expected; a value clearly above 0.5 would mean the shortcut carries signal even inside a group.
- Stroke-feature rows are the non-deep-learning reference under the same shift.
- Only three groups and two drawing types: differences of a few points are within the noise.