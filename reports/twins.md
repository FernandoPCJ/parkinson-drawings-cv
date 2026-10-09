# Near-duplicate ('twin') audit on binarised drawings

Similarity: cosine on 32x32 thumbnails of the binarised ink map, best of 8 rotations / mirrors. Groups = terciles of frame_mean (as in the background hold-out); folds = blocked 5-fold (offset 0).

## spiral (n=1839)

best similarity to any other image: median 1.000, p10 0.878

| threshold | images with a twin | clusters of size >=2 | largest cluster | label-pure clusters | twin pairs | same label | different bg group | different CV fold |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 96.4% | 161 | 173 | 97.5% | 9851 | 91.7% | 26.3% | 70.7% |
| 0.90 | 88.4% | 198 | 16 | 98.0% | 6439 | 99.8% | 16.3% | 84.5% |
| 0.95 | 86.8% | 196 | 16 | 99.5% | 5857 | 99.9% | 9.3% | 85.2% |

## wave (n=1382)

best similarity to any other image: median 1.000, p10 0.993

| threshold | images with a twin | clusters of size >=2 | largest cluster | label-pure clusters | twin pairs | same label | different bg group | different CV fold |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 99.8% | 149 | 41 | 99.3% | 6335 | 98.6% | 25.5% | 84.9% |
| 0.90 | 99.4% | 167 | 16 | 100.0% | 5233 | 100.0% | 12.9% | 85.7% |
| 0.95 | 99.3% | 174 | 16 | 100.0% | 4837 | 100.0% | 6.4% | 86.3% |

Reading: 'different bg group' close to 100% among twin pairs means the background hold-out puts copies on both sides of the split (so it is easier than it looks). 'different CV fold' high means the blocked CV does too. 'same label' near 100% means the copies carry the label with them.
Cluster ids at threshold 0.90 were saved for a cluster-grouped evaluation.