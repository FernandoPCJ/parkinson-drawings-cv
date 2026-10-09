# Near-duplicate ('twin') audit on binarised drawings

Similarity: cosine on 32x32 thumbnails of the binarised and cropped (position and size removed) ink map, best of 8 rotations / mirrors. Groups = terciles of frame_mean (as in the background hold-out); folds = blocked 5-fold (offset 0).

## spiral (n=1839)

best similarity to any other image: median 0.999, p10 0.911

| threshold | images with a twin | clusters of size >=2 | largest cluster | label-pure clusters | twin pairs | same label | different bg group | different CV fold |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 98.0% | 157 | 203 | 98.1% | 15504 | 76.0% | 31.6% | 43.0% |
| 0.90 | 91.6% | 191 | 82 | 98.4% | 6299 | 98.3% | 14.0% | 81.1% |
| 0.95 | 85.7% | 308 | 16 | 99.7% | 3853 | 99.8% | 6.9% | 85.3% |

## wave (n=1382)

best similarity to any other image: median 0.999, p10 0.994

| threshold | images with a twin | clusters of size >=2 | largest cluster | label-pure clusters | twin pairs | same label | different bg group | different CV fold |
|---|---|---|---|---|---|---|---|---|
| 0.80 | 99.8% | 146 | 73 | 99.3% | 6653 | 95.4% | 25.2% | 84.5% |
| 0.90 | 99.1% | 179 | 16 | 100.0% | 4837 | 100.0% | 10.7% | 85.8% |
| 0.95 | 98.8% | 204 | 16 | 100.0% | 4212 | 100.0% | 5.8% | 86.1% |

Reading: 'different bg group' close to 100% among twin pairs means the background hold-out puts copies on both sides of the split (so it is easier than it looks). 'different CV fold' high means the blocked CV does too. 'same label' near 100% means the copies carry the label with them.
Cluster ids at threshold 0.90 were saved for a cluster-grouped evaluation.