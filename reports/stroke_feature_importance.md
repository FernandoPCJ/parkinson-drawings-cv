# Stroke feature importance (descriptive)

single AUC: 0.5 = no separation. coef: standardized logistic regression, all features.

## spiral (n=1839)

| feature | in parkinson | single AUC | LR coef |
|---|---|---|---|
| cross_h | higher | 0.683 | -0.16 |
| cross_v | higher | 0.665 | -0.42 |
| r_mean | lower | 0.622 | -1.79 |
| sx | lower | 0.618 | +0.48 |
| sy | lower | 0.609 | +0.48 |
| ink_frac | higher | 0.605 | +4.43 |
| ink_contrast | higher | 0.578 | +0.27 |
| thickness_idx | higher | 0.555 | -1.76 |
| paper_noise | higher | 0.541 | +0.27 |
| bbox_w | higher | 0.529 | +0.32 |
| r_std | lower | 0.528 | -0.52 |
| roughness | lower | 0.524 | +2.13 |
| bbox_h | higher | 0.518 | +0.28 |
| cx | higher | 0.518 | -0.03 |
| cy | lower | 0.510 | -0.01 |

## wave (n=1382)

| feature | in parkinson | single AUC | LR coef |
|---|---|---|---|
| r_mean | lower | 0.652 | +0.90 |
| sx | lower | 0.628 | -1.63 |
| sy | lower | 0.621 | -1.61 |
| roughness | lower | 0.591 | -0.21 |
| thickness_idx | higher | 0.590 | +1.31 |
| bbox_w | lower | 0.562 | -0.06 |
| ink_frac | higher | 0.558 | -1.28 |
| bbox_h | lower | 0.549 | -0.04 |
| cross_v | higher | 0.534 | -0.00 |
| paper_noise | higher | 0.528 | +1.01 |
| ink_contrast | higher | 0.524 | -0.34 |
| cross_h | higher | 0.517 | -0.05 |
| cx | lower | 0.504 | -0.02 |
| r_std | lower | 0.500 | -0.11 |
| cy | lower | 0.500 | -0.00 |
