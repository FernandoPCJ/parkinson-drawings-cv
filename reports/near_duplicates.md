# Near-duplicate report (train vs val originals)

Similarity = correlation of 48x48 inverted thumbnails (1.0 = identical).
Calibration: centred spirals of DIFFERENT people already correlate around the
'median best-match' value below, so only >= 0.99 is treated as a near-duplicate;
0.95-0.99 needs a visual check in near_dup_examples.png.

## spiral: 397 val images vs 1601 train images

- val images with a train image at correlation >= 0.995: 100 (25.2%)
- val images with a train image at correlation >= 0.99: 100 (25.2%)
- val images with a train image at correlation >= 0.98: 100 (25.2%)
- val images with a train image at correlation >= 0.95: 101 (25.4%)
- median best-match correlation: 0.289

- pairs inside train with correlation >= 0.99: 0

## wave: 350 val images vs 1384 train images

- val images with a train image at correlation >= 0.995: 161 (46.0%)
- val images with a train image at correlation >= 0.99: 163 (46.6%)
- val images with a train image at correlation >= 0.98: 163 (46.6%)
- val images with a train image at correlation >= 0.95: 168 (48.0%)
- median best-match correlation: 0.931

- pairs inside train with correlation >= 0.99: 156

## Cross-split pairs >= 0.98

- total: 263
- with DIFFERENT diagnosis label (possible label noise): 51

Top pairs saved to reports/near_dup_examples.png (left = val, right = train).
