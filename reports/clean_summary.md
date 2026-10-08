# Clean manifest summary

Images in manifest: 3732
Links found: 219 byte-identical, 231 near-identical (pixel-verified)
Distinct drawings (groups): **3282**
Groups with more than one file: 417 (867 files)
Groups that span authors' train AND val: 251
Authors' val images with a twin in train: 263 of 747 (35.2%)
Groups with contradictory diagnosis (dropped): 61 (126 files)

## After cleaning (one image per drawing, conflicts dropped)

- train / healthy spiral: 762
- train / healthy wave: 533
- train / parkinson spiral: 805
- train / parkinson wave: 678
- val / healthy spiral: 81
- val / healthy wave: 55
- val / parkinson spiral: 191
- val / parkinson wave: 116

Totals by split_clean: {'train': 2778, 'dup': 385, 'drop': 126, 'val': 443}

- train: 2778 images, 53.4% parkinson
- val: 443 images, 69.3% parkinson
