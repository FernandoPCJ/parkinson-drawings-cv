# YOLO dataset report (v2)

Root: `cornelioac/parkinson-yolo-dataset` (Kaggle, version 5)

## Images per top-level folder and split

- YOLODatasetFull / train / original: 2985
- YOLODatasetFull / val / original: 747
- YOLODatasetFull_Augmented / train / augmented: 8054
- YOLODatasetFull_Augmented / train / original: 1377
- YOLODatasetFull_Augmented / val / original: 2355

Distinct original images (after removing _augN suffixes): **3732**
Total image files: 15518

## Leakage: validation/test images that have a sibling in train

Overall (all folders together):

- val: 1608 of 3102 images (51.8%) have the same original (or an augmented copy of it) in train

Inside `YOLODatasetFull` only:

- val: 0 of 747 images (0.0%) have the same original (or an augmented copy of it) in train

Inside `YOLODatasetFull_Augmented` only:

- val: 1356 of 2355 images (57.6%) have the same original (or an augmented copy of it) in train

Originals that appear (unaugmented) in more than one split: 1608

## Labels

- images without a label file: 0
- class healthy spiral: 3779
- class healthy wave: 3846
- class parkinson spiral: 4135
- class parkinson wave: 3758
- boxes covering >= 90% of the image: 15518 of 15518 (100.0%)  -> if this is ~100%, the 'detection' format is really whole-image classification
- mean box width/height (normalised): 1.00 / 1.00

## Sample names

- YOLODatasetFull / train: healthy_1.png, healthy_10.png, healthy_100.png, healthy_1000.png, healthy_1001.png
- YOLODatasetFull / val: healthy_1598.png, healthy_1599.png, healthy_1600.png, healthy_1601.png, healthy_1602.png
- YOLODatasetFull_Augmented / train: healthy_1000_aug1017.png, healthy_1000_aug281.png, healthy_1000_aug281_aug246.png, healthy_1000_aug281_aug832.png, healthy_1000_aug5.png
- YOLODatasetFull_Augmented / val: healthy_1.png, healthy_10.png, healthy_100.png, healthy_1000.png, healthy_1002.png
