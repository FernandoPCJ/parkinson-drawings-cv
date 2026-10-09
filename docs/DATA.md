# Data sources, license and attribution

## Dataset used
- Kaggle dataset **Parkinson Yolo Dataset**, published by CornelioAC:
  https://www.kaggle.com/datasets/cornelioac/parkinson-yolo-dataset
- License declared on the Kaggle page (checked 2026-10-09): **Attribution 4.0 International (CC BY 4.0)**.
- Only the folder `YOLODatasetFull` is used. `YOLODatasetFull_Augmented` is not used.
- The images are not redistributed in this repository. Download them from Kaggle.

## Upstream sources listed by the publisher
The Kaggle page says the data "came from various sources":
1. https://www.kaggle.com/datasets/kmader/parkinsons-drawings
2. https://data.mendeley.com/datasets/fd5wd6wmdj/1 (Parkinson's Disease Detection Using Spiral Images, hand drawings)
3. https://www.kaggle.com/datasets/banilkumar20phd7071/handwritten-parkinsons-disease-augmented-data
4. https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/ (HandPD)

## What this means for the results
- CC BY 4.0 is the license declared by the publisher of the compilation. The licenses and terms of the four
  upstream sources were **not** checked here. Anyone reusing the images should check them and credit both the
  publisher and the original sources.
- The images come from different acquisition conditions. The background-style groups found in the audit
  (grey frame vs. speckled paper) have very different parkinson rates, which is consistent with different
  sources but does not prove it. If the source correlates with the label, a model can use the source
  as a shortcut.
- One upstream source is named as augmented data. Transformed copies of the same drawing may exist that the
  deduplication (byte-identical or near-identical files) does not detect. Overlap between upstream
  sources was not checked.
- There are no subject identifiers, so a subject-level split cannot be proven.

Possible future check: train on some background groups and test on another, as a proxy for a new source.

## Code license
The code is under the MIT license (`LICENSE`). It does not cover the images.
