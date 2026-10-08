# Grad-CAM audit: wave

One model, trained on folds 1-4, explained on held-out fold 0 (277 images). Held-out accuracy 0.758. Ink = pixel >= 15 widened by 3 px.

| group | n | ink enrichment (1 = ignores stroke) | heat in outer 10% frame |
|---|---|---|---|
| true positive | 143 | 1.02 | 28.6% |
| true negative | 67 | 0.94 | 18.2% |
| false positive | 51 | 0.88 | 40.8% |
| false negative | 16 | 0.89 | 18.6% |

all images: ink enrichment 0.97, heat in frame 27.7% (uniform heat would put 36% in the frame)

Reading: enrichment near 1.0 or a frame share above the uniform value means the model is not
focused on the stroke. Enrichment clearly above 1 with a low frame share means it uses the stroke.
Look at the PNG as well: numbers do not show WHICH part of the drawing is used.