# Grad-CAM audit: spiral

One model, trained on folds 1-4, explained on held-out fold 0 (369 images). Held-out accuracy 0.778. Ink = pixel >= 15 widened by 3 px.

| group | n | ink enrichment (1 = ignores stroke) | heat in outer 10% frame |
|---|---|---|---|
| true positive | 184 | 1.28 | 25.1% |
| true negative | 103 | 1.06 | 32.0% |
| false positive | 66 | 1.38 | 22.6% |
| false negative | 16 | 1.06 | 30.4% |

all images: ink enrichment 1.22, heat in frame 26.8% (uniform heat would put 36% in the frame)

Reading: enrichment near 1.0 or a frame share above the uniform value means the model is not
focused on the stroke. Enrichment clearly above 1 with a low frame share means it uses the stroke.
Look at the PNG as well: numbers do not show WHICH part of the drawing is used.