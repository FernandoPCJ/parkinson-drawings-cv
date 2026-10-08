# Export report

Final models trained on all usable drawings of each type. No held-out evaluation here: expected quality is the cross-validated result in the README.

| type | n train | ONNX size (KB) | max abs logit diff | same decisions | torch median / p95 (ms) | ONNX median / p95 (ms) |
|---|---|---|---|---|---|---|
| spiral | 1839 | 239 | 3.81e-06 | True | 1.94 / 2.05 | 0.27 / 0.29 |
| wave | 1382 | 239 | 3.81e-06 | True | 1.68 / 2.15 | 0.27 / 0.29 |