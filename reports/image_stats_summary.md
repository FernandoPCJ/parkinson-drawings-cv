# Image audit

Images: 3221

## Per class

### healthy spiral (n=843)

- distinct (width, height): 1; most common (512, 512) (100%)
- width min/median/max: 512 / 512 / 512
- height min/median/max: 512 / 512 / 512
- color modes: {'RGB': 799, 'RGBA': 44}
- file size median: 228423 bytes
- paper brightness (median gray) median: 233.0
- darkest 5% gray level, median: 162.0
- ink share (pixels >40 darker than paper), median: 4.66%

### healthy wave (n=588)

- distinct (width, height): 1; most common (512, 512) (100%)
- width min/median/max: 512 / 512 / 512
- height min/median/max: 512 / 512 / 512
- color modes: {'RGB': 553, 'RGBA': 35}
- file size median: 251472 bytes
- paper brightness (median gray) median: 193.0
- darkest 5% gray level, median: 163.0
- ink share (pixels >40 darker than paper), median: 3.73%

### parkinson spiral (n=996)

- distinct (width, height): 1; most common (512, 512) (100%)
- width min/median/max: 512 / 512 / 512
- height min/median/max: 512 / 512 / 512
- color modes: {'RGB': 996}
- file size median: 230900 bytes
- paper brightness (median gray) median: 233.0
- darkest 5% gray level, median: 158.0
- ink share (pixels >40 darker than paper), median: 5.65%

### parkinson wave (n=794)

- distinct (width, height): 1; most common (512, 512) (100%)
- width min/median/max: 512 / 512 / 512
- height min/median/max: 512 / 512 / 512
- color modes: {'RGB': 794}
- file size median: 248307 bytes
- paper brightness (median gray) median: 192.0
- darkest 5% gray level, median: 162.0
- ink share (pixels >40 darker than paper), median: 4.21%

## Can metadata alone tell the classes apart?

Single-feature AUC-ROC for parkinson vs healthy, folded so 0.5 = no information (max(AUC, 1-AUC)). Values near 0.5 are good news; values >= 0.65 are shortcut candidates.

| drawing | width | height | filesize | mean_gray | median_gray | p5_gray | ink_frac |
|---|---|---|---|---|---|---|---|
| spiral | 0.500 | 0.500 | 0.517 | 0.518 | 0.506 | 0.594 | 0.605 |
| wave | 0.500 | 0.500 | 0.505 | 0.564 | 0.563 | 0.548 | 0.558 |

Shortcut candidates (>= 0.65): none
