# ReplaceSIMCard deviations

Distance is the mean cosine distance along the DTW path, averaged over the 48 reference videos. A step is a deviation when that distance is above the cutoff.

Source: 1,128 correct COIN pairs and six phone recordings, 5 October 2026.

> **Histogram numbers predate the histogram fix.** All histogram tables below were computed before `frame_feature` normalized its colour and edge parts separately. The cutoffs are now 0.66, 0.63, 0.64 and 0.62 (see the README). Rerun `scripts/score_recordings.py` on the six recordings to refresh the histogram tables. CLIP numbers are unaffected.

## Cutoff from the 48 correct videos

Cutoff is the mean of 1,128 correct pairs plus two standard deviations.

| Step | Histograms | Histogram cutoff | CLIP | CLIP cutoff |
| --- | --- | --- | --- | --- |
| Open the slot | 0.537 ± 0.101 | 0.739 | 0.196 ± 0.044 | 0.283 |
| Put the SIM in | 0.486 ± 0.088 | 0.661 | 0.197 ± 0.050 | 0.298 |
| Press the tray back | 0.526 ± 0.097 | 0.719 | 0.202 ± 0.047 | 0.296 |
| Whole video | 0.504 ± 0.077 | 0.659 | 0.200 ± 0.039 | 0.277 |

## Experiment: deviations from DTW labels

The recordings have no timestamps. DTW copies each step boundary from the reference video it aligns to.

### Histogram features

| Recording | Open | Insert | Close | Whole |
| --- | --- | --- | --- | --- |
| correct-1.mp4 | 0.605 | 0.530 | 0.599 | 0.580 |
| correct-2.mp4 | 0.643 | 0.546 | 0.579 | 0.598 |
| correct-3.mp4 | 0.672 | 0.552 | 0.643 | 0.624 |
| correct-4.mp4 | 0.628 | 0.579 | 0.659 | 0.621 |
| incorrect-1.mp4 | 0.630 | 0.582 | 0.625 | 0.602 |
| incorrect-2.mp4 | 0.681 | 0.571 | 0.656 | 0.623 |

### CLIP features

| Recording | Open | Insert | Close | Whole |
| --- | --- | --- | --- | --- |
| correct-1.mp4 | 0.153 | 0.171 | 0.165 | 0.170 |
| correct-2.mp4 | 0.183 | 0.180 | 0.187 | 0.185 |
| correct-3.mp4 | 0.167 | 0.171 | 0.169 | 0.174 |
| correct-4.mp4 | 0.201 | 0.215 | 0.182 | 0.205 |
| incorrect-1.mp4 | 0.175 | 0.183 | 0.208 | 0.187 |
| incorrect-2.mp4 | 0.192 | 0.196 | 0.183 | 0.194 |

## Ablation: DTW timestamps against the marked timestamps

CLIP alignment. The DTW interval is the median window transferred from the 48 references. Overlap is 1 when the intervals match. Positive start error means DTW begins late. Times are seconds.

| Recording | Step | Marked | DTW | Start error | End error | Overlap |
| --- | --- | --- | --- | --- | --- | --- |
| correct-1.mp4 | 1 | 0.0–10.6 | 2.2–8.4 | +2.2 | −2.2 | 0.49 |
| correct-1.mp4 | 2 | 12.4–19.9 | 14.2–17.6 | +1.8 | −2.3 | 0.35 |
| correct-1.mp4 | 3 | 20.5–24.9 | 20.0–23.2 | −0.5 | −1.7 | 0.45 |
| correct-2.mp4 | 1 | 0.0–9.0 | 2.6–10.0 | +2.6 | +1.0 | 0.50 |
| correct-2.mp4 | 2 | 15.2–23.0 | 17.0–19.8 | +1.8 | −3.2 | 0.16 |
| correct-2.mp4 | 3 | 23.0–28.0 | 21.6–26.8 | −1.4 | −1.2 | 0.44 |
| correct-3.mp4 | 1 | 0.0–8.7 | 2.4–7.9 | +2.4 | −0.8 | 0.37 |
| correct-3.mp4 | 2 | 8.7–17.4 | 12.6–15.4 | +3.9 | −2.0 | 0.22 |
| correct-3.mp4 | 3 | 17.4–22.4 | 16.8–20.8 | −0.6 | −1.6 | 0.38 |
| correct-4.mp4 | 1 | 0.0–15.2 | 2.5–9.4 | +2.5 | −5.8 | 0.44 |
| correct-4.mp4 | 2 | 16.9–28.0 | 17.0–23.0 | +0.1 | −5.0 | 0.29 |
| correct-4.mp4 | 3 | 28.0–34.9 | 27.0–31.4 | −1.0 | −3.5 | 0.28 |
| incorrect-1.mp4 | 1 | 0.0–7.5 | 3.6–12.5 | +3.6 | +5.0 | 0.29 |
| incorrect-1.mp4 | 2 | 9.3–57.4 | 25.4–43.8 | +16.1 | −13.6 | 0.16 |
| incorrect-1.mp4 | 3 | 57.4–61.2 | 49.4–57.4 | −8.0 | −3.8 | 0.02 |
| incorrect-2.mp4 | 1 | 1.6–9.4 | 4.4–14.8 | +2.8 | +5.4 | 0.34 |
| incorrect-2.mp4 | 2 | 13.3–40.1 | 27.5–37.2 | +14.2 | −2.9 | 0.15 |
| incorrect-2.mp4 | 3 | 40.2–45.8 | 39.4–44.2 | −0.8 | −1.6 | 0.47 |

## Ablation: deviations given the marked timestamps

The same 48 alignments, with each recording’s manual step interval included in the step cost.

### Histogram features

| Recording | Open | Insert | Close | Whole |
| --- | --- | --- | --- | --- |
| correct-1.mp4 | 0.616 | 0.578 | 0.594 | 0.579 |
| correct-2.mp4 | 0.694 | 0.579 | 0.627 | 0.603 |
| correct-3.mp4 | 0.680 | 0.596 | 0.664 | 0.626 |
| correct-4.mp4 | 0.658 | 0.609 | 0.660 | 0.625 |
| incorrect-1.mp4 | 0.640 | 0.581 | 0.624 | 0.593 |
| incorrect-2.mp4 | 0.679 | 0.591 | 0.653 | 0.619 |

### CLIP features

| Recording | Open | Insert | Close | Whole |
| --- | --- | --- | --- | --- |
| correct-1.mp4 | 0.161 | 0.174 | 0.167 | 0.167 |
| correct-2.mp4 | 0.187 | 0.182 | 0.192 | 0.184 |
| correct-3.mp4 | 0.175 | 0.173 | 0.175 | 0.172 |
| correct-4.mp4 | 0.205 | 0.218 | 0.189 | 0.206 |
| incorrect-1.mp4 | 0.166 | 0.180 | 0.195 | 0.177 |
| incorrect-2.mp4 | 0.188 | 0.198 | 0.183 | 0.190 |
