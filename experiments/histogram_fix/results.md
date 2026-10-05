# Results

Histogram step costs before and after normalizing the hue, saturation and gradient blocks separately. CLIP is unchanged by the fix and shown for reference.

## ReplaceSIMCard

48 of 86 annotated videos (the 48 in `replace_simcard_videos.json`), all 1128 pairs, via `scripts/classical_pairwise_dtw.py`. The *before* column reproduces the top-level README table. CLIP numbers are the README's.

| Step | Histograms before | Histograms after | CLIP | Hist. cutoff before | Hist. cutoff after | CLIP cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| use the needle to open the SIM card slot | 0.537 ± 0.101 | 0.431 ± 0.114 | 0.196 ± 0.044 | 0.74 | 0.66 | 0.28 |
| put the SIM card into the SIM card slot | 0.486 ± 0.088 | 0.401 ± 0.114 | 0.197 ± 0.050 | 0.66 | 0.63 | 0.30 |
| press the SIM card slot back | 0.526 ± 0.097 | 0.417 ± 0.113 | 0.202 ± 0.047 | 0.72 | 0.64 | 0.30 |
| Whole video | 0.504 ± 0.077 | 0.413 ± 0.103 | 0.200 ± 0.039 | 0.66 | 0.62 | 0.28 |

## MakeStrawberrySmoothie

17 of 97 annotated videos; reference `-3C-VGhs2mo` against 16 others.

| Step | Histograms before | Histograms after | CLIP | Hist. cutoff before | Hist. cutoff after | CLIP cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| put strawberries and other fruits into the juicer | 0.436 ± 0.094 | 0.353 ± 0.082 | 0.297 ± 0.067 | 0.62 | 0.52 | 0.43 |
| put yogurt, honey and other ingredients into the juicer | 0.444 ± 0.057 | 0.348 ± 0.092 | 0.308 ± 0.056 | 0.56 | 0.53 | 0.42 |
| shake and juice | 0.444 ± 0.113 | 0.355 ± 0.118 | 0.302 ± 0.058 | 0.67 | 0.59 | 0.42 |
| Whole video | 0.431 ± 0.069 | 0.339 ± 0.080 | 0.305 ± 0.049 | 0.57 | 0.50 | 0.40 |

## UseRiceCookerToCookRice

31 of 101 annotated videos; reference `VUWV9rEme7c` against 30 others.

| Step | Histograms before | Histograms after | CLIP | Hist. cutoff before | Hist. cutoff after | CLIP cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| take out some rice | 0.498 ± 0.072 | 0.393 ± 0.103 | 0.247 ± 0.067 | 0.64 | 0.60 | 0.38 |
| soak and wash the rice | 0.504 ± 0.079 | 0.400 ± 0.110 | 0.227 ± 0.052 | 0.66 | 0.62 | 0.33 |
| put the washed rice into the rice cooker | 0.516 ± 0.077 | 0.401 ± 0.090 | 0.267 ± 0.061 | 0.67 | 0.58 | 0.39 |
| cook the rice by rice cooker | 0.476 ± 0.070 | 0.335 ± 0.093 | 0.290 ± 0.056 | 0.62 | 0.52 | 0.40 |
| Whole video | 0.489 ± 0.057 | 0.361 ± 0.078 | 0.267 ± 0.042 | 0.60 | 0.52 | 0.35 |

## MakePaperWindMill

25 of 106 annotated videos; reference `4ufBW5Cfpgw` against 24 others.

| Step | Histograms before | Histograms after | CLIP | Hist. cutoff before | Hist. cutoff after | CLIP cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| fold the edges of the paper | 0.516 ± 0.045 | 0.580 ± 0.070 | 0.233 ± 0.053 | 0.61 | 0.72 | 0.34 |
| cut along the edges | 0.505 ± 0.096 | 0.518 ± 0.098 | 0.201 ± 0.063 | 0.70 | 0.71 | 0.33 |
| fold the squares inward and fix them | 0.479 ± 0.081 | 0.575 ± 0.080 | 0.214 ± 0.054 | 0.64 | 0.74 | 0.32 |
| fix the wind mill on the bracket | 0.519 ± 0.088 | 0.604 ± 0.075 | 0.230 ± 0.050 | 0.70 | 0.76 | 0.33 |
| Whole video | 0.493 ± 0.059 | 0.565 ± 0.063 | 0.221 ± 0.049 | 0.61 | 0.69 | 0.32 |

## PutOnQuiltCover

26 of 40 annotated videos; reference `H52vAkIp80A` against 25 others.

| Step | Histograms before | Histograms after | CLIP | Hist. cutoff before | Hist. cutoff after | CLIP cutoff |
| --- | --- | --- | --- | --- | --- | --- |
| put nicely and align the quilt and the cover | 0.492 ± 0.085 | 0.344 ± 0.123 | 0.277 ± 0.059 | 0.66 | 0.59 | 0.40 |
| roll the quilt cover and the quilt together | 0.458 ± 0.075 | 0.315 ± 0.106 | 0.258 ± 0.060 | 0.61 | 0.53 | 0.38 |
| take out the quilt cover from another side | 0.452 ± 0.056 | 0.304 ± 0.112 | 0.264 ± 0.053 | 0.56 | 0.53 | 0.37 |
| put and arrange nicely | 0.453 ± 0.053 | 0.343 ± 0.093 | 0.323 ± 0.059 | 0.56 | 0.53 | 0.44 |
| Whole video | 0.454 ± 0.050 | 0.322 ± 0.089 | 0.284 ± 0.050 | 0.55 | 0.50 | 0.38 |
