# 10718-MLIP-Fall26

Repository for the 10718 course project.

## Goal

A test pair is one full video of a task done correctly and one cropped video of someone doing it. The crop should match the part of the full video it actually covers. If that match is much worse than matches between correct videos, that step was done wrong.

The scripts do not score a crop yet. They build the normal range: how far apart two correct videos of the same task look. A crop is a deviation when its cost on a step is above that range.

## How similarity is measured

1. Download the task videos and the COIN annotations.
2. Keep only videos that share one step sequence: the same steps, once each, in the same order.
3. Sample each annotated region at 5 fps. A frame becomes either a histogram descriptor or a CLIP embedding. Both are L2-normalized.
4. DTW aligns the two sequences. It can match one frame to several frames, so a slower or longer clip still lines up. The local cost of pairing frame \(i\) with frame \(j\) is the cosine distance \(c(i,j) = 1 - f_i \cdot g_j\).
5. The reported distance is the average of \(c(i,j)\) along the alignment path, inside each step. The raw DTW sum is not used, because it grows just because the path is longer.

| Script | Features |
| --- | --- |
| `scripts/classical_pairwise_dtw.py` | 16-bin hue histogram, 8-bin saturation histogram, and a 4×4 grid of 9-bin gradient-orientation histograms, from a 64×64 frame |
| `scripts/ml_pairwise_dtw.py` | 512-D CLIP ViT-B/32 embedding (OpenAI weights), from a 224×224 frame |

## What the values mean

- **0** means the aligned frames point the same direction.
- **2** means they point in opposite directions.
- The **mean** is the typical cost across the correct pairs.
- The **standard deviation** is how much that cost varies among correct pairs.
- The **cutoff** is mean + 2 standard deviations. A step whose cost is above that is a deviation.

Correct videos are not at 0. The phone, the hands, and the camera differ. CLIP puts correct videos closer together than the histograms do.

## Run

Set the task name in `TASKS` inside `scripts/coin_tasks.py`. The name must match the `class` field in COIN. From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_coin.py
.venv/bin/python scripts/prepare_coin.py
.venv/bin/python scripts/classical_pairwise_dtw.py
.venv/bin/python scripts/ml_pairwise_dtw.py
```

`download_coin.py` saves videos to `data/videos/<task>/<youtube_id>.mp4`. `prepare_coin.py` writes `data/prepared/manifest.jsonl` with the official `training` and `testing` split. The DTW scripts reuse cached features on a re-run.

Each DTW script writes a summary JSON (mean, standard deviation, median, 10th and 90th percentiles, per step and for the whole video) and a square distance matrix. Rows of the matrix follow `video_ids` in the summary. The diagonal is 0. Use the per-step blocks.

## Example: ReplaceSIMCard

The run so far is ReplaceSIMCard, task id 141. The scripts keep the 48 downloaded videos whose annotation is exactly these three steps, once each, in this order:

1. use the needle to open the SIM card slot
2. put the SIM card into the SIM card slot
3. press the SIM card slot back

Those clips are listed in `replace_simcard_videos.json` and stored in `data/videos/ReplaceSIMCard/used/`. There are 48 choose 2 = 1128 pairs. The summaries are `data/prepared/sim_classical_dtw_summary.json` and `data/prepared/sim_clip_dtw_summary.json`.

| Step | Histograms | CLIP | Histogram cutoff | CLIP cutoff |
| --- | --- | --- | --- | --- |
| Open the slot | 0.537 ± 0.101 | 0.196 ± 0.044 | 0.74 | 0.28 |
| Put the SIM in | 0.486 ± 0.088 | 0.197 ± 0.050 | 0.66 | 0.30 |
| Press the tray back | 0.526 ± 0.097 | 0.202 ± 0.047 | 0.72 | 0.30 |
| Whole video | 0.504 ± 0.077 | 0.200 ± 0.039 | 0.66 | 0.28 |

A cropped insert that costs more than about 0.30 in CLIP, or 0.66 with the histograms, is outside the normal range for "put the SIM in." A backwards card should push that step over the cutoff. A tray left halfway out should push "press the SIM card slot back" over 0.30 in CLIP, or 0.72 with the histograms.
