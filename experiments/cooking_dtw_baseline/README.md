# Cooking DTW baseline

## Question

Do the repo's two baselines (histogram DTW and CLIP DTW) run end to end on cooking tasks from COIN, and how far apart do correct cooking videos look?

## Method

Two COIN cooking tasks from the proposal shortlist. For each, we keep only videos whose annotated steps match the reference exactly, in the same order.

| Task | Steps | Reference (training) |
| --- | --- | --- |
| BoilNoodles | 1. pour the noodles into the water and stir · 2. pour the cooked noodles | `krNVdmW0Nto` |
| MakeBurger | 1. knead the meat · 2. fry meat · 3. combine meat and bread to make burger | `aogu4SX7ULg` |

We download only the 8 smallest matching videos per task from the Hugging Face mirror (`scripts/download_subset.py`), not the full 3.8 GB task folders. Each task then runs through `scripts.run_coin_reference` twice: once with histogram features, once with CLIP. Each run aligns the reference against the other 7 videos and reports a per-step cost. `scripts/summarize.py` turns the four reports into a mean ± sd table with a mean + 2 sd cutoff, in the same form as the ReplaceSIMCard table in the top-level README.

Seven comparisons per task is small. Treat the cutoffs as a smoke test of the pipeline, not as calibrated thresholds.

## Result

Both tasks ran end to end with both encoders: 7 comparisons each, 0 failures, no empty steps. Full tables in `results.md`.

| Task | Whole-video cost, histograms | Whole-video cost, CLIP |
| --- | --- | --- |
| BoilNoodles | 0.469 ± 0.077 (cutoff 0.62) | 0.261 ± 0.062 (cutoff 0.38) |
| MakeBurger | 0.423 ± 0.033 (cutoff 0.49) | 0.274 ± 0.042 (cutoff 0.36) |

As with ReplaceSIMCard, CLIP puts correct videos closer together than the histograms do. Cooking costs under CLIP (about 0.26 to 0.27) sit above the SIM-card ones (0.20), which is consistent with kitchens varying more than close-ups of a phone. That comparison is loose: the SIM numbers average all 1,128 pairs of 48 videos, while these use one reference against 7. Scoring a wrong execution against these cutoffs is not built yet; this run only sets the normal range.

Rerun: `bash experiments/cooking_dtw_baseline/scripts/run_all.sh` (the runner refuses to overwrite, so move old `runs/*.json` first), then `scripts/summarize.py`.

## Files

- `scripts/download_subset.py`: downloads the chosen videos into `data/videos/<Task>/`
- `scripts/summarize.py`: reads `runs/*.json`, writes `results.md`
- `runs/<task>_<encoder>.json`: runner reports
- `logs/`: one log per download and per run
