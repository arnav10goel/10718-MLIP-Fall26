# Histogram feature fix

## Question

The histogram feature in `scripts/classical_pairwise_dtw.py::frame_feature` joined hue counts, saturation counts and gradient-orientation sums, then normalized once. Gradient sums are far larger, so they held ~99% of the vector and colour barely counted (see `experiments/candidate_task_check/README.md`). After normalizing each part separately, how do the histogram step costs change on the five tasks?

## Method

- **Fix** (branch `fix-histogram-normalization`): L2-normalize hue, saturation and gradient blocks separately, then join and normalize. Each block now holds a third of the vector's energy.
- **Tasks**: ReplaceSIMCard (the repo example), MakeStrawberrySmoothie, UseRiceCookerToCookRice, MakePaperWindMill, PutOnQuiltCover. Every COIN video with the task's most common exact step sequence.
- **ReplaceSIMCard** reruns the repo's own all-pairs script, `scripts/classical_pairwise_dtw.py` (48 videos, 1128 pairs), before and after the fix, so the "before" run also checks that we reproduce the README table.
- **The other four** use the team's runner `scripts.run_coin_reference` (one reference against every other matching video), as in `experiments/candidate_task_check/`. Smoothie uses the README's example reference `-3C-VGhs2mo`.
- CLIP does not use this feature, so CLIP numbers do not change; smoothie gets a CLIP run because it had none.

## Result

All 10 runs finished with 0 failures. The *before* SIM run reproduces the top-level README table exactly (0.537 ± 0.101, 0.486 ± 0.088, 0.526 ± 0.097, whole video 0.504 ± 0.077), which confirms the setup.

After the fix, histogram costs fall by 0.08 to 0.15 on four tasks (SIM, smoothie, rice cooker, duvet), and their spread grows a little: colour now counts, and within one task the colours are often alike. The windmill goes the other way (whole video 0.493 → 0.565): its videos use paper of many colours, which the old feature ignored. Cutoffs move by up to 0.10. CLIP is unchanged. Whether the fixed feature tracks steps better is not tested here; it does now match its description. Full tables in `results.md`.

If this branch is merged, the ReplaceSIMCard histogram numbers in the top-level README need replacing with the *after* column.

## Files

- `scripts/download.py`: SIM and smoothie videos
- `scripts/run_sim_and_smoothie.sh <before|after>`, `scripts/run_after.sh`: runs (the *before* run was made with the fix stashed)
- `scripts/summarize.py`: writes `results.md`
- `runs/`: runner reports and copies of the SIM summaries
- `logs/`: one log per run
