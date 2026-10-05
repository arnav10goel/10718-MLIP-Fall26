# Candidate task check: rice cooker, paper windmill, duvet cover

## Question

For three tasks the team could record (COIN `UseRiceCookerToCookRice`, `MakePaperWindMill`, `PutOnQuiltCover`): how many COIN videos share one exact step sequence, do those videos look usable as references, and what is the normal range of step costs between correct videos, as in the repo's ReplaceSIMCard example?

## Method

1. Keep the videos whose COIN step labels are the task's most common sequence: same steps, once each, same order. Download all of them from the Hugging Face mirror (`scripts/download.py`).
2. Make a contact sheet per task: one frame from the middle of each step, one row per video (`scripts/contact_sheets.py`). Look at them to judge camera view, hands, clutter and overlays, and pick a reference.
3. Run the collaborators' runner, `scripts.run_coin_reference`, unchanged, with histogram and CLIP features. It aligns the reference against every other matching video with DTW.
4. Summarize per step: mean ± sd, and cutoff = mean + 2 sd (`scripts/summarize.py`), in the form of the ReplaceSIMCard table.

The runner compares one reference against each other video. The SIM table averages all pairs. So the spreads are comparable in kind, not exactly.

## Result

See `results.md`.

## Files

- `scripts/`: download, contact sheets, run, summarize
- `contact_sheets/<task>.jpg`: frames per step per video
- `runs/<task>_<encoder>.json`: runner reports
- `logs/`: one log per step

## Metric check (`scripts/check_metric.py`, output `runs/metric_checks.json`)

- **DTW code is correct.** `dtw_pair` matches a slow plain DTW on 200 random cases (max error 5e-15). Frame counts match task length × 5 fps within 0.2%. Table cells recompute from the reports.
- **It separates tasks.** A reference is closer to its own task's videos than to other tasks' (CLIP whole-video cost, e.g. SIM card 0.20 vs 0.39, matcha 0.27 vs 0.35).
- **The per-step numbers are mostly about the pair, not the step.** Across videos, a reference step is only 0.000 to 0.04 closer to the *same* step than to a *different* step. In the per-step table, 50 to 84% of the spread in step costs is explained by how alike the two videos are overall; a pair's step costs move together (correlation 0.4 to 0.8). So "cutoff for step k" is close to "cutoff for this pair of videos".
- **The histogram feature is almost all edges.** Gradient-orientation bins hold 99.0 to 99.4% of the vector's energy (29 frames from 10 videos per task, all three tasks); hue and saturation together hold under 1%. The raw counts are on different scales and are joined before one normalization. The "Histograms" column is in effect an edge descriptor; colour barely counts.
- **It needs the comparison video's step labels.** The runner reports `uses_comparison_annotations: true`, and `step_distance` averages the reference side and the comparison side. A live user has no labels, so the per-step costs cannot be computed on a live stream as written.
- **"Correct" means correct by COIN labels, not the same procedure.** About 10 of the 31 rice cooker videos use a pressure cooker, a pot on the stove or a microwave; a few duvet videos come from TV studios. These widen the normal range. A hand-picked set would tighten it.
- **Histogram costs cannot reach 2.** With non-negative features, cosine distance stays in [0, 1], so the top-level README's "2 means opposite" only applies to CLIP in principle.
