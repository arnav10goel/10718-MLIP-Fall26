# Offline VLM alignment test

## Question

Given an annotated reference video and a whole unlabelled execution video, does Gemini (`scripts/run_offline_vlm.py`) say, for each reference step, whether and where it happened, and does it catch a skipped step and a swapped pair?

## Method

- **Reference:** COIN `MakePaperWindMill` video `TULczOB5joI` (training, overhead camera). Its COIN annotation becomes the checklist (4 steps). The clip is cut to its COIN task section; audio is removed.
- **Executions** (`scripts/make_cases.py`), all from COIN `e7p9QHRmd4k`, another correct windmill video, cut to its task section with audio removed so narration cannot reveal the steps:
  - `correct`: unchanged.
  - `skip_s002`: step 2 (cut along the edges) removed.
  - `swap_s003_s004`: steps 3 and 4 swapped.
  Both videos share the same 4 COIN steps in the same order (25 of 106 annotated windmill videos do).
- Each case has a private answer key (`cases/<case>_truth.json`) with expected status and execution times per step. It is never sent to the model.
- `scripts/score.py` compares each report with its answer key.

## Result

See `results.md`.

## Files

- `scripts/make_cases.py`, `scripts/run_cases.sh`, `scripts/score.py`
- `cases/`: execution clips and answer keys (clips are gitignored by size, see below)
- `runs/<case>.json`: run_offline_vlm reports; `runs/<case>_clips/`: the exact clips sent
- `logs/`
