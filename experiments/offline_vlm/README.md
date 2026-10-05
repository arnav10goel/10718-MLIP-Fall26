# Offline VLM alignment test

## Question

Given an annotated reference video and a whole unlabelled execution video, does Gemini (`scripts/run_offline_vlm.py`) say, for each reference step, whether and where it happened, and does it catch a skipped step and a swapped pair?

## Method

- **Reference:** COIN `MakePaperWindMill` video `TULczOB5joI` (training, overhead camera). Its COIN annotation becomes the checklist (4 steps). The clip is cut to its COIN task section; audio is removed.
- **Executions** (`scripts/make_cases.py <id>`), from another correct windmill video:
  first `e7p9QHRmd4k` (cases in `cases/`), then `I33S4c7zrcg` (cases in `cases/I33S4c7zrcg/`), which folds by hand like the reference. `e7p9QHRmd4k` rules guide lines instead of folding, so Gemini fairly called its step 1 different. Each is cut to its task section with audio removed so narration cannot reveal the steps:
  - `correct`: unchanged.
  - `skip_s002`: step 2 (cut along the edges) removed.
  - `swap_s003_s004`: steps 3 and 4 swapped.
  Both videos share the same 4 COIN steps in the same order (25 of 106 annotated windmill videos do).
- Each case has a private answer key (`cases/<case>_truth.json`) with expected status and execution times per step. It is never sent to the model.
- `scripts/score.py` compares each report with its answer key.

## Result

Single runs, three cases each, on execution `I33S4c7zrcg` (full tables: `results_flash_I33.md`, `results_lite_I33.md`).

| Model | Steps with the right status | Skipped step caught | Swapped steps caught | False alarm on the correct video |
| --- | --- | --- | --- | --- |
| gemini-3.5-flash-lite | 10 of 12 | yes | no | no |
| gemini-3.5-flash | 9 of 12 | no | no | no |

- Both models placed steps 1 and 2 well on the correct video (overlap 0.80 to 0.98 with the true interval) and raised no false alarm.
- Neither caught the swap. Flash also missed the skip.
- The spliced cases are a weak test: cutting out the cutting step still leaves cut paper in every later frame, and the swap shows the finished windmill on its stick before its blades are folded. A real skipped or swapped step needs a real recording.
- Earlier runs on `e7p9QHRmd4k` (`results.md`, tag `live2`) mostly failed the answer checks because of timestamps past the end of the video. Timing problems are now repaired and listed as warnings instead of discarding the answer.
- Response time varied from about 9 s to over 5 minutes per call; one upload failed with a Google server error (500) and was rerun (`runs/correct_lite_I33_server500.json`).

## Files

- `scripts/make_cases.py`, `scripts/run_cases.sh`, `scripts/score.py` (writes `results_<tag>.md`)
- `cases/`: execution clips and answer keys (clips are gitignored by size, see below)
- `runs/<case>.json`: run_offline_vlm reports; `runs/<case>_clips/`: the exact clips sent
- `logs/`
