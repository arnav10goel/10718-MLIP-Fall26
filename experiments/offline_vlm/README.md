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

## Other models (OpenAI-compatible servers, such as GLM or Qwen on Modal)

`scripts/run_offline_vlm.py --backend openai` sends the same prompt, checklist and JSON schema to any OpenAI-compatible chat server and checks the answer the same way. By default each video is sent as JPEG frames (1 fps, at most 256 per video, longer side 640 px), each preceded by its time in seconds. With `--video` the execution is sent as one video instead, for models that take video; the reference stays as frames because Modal's GLM endpoint accepts one video per request.

Setup, once per person. Each person uses their own Modal login, and keys stay in `.env`, which git ignores:

```bash
.venv/bin/pip install -r requirements.txt modal
.venv/bin/modal token new                        # your own Modal account
.venv/bin/modal workspace proxy-tokens create    # prints wk-... and ws-...
echo "VLM_API_KEY=wk-<id>.ws-<secret>" >> .env
```

GLM-5.3-Flash on a Shared Endpoint (billed per token, no GPU start-up, takes video):

```bash
.venv/bin/modal endpoint create --shared --model zai-org/GLM-5.3-Flash   # prints ep-...
.venv/bin/modal endpoint info <ep-id>                                    # prints its URL
.venv/bin/python -m scripts.run_offline_vlm --backend openai --video \
    --model zai-org/GLM-5.3-Flash --max-tokens 32768 --base-url <URL>/v1 \
    --reference ref.mp4 --checklist ref.csv --execution my_attempt.mp4 \
    --output runs/attempt_glm.json
```

Qwen3.5-9B on a Dedicated Endpoint (images only; bills per GPU-second until stopped):

```bash
.venv/bin/modal endpoint create --name guideme-qwen9b --model Qwen/Qwen3.5-9B
.venv/bin/modal endpoint info <ep-id>            # ready once <URL>/v1/models stops returning 503
.venv/bin/python -m scripts.run_offline_vlm --backend openai --model guideme-qwen9b \
    --base-url <URL>/v1 --reference ref.mp4 --checklist ref.csv \
    --execution my_attempt.mp4 --output runs/attempt_qwen9b.json
.venv/bin/modal endpoint stop <ep-id> -y         # stop billing
```

- On a Dedicated Endpoint, `--model` is the endpoint name; `<URL>/v1/models` lists the name to use.
- `--max-tokens` (default 16384) covers thinking and the answer. GLM thought for about 14,000 tokens on a 25 s execution, so give it 32768. `--disable-thinking` turns Qwen's thinking off.
- On a login node, ffmpeg and NumPy fail with `pthread_create failed: Resource temporarily unavailable` when your processes already use most of the per-user thread limit (256 on Babel; VS Code servers use many). Close unused VS Code windows or run on a compute node.

Smoke test on COIN `ReplaceSIMCard` (reference `0b6joFdBwUw`, execution `V_BwPfCh1Z0`), two cuts: the whole correct execution, and a cut in the middle of the insert step.
- GLM-5.3-Flash with `--video`, one run: both right, with step times within about 2 s of the COIN annotation.
- Qwen3.5-9B with frames, two runs: the mid-insert cut was right both times. The whole execution was right once; the other time Qwen left out s003 and flagged a deviation that is not there, so the answer was rejected.

## Files

- `scripts/make_cases.py`, `scripts/run_cases.sh`, `scripts/score.py` (writes `results_<tag>.md`)
- `cases/`: execution clips and answer keys (clips are gitignored by size, see below)
- `runs/<case>.json`: run_offline_vlm reports; `runs/<case>_clips/`: the exact clips sent
- `logs/`
