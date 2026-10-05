# Deviation detection

## Question

When someone skips a step or does steps out of order, does the step cost rise above the normal range (mean + 2 sd of correct videos)? Does it rise on the right step and stay normal on the others?

## Method

**Correct videos.** For a task, "correct" means the videos whose COIN step labels match the task's most common sequence exactly: same steps, same order, once each. `scripts/common.py` asserts this.

**Features.** Each video's task section is sampled at 5 fps. Each frame becomes a CLIP ViT-B/32 vector or a histogram vector, as in the repo baselines. Features are cached in `data/features/<encoder>/<id>.npz`.

**Per-step cost.** We line up a query video with a reference video using full DTW (both ends pinned), with the cost of a frame pair set to 1 minus cosine similarity. The cost of step k is the average cost along the path over the *reference's* frames in step k. We use only the reference side, because a deviant video has no frames for a step it skipped.

**Many references.** Every training-split correct video serves as a reference. A query's cost for step k is the median over all references, leaving out the query itself.

**Normal range.** We score every training correct video against the other training correct videos (leave-one-out). Per step, cutoff = mean + 2 sd of those scores.

**Deviants.**
- *Synthetic* (`scripts/score.py`): taken from each correct video. `skip_k` removes step k's frames. `swap_k` swaps the frame blocks of steps k and k+1. The video is otherwise unchanged, so only the step differs.
- *Natural* (`scripts/score.py`): COIN videos of the same task whose labels lack a step or put the steps in another order. Caveat: a missing COIN label almost always means the step was not *filmed*, not that the cook did it wrong. To the camera they look the same, so it is a fair stand-in, not a real error.

**What counts.**
- *Flagged on the right step*: every step that was skipped or moved is above its cutoff.
- *Localized*: flagged on the right step, and every other step stays under its cutoff.
- *False alarm*: a held-out (testing-split) correct video with any step above its cutoff.

## Result

Negative on COIN. Across 10 tasks, a skipped step moves its cost by about z = 0 to 0.4, never near the cutoff of z = 2, when the reference video comes from another kitchen. Swaps fare no better. That includes ReplaceSIMCard, the repo's worked example. Steps do look different within one video (gap 0.06 to 0.17), but across videos the gap is 0.001 to 0.04, below the spread between correct videos. Relative costs, mean-step matching and CLIP text matching did not fix it. MakeBurger (20 of 88 annotated) separates steps best, so it is the cooking pick for the team's own same-kitchen recordings. Full table and caveats in `results.md`.

Steps tried, in order: absolute DTW cost on 16 local videos → relative cost and mean-step matching → screen of 8 more tasks with 10 videos each → within- vs cross-video gap → CLIP text matching (`scripts/text_steps.py`).

## Files

- `scripts/common.py`: cohorts, feature extraction and cache, DTW step cost
- `scripts/score.py`: normal range, synthetic and natural deviants, writes `runs/<task>_<encoder>.json`
- `scripts/report.py`: writes `results.md`
- `scripts/text_steps.py`: zero-shot step recognition with CLIP text vectors
- `scripts/run.sh`, `scripts/screen.sh`: run one task / screen the candidate tasks
- `logs/`: one log per run
