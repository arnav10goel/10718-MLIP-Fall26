# Results

All numbers use CLIP ViT-B/32 frame vectors at 5 fps. "Correct" means the same steps, in the same order, once each, as the task's most common COIN sequence. Screening used up to 10 correct videos per task (smallest files first); BoilNoodles and MakeBurger reuse the 8 already downloaded. Natural deviants were counted, not downloaded or scored.

## Task screen

- **Within-video gap**: in one video, how much further apart frames of *different* steps are than frames of the *same* step (mean cosine distance).
- **Cross-video gap**: the same, comparing frames from two *different* videos. This is what a reference from another kitchen can see.
- **Skip z**: how far a skipped step's cost moves, in standard deviations of correct videos. The cutoff is z = 2.
- **Skip / swap caught**: synthetic deviants where every changed step is above the cutoff (absolute DTW cost).
- **False alarms**: held-out correct videos with any step above the cutoff.
- **Text accuracy**: frames whose closest CLIP text vector is their own step name.

| Task | Steps | Correct of annotated (on mirror) | Natural deviants | Used | Within-video gap | Cross-video gap | Cost sd | Skip z | Skip caught | Swap caught | False alarms | Text accuracy (chance) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MakeBurger | 3 | 20 of 88 (76) | 24 | 8 | 0.172 | 0.042 | 0.045 | 0.17 | 8% of 24 | 0% of 16 | 0% of 1 | 0.61 (0.33) |
| ChangeTonerCartridge | 4 | 48 of 92 (81) | 15 | 10 | 0.094 | 0.016 | 0.035 | 0.41 | 12% of 40 | 3% of 30 | 0% of 2 | 0.39 (0.25) |
| MakeOrangeJuice | 3 | 23 of 54 (45) | 10 | 10 | 0.126 | 0.015 | 0.026 | 0.22 | 17% of 30 | 10% of 20 | 25% of 4 | 0.34 (0.33) |
| CookOmelet | 3 | 20 of 93 (83) | 11 | 9 | 0.089 | 0.015 | 0.033 | -0.38 | 0% of 27 | 0% of 18 | 0% of 5 | 0.28 (0.33) |
| ReplaceSIMCard | 3 | 48 of 86 (82) | 11 | 10 | 0.081 | 0.011 | 0.028 | 0.34 | 7% of 30 | 0% of 20 | 50% of 2 | 0.52 (0.33) |
| MakeStrawberrySmoothie | 3 | 17 of 97 (82) | 20 | 10 | 0.092 | 0.009 | 0.037 | -0.01 | 7% of 30 | 0% of 20 | 33% of 3 | 0.29 (0.33) |
| MakePickles | 3 | 36 of 85 (78) | 16 | 10 | 0.106 | 0.005 | 0.034 | 0.20 | 10% of 30 | 5% of 20 | 33% of 3 | 0.45 (0.33) |
| MakeMatchaTea | 2 | 64 of 102 (93) | 0 | 10 | 0.067 | 0.001 | 0.036 | 0.31 | 15% of 20 | 10% of 10 | 100% of 2 | 0.33 (0.50) |
| Transplant | 3 | 24 of 91 (76) | 19 | 10 | 0.063 | 0.001 | 0.040 | 0.10 | 7% of 30 | 10% of 20 | 50% of 2 | 0.27 (0.33) |
| BoilNoodles | 2 | 32 of 97 (80) | 0 | 8 | 0.113 | 0.000 | 0.042 | 0.29 | 12% of 16 | 12% of 8 | 33% of 3 | 0.55 (0.50) |

## What this says

1. **No task shows a skipped step above the cutoff when the reference comes from another kitchen.** Skip z is about 0 to 0.4 everywhere, against a cutoff of 2. That includes ReplaceSIMCard, the repo's worked example.
2. **Steps do look different inside one video**, but across videos the gap shrinks to 0.001 to 0.04, below the spread between correct videos (cost sd 0.02 to 0.06). Whole-frame CLIP vectors mostly encode the kitchen, not the action.
3. **Other measures did not rescue it.** Subtracting each video's mean step cost, matching frames to mean step vectors, and matching frames to CLIP text vectors of the step names all left skipped steps under the cutoff. Text vectors beat chance clearly only for MakeBurger and ReplaceSIMCard.
4. **The cutoffs are noisy.** With 4 to 8 references, correct held-out videos often land over the cutoff. More references would steady it, but cannot move a z of 0.3 to 2.

## For the team's setup

The team records the reference and the live attempt in the same kitchen with the same camera, which removes most of the kitchen offset. The within-video gap is the closer guide there. MakeBurger has the largest gap on both measures and the best text accuracy, so it is the best cooking pick. The real test needs our own recordings: 3 to 4 correct runs plus deliberate skips and swaps, same kitchen. COIN cannot stand in for that.

Per-video numbers: `runs/<Task>_clip_<tag>.json`. Text test: `runs/text_steps.json`.
