"""Write results.md from runs/*_clip_*.json and runs/text_steps.json.

Usage (repo root): python experiments/deviation_detection/scripts/report.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EXPERIMENT, FEATURE_DIR, label_frames, load_database, row_for  # noqa: E402


def gaps(database, vids, n):
    """Within-video and cross-video step gaps (different-step minus same-step distance)."""
    feats, labels = {}, {}
    for v in vids:
        data = np.load(FEATURE_DIR / "clip" / f"{v}.npz")
        feats[v] = data["features"]
        labels[v] = label_frames(data["times"], row_for(database, v)["steps"])
    groups = {"ws": [], "wd": [], "cs": [], "cd": []}
    for a in vids:
        for b in vids:
            for k in range(n):
                fa = feats[a][labels[a] == k]
                for j in range(n):
                    fb = feats[b][labels[b] == j]
                    if len(fa) and len(fb):
                        key = ("w" if a == b else "c") + ("s" if k == j else "d")
                        groups[key].append(float((1.0 - fa @ fb.T).mean()))
    mean = {k: np.mean(v) for k, v in groups.items()}
    return mean["wd"] - mean["ws"], mean["cd"] - mean["cs"]


def main() -> None:
    database = load_database()
    text = json.loads((EXPERIMENT / "runs" / "text_steps.json").read_text())
    rows = []
    for path in sorted(glob.glob(str(EXPERIMENT / "runs" / "*_clip_*.json"))):
        r = json.loads(Path(path).read_text())
        n = len(r["steps"])
        held = r["held_out_correct"]
        vids = r["references"] + [h["video_id"] for h in held]
        within, cross = gaps(database, vids, n)
        skip = [x for x in r["synthetic"] if x["kind"] == "skip"]
        swap = [x for x in r["synthetic"] if x["kind"] == "swap"]
        rows.append({
            "task": r["task"], "n": n, "c": r["counts"], "within": within, "cross": cross,
            "sd": float(np.mean(r["normal_range"]["absolute"]["sd"])),
            "zskip": np.mean([x["z_absolute"][x["expected_steps"][0]] for x in skip]),
            "skip": np.mean([x["judge_absolute"]["right_step"] for x in skip]),
            "swap": np.mean([x["judge_absolute"]["right_step"] for x in swap]),
            "fa": np.mean([x["judge_absolute"]["any_flag"] for x in held]) if held else float("nan"),
            "nh": len(held), "nskip": len(skip), "nswap": len(swap),
            "tacc": text[r["task"]]["frame_accuracy"], "chance": text[r["task"]]["chance"],
        })
    rows.sort(key=lambda r: -r["cross"])

    lines = [
        "# Results", "",
        "All numbers use CLIP ViT-B/32 frame vectors at 5 fps. \"Correct\" means the same steps, "
        "in the same order, once each, as the task's most common COIN sequence. Screening used up "
        "to 10 correct videos per task (smallest files first); BoilNoodles and MakeBurger reuse "
        "the 8 already downloaded. Natural deviants were counted, not downloaded or scored.", "",
        "## Task screen", "",
        "- **Within-video gap**: in one video, how much further apart frames of *different* steps "
        "are than frames of the *same* step (mean cosine distance).",
        "- **Cross-video gap**: the same, comparing frames from two *different* videos. This is what "
        "a reference from another kitchen can see.",
        "- **Skip z**: how far a skipped step's cost moves, in standard deviations of correct videos. "
        "The cutoff is z = 2.",
        "- **Skip / swap caught**: synthetic deviants where every changed step is above the cutoff "
        "(absolute DTW cost).",
        "- **False alarms**: held-out correct videos with any step above the cutoff.",
        "- **Text accuracy**: frames whose closest CLIP text vector is their own step name.", "",
        "| Task | Steps | Correct of annotated (on mirror) | Natural deviants | Used | "
        "Within-video gap | Cross-video gap | Cost sd | Skip z | Skip caught | Swap caught | "
        "False alarms | Text accuracy (chance) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        c = r["c"]
        lines.append(
            f"| {r['task']} | {r['n']} | {c['correct_on_mirror']} of {c['annotated']} "
            f"({c['on_mirror']}) | {c['natural_deviants_on_mirror']} | {c['correct_used']} | "
            f"{r['within']:.3f} | {r['cross']:.3f} | {r['sd']:.3f} | {r['zskip']:.2f} | "
            f"{r['skip']:.0%} of {r['nskip']} | {r['swap']:.0%} of {r['nswap']} | "
            f"{r['fa']:.0%} of {r['nh']} | {r['tacc']:.2f} ({r['chance']:.2f}) |"
        )
    lines += [
        "", "## What this says", "",
        "1. **No task shows a skipped step above the cutoff when the reference comes from another "
        "kitchen.** Skip z is about 0 to 0.4 everywhere, against a cutoff of 2. That includes "
        "ReplaceSIMCard, the repo's worked example.",
        "2. **Steps do look different inside one video**, but across videos the gap shrinks to "
        "0.001 to 0.04, below the spread between correct videos (cost sd 0.02 to 0.06). Whole-frame "
        "CLIP vectors mostly encode the kitchen, not the action.",
        "3. **Other measures did not rescue it.** Subtracting each video's mean step cost, matching "
        "frames to mean step vectors, and matching frames to CLIP text vectors of the step names "
        "all left skipped steps under the cutoff. Text vectors beat chance clearly only for "
        "MakeBurger and ReplaceSIMCard.",
        "4. **The cutoffs are noisy.** With 4 to 8 references, correct held-out videos often land "
        "over the cutoff. More references would steady it, but cannot move a z of 0.3 to 2.", "",
        "## For the team's setup", "",
        "The team records the reference and the live attempt in the same kitchen with the same "
        "camera, which removes most of the kitchen offset. The within-video gap is the closer "
        "guide there. MakeBurger has the largest gap on both measures and the best text accuracy, "
        "so it is the best cooking pick. The real test needs our own recordings: 3 to 4 correct "
        "runs plus deliberate skips and swaps, same kitchen. COIN cannot stand in for that.", "",
        "Per-video numbers: `runs/<Task>_clip_<tag>.json`. Text test: `runs/text_steps.json`.", "",
    ]
    (EXPERIMENT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines[13:15 + len(rows)]))


if __name__ == "__main__":
    main()
