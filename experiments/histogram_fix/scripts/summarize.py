"""Write results.md: per-step histogram costs before and after the fix, with CLIP alongside.

Per step: mean ± sd (population sd, as in the repo scripts) and cutoff = mean + 2 sd.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
CANDIDATE = EXPERIMENT.parent / "candidate_task_check" / "runs"
RUNS = EXPERIMENT / "runs"

# ReplaceSIMCard CLIP numbers are the repo README's (CLIP does not use the histogram feature).
SIM_CLIP = {
    "use the needle to open the SIM card slot": (0.196, 0.044),
    "put the SIM card into the SIM card slot": (0.197, 0.050),
    "press the SIM card slot back": (0.202, 0.047),
    "Whole video": (0.200, 0.039),
}


def runner_stats(path: Path) -> tuple[dict, dict]:
    """Per-step (mean, sd) from a run_coin_reference report, plus its summary."""
    r = json.loads(path.read_text())
    steps = [s["label"] for s in r["scoring"]["steps"]]
    out = {}
    for i, label in enumerate(steps):
        v = np.array([s["step_distances"][i] for s in r["scoring"]["scores"]
                      if s["step_distances"][i] is not None])
        out[label] = (v.mean(), v.std())
    g = np.array([s["global_distance"] for s in r["scoring"]["scores"]])
    out["Whole video"] = (g.mean(), g.std())
    return out, r


def sim_stats(path: Path) -> dict:
    s = json.loads(path.read_text())
    out = {label: (d["mean"], d["std"]) for label, d in s["steps"].items()}
    out["Whole video"] = (s["global"]["mean"], s["global"]["std"])
    return out


def cell(ms):
    return f"{ms[0]:.3f} ± {ms[1]:.3f}"


def cut(ms):
    return f"{ms[0] + 2 * ms[1]:.2f}"


def table(title: str, intro: str, before: dict, after: dict, clip: dict) -> list[str]:
    lines = [f"## {title}", "", intro, "",
             "| Step | Histograms before | Histograms after | CLIP | Hist. cutoff before | Hist. cutoff after | CLIP cutoff |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label in after:
        lines.append(f"| {label} | {cell(before[label])} | {cell(after[label])} | {cell(clip[label])} | "
                     f"{cut(before[label])} | {cut(after[label])} | {cut(clip[label])} |")
    return lines + [""]


def main() -> None:
    lines = ["# Results", "",
             "Histogram step costs before and after normalizing the hue, saturation and gradient "
             "blocks separately. CLIP is unchanged by the fix and shown for reference.", ""]

    lines += table(
        "ReplaceSIMCard",
        "48 of 86 annotated videos (the 48 in `replace_simcard_videos.json`), all 1128 pairs, "
        "via `scripts/classical_pairwise_dtw.py`. The *before* column reproduces the top-level "
        "README table. CLIP numbers are the README's.",
        sim_stats(RUNS / "sim_classical_before.json"),
        sim_stats(RUNS / "sim_classical_after.json"),
        SIM_CLIP,
    )

    before, r = runner_stats(RUNS / "MakeStrawberrySmoothie_classical_before.json")
    after, _ = runner_stats(RUNS / "MakeStrawberrySmoothie_classical_after.json")
    clip, _ = runner_stats(RUNS / "MakeStrawberrySmoothie_clip_same.json")
    s = r["summary"]
    lines += table(
        "MakeStrawberrySmoothie",
        f"{s['matching_training'] + s['matching_testing']} of {s['annotated_videos']} annotated videos; "
        f"reference `{r['reference_id']}` against {s['comparisons_scored']} others.",
        before, after, clip,
    )

    for task in ("UseRiceCookerToCookRice", "MakePaperWindMill", "PutOnQuiltCover"):
        before, r = runner_stats(CANDIDATE / f"{task}_classical.json")
        after, _ = runner_stats(RUNS / f"{task}_classical_after.json")
        clip, _ = runner_stats(CANDIDATE / f"{task}_clip.json")
        s = r["summary"]
        lines += table(
            task,
            f"{s['matching_training'] + s['matching_testing']} of {s['annotated_videos']} annotated videos; "
            f"reference `{r['reference_id']}` against {s['comparisons_scored']} others.",
            before, after, clip,
        )

    (EXPERIMENT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
