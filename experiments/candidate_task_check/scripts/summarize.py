"""Turn the runner reports in runs/ into results.md, in the form of the ReplaceSIMCard table.

Per step: mean ± sd of the DTW cost between the reference and every other correct
video, and cutoff = mean + 2 sd (population sd, as in scripts/classical_pairwise_dtw.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
TASKS = ("UseRiceCookerToCookRice", "MakePaperWindMill", "PutOnQuiltCover")


def stats(values):
    a = np.array(values, dtype=np.float64)
    return a.mean(), a.std(), a.mean() + 2 * a.std()


def main() -> None:
    lines = ["# Results", ""]
    for task in TASKS:
        if not all((EXPERIMENT / "runs" / f"{task}_{enc}.json").exists() for enc in ("classical", "clip")):
            continue
        reports = {
            enc: json.loads((EXPERIMENT / "runs" / f"{task}_{enc}.json").read_text())
            for enc in ("classical", "clip")
        }
        r = reports["clip"]
        s = r["summary"]
        steps = [step["label"] for step in r["scoring"]["steps"]]
        lines += [
            f"## {task}", "",
            f"{s['matching_training'] + s['matching_testing']} of {s['annotated_videos']} annotated "
            f"videos share these {len(steps)} steps, once each, in this order "
            f"({s['usable_media']} decoded cleanly on disk). Reference `{r['reference_id']}` "
            f"against {s['comparisons_scored']} others "
            f"({s['comparison_failures']} failures).", "",
        ]
        lines += [f"{i}. {label}" for i, label in enumerate(steps, 1)] + [""]
        lines += [
            "| Step | Histograms | CLIP | Histogram cutoff | CLIP cutoff |",
            "| --- | --- | --- | --- | --- |",
        ]
        for index, label in enumerate(steps + ["Whole video"]):
            cells = []
            for enc in ("classical", "clip"):
                scores = reports[enc]["scoring"]["scores"]
                if label == "Whole video":
                    values = [x["global_distance"] for x in scores]
                else:
                    values = [x["step_distances"][index] for x in scores
                              if x["step_distances"][index] is not None]
                cells.append(stats(values))
            (hm, hs, hc), (cm, cs, cc) = cells
            lines.append(f"| {label} | {hm:.3f} ± {hs:.3f} | {cm:.3f} ± {cs:.3f} | {hc:.2f} | {cc:.2f} |")
        lines.append("")
    (EXPERIMENT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
