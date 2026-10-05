"""Turn the runner reports in runs/ into results.md and results.json.

For each task and encoder: per-step mean, sd, and cutoff (mean + 2 sd) of the
DTW cost between the reference and each other correct video.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
TASKS = (("BoilNoodles", "boilnoodles"), ("MakeBurger", "makeburger"))
ENCODERS = ("classical", "clip")


def stats(values: list[float]) -> dict:
    array = np.array(values, dtype=np.float64)
    mean, sd = float(array.mean()), float(array.std())
    return {"n": len(values), "mean": mean, "sd": sd, "cutoff": mean + 2 * sd}


def main() -> None:
    results: dict = {}
    lines = ["# Results", ""]
    for task, short in TASKS:
        reports = {
            enc: json.loads((EXPERIMENT / "runs" / f"{short}_{enc}.json").read_text())
            for enc in ENCODERS
        }
        first = reports["classical"]
        steps = [step["label"] for step in first["scoring"]["steps"]]
        lines += [
            f"## {task}",
            "",
            f"Reference `{first['reference_id']}` against "
            f"{first['summary']['comparisons_scored']} other correct videos "
            f"({first['summary']['comparison_failures']} failures).",
            "",
            "| Step | Histograms | CLIP | Histogram cutoff | CLIP cutoff |",
            "| --- | --- | --- | --- | --- |",
        ]
        results[task] = {"reference_id": first["reference_id"], "steps": {}}
        rows = [(label, i) for i, label in enumerate(steps)] + [("Whole video", None)]
        for label, index in rows:
            cell = {}
            for enc in ENCODERS:
                scores = reports[enc]["scoring"]["scores"]
                if index is None:
                    values = [s["global_distance"] for s in scores]
                else:
                    values = [s["step_distances"][index] for s in scores
                              if s["step_distances"][index] is not None]
                cell[enc] = stats(values)
            results[task]["steps"][label] = cell
            h, c = cell["classical"], cell["clip"]
            lines.append(
                f"| {label} | {h['mean']:.3f} ± {h['sd']:.3f} | "
                f"{c['mean']:.3f} ± {c['sd']:.3f} | {h['cutoff']:.2f} | {c['cutoff']:.2f} |"
            )
        lines += ["", "Per video (whole-video cost):", "",
                  "| Video | Split | Histograms | CLIP |", "| --- | --- | --- | --- |"]
        clip_by_id = {s["video_id"]: s for s in reports["clip"]["scoring"]["scores"]}
        for s in reports["classical"]["scoring"]["scores"]:
            lines.append(
                f"| `{s['video_id']}` | {s['split']} | {s['global_distance']:.3f} | "
                f"{clip_by_id[s['video_id']]['global_distance']:.3f} |"
            )
        lines.append("")
    (EXPERIMENT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (EXPERIMENT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
