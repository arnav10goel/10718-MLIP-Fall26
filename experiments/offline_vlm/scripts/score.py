"""Compare run_offline_vlm reports with the private answer keys; write results.md.

Usage (repo root): python experiments/offline_vlm/scripts/score.py <tag> [cases subfolder]
"""

import json
import sys
from pathlib import Path

E = Path(__file__).resolve().parents[1]
CASES = ("correct", "skip_s002", "swap_s003_s004")


def overlap(a, b):
    """Intersection over union of two [start, end] intervals; None if either is missing."""
    if None in a or None in b:
        return None
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main(tag: str, subfolder: str = "") -> None:
    cases_dir = E / "cases" / subfolder
    lines = [f"# Results: {tag}", "", f"Run tag `{tag}`; cases in `cases/{subfolder}`.", ""]
    totals = {"steps": 0, "status_right": 0, "deviation_cases": 0, "deviation_caught": 0,
              "false_deviation_cases": 0}
    for case in CASES:
        truth = json.loads((cases_dir / f"{case}_truth.json").read_text())
        report = json.loads((E / "runs" / f"{case}_{tag}.json").read_text())
        lines += [f"## {case}", "", f"Status: `{report['status']}`; model `{report['requested_model']}`."]
        if (report.get("result") or {}).get("warnings"):
            lines.append("Timing warnings: " + "; ".join(report["result"]["warnings"]))
        result = report.get("result") or {}
        prediction = result.get("prediction")
        if prediction is None:
            lines += [f"No valid prediction ({result.get('validation_error') or report.get('error')}).", ""]
            continue
        by_id = {s["reference_step_id"]: s for s in prediction["steps"]}
        lines += ["", "| Step | Expected | Gemini | Expected time (s) | Gemini time (s) | Overlap |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for t in truth["steps"]:
            p = by_id[t["reference_step_id"]]
            right = p["status"] == t["status"]
            totals["steps"] += 1
            totals["status_right"] += right
            iou = overlap((t["execution_start_s"], t["execution_end_s"]),
                          (p["execution_start_s"], p["execution_end_s"]))

            def fmt(a, b):
                return "–" if a is None else f"{a:.0f}–{b:.0f}"

            lines.append(
                f"| {t['reference_step_id']} {t['label']} | {t['status']} | "
                f"{p['status']}{'' if right else ' ✗'} | {fmt(t['execution_start_s'], t['execution_end_s'])} | "
                f"{fmt(p['execution_start_s'], p['execution_end_s'])} | {'–' if iou is None else f'{iou:.2f}'} |"
            )
        expected_bad = {t["reference_step_id"] for t in truth["steps"] if t["status"] != "done"}
        flagged = {d["reference_step_id"] for d in prediction["deviations"]}
        if expected_bad:
            totals["deviation_cases"] += 1
            caught = bool(expected_bad & flagged)
            totals["deviation_caught"] += caught
            lines.append(f"\nDeviation on {sorted(expected_bad)} caught: {'yes' if caught else 'no'}.")
        elif prediction["deviations"]:
            totals["false_deviation_cases"] += 1
            lines.append("\nFalse alarm: deviations reported on a correct execution.")
        lines += ["", "Deviations reported:"] + [
            f"- {d['type']} on {d['reference_step_id']} at {d['execution_time_s']}: {d['message']}"
            for d in prediction["deviations"]
        ] + ["", f"Summary: {prediction['summary']}", ""]
    lines[3:3] = [
        f"Steps with the right status: {totals['status_right']} of {totals['steps']}. "
        f"Deviation cases caught: {totals['deviation_caught']} of {totals['deviation_cases']}. "
        f"Correct cases with a false alarm: {totals['false_deviation_cases']} of 1.", "",
    ]
    (E / f"results_{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
