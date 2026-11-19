"""Histogram DTW on phone recordings for tasks whose cutoff is already saved.

Each incorrect recording is cut in the middle of its mistake. A skipped step
has no frames, so the cut is the middle of the next step that was performed.
Each correct recording is cut in the middle of that same step. The reference
is one COIN video. A step is a deviation when its mean cosine distance is
above the whole-video cutoff.

    .venv/bin/python scripts/recorded_histogram.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_pairwise_dtw import FPS, open_end_alignment  # noqa: E402
from coin_tasks import COIN_JSON, PREPARED_DIR, REPO_ROOT  # noqa: E402
from score_recordings import _dtw_segment_distance  # noqa: E402
from smoothie_recorded import features_for, frame_labels  # noqa: E402

# correct_step is the step that is wrong in the incorrect recordings.
TASKS = (
    {
        "name": "toilet",
        "module": "clean_toilet",
        "reference_id": "d2gMl6jwEEc",
        "correct_step": 1,
    },
    {
        "name": "battery",
        "module": "replace_battery",
        "reference_id": "ROGBQhc9tCs",
        "correct_step": 3,
    },
)


def reference_video(spec, reference_id: str) -> dict:
    database = json.loads(COIN_JSON.read_text())["database"]
    info = database[reference_id]
    labels = [step["label"] for step in info["annotation"]]
    if labels != list(spec.step_labels):
        raise ValueError(f"{reference_id} steps do not match {spec.task}")
    return {
        "file": f"used/{reference_id}.mp4",
        "steps": [
            {
                "index": index,
                "label": step["label"],
                "start": float(step["segment"][0]),
                "end": float(step["segment"][1]),
            }
            for index, step in enumerate(info["annotation"], start=1)
        ],
    }


def test_cases(recordings: list[dict], correct_step: int) -> list[dict]:
    cases = []
    for video in recordings:
        if video["outcome"] == "correct":
            cases.append({
                "file": video["file"],
                "test_step": correct_step,
                "cut_step": correct_step,
            })
            continue
        test_step = int(video["deviation_step"])
        cut_step = next(
            step["index"]
            for step in video["steps"]
            if step["index"] >= test_step and step["start"] is not None
        )
        cases.append({"file": video["file"], "test_step": test_step, "cut_step": cut_step})
    return cases


def score_task(task: dict, encoder: str) -> dict:
    module = __import__(task["module"])
    spec = module.SPEC
    recorded = json.loads(spec.annotations_path().read_text())
    reference = reference_video(spec, task["reference_id"])
    recording_dir = spec.recording_dir()
    recordings = [
        video for video in recorded["videos"]
        if (recording_dir / video["file"]).is_file()
    ]
    if spec.recording_count is not None and len(recordings) != spec.recording_count:
        raise ValueError(f"expected {spec.recording_count} recordings, found {len(recordings)}")
    by_file = {video["file"]: video for video in recordings}
    cases = test_cases(recordings, task["correct_step"])
    summary_name = "classical" if encoder == "classical" else "clip"
    summary = json.loads(spec.summary_path(summary_name).read_text())
    cutoff = float(summary["global"]["mean"] + 2.0 * summary["global"]["std"])

    reference_features = features_for(
        recording_dir.parent, [reference], encoder,
        f"{task['name']}_reference_{summary_name}_features.npz",
    )[0]
    recording_features = dict(zip(
        [video["file"] for video in recordings],
        features_for(
            recording_dir, recordings, encoder,
            spec.recording_cache(encoder).name,
        ),
    ))
    reference_labels = frame_labels(len(reference_features), reference["steps"])
    annotated = np.flatnonzero(reference_labels >= 0)
    region = slice(int(annotated[0]), int(annotated[-1]) + 1)
    reference_region = reference_features[region]
    labels = reference_labels[region]
    reference_times = np.arange(len(reference_features), dtype=np.float64)[region] / FPS
    n_steps = len(spec.step_labels)

    print(f"\n{task['name']} {encoder}: cutoff {cutoff:.3f}", flush=True)
    rows = []
    for case in cases:
        video = by_file[case["file"]]
        features = recording_features[case["file"]]
        cut = video["steps"][case["cut_step"] - 1]
        cut_time = 0.5 * (float(cut["start"]) + float(cut["end"]))
        prefix_count = min(max(int(np.ceil(cut_time * FPS - 1e-9)), 2), len(features))
        prefix = features[:prefix_count]
        cost = 1.0 - prefix @ reference_region.T
        np.clip(cost, 0.0, 2.0, out=cost)
        _, step_sum, step_count, end_j, last_step = open_end_alignment(
            cost.astype(np.float64), labels, n_steps
        )
        test_index = case["test_step"] - 1
        open_end = (
            float(step_sum[test_index] / step_count[test_index])
            if step_count[test_index] > 0 else None
        )
        stop_step = int(last_step) + 1 if int(last_step) >= 0 else None
        recording_labels = frame_labels(len(features), video["steps"])
        times = np.arange(len(features), dtype=np.float64) / FPS
        query_index = np.flatnonzero(
            (recording_labels == case["cut_step"] - 1) & (times < cut_time)
        )
        reference_index = np.flatnonzero(labels == test_index)
        segment = (
            _dtw_segment_distance(features[query_index], reference_region[reference_index])
            if len(query_index) and len(reference_index) else None
        )
        row = {
            **case,
            "encoder": encoder,
            "outcome": video["outcome"],
            "cut_time_s": cut_time,
            "stop_time_s": float(reference_times[int(end_j)]),
            "stop_step": stop_step,
            "mapped_to_cut_step": stop_step == case["cut_step"],
            "open_end_distance": open_end,
            "open_end_deviation": open_end is not None and open_end > cutoff,
            "segment_distance": segment,
            "segment_deviation": segment is not None and segment > cutoff,
            "cutoff": cutoff,
        }
        rows.append(row)
        print(
            f"  {case['file']:<24} test {case['test_step']} cut {case['cut_step']} "
            f"@ {cut_time:.1f}s stop {stop_step} "
            f"open {None if open_end is None else round(open_end, 3)} "
            f"segment {None if segment is None else round(segment, 3)}",
            flush=True,
        )
    return {
        "task": task["name"],
        "encoder": encoder,
        "reference_id": task["reference_id"],
        "cutoff": cutoff,
        "results": rows,
    }


def main() -> int:
    payload = [score_task(task, encoder) for task in TASKS for encoder in ("classical", "clip")]
    output = PREPARED_DIR / "recorded_dtw.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {output.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
