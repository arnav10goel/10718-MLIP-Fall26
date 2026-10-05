"""Score an annotated COIN reference against a comparable offline cohort."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

import numpy as np

from scripts.classical_pairwise_dtw import (
    dtw_pair,
    label_frames,
    sample_video,
    step_distance,
)


def score_reference_cohort(
    rows: list[dict],
    reference_id: str,
    sample_features: Callable[[dict], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Score a cohort from L2-normalized feature rows and matching frame times."""
    sampler = sample_video if sample_features is None else sample_features
    ids = [row["youtube_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Cohort contains duplicate video IDs")
    if reference_id not in ids:
        raise ValueError(f"Reference is absent from cohort: {reference_id}")
    reference = rows[ids.index(reference_id)]
    if reference["subset"] != "training":
        raise ValueError(f"Reference is not a training video: {reference_id}")
    reference_steps = reference["steps"]
    step_ids = [str(step["id"]) for step in reference_steps]
    if not step_ids:
        raise ValueError("Reference has no annotated steps")
    for row in rows:
        if row["task"] != reference["task"]:
            raise ValueError(f"Video has a different task: {row['youtube_id']}")
        if [str(step["id"]) for step in row["steps"]] != step_ids:
            raise ValueError(f"Video has different ordered steps: {row['youtube_id']}")
        if not Path(row["video_path"]).is_absolute():
            raise ValueError(f"Media path must be absolute: {row['youtube_id']}")

    try:
        reference_features, reference_times = sampler(reference)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not sample reference {reference_id}: {exc}") from exc
    reference_labels = label_frames(reference_times, reference_steps)
    n_steps = len(reference_steps)
    scores: list[dict] = []
    failures: list[dict] = []

    for row in rows:
        video_id = row["youtube_id"]
        if video_id == reference_id:
            continue
        try:
            features, times = sampler(row)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            failures.append({
                "video_id": video_id,
                "split": row["subset"],
                "reason": "sampling_error",
                "detail": str(exc),
            })
            continue

        cost = 1.0 - reference_features @ features.T
        np.clip(cost, 0.0, 2.0, out=cost)
        path_mean, sum_a, count_a, sum_b, count_b = dtw_pair(
            cost.astype(np.float64),
            reference_labels,
            label_frames(times, row["steps"]),
            n_steps,
        )
        step_distances: list[float | None] = []
        step_path_counts: list[dict[str, int]] = []
        step_issues: list[dict] = []
        for step_index in range(n_steps):
            reference_count = int(count_a[step_index])
            comparison_count = int(count_b[step_index])
            step_path_counts.append({
                "reference": reference_count,
                "comparison": comparison_count,
            })
            if reference_count == 0 and comparison_count == 0:
                step_distances.append(None)
                step_issues.append({
                    "step_index": step_index,
                    "reason": "no_labeled_frames",
                })
            else:
                step_distances.append(float(step_distance(
                    sum_a, count_a, sum_b, count_b, step_index
                )))
        scores.append({
            "video_id": video_id,
            "split": row["subset"],
            "sampled_frames": len(features),
            "global_distance": float(path_mean),
            "step_distances": step_distances,
            "step_path_counts": step_path_counts,
            "step_issues": step_issues,
        })

    return {
        "task": reference["task"],
        "reference_id": reference_id,
        "reference_frames": len(reference_features),
        "steps": [
            {"id": step["id"], "label": step["label"]}
            for step in reference_steps
        ],
        "scores": scores,
        "failures": failures,
    }
