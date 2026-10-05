"""Shared DTW runner. Task scripts pass a TaskSpec and call run()."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_pairwise_dtw import (  # noqa: E402
    FPS,
    HISTOGRAM_FEATURE_VERSION,
    dtw_pair,
    label_frames,
    sample_video,
    step_distance,
)
from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402
from task_spec import TaskSpec, load_manifest_rows, matching_rows  # noqa: E402

CLASSICAL_FEATURE = (
    "L2-normalized 16-bin hue histogram, 8-bin saturation histogram, "
    "and a 4x4 grid of 9-bin gradient orientation histograms"
)
CLIP_MODEL = "CLIP ViT-B/32, OpenAI weights"
CLIP_FEATURE = "L2-normalized 512-D image embedding"


def distance_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_pairs(
    features: list[np.ndarray],
    labels: list[np.ndarray],
    step_labels: tuple[str, ...],
    video_ids: list[str],
    encoder: str,
) -> tuple[dict, np.ndarray]:
    """All-pairs mean cosine distance along the global DTW path."""
    n = len(features)
    n_steps = len(step_labels)
    n_pairs = n * (n - 1) // 2
    dtw_pair(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(2, dtype=np.int8),
        np.zeros(2, dtype=np.int8),
        n_steps,
    )

    matrix = np.zeros((n, n), dtype=np.float64)
    global_costs = np.empty(n_pairs, dtype=np.float64)
    step_costs = np.empty((n_pairs, n_steps), dtype=np.float64)
    pair_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            cost = 1.0 - features[i] @ features[j].T
            np.clip(cost, 0.0, 2.0, out=cost)
            path_mean, sum_a, count_a, sum_b, count_b = dtw_pair(
                cost.astype(np.float64),
                labels[i].astype(np.int8),
                labels[j].astype(np.int8),
                n_steps,
            )
            matrix[i, j] = matrix[j, i] = path_mean
            global_costs[pair_index] = path_mean
            for step_index in range(n_steps):
                step_costs[pair_index, step_index] = step_distance(
                    sum_a, count_a, sum_b, count_b, step_index
                )
            pair_index += 1
        print(f"finished reference {i + 1}/{n}", flush=True)

    summary = {
        "n_videos": n,
        "n_pairs": n_pairs,
        "fps": FPS,
        "distance": "mean cosine distance along the full DTW path",
        "video_ids": video_ids,
        "global": distance_stats(global_costs),
        "steps": {
            label: distance_stats(step_costs[:, step_index])
            for step_index, label in enumerate(step_labels)
        },
    }
    if encoder == "clip":
        summary["model"] = CLIP_MODEL
        summary["feature"] = CLIP_FEATURE
    else:
        summary["feature"] = CLASSICAL_FEATURE
    return summary, matrix


def load_or_extract(
    spec: TaskSpec, rows: list[dict], encoder: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cache_path = spec.reference_cache(encoder)
    ids = [row["youtube_id"] for row in rows]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        version_ok = encoder != "classical" or (
            "feature_version" in cached.files
            and str(cached["feature_version"]) == HISTOGRAM_FEATURE_VERSION
        )
        if list(cached["ids"]) == ids and version_ok:
            print(
                f"loaded cached {encoder} features for {len(ids)} videos",
                flush=True,
            )
            return list(cached["features"]), list(cached["labels"])

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    if encoder == "clip":
        from ml_pairwise_dtw import embed_frames, load_clip, sample_frames

        model, device = load_clip()
        print(f"CLIP ViT-B/32 on {device}", flush=True)
        for index, row in enumerate(rows, start=1):
            frames, times = sample_frames(row)
            features.append(embed_frames(model, device, frames))
            labels.append(label_frames(times, row["steps"]))
            print(
                f"[{index}/{len(rows)}] {row['youtube_id']} {len(frames)} frames",
                flush=True,
            )
    elif encoder == "classical":
        for index, row in enumerate(rows, start=1):
            video_features, times = sample_video(row)
            features.append(video_features)
            labels.append(label_frames(times, row["steps"]))
            print(
                f"[{index}/{len(rows)}] {row['youtube_id']} {len(video_features)} frames",
                flush=True,
            )
    else:
        raise ValueError(f"unknown encoder: {encoder}")

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=np.array(ids),
        features=np.array(features, dtype=object),
        labels=np.array(labels, dtype=object),
        feature_version=np.array(
            HISTOGRAM_FEATURE_VERSION if encoder == "classical" else "clip"
        ),
    )
    return features, labels


def pairwise_cutoff(spec: TaskSpec, encoder: str) -> dict:
    rows = matching_rows(load_manifest_rows(), spec)
    features, labels = load_or_extract(spec, rows, encoder)
    n_pairs = len(rows) * (len(rows) - 1) // 2
    print(f"aligning {n_pairs} {encoder} pairs", flush=True)
    summary, matrix = summarize_pairs(
        features,
        labels,
        spec.step_labels,
        [row["youtube_id"] for row in rows],
        encoder,
    )
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(spec.matrix_path(encoder), matrix)
    summary_path = spec.summary_path(encoder)
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print(
        f"global mean {summary['global']['mean']:.4f} "
        f"std {summary['global']['std']:.4f}"
    )
    for label, step_stats in summary["steps"].items():
        print(f"{label}: mean {step_stats['mean']:.4f} std {step_stats['std']:.4f}")
    print(f"wrote {summary_path.relative_to(REPO_ROOT)}")
    return summary


def run(spec: TaskSpec, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"DTW cutoff and recording scores for {spec.task}."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cutoff = subparsers.add_parser(
        "cutoff", help="all-pairs distance among the canonical videos"
    )
    cutoff.add_argument(
        "--encoder",
        choices=("classical", "clip", "both"),
        default="both",
    )
    score = subparsers.add_parser(
        "score", help="compare recordings with the canonical videos"
    )
    score.add_argument(
        "--no-recording-annotations",
        action="store_true",
        help="score recordings without reading their step timestamps",
    )
    subparsers.add_parser(
        "recorded-prefix",
        help="mid-insert recordings against the first two canonical videos",
    )
    args = parser.parse_args(argv)
    if args.command == "cutoff":
        encoders = ("classical", "clip") if args.encoder == "both" else (args.encoder,)
        for encoder in encoders:
            pairwise_cutoff(spec, encoder)
        return 0

    if args.command == "recorded-prefix":
        from score_recordings import score_recorded_prefixes

        score_recorded_prefixes(spec)
        return 0

    from score_recordings import score_task

    score_task(spec, use_annotations=not args.no_recording_annotations)
    return 0
