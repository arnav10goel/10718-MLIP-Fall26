"""Compare annotated recordings with all 48 canonical ReplaceSIMCard videos.

The scoring matches the pairwise baselines: sample at 5 fps, run global DTW
with cosine distance, and average path costs globally and within each step.
Recorded step boundaries come from recorded_annotations.json. Correctness and
deviation labels are copied to the output for evaluation but never used by DTW.

Output:
    data/prepared/recorded_similarity.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotate_recordings import STEP_LABELS  # noqa: E402
from classical_pairwise_dtw import (  # noqa: E402
    FPS,
    dtw_alignment,
    dtw_pair,
    frame_feature,
    label_frames,
    step_distance,
)
from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402
from ml_pairwise_dtw import embed_frames, load_clip  # noqa: E402

RECORDING_DIR = REPO_ROOT / "data/videos/ReplaceSIMCard/recorded"
ANNOTATIONS_PATH = PREPARED_DIR / "recorded_annotations.json"
OUTPUT_PATH = PREPARED_DIR / "recorded_similarity.json"
UNLABELED_OUTPUT_PATH = PREPARED_DIR / "recorded_similarity_unlabeled.json"
CLASSICAL_REFERENCE_CACHE = PREPARED_DIR / "sim_classical_features.npz"
CLIP_REFERENCE_CACHE = PREPARED_DIR / "sim_clip_features.npz"
CLASSICAL_RECORDING_CACHE = PREPARED_DIR / "recorded_classical_features.npz"
CLIP_RECORDING_CACHE = PREPARED_DIR / "recorded_clip_features.npz"
CLASSICAL_SUMMARY = PREPARED_DIR / "sim_classical_dtw_summary.json"
CLIP_SUMMARY = PREPARED_DIR / "sim_clip_dtw_summary.json"


def canonical_videos() -> list[dict]:
    rows = []
    with (PREPARED_DIR / "manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["task"] != "ReplaceSIMCard" or len(row["steps"]) != 3:
                continue
            if [step["label"] for step in row["steps"]] != list(STEP_LABELS):
                continue
            rows.append(row)
    rows.sort(key=lambda row: row["youtube_id"])
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 canonical videos, found {len(rows)}")
    return rows


def load_annotations() -> list[dict]:
    with ANNOTATIONS_PATH.open() as file:
        payload = json.load(file)
    if payload.get("task") != "ReplaceSIMCard":
        raise ValueError(f"{ANNOTATIONS_PATH} is not for ReplaceSIMCard")

    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{ANNOTATIONS_PATH} has no video annotations")
    for video in videos:
        path = RECORDING_DIR / video["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        steps = video.get("steps", [])
        if len(steps) != 3:
            raise ValueError(f"{video['file']} does not have three steps")
        if [step.get("label") for step in steps] != list(STEP_LABELS):
            raise ValueError(f"{video['file']} has unexpected step labels")
        for step in steps:
            start = step.get("start")
            end = step.get("end")
            if start is None or end is None or end <= start:
                raise ValueError(
                    f"{video['file']} step {step.get('index')} is incomplete"
                )
    return videos


def sample_frames(path: Path, size: int, bicubic: bool = False) -> np.ndarray:
    scale = f"scale={size}:{size}"
    if bicubic:
        scale += ":flags=bicubic"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={FPS},{scale}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = size * size * 3
    count = len(raw) // frame_bytes
    if count < 2:
        raise RuntimeError(f"{path.name} produced only {count} frames")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, size, size, 3)


def recording_signatures(annotations: list[dict]) -> list[str]:
    signatures = []
    for annotation in annotations:
        path = RECORDING_DIR / annotation["file"]
        stat = path.stat()
        signatures.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    return signatures


def load_cached_recordings(
    cache_path: Path, annotations: list[dict]
) -> list[np.ndarray] | None:
    if not cache_path.exists():
        return None
    cache = np.load(cache_path, allow_pickle=True)
    names = [annotation["file"] for annotation in annotations]
    if (
        list(cache["names"]) != names
        or list(cache["signatures"]) != recording_signatures(annotations)
    ):
        return None
    print(f"loaded {cache_path.name}", flush=True)
    return [np.asarray(features, dtype=np.float32) for features in cache["features"]]


def save_recording_cache(
    cache_path: Path, annotations: list[dict], features: list[np.ndarray]
) -> None:
    np.savez_compressed(
        cache_path,
        names=np.array([annotation["file"] for annotation in annotations]),
        signatures=np.array(recording_signatures(annotations)),
        features=np.array(features, dtype=object),
    )


def classical_recording_features(annotations: list[dict]) -> list[np.ndarray]:
    cached = load_cached_recordings(CLASSICAL_RECORDING_CACHE, annotations)
    if cached is not None:
        return cached
    result = []
    for index, annotation in enumerate(annotations, start=1):
        frames = sample_frames(RECORDING_DIR / annotation["file"], 64)
        result.append(np.stack([frame_feature(frame) for frame in frames]))
        print(
            f"classical [{index}/{len(annotations)}] "
            f"{annotation['file']} {len(frames)} frames",
            flush=True,
        )
    save_recording_cache(CLASSICAL_RECORDING_CACHE, annotations, result)
    return result


def clip_recording_features(annotations: list[dict]) -> list[np.ndarray]:
    cached = load_cached_recordings(CLIP_RECORDING_CACHE, annotations)
    if cached is not None:
        return cached
    model, device = load_clip()
    print(f"CLIP ViT-B/32 on {device}", flush=True)
    result = []
    for index, annotation in enumerate(annotations, start=1):
        frames = sample_frames(
            RECORDING_DIR / annotation["file"], 224, bicubic=True
        )
        result.append(embed_frames(model, device, frames))
        print(
            f"CLIP [{index}/{len(annotations)}] "
            f"{annotation['file']} {len(frames)} frames",
            flush=True,
        )
    save_recording_cache(CLIP_RECORDING_CACHE, annotations, result)
    return result


def recording_labels(
    annotations: list[dict], features: list[np.ndarray]
) -> list[np.ndarray]:
    labels = []
    for annotation, video_features in zip(annotations, features):
        times = np.arange(len(video_features), dtype=np.float64) / FPS
        labels.append(label_frames(times, annotation["steps"]))
    return labels


def load_reference_cache(
    cache_path: Path, expected_ids: list[str]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cache = np.load(cache_path, allow_pickle=True)
    if list(cache["ids"]) != expected_ids:
        raise ValueError(f"{cache_path} does not match the canonical videos")
    features = [
        np.asarray(video_features, dtype=np.float32)
        for video_features in cache["features"]
    ]
    labels = [
        np.asarray(video_labels, dtype=np.int8) for video_labels in cache["labels"]
    ]
    return features, labels


def crop_to_annotated_region(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    annotated = np.flatnonzero(labels >= 0)
    if not len(annotated):
        raise ValueError("video has no annotated frames")
    region = slice(annotated[0], annotated[-1] + 1)
    return features[region], labels[region]


def pair_score(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    cost = 1.0 - query_features @ reference_features.T
    np.clip(cost, 0.0, 2.0, out=cost)
    n_steps = len(STEP_LABELS)
    global_cost, sum_a, count_a, sum_b, count_b = dtw_pair(
        cost.astype(np.float64), query_labels, reference_labels, n_steps
    )
    steps = np.array(
        [
            step_distance(sum_a, count_a, sum_b, count_b, index)
            for index in range(n_steps)
        ],
        dtype=np.float64,
    )
    return float(global_cost), steps


def baseline_blocks(summary: dict) -> list[dict]:
    return [summary["global"]] + [summary["steps"][label] for label in STEP_LABELS]


def aggregate(values: np.ndarray, baseline: dict) -> dict:
    mean = float(values.mean())
    cutoff = float(baseline["mean"] + 2.0 * baseline["std"])
    return {
        "cosine_distance_mean": mean,
        "cosine_similarity_mean": 1.0 - mean,
        "cosine_distance_median": float(np.median(values)),
        "cosine_distance_min": float(values.min()),
        "cosine_distance_max": float(values.max()),
        "baseline_mean": float(baseline["mean"]),
        "baseline_std": float(baseline["std"]),
        "cutoff_mean_plus_2std": cutoff,
        "z_score": float((mean - baseline["mean"]) / baseline["std"]),
        "references_above_cutoff": int(np.sum(values > cutoff)),
        "n_references": int(len(values)),
        "mean_above_cutoff": bool(mean > cutoff),
    }


def score_method(
    method: str,
    annotations: list[dict],
    recording_features: list[np.ndarray],
    reference_ids: list[str],
    reference_features: list[np.ndarray],
    reference_labels: list[np.ndarray],
    summary: dict,
) -> dict:
    n_recordings = len(annotations)
    n_references = len(reference_ids)
    scores = np.empty((n_recordings, n_references, 4), dtype=np.float64)
    recording_step_labels = recording_labels(annotations, recording_features)

    # Compile the numba DTW loop before scoring.
    dtw_pair(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(2, dtype=np.int8),
        np.zeros(2, dtype=np.int8),
        len(STEP_LABELS),
    )

    cropped_references = [
        crop_to_annotated_region(features, labels)
        for features, labels in zip(reference_features, reference_labels)
    ]
    for recording_index, (features, labels) in enumerate(
        zip(recording_features, recording_step_labels)
    ):
        query_features, query_labels = crop_to_annotated_region(features, labels)
        for reference_index, (ref_features, ref_labels) in enumerate(
            cropped_references
        ):
            global_cost, step_costs = pair_score(
                query_features,
                query_labels,
                ref_features,
                ref_labels,
            )
            scores[recording_index, reference_index, 0] = global_cost
            scores[recording_index, reference_index, 1:] = step_costs
        print(
            f"{method}: scored {annotations[recording_index]['file']} "
            f"against {n_references} references",
            flush=True,
        )

    blocks = baseline_blocks(summary)
    results = []
    for recording_index, annotation in enumerate(annotations):
        references = []
        for reference_index, reference_id in enumerate(reference_ids):
            references.append(
                {
                    "youtube_id": reference_id,
                    "global_cosine_distance": float(
                        scores[recording_index, reference_index, 0]
                    ),
                    "step_cosine_distances": {
                        label: float(
                            scores[recording_index, reference_index, step_index + 1]
                        )
                        for step_index, label in enumerate(STEP_LABELS)
                    },
                }
            )
        results.append(
            {
                "file": annotation["file"],
                "ground_truth_outcome": annotation["outcome"],
                "ground_truth_deviation_step": annotation["deviation_step"],
                "global": aggregate(scores[recording_index, :, 0], blocks[0]),
                "steps": {
                    label: aggregate(
                        scores[recording_index, :, step_index + 1],
                        blocks[step_index + 1],
                    )
                    for step_index, label in enumerate(STEP_LABELS)
                },
                "references": references,
            }
        )

    return {
        "feature": summary.get("model", summary.get("feature")),
        "distance": (
            "mean cosine distance along global DTW path; recording and reference "
            "step labels both contribute to each per-step score"
        ),
        "baseline": "1,128 pairwise comparisons among 48 canonical videos",
        "recordings": results,
    }


def load_cached_feature_map(cache_path: Path) -> dict[str, np.ndarray]:
    cache = np.load(cache_path, allow_pickle=True)
    return {
        name: np.asarray(features, dtype=np.float32)
        for name, features in zip(cache["names"], cache["features"])
    }


def score_unlabeled_method(
    method: str,
    names: list[str],
    recording_features: list[np.ndarray],
    reference_ids: list[str],
    reference_features: list[np.ndarray],
    reference_labels: list[np.ndarray],
    summary: dict,
) -> dict:
    n_recordings = len(names)
    n_references = len(reference_ids)
    n_steps = len(STEP_LABELS)
    scores = np.empty((n_recordings, n_references, n_steps + 1), dtype=np.float64)
    starts = np.empty((n_recordings, n_references, n_steps), dtype=np.float64)
    ends = np.empty((n_recordings, n_references, n_steps), dtype=np.float64)
    dtw_alignment(
        np.zeros((2, 2), dtype=np.float64),
        np.full(2, -1, dtype=np.int8),
        np.zeros(2, dtype=np.int8),
        n_steps,
    )

    for recording_index, features in enumerate(recording_features):
        query = features.astype(np.float32)
        for reference_index, (ref_features, ref_labels) in enumerate(
            zip(reference_features, reference_labels)
        ):
            cost = 1.0 - query @ ref_features.T
            np.clip(cost, 0.0, 2.0, out=cost)
            blank = np.full(cost.shape[0], -1, dtype=np.int8)
            (
                path_mean,
                _,
                _,
                step_sum,
                step_count,
                first,
                last,
            ) = dtw_alignment(
                cost.astype(np.float64),
                blank,
                ref_labels.astype(np.int8),
                n_steps,
            )
            scores[recording_index, reference_index, 0] = path_mean
            for step_index in range(n_steps):
                if step_count[step_index] > 0:
                    scores[recording_index, reference_index, step_index + 1] = (
                        step_sum[step_index] / step_count[step_index]
                    )
                    starts[recording_index, reference_index, step_index] = (
                        first[step_index] / FPS
                    )
                    ends[recording_index, reference_index, step_index] = (
                        (last[step_index] + 1) / FPS
                    )
                else:
                    scores[recording_index, reference_index, step_index + 1] = np.nan
                    starts[recording_index, reference_index, step_index] = np.nan
                    ends[recording_index, reference_index, step_index] = np.nan
        print(
            f"{method}: scored {names[recording_index]} "
            f"against {n_references} references",
            flush=True,
        )

    blocks = baseline_blocks(summary)
    results = []
    for recording_index, name in enumerate(names):
        results.append(
            {
                "file": name,
                "global": aggregate(scores[recording_index, :, 0], blocks[0]),
                "steps": {
                    label: aggregate(
                        scores[recording_index, :, step_index + 1],
                        blocks[step_index + 1],
                    )
                    for step_index, label in enumerate(STEP_LABELS)
                },
                "inferred_step_seconds": {
                    label: {
                        "median_start": float(
                            np.nanmedian(starts[recording_index, :, step_index])
                        ),
                        "median_end": float(
                            np.nanmedian(ends[recording_index, :, step_index])
                        ),
                    }
                    for step_index, label in enumerate(STEP_LABELS)
                },
            }
        )
    return {
        "feature": summary.get("model", summary.get("feature")),
        "distance": (
            "mean cosine distance along the global DTW path while the reference "
            "frame is inside each annotated reference step"
        ),
        "recording_annotations_used": False,
        "baseline": "1,128 pairwise comparisons among 48 canonical videos",
        "recordings": results,
    }


def score_without_recording_annotations() -> None:
    names = sorted(path.name for path in RECORDING_DIR.glob("*.mp4"))
    if len(names) != 6:
        raise RuntimeError(f"expected 6 recordings, found {len(names)}")
    rows = canonical_videos()
    reference_ids = [row["youtube_id"] for row in rows]
    classical_map = load_cached_feature_map(CLASSICAL_RECORDING_CACHE)
    clip_map = load_cached_feature_map(CLIP_RECORDING_CACHE)
    missing = [name for name in names if name not in classical_map or name not in clip_map]
    if missing:
        raise RuntimeError(f"missing cached features for {missing}")

    classical_refs, classical_labels = load_reference_cache(
        CLASSICAL_REFERENCE_CACHE, reference_ids
    )
    clip_refs, clip_labels = load_reference_cache(CLIP_REFERENCE_CACHE, reference_ids)
    with CLASSICAL_SUMMARY.open() as file:
        classical_summary = json.load(file)
    with CLIP_SUMMARY.open() as file:
        clip_summary = json.load(file)

    classical_result = score_unlabeled_method(
        "classical",
        names,
        [classical_map[name] for name in names],
        reference_ids,
        classical_refs,
        classical_labels,
        classical_summary,
    )
    clip_result = score_unlabeled_method(
        "CLIP",
        names,
        [clip_map[name] for name in names],
        reference_ids,
        clip_refs,
        clip_labels,
        clip_summary,
    )
    output = {
        "task": "ReplaceSIMCard",
        "fps": FPS,
        "recording_annotations_used": False,
        "n_references": len(reference_ids),
        "reference_ids": reference_ids,
        "methods": {"classical": classical_result, "clip": clip_result},
    }
    with UNLABELED_OUTPUT_PATH.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print_results("Classical", classical_result)
    print_results("CLIP", clip_result)
    for recording in clip_result["recordings"]:
        windows = recording["inferred_step_seconds"]
        print(
            recording["file"],
            " ".join(
                f"{index}:{windows[label]['median_start']:.1f}-{windows[label]['median_end']:.1f}s"
                for index, label in enumerate(STEP_LABELS, start=1)
            ),
        )
    print(f"\nwrote {UNLABELED_OUTPUT_PATH.relative_to(REPO_ROOT)}")


def print_results(method: str, result: dict) -> None:
    print(f"\n{method} mean cosine distance across 48 references")
    print(
        f"{'video':<16}{'open':>9}{'insert':>9}"
        f"{'close':>9}{'global':>9}{'insert z':>10}{'>cut':>7}"
    )
    insert_label = STEP_LABELS[1]
    for recording in result["recordings"]:
        steps = recording["steps"]
        insert = steps[insert_label]
        print(
            f"{recording['file']:<16}"
            f"{steps[STEP_LABELS[0]]['cosine_distance_mean']:9.3f}"
            f"{insert['cosine_distance_mean']:9.3f}"
            f"{steps[STEP_LABELS[2]]['cosine_distance_mean']:9.3f}"
            f"{recording['global']['cosine_distance_mean']:9.3f}"
            f"{insert['z_score']:10.2f}"
            f"{insert['references_above_cutoff']:7d}"
        )
    insert_baseline = result["recordings"][0]["steps"][insert_label]
    print(
        f"insert baseline: {insert_baseline['baseline_mean']:.3f} ± "
        f"{insert_baseline['baseline_std']:.3f}; cutoff "
        f"{insert_baseline['cutoff_mean_plus_2std']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-recording-annotations",
        action="store_true",
        help="score recordings without reading their step timestamps",
    )
    args = parser.parse_args()
    if args.no_recording_annotations:
        score_without_recording_annotations()
        return

    annotations = load_annotations()
    rows = canonical_videos()
    reference_ids = [row["youtube_id"] for row in rows]

    classical_refs, classical_ref_labels = load_reference_cache(
        CLASSICAL_REFERENCE_CACHE, reference_ids
    )
    clip_refs, clip_ref_labels = load_reference_cache(
        CLIP_REFERENCE_CACHE, reference_ids
    )
    with CLASSICAL_SUMMARY.open() as file:
        classical_summary = json.load(file)
    with CLIP_SUMMARY.open() as file:
        clip_summary = json.load(file)

    classical_result = score_method(
        "classical",
        annotations,
        classical_recording_features(annotations),
        reference_ids,
        classical_refs,
        classical_ref_labels,
        classical_summary,
    )
    clip_result = score_method(
        "CLIP",
        annotations,
        clip_recording_features(annotations),
        reference_ids,
        clip_refs,
        clip_ref_labels,
        clip_summary,
    )

    output = {
        "task": "ReplaceSIMCard",
        "fps": FPS,
        "n_references": len(reference_ids),
        "reference_ids": reference_ids,
        "methods": {
            "classical": classical_result,
            "clip": clip_result,
        },
    }
    with OUTPUT_PATH.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")

    print_results("Classical", classical_result)
    print_results("CLIP", clip_result)
    print(f"\nwrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
