"""Score recordings against the canonical videos of a TaskSpec.

The scoring matches the pairwise baselines: sample at 5 fps, run global DTW
with cosine distance, and average path costs globally and within each step.
Recorded step boundaries come from the task annotation file. Correctness and
deviation labels are copied to the output for evaluation but never used by DTW.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_pairwise_dtw import (  # noqa: E402
    FPS,
    HISTOGRAM_FEATURE_VERSION,
    dtw_alignment,
    dtw_pair,
    frame_feature,
    label_frames,
    open_end_alignment,
    step_distance,
)
from coin_tasks import REPO_ROOT  # noqa: E402
from ml_pairwise_dtw import embed_frames, load_clip  # noqa: E402
from task_spec import TaskSpec, load_manifest_rows, matching_rows  # noqa: E402


def canonical_videos(spec: TaskSpec) -> list[dict]:
    return matching_rows(load_manifest_rows(), spec)


def load_annotations(spec: TaskSpec) -> list[dict]:
    annotations_path = spec.annotations_path()
    recording_dir = spec.recording_dir()
    step_labels = list(spec.step_labels)
    with annotations_path.open() as file:
        payload = json.load(file)
    if payload.get("task") != spec.task:
        raise ValueError(f"{annotations_path} is not for {spec.task}")

    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{annotations_path} has no video annotations")
    for video in videos:
        path = recording_dir / video["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        steps = video.get("steps", [])
        if len(steps) != len(step_labels):
            raise ValueError(
                f"{video['file']} does not have {len(step_labels)} steps"
            )
        if [step.get("label") for step in steps] != step_labels:
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


def recording_signatures(recording_dir: Path, annotations: list[dict]) -> list[str]:
    signatures = []
    for annotation in annotations:
        path = recording_dir / annotation["file"]
        stat = path.stat()
        signatures.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    return signatures


def load_cached_recordings(
    cache_path: Path,
    recording_dir: Path,
    annotations: list[dict],
    feature_version: str | None = None,
) -> list[np.ndarray] | None:
    if not cache_path.exists():
        return None
    cache = np.load(cache_path, allow_pickle=True)
    names = [annotation["file"] for annotation in annotations]
    if (
        list(cache["names"]) != names
        or list(cache["signatures"]) != recording_signatures(recording_dir, annotations)
    ):
        return None
    if feature_version is not None and (
        "feature_version" not in cache.files
        or str(cache["feature_version"]) != feature_version
    ):
        return None
    print(f"loaded {cache_path.name}", flush=True)
    return [np.asarray(features, dtype=np.float32) for features in cache["features"]]


def save_recording_cache(
    cache_path: Path,
    recording_dir: Path,
    annotations: list[dict],
    features: list[np.ndarray],
    feature_version: str | None = None,
) -> None:
    payload = {
        "names": np.array([annotation["file"] for annotation in annotations]),
        "signatures": np.array(recording_signatures(recording_dir, annotations)),
        "features": np.array(features, dtype=object),
    }
    if feature_version is not None:
        payload["feature_version"] = np.array(feature_version)
    np.savez_compressed(cache_path, **payload)


def classical_recording_features(
    spec: TaskSpec, annotations: list[dict]
) -> list[np.ndarray]:
    recording_dir = spec.recording_dir()
    cache_path = spec.recording_cache("classical")
    cached = load_cached_recordings(
        cache_path, recording_dir, annotations, HISTOGRAM_FEATURE_VERSION
    )
    if cached is not None:
        return cached
    result = []
    for index, annotation in enumerate(annotations, start=1):
        frames = sample_frames(recording_dir / annotation["file"], 64)
        result.append(np.stack([frame_feature(frame) for frame in frames]))
        print(
            f"classical [{index}/{len(annotations)}] "
            f"{annotation['file']} {len(frames)} frames",
            flush=True,
        )
    save_recording_cache(
        cache_path, recording_dir, annotations, result, HISTOGRAM_FEATURE_VERSION
    )
    return result


def clip_recording_features(
    spec: TaskSpec, annotations: list[dict]
) -> list[np.ndarray]:
    recording_dir = spec.recording_dir()
    cache_path = spec.recording_cache("clip")
    cached = load_cached_recordings(cache_path, recording_dir, annotations)
    if cached is not None:
        return cached
    model, device = load_clip()
    print(f"CLIP ViT-B/32 on {device}", flush=True)
    result = []
    for index, annotation in enumerate(annotations, start=1):
        frames = sample_frames(
            recording_dir / annotation["file"], 224, bicubic=True
        )
        result.append(embed_frames(model, device, frames))
        print(
            f"CLIP [{index}/{len(annotations)}] "
            f"{annotation['file']} {len(frames)} frames",
            flush=True,
        )
    save_recording_cache(cache_path, recording_dir, annotations, result)
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


def baseline_text(n_videos: int) -> str:
    pairs = n_videos * (n_videos - 1) // 2
    return f"{pairs:,} pairwise comparisons among {n_videos} canonical videos"


def pair_score(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    n_steps: int,
) -> tuple[float, np.ndarray]:
    cost = 1.0 - query_features @ reference_features.T
    np.clip(cost, 0.0, 2.0, out=cost)
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


def baseline_blocks(summary: dict, step_labels: tuple[str, ...]) -> list[dict]:
    return [summary["global"]] + [summary["steps"][label] for label in step_labels]


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
    step_labels: tuple[str, ...],
) -> dict:
    n_recordings = len(annotations)
    n_references = len(reference_ids)
    n_steps = len(step_labels)
    scores = np.empty((n_recordings, n_references, n_steps + 1), dtype=np.float64)
    recording_step_labels = recording_labels(annotations, recording_features)

    # Compile the numba DTW loop before scoring.
    dtw_pair(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(2, dtype=np.int8),
        np.zeros(2, dtype=np.int8),
        n_steps,
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
                n_steps,
            )
            scores[recording_index, reference_index, 0] = global_cost
            scores[recording_index, reference_index, 1:] = step_costs
        print(
            f"{method}: scored {annotations[recording_index]['file']} "
            f"against {n_references} references",
            flush=True,
        )

    blocks = baseline_blocks(summary, step_labels)
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
                        for step_index, label in enumerate(step_labels)
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
                    for step_index, label in enumerate(step_labels)
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
        "baseline": baseline_text(n_references),
        "recordings": results,
    }


def load_cached_feature_map(
    cache_path: Path, feature_version: str | None = None
) -> dict[str, np.ndarray]:
    cache = np.load(cache_path, allow_pickle=True)
    if feature_version is not None and (
        "feature_version" not in cache.files
        or str(cache["feature_version"]) != feature_version
    ):
        raise RuntimeError(
            f"{cache_path.name} was saved before the histogram normalization change"
        )
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
    step_labels: tuple[str, ...],
) -> dict:
    n_recordings = len(names)
    n_references = len(reference_ids)
    n_steps = len(step_labels)
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

    blocks = baseline_blocks(summary, step_labels)
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
                    for step_index, label in enumerate(step_labels)
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
                    for step_index, label in enumerate(step_labels)
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
        "baseline": baseline_text(n_references),
        "recordings": results,
    }


def score_without_recording_annotations(spec: TaskSpec) -> None:
    recording_dir = spec.recording_dir()
    names = sorted(path.name for path in recording_dir.glob("*.mp4"))
    if spec.recording_count is not None and len(names) != spec.recording_count:
        raise RuntimeError(
            f"expected {spec.recording_count} recordings, found {len(names)}"
        )
    rows = canonical_videos(spec)
    reference_ids = [row["youtube_id"] for row in rows]
    classical_map = load_cached_feature_map(
        spec.recording_cache("classical"), HISTOGRAM_FEATURE_VERSION
    )
    clip_map = load_cached_feature_map(spec.recording_cache("clip"))
    missing = [name for name in names if name not in classical_map or name not in clip_map]
    if missing:
        raise RuntimeError(f"missing cached features for {missing}")

    classical_refs, classical_labels = load_reference_cache(
        spec.reference_cache("classical"), reference_ids
    )
    clip_refs, clip_labels = load_reference_cache(
        spec.reference_cache("clip"), reference_ids
    )
    with spec.summary_path("classical").open() as file:
        classical_summary = json.load(file)
    with spec.summary_path("clip").open() as file:
        clip_summary = json.load(file)

    classical_result = score_unlabeled_method(
        "classical",
        names,
        [classical_map[name] for name in names],
        reference_ids,
        classical_refs,
        classical_labels,
        classical_summary,
        spec.step_labels,
    )
    clip_result = score_unlabeled_method(
        "CLIP",
        names,
        [clip_map[name] for name in names],
        reference_ids,
        clip_refs,
        clip_labels,
        clip_summary,
        spec.step_labels,
    )
    output = {
        "task": spec.task,
        "fps": FPS,
        "recording_annotations_used": False,
        "n_references": len(reference_ids),
        "reference_ids": reference_ids,
        "methods": {"classical": classical_result, "clip": clip_result},
    }
    output_path = spec.unlabeled_output()
    with output_path.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print_results("Classical", classical_result, spec.step_labels)
    print_results("CLIP", clip_result, spec.step_labels)
    for recording in clip_result["recordings"]:
        windows = recording["inferred_step_seconds"]
        print(
            recording["file"],
            " ".join(
                f"{index}:{windows[label]['median_start']:.1f}-{windows[label]['median_end']:.1f}s"
                for index, label in enumerate(spec.step_labels, start=1)
            ),
        )
    print(f"\nwrote {output_path.relative_to(REPO_ROOT)}")


def print_results(method: str, result: dict, step_labels: tuple[str, ...]) -> None:
    n_references = result["recordings"][0]["global"]["n_references"]
    focus_index = 1 if len(step_labels) > 1 else 0
    focus = step_labels[focus_index]
    print(f"\n{method} mean cosine distance across {n_references} references")
    header = f"{'video':<16}" + "".join(
        f"{index:>9d}" for index in range(1, len(step_labels) + 1)
    )
    print(f"{header}{'global':>9}{'focus z':>10}{'>cut':>7}")
    for recording in result["recordings"]:
        steps = recording["steps"]
        focus_step = steps[focus]
        line = f"{recording['file']:<16}" + "".join(
            f"{steps[label]['cosine_distance_mean']:9.3f}" for label in step_labels
        )
        print(
            f"{line}"
            f"{recording['global']['cosine_distance_mean']:9.3f}"
            f"{focus_step['z_score']:10.2f}"
            f"{focus_step['references_above_cutoff']:7d}"
        )
    focus_baseline = result["recordings"][0]["steps"][focus]
    print(
        f"step {focus_index + 1} baseline: {focus_baseline['baseline_mean']:.3f} ± "
        f"{focus_baseline['baseline_std']:.3f}; cutoff "
        f"{focus_baseline['cutoff_mean_plus_2std']:.3f}"
    )


def score_with_annotations(spec: TaskSpec) -> None:
    annotations = load_annotations(spec)
    rows = canonical_videos(spec)
    reference_ids = [row["youtube_id"] for row in rows]

    classical_refs, classical_ref_labels = load_reference_cache(
        spec.reference_cache("classical"), reference_ids
    )
    clip_refs, clip_ref_labels = load_reference_cache(
        spec.reference_cache("clip"), reference_ids
    )
    with spec.summary_path("classical").open() as file:
        classical_summary = json.load(file)
    with spec.summary_path("clip").open() as file:
        clip_summary = json.load(file)

    classical_result = score_method(
        "classical",
        annotations,
        classical_recording_features(spec, annotations),
        reference_ids,
        classical_refs,
        classical_ref_labels,
        classical_summary,
        spec.step_labels,
    )
    clip_result = score_method(
        "CLIP",
        annotations,
        clip_recording_features(spec, annotations),
        reference_ids,
        clip_refs,
        clip_ref_labels,
        clip_summary,
        spec.step_labels,
    )

    output = {
        "task": spec.task,
        "fps": FPS,
        "n_references": len(reference_ids),
        "reference_ids": reference_ids,
        "methods": {
            "classical": classical_result,
            "clip": clip_result,
        },
    }
    output_path = spec.annotated_output()
    with output_path.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")

    print_results("Classical", classical_result, spec.step_labels)
    print_results("CLIP", clip_result, spec.step_labels)
    print(f"\nwrote {output_path.relative_to(REPO_ROOT)}")


def score_recorded_prefixes(spec: TaskSpec) -> None:
    """Cut each recording at mid-insert and score it against the first two videos.

    Thresholds come from global DTW on all canonical videos. Open-end DTW
    finds the mapping. The ablation uses the recorded annotations. The phone
    insert is cut at its midpoint and compared with the whole reference insert
    by DTW inside that segment.
    """
    from task_runner import load_or_extract, pairwise_cutoff

    reference_rows = canonical_videos(spec)
    reference_ids = [
        path.stem
        for path in sorted((spec.recording_dir().parent / "used").glob("*.mp4"))
    ]
    chosen_ids = reference_ids[:2]
    if len(chosen_ids) != 2:
        raise RuntimeError("the used-video folder does not contain two videos")
    by_id = {row["youtube_id"]: row for row in reference_rows}
    chosen_rows = [by_id[video_id] for video_id in chosen_ids]
    annotations = load_annotations(spec)
    n_steps = len(spec.step_labels)
    print(f"references: {', '.join(chosen_ids)}", flush=True)

    for encoder in ("classical", "clip"):
        summary = pairwise_cutoff(spec, encoder)
        features, labels = load_or_extract(spec, reference_rows, encoder)
        feature_by_id = dict(zip([row["youtube_id"] for row in reference_rows], features))
        label_by_id = dict(zip([row["youtube_id"] for row in reference_rows], labels))
        if encoder == "classical":
            recording_features = classical_recording_features(spec, annotations)
        else:
            recording_features = clip_recording_features(spec, annotations)
        dtw_pair(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.int8),
            np.zeros(2, dtype=np.int8),
            1,
        )
        open_end_alignment(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.int8),
            n_steps,
        )
        cutoffs = [
            float(summary["steps"][label]["mean"] + 2.0 * summary["steps"][label]["std"])
            for label in spec.step_labels
        ]
        print(
            f"\n{encoder} cutoffs from {summary['n_pairs']} full-video pairs: "
            + ", ".join(
                f"step {index + 1}={cutoff:.3f}" for index, cutoff in enumerate(cutoffs)
            ),
            flush=True,
        )
        methods = ("open-end DTW", "annotated segment DTW")
        for method in methods:
            print(f"\n{encoder} {method}", flush=True)
            print(
                f"  {'video':<16} {'step 1':>8} {'dev':>4} {'step 2':>8} {'dev':>4}  mapped",
                flush=True,
            )
            for annotation, video_features in zip(annotations, recording_features):
                times = np.arange(len(video_features), dtype=np.float64) / FPS
                recording_labels_for_video = label_frames(times, annotation["steps"])
                insert = annotation["steps"][1]
                cut_time = 0.5 * (float(insert["start"]) + float(insert["end"]))
                prefix_count = int(np.ceil(cut_time * FPS - 1e-9))
                prefix_count = min(max(prefix_count, 2), len(video_features))
                prefix = video_features[:prefix_count]
                step_values = {0: [], 1: []}
                mapped = []
                for row in chosen_rows:
                    reference = feature_by_id[row["youtube_id"]]
                    reference_labels = np.asarray(label_by_id[row["youtube_id"]], dtype=np.int8)
                    reference_times = row["roi_start"] + np.arange(len(reference)) / FPS
                    if method == "open-end DTW":
                        cost = 1.0 - prefix @ reference.T
                        np.clip(cost, 0.0, 2.0, out=cost)
                        _, step_sum, step_count, _, last_step = open_end_alignment(
                            cost.astype(np.float64),
                            reference_labels,
                            n_steps,
                        )
                        mapped.append(str(int(last_step) + 1) if int(last_step) >= 0 else "gap")
                        for step_index in (0, 1):
                            if step_count[step_index] > 0:
                                step_values[step_index].append(
                                    float(step_sum[step_index] / step_count[step_index])
                                )
                            else:
                                step_values[step_index].append(None)
                        continue
                    mapped.append("steps 1-2")
                    for step_index in (0, 1):
                        end_time = cut_time if step_index == 1 else None
                        query_index = _segment_indexes(
                            times, recording_labels_for_video, step_index, end_time
                        )
                        reference_index = _segment_indexes(
                            reference_times, reference_labels, step_index, None
                        )
                        if len(query_index) == 0 or len(reference_index) == 0:
                            step_values[step_index].append(None)
                            continue
                        step_values[step_index].append(
                            _dtw_segment_distance(
                                video_features[query_index], reference[reference_index]
                            )
                        )
                cells = []
                for step_index in (0, 1):
                    distance = _mean_present(step_values[step_index])
                    deviation = (
                        "yes"
                        if distance is not None and distance > cutoffs[step_index]
                        else "no"
                    )
                    text = "none" if distance is None else f"{distance:.3f}"
                    cells.extend((f"{text:>8}", f"{deviation:>4}"))
                print(
                    f"  {annotation['file']:<16} {cells[0]} {cells[1]} {cells[2]} {cells[3]}"
                    f"  {', '.join(mapped)}",
                    flush=True,
                )


def score_task(spec: TaskSpec, *, use_annotations: bool) -> None:
    if use_annotations:
        score_with_annotations(spec)
    else:
        score_without_recording_annotations(spec)

