"""Score smoothie recordings cut mid-step against the annotated reference video.

Each wrong recording is tested on its deviation step. A skipped step has no
frames, so that recording is cut in the middle of the next step it performs.
Each correct recording is tested on three steps drawn with a fixed seed.

Open-end DTW maps the crop onto the reference and reports where it stops.
Segment DTW uses the annotations: the recording frames of the step being done
at the cut, up to the cut, against the whole reference test step. A step is a
deviation when its mean cosine distance is above the COIN whole-video cutoff.

    .venv/bin/python scripts/make_strawberry_smoothie.py cutoff
    .venv/bin/python scripts/smoothie_recorded.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_pairwise_dtw import (  # noqa: E402
    FPS,
    HISTOGRAM_FEATURE_VERSION,
    frame_feature,
    open_end_alignment,
)
from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402
from make_strawberry_smoothie import SPEC  # noqa: E402
from score_recordings import (  # noqa: E402
    _dtw_segment_distance,
    load_cached_recordings,
    sample_frames,
    save_recording_cache,
)

REFERENCE_ANNOTATIONS = PREPARED_DIR / "smoothie_reference_annotations.json"
OUTPUT = PREPARED_DIR / "smoothie_recorded_steps.json"
SEED = 0
STEPS_PER_CORRECT_VIDEO = 3


def frame_labels(n_frames: int, steps: list[dict]) -> np.ndarray:
    """Step index per frame; unannotated steps keep their index but get no frames."""
    times = np.arange(n_frames, dtype=np.float64) / FPS
    labels = np.full(n_frames, -1, dtype=np.int8)
    for step_index, step in enumerate(steps):
        if step["start"] is None or step["end"] is None:
            continue
        labels[(times >= step["start"]) & (times < step["end"])] = step_index
    return labels


def features_for(directory: Path, videos: list[dict], encoder: str, cache_name: str) -> list[np.ndarray]:
    cache_path = PREPARED_DIR / cache_name
    version = HISTOGRAM_FEATURE_VERSION if encoder == "classical" else None
    cached = load_cached_recordings(cache_path, directory, videos, version)
    if cached is not None:
        return cached
    if encoder == "clip":
        from ml_pairwise_dtw import embed_frames, load_clip

        model, device = load_clip()
    features = []
    for index, video in enumerate(videos, start=1):
        if encoder == "classical":
            frames = sample_frames(directory / video["file"], 64)
            features.append(np.stack([frame_feature(frame) for frame in frames]))
        else:
            frames = sample_frames(directory / video["file"], 224, bicubic=True)
            features.append(embed_frames(model, device, frames))
        print(f"{encoder} [{index}/{len(videos)}] {video['file']} {len(frames)} frames", flush=True)
    save_recording_cache(cache_path, directory, videos, features, version)
    return features


def test_cases(recordings: list[dict], n_steps: int) -> list[dict]:
    """One case per wrong video and three per correct video, as 1-based steps."""
    rng = random.Random(SEED)
    cases = []
    for video in recordings:
        if video["outcome"] == "correct":
            for step in sorted(rng.sample(range(1, n_steps + 1), STEPS_PER_CORRECT_VIDEO)):
                cases.append({"file": video["file"], "test_step": step, "cut_step": step})
            continue
        test_step = int(video["deviation_step"])
        cut_step = next(
            step["index"]
            for step in video["steps"]
            if step["index"] >= test_step and step["start"] is not None
        )
        cases.append({"file": video["file"], "test_step": test_step, "cut_step": cut_step})
    return cases


def main() -> int:
    reference_payload = json.loads(REFERENCE_ANNOTATIONS.read_text())
    recorded_payload = json.loads(SPEC.annotations_path().read_text())
    if recorded_payload["step_labels"] != reference_payload["step_labels"]:
        raise ValueError("reference and recorded annotations use different steps")
    n_steps = len(reference_payload["step_labels"])
    reference_video = reference_payload["videos"][0]
    recording_dir = SPEC.recording_dir()
    recordings = [
        video for video in recorded_payload["videos"]
        if (recording_dir / video["file"]).is_file()
    ]
    by_file = {video["file"]: video for video in recordings}
    cases = test_cases(recordings, n_steps)

    results = []
    for encoder in ("classical", "clip"):
        summary = json.loads(SPEC.summary_path(encoder).read_text())
        cutoff = float(summary["global"]["mean"] + 2.0 * summary["global"]["std"])
        reference_features = features_for(
            recording_dir.parent, [reference_video], encoder,
            f"smoothie_reference_{encoder}_features.npz",
        )[0]
        recording_features = dict(zip(
            [video["file"] for video in recordings],
            features_for(recording_dir, recordings, encoder, SPEC.recording_cache(encoder).name),
        ))

        reference_labels = frame_labels(len(reference_features), reference_video["steps"])
        annotated = np.flatnonzero(reference_labels >= 0)
        region = slice(annotated[0], annotated[-1] + 1)
        reference = reference_features[region]
        labels = reference_labels[region]
        reference_times = np.arange(len(reference_features), dtype=np.float64)[region] / FPS

        print(f"\n{encoder}: COIN whole-video cutoff {cutoff:.3f}", flush=True)
        print(
            f"  {'video':<18} {'test':>4} {'cut':>4} {'cut s':>6} | {'stop s':>6} {'stop':>5} "
            f"{'mapped':>6} {'open-end':>8} {'dev':>4} | {'segment':>7} {'dev':>4}",
            flush=True,
        )
        for case in cases:
            video = by_file[case["file"]]
            features = recording_features[case["file"]]
            cut = video["steps"][case["cut_step"] - 1]
            cut_time = 0.5 * (cut["start"] + cut["end"])
            prefix_count = min(max(int(np.ceil(cut_time * FPS - 1e-9)), 2), len(features))
            prefix = features[:prefix_count]

            cost = 1.0 - prefix @ reference.T
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
                _dtw_segment_distance(features[query_index], reference[reference_index])
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
                "reached_test_step": bool(step_count[test_index] > 0),
                "open_end_distance": open_end,
                "open_end_deviation": open_end is not None and open_end > cutoff,
                "segment_distance": segment,
                "segment_deviation": segment is not None and segment > cutoff,
                "cutoff": cutoff,
            }
            results.append(row)
            open_text = "none" if open_end is None else f"{open_end:.3f}"
            segment_text = "none" if segment is None else f"{segment:.3f}"
            print(
                f"  {case['file']:<18} {case['test_step']:>4} {case['cut_step']:>4} "
                f"{cut_time:>6.1f} | {row['stop_time_s']:>6.1f} {stop_step or 'gap':>5} "
                f"{'yes' if row['mapped_to_cut_step'] else 'no':>6} {open_text:>8} "
                f"{'yes' if row['open_end_deviation'] else 'no':>4} | {segment_text:>7} "
                f"{'yes' if row['segment_deviation'] else 'no':>4}",
                flush=True,
            )

    OUTPUT.write_text(json.dumps({"seed": SEED, "results": results}, indent=2) + "\n")
    print(f"\nwrote {OUTPUT.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
