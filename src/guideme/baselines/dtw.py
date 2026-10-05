"""Connect timestamped sampling to raw subsequence DTW checkpoint matches.

The descriptor temporarily comes from the existing pairwise script, so the
repository root must be importable alongside src. That legacy import also
loads its existing dependencies and path setup; its entry point is not run.
"""

import math
from pathlib import Path

import numpy as np

from scripts.classical_pairwise_dtw import frame_feature

from ..alignment import align_feature_window
from ..annotations import load_reference_annotations, reference_step_at_time
from ..features import encode_clip_frames
from ..video import sample_video_frames


def run_classical_dtw_checkpoint(
    reference_path: str | Path,
    execution_path: str | Path,
    reference_annotations_path: str | Path,
    *,
    reference_duration_s: float,
    checkpoint_s: float,
    window_s: float,
    fps: float = 5.0,
) -> dict:
    """Keep the classical-only entry point and its existing result format."""
    return run_dtw_checkpoint(
        reference_path, execution_path, reference_annotations_path,
        reference_duration_s=reference_duration_s, checkpoint_s=checkpoint_s,
        window_s=window_s, fps=fps,
    )


def run_dtw_checkpoint(
    reference_path: str | Path,
    execution_path: str | Path,
    reference_annotations_path: str | Path,
    *,
    reference_duration_s: float,
    checkpoint_s: float,
    window_s: float,
    fps: float = 5.0,
    encoder: str = "classical",
    clip_model=None,
    device=None,
    batch_size: int = 32,
    warp_penalty: float = 0.0,
) -> dict:
    """Return a raw reference match for one causal execution window.

    The caller supplies explicit paths, the reference's duration bound and
    their correspondence. Sample reference [0, reference_duration_s) and
    execution [max(0, checkpoint_s-window_s), checkpoint_s) at up to fps in
    BGR. Classical features use 64x64 frames; CLIP uses 224x224 frames and
    encode_clip_frames with a caller-owned, already loaded evaluation model
    on the supplied device. CLIP loading, checkpoint identity and provenance
    remain caller responsibilities. No model loading or downloads occur.
    The supplied reference bound does not guarantee sampling to physical EOF.
    Actual source timestamps come from the existing sampler.

    Both encoders use subsequence DTW: match every execution-window row,
    with free reference start and end. This is not full-sequence DTW or
    alignment of the entire execution prefix. No future execution frames
    are features, and no past-window state is retained.
    An optional nonnegative warp_penalty discourages repeated frames in
    either encoder's alignment. Costs include this penalty. Positive values
    are recorded in the result; zero preserves the previous result structure.

    Require a nonempty reference CSV. Map every exactly tied best endpoint
    through its reference timestamp: a candidate needs one unambiguous row
    at every endpoint, all with the same step ID. Gaps and overlapping labels
    are unknown. Retain the matcher's earliest representative and all ties.
    A candidate is an unfiltered visual match, not accepted progress or proof
    of execution correctness. No execution annotations or history are read.

    Numeric parameters must be finite positive int/float values representable
    as floats, except warp_penalty also permits zero; validate before file
    access. Paths expand ~ and resolve from the caller's location.
    Return only a JSON-serializable in-memory report;
    source files are read only and validation/decode errors propagate.
    """
    parameters = []
    for name, value in (
        ("reference_duration_s", reference_duration_s),
        ("checkpoint_s", checkpoint_s),
        ("window_s", window_s),
        ("fps", fps),
        ("warp_penalty", warp_penalty),
    ):
        bound = "nonnegative" if name == "warp_penalty" else "positive"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite {bound} number")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite {bound} number") from error
        invalid_bound = value < 0 if name == "warp_penalty" else value <= 0
        if not math.isfinite(value) or invalid_bound:
            raise ValueError(f"{name} must be a finite {bound} number")
        parameters.append(value)
    reference_duration_s, checkpoint_s, window_s, fps, warp_penalty = parameters

    if encoder not in ("classical", "clip"):
        raise ValueError("encoder must be classical or clip")
    if encoder == "clip":
        if clip_model is None or device is None:
            raise ValueError("CLIP requires an already loaded model and its device")
        if getattr(clip_model, "training", None) is not False:
            raise ValueError("CLIP model must already be in evaluation mode")
        if not callable(getattr(clip_model, "encode_image", None)):
            raise ValueError("CLIP model must provide encode_image")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
    elif clip_model is not None or device is not None:
        raise ValueError("CLIP model and device apply only to the clip encoder")
    frame_size = (224, 224) if encoder == "clip" else (64, 64)

    reference = Path(reference_path).expanduser().resolve()
    execution = Path(execution_path).expanduser().resolve()
    annotations = Path(reference_annotations_path).expanduser().resolve()
    rows = load_reference_annotations(annotations, reference_duration_s)
    if not rows:
        raise ValueError("Reference annotations must contain at least one step")

    window_start_s = max(0.0, checkpoint_s - window_s)
    reference_frames, reference_times = sample_video_frames(
        reference, 0.0, reference_duration_s, fps=fps, size=frame_size,
    )
    execution_frames, execution_times = sample_video_frames(
        execution, window_start_s, checkpoint_s, fps=fps, size=frame_size,
    )
    if encoder == "clip":
        reference_features = encode_clip_frames(
            clip_model, device, reference_frames, batch_size=batch_size,
        )
        execution_features = encode_clip_frames(
            clip_model, device, execution_frames, batch_size=batch_size,
        )
    else:
        reference_features = np.stack([frame_feature(frame) for frame in reference_frames])
        execution_features = np.stack([frame_feature(frame) for frame in execution_frames])
    alignment = align_feature_window(
        reference_features, execution_features, warp_penalty=warp_penalty,
    )

    reference_time_s = float(reference_times[alignment["end_reference_index"]])
    tied_reference_times_s = [
        float(reference_times[index]) for index in alignment["tied_endpoints"]
    ]
    tied_steps = [reference_step_at_time(rows, time) for time in tied_reference_times_s]
    candidate_step = None
    unknown_reason = None
    if any(step is None for step in tied_steps):
        unknown_reason = "unmapped_reference_time"
    elif len({step["step_id"] for step in tied_steps}) > 1:
        unknown_reason = "tied_reference_steps"
    else:
        candidate_step = tied_steps[0].copy()

    result = {
        "method": f"{encoder}_subsequence_dtw",
        "reference_path": str(reference),
        "execution_path": str(execution),
        "reference_annotations_path": str(annotations),
        "reference_duration_s": reference_duration_s,
        "checkpoint_s": checkpoint_s,
        "window_s": window_s,
        "window_start_s": window_start_s,
        "fps": fps,
        "frame_size": list(frame_size),
        "reference_times_s": reference_times.tolist(),
        "execution_times_s": execution_times.tolist(),
        "alignment": alignment,
        "reference_time_s": reference_time_s,
        "tied_reference_times_s": tied_reference_times_s,
        "candidate_step": candidate_step,
        "unknown_reason": unknown_reason,
    }
    if warp_penalty > 0:
        result["warp_penalty"] = warp_penalty
    return result


def run_dtw_replay(
    reference_path: str | Path,
    execution_path: str | Path,
    reference_annotations_path: str | Path,
    *,
    reference_duration_s: float,
    checkpoint_times_s: list[float],
    window_s: float,
    fps: float = 5.0,
    encoder: str = "classical",
    clip_model=None,
    device=None,
    batch_size: int = 32,
    warp_penalty: float = 0.0,
) -> list[dict]:
    """Run the existing causal window matcher at explicit ordered checkpoints.

    Validate shared settings and the entire checkpoint list before file work.
    Times must be positive finite int/float values representable as floats,
    strictly increasing after conversion; do not sort or generate a cadence.
    An empty list returns [] after shared settings have been validated.

    Call run_dtw_checkpoint once per time, using the same paths, settings and
    caller-loaded CLIP model. Each call uses execution frames strictly before
    that checkpoint and the complete supplied reference interval. Retain one
    minimal in-memory report per checkpoint, including ordinary exceptions as
    typed errors, and continue without retrying or dropping failed checks.
    Reports work with dtw_prediction_from_report; exception messages are omitted.
    KeyboardInterrupt and SystemExit propagate.

    This initial loop processes source videos again at each checkpoint. It
    does not sleep to match playback, cache reference features, load a model,
    write files, score labels, retain recognized-step history or emit warnings.
    The caller supplies a shared grid within the execution duration and records
    source/model identity and processing time when persisting an experiment.
    """
    parameters = []
    for name, value in (
        ("reference_duration_s", reference_duration_s), ("window_s", window_s),
        ("fps", fps), ("warp_penalty", warp_penalty),
    ):
        bound = "nonnegative" if name == "warp_penalty" else "positive"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite {bound} number")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite {bound} number") from error
        if not math.isfinite(value) or (value < 0 if name == "warp_penalty" else value <= 0):
            raise ValueError(f"{name} must be a finite {bound} number")
        parameters.append(value)
    reference_duration_s, window_s, fps, warp_penalty = parameters

    if encoder not in ("classical", "clip"):
        raise ValueError("encoder must be classical or clip")
    if encoder == "clip":
        if clip_model is None or device is None:
            raise ValueError("CLIP requires an already loaded model and its device")
        if getattr(clip_model, "training", None) is not False:
            raise ValueError("CLIP model must already be in evaluation mode")
        if not callable(getattr(clip_model, "encode_image", None)):
            raise ValueError("CLIP model must provide encode_image")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
    elif clip_model is not None or device is not None:
        raise ValueError("CLIP model and device apply only to the clip encoder")

    if not isinstance(checkpoint_times_s, list):
        raise ValueError("checkpoint_times_s must be a list")
    times = []
    for index, value in enumerate(checkpoint_times_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"checkpoint {index}: require a finite positive number")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"checkpoint {index}: require a finite positive number") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"checkpoint {index}: require a finite positive number")
        if times and value <= times[-1]:
            raise ValueError("checkpoint_times_s must be strictly increasing")
        times.append(value)

    reports = []
    for time_s in times:
        checkpoint = {"end_s": time_s, "window_s": window_s, "fps": fps}
        if warp_penalty > 0:
            checkpoint["warp_penalty"] = warp_penalty
        report = {
            "schema_version": 1, "status": "running", "encoder": {"name": encoder},
            "checkpoint": checkpoint, "result": None,
        }
        try:
            report["result"] = run_dtw_checkpoint(
                reference_path, execution_path, reference_annotations_path,
                reference_duration_s=reference_duration_s, checkpoint_s=time_s,
                window_s=window_s, fps=fps, encoder=encoder,
                clip_model=clip_model, device=device, batch_size=batch_size,
                warp_penalty=warp_penalty,
            )
            report["status"] = "ok"
        except Exception as error:
            report["status"] = "error"
            report["error"] = {"stage": "run_checkpoint", "type": type(error).__name__}
        reports.append(report)
    return reports
