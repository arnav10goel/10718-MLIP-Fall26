"""Visual-only reference preparation, independent of datasets and predictors."""

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from time import perf_counter

from .media import FileFrameSource, TimestampFrameSampler, encode_frame_jpeg


def prepare_reference(
    path: str | Path,
    *,
    fps: float = 1,
    max_frames: int = 256,
    max_side: int = 640,
    jpeg_quality: int = 85,
    data_root: str | Path | None = None,
    timeout_s: float = 30,
) -> dict:
    """Decode a complete reference into a bounded in-memory JPEG preparation.

    The full reference is available before execution monitoring. Preserve fixed
    cadence and include the actual final frame if not already selected. Count
    unique images against max_frames; overflow raises rather than thinning or
    returning a truncated ready object. A one-frame source fits a budget of one.
    Successful preparation requires clean EOF and unchanged source bytes/stat
    identity. The caller logs failures; no partial artifact is published here.

    Ready means technically prepared, not semantically certified correct.
    last_elapsed_s is the last frame's timestamp, not a duration estimate.
    Source/config/frame metadata bind reference_id; paths and measured runtime
    are excluded, so relocating identical bytes does not change visual identity.
    Operational timeout is per decoder read and is not a visual configuration.

    Relative paths follow GUIDEME_DATA_ROOT or repository data/. No file is
    written, no execution or evaluation labels are read, and no model is called.
    Retain only bounded JPEG samples and the most recent raw frame. The caller
    owns the returned object; hashes are provenance, not mutation protection.
    """
    started = perf_counter()
    sampler = TimestampFrameSampler(fps=fps, max_frames=max_frames)
    if type(max_side) is not int or max_side <= 0:
        raise ValueError("max_side must be a positive integer")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer from 1 to 100")
    try:
        valid_timeout = type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("timeout_s must be finite and positive")
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("path must be a nonempty local path")
    source_path = Path(path).expanduser()
    if not source_path.is_absolute():
        configured = data_root if data_root is not None else os.environ.get("GUIDEME_DATA_ROOT") or Path(__file__).resolve().parents[2] / "data"
        if not isinstance(configured, (str, Path)) or not str(configured):
            raise ValueError("data_root must be a nonempty absolute path")
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("data_root must be absolute")
        source_path = root / source_path
    source_path = source_path.resolve()

    def snapshot():
        # Check the opened descriptor before any read. Nonblocking open also
        # avoids hanging if a nonregular input is a FIFO (or replaces the path).
        descriptor = os.open(source_path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Reference must be a regular video file")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
        finally:
            if descriptor is not None:
                os.close(descriptor)
        current = source_path.stat()
        identities = [(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
                      for item in (before, after, current)]
        if identities[0] != identities[1] or identities[0] != identities[2]:
            raise RuntimeError("Reference source changed while hashing")
        return digest, identities[0]

    def json_hash(value):
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    source_hash, source_identity = snapshot()
    config = {"fps": sampler.fps, "max_frames": max_frames, "max_side": max_side,
              "jpeg_quality": jpeg_quality, "endpoint_policy": "include_first_and_last",
              "budget_policy": "error"}
    frames = []
    decoded_count = 0
    first_pts = None
    last = None
    with FileFrameSource(source_path, timeout_s=timeout_s) as source:
        # All reference visuals are permitted. Never use this full-source gate
        # for execution observations, which require their actual causal cutoff.
        while (frame := source.read(sys.float_info.max)) is not None:
            if first_pts is None:
                first_pts = frame["source_pts_s"]
            decoded_count += 1
            last = frame
            selection = sampler.consider(frame, frame["elapsed_s"])
            if selection is not None:
                image = encode_frame_jpeg(selection["frame"], max_side=max_side, jpeg_quality=jpeg_quality)
                image.update(selection_reason="cadence", target_s=selection["target_s"],
                             skipped_targets=selection["skipped_targets"])
                frames.append(image)
        if not source.eof:
            raise RuntimeError("Reference decoding did not reach clean EOF")
    if last is None:
        raise RuntimeError("Reference contains no decoded frames")
    if not frames or frames[-1]["frame_index"] != last["frame_index"]:
        if len(frames) >= max_frames:
            raise RuntimeError("Reference frame budget cannot include the final frame")
        image = encode_frame_jpeg(last, max_side=max_side, jpeg_quality=jpeg_quality)
        image.update(selection_reason="final_frame", target_s=None, skipped_targets=None)
        frames.append(image)
    try:
        final_hash, final_identity = snapshot()
    except (OSError, ValueError) as exc:
        raise RuntimeError("Reference source became invalid or unavailable during preparation") from exc
    if final_hash != source_hash or final_identity != source_identity:
        raise RuntimeError("Reference source changed during preparation")

    result = {
        "schema_version": 1, "status": "ready", "source_sha256": source_hash,
        "config_sha256": json_hash(config), "config": config, "frames": frames,
        "decoded_frame_count": decoded_count, "sample_count": len(frames),
        "first_source_pts_s": first_pts, "last_source_pts_s": last["source_pts_s"],
        "last_elapsed_s": last["elapsed_s"],
        "total_jpeg_bytes": sum(len(image["jpeg_bytes"]) for image in frames),
    }
    identity = {key: result[key] for key in (
        "schema_version", "source_sha256", "config_sha256", "decoded_frame_count",
        "first_source_pts_s", "last_source_pts_s", "last_elapsed_s")}
    identity["frames"] = [{key: value for key, value in image.items() if key != "jpeg_bytes"}
                          for image in frames]
    result["reference_id"] = json_hash(identity)
    result["preparation_s"] = perf_counter() - started
    return result
