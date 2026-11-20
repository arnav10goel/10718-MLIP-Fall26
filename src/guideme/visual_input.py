"""Validate visual model input before request construction."""

import math
from uuid import UUID


def validate_visual_input(model_input):
    """Validate causal JPEG frames and return their available timestamps."""
    def keys(value, required, optional=()):
        if (not isinstance(value, dict) or not set(required) <= value.keys()
                or not value.keys() <= set(required) | set(optional)):
            raise ValueError("Record has missing or unsupported fields")

    def finite_number(value):
        if type(value) not in (int, float):
            raise ValueError("Timestamps and scores must be finite numbers, not booleans")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Timestamps and scores must be finite")
        return value

    def version(record):
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("Expected schema version 1")

    keys(model_input, ("schema_version", "session_id", "observation_id", "cutoff_s",
                       "reference_frames", "execution_frames"))
    version(model_input)
    for name in ("session_id", "observation_id"):
        value = model_input[name]
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a canonical UUID string")
    cutoff = finite_number(model_input["cutoff_s"])
    if cutoff < 0:
        raise ValueError("Cutoff cannot be negative")
    available_times = {}
    for name in ("reference_frames", "execution_frames"):
        frames = model_input[name]
        if not isinstance(frames, list) or not frames:
            raise ValueError("Reference and execution frame lists must be nonempty")
        times = set()
        previous = None
        for frame in frames:
            keys(frame, ("timestamp_s", "mime_type", "jpeg_bytes"))
            timestamp = finite_number(frame["timestamp_s"])
            if timestamp < 0 or (previous is not None and timestamp <= previous):
                raise ValueError("Frame timestamps must be nonnegative and strictly increasing")
            if name == "execution_frames" and timestamp > cutoff:
                raise ValueError("Execution frames cannot follow the cutoff")
            payload = frame["jpeg_bytes"]
            if (frame["mime_type"] != "image/jpeg" or type(payload) is not bytes
                    or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9")):
                raise ValueError("Frames require immutable JPEG bytes and image/jpeg MIME type")
            times.add(timestamp)
            previous = timestamp
        available_times[name] = times

    return available_times
