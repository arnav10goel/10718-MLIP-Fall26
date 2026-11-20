"""Validate source-frame provenance shared by assessment state and policy."""

import math
import re


def validate_assessment_frame(frame, source_sha256, cutoff_s):
    """Return the actual source elapsed time of a native or JPEG frame.

    Native frames retain their historical source/video/cutoff clock checks.
    JPEG source time is separate from observation capture/receipt time; its
    causal availability is established by the binder, not this provenance check.
    This function neither validates image bytes nor mutates its inputs.
    """
    def number(value, *, signed=False):
        try:
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and (signed or value >= 0))
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite source-frame number")
        return value

    if (not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None):
        raise ValueError("Expected a lowercase source SHA-256 identity")
    cutoff = number(cutoff_s)
    native_fields = {"source_sha256", "frame_index", "source_pts_s", "elapsed_s", "video_pts_s"}
    jpeg_fields = {"media_type", "source_sha256", "frame_index", "source_pts_s", "elapsed_s",
                   "jpeg_sha256", "jpeg_size_bytes"}
    if not isinstance(frame, dict) or set(frame) not in (native_fields, jpeg_fields):
        raise ValueError("Source frame has missing or unsupported fields")
    if (frame["source_sha256"] != source_sha256 or type(frame["frame_index"]) is not int
            or frame["frame_index"] < 0):
        raise ValueError("Source frame identity disagrees")
    elapsed = number(frame["elapsed_s"])
    if set(frame) == native_fields:
        source_pts, video_pts = number(frame["source_pts_s"]), number(frame["video_pts_s"])
        if source_pts != elapsed or elapsed > cutoff or abs(video_pts - elapsed) > 0.000001:
            raise ValueError("Native frame disagrees with its source or cutoff")
    else:
        number(frame["source_pts_s"], signed=True)
        if (frame["media_type"] != "image/jpeg" or not isinstance(frame["jpeg_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", frame["jpeg_sha256"]) is None
                or type(frame["jpeg_size_bytes"]) is not int or frame["jpeg_size_bytes"] <= 0):
            raise ValueError("JPEG frame provenance is invalid")
    return elapsed
