"""Upload a local MP4 so Gemini can read it."""

import math
from pathlib import Path
from time import perf_counter, sleep

from google import genai
from google.genai import types


def upload_video(
    client: genai.Client,
    path: str | Path,
    *,
    processing_timeout_s: float = 120,
    poll_interval_s: float = 2,
) -> dict:
    """Upload a local MP4 and wait for an ACTIVE file usable by Gemini.

    Returns uri/name and observed upload/processing durations. The caller
    prepares the media, supplies authentication, and manages remote reuse and
    deletion; this helper neither decodes nor clips the video.

    The processing deadline starts after upload and is checked between SDK
    calls. Configure network request timeouts on the supplied client; the
    deadline cannot interrupt an in-flight SDK call. API errors propagate;
    failed/unknown file states raise RuntimeError and expiry raises TimeoutError.
    Uploaded files remain on the server even if readiness checking fails.
    """
    for name, value in (
        ("processing_timeout_s", processing_timeout_s),
        ("poll_interval_s", poll_interval_s),
    ):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")
    video_path = Path(path).expanduser().resolve()
    if (
        video_path.suffix.lower() != ".mp4" or not video_path.is_file()
        or video_path.stat().st_size == 0
    ):
        raise ValueError("path must point to a nonempty MP4 file")

    started = perf_counter()
    media = client.files.upload(file=str(video_path), config={"mime_type": "video/mp4"})
    uploaded_at = perf_counter()
    file_name = media.name
    if not file_name:
        raise RuntimeError("Upload response contains no file name")

    while True:
        elapsed = perf_counter() - uploaded_at
        if media.state == types.FileState.FAILED:
            raise RuntimeError(f"Video processing failed for {file_name}: {media.error}")
        if media.state not in (types.FileState.PROCESSING, types.FileState.ACTIVE):
            raise RuntimeError(f"Unexpected file state for {file_name}: {media.state}")
        if elapsed >= processing_timeout_s:
            raise TimeoutError(f"Timed out waiting for video processing: {file_name}")
        if media.state == types.FileState.ACTIVE:
            if not media.uri:
                raise RuntimeError(f"Ready file contains no video URI: {file_name}")
            return {
                "uri": media.uri,
                "name": file_name,
                "upload_duration_s": uploaded_at - started,
                "processing_duration_s": elapsed,
            }
        sleep(min(poll_interval_s, processing_timeout_s - elapsed))
        if perf_counter() - uploaded_at >= processing_timeout_s:
            raise TimeoutError(f"Timed out waiting for video processing: {file_name}")
        media = client.files.get(name=file_name)
