"""One Gemini prediction from a tutorial and an observed execution prefix."""

import json
import math
from pathlib import Path
from time import perf_counter, sleep

from google import genai
from google.genai import types
from jsonschema import Draft202012Validator, ValidationError


PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "current_step": {"type": "string"},
        "reference_step_id": {"type": ["string", "null"]},
        "reference_time_s": {"type": ["number", "null"], "minimum": 0},
        "problem": {
            "type": "string",
            "enum": ["none", "skipped_step", "wrong_order", "execution_error"],
        },
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "message": {"type": "string"},
    },
    "required": [
        "current_step", "reference_step_id", "reference_time_s", "problem", "evidence",
        "confidence", "message",
    ],
    "additionalProperties": False,
}


def predict_deviation(
    client: genai.Client,
    reference_uri: str,
    execution_prefix_uri: str,
    checklist: str | None = None,
    fps: float = 2,
    *,
    model: str = "gemini-3.5-flash-lite",
) -> dict:
    """Request one candidate diagnosis and retain its request and raw response.

    Both URIs must refer to ready, uploaded MP4s. The caller must physically
    clip the execution to the observed prefix and remove its audio before
    uploading; this function cannot verify those properties from a URI.
    The optional checklist must describe only the reference tutorial.
    A JSON-array checklist supplies exact nonempty string step_id values;
    other checklist text remains notes without IDs. reference_step_id names
    the observed step, or is null when unknown or no IDs are supplied.
    ID membership does not verify the observed action or reference timestamp.
    The requested model is recorded separately from the returned model version.

    Invalid model output returns prediction=None with a validation_error.
    Invalid arguments and SDK/API errors raise. Token usage and returned model
    information are retained in response when the API supplies them.
    request_duration_s measures the SDK call, excluding upload/processing.
    This function does not apply alert thresholds or suppress repeat warnings.
    """
    for name, uri in (
        ("reference_uri", reference_uri),
        ("execution_prefix_uri", execution_prefix_uri),
    ):
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"{name} must be a nonempty video URI")
    if (
        isinstance(fps, bool) or not isinstance(fps, (int, float))
        or not math.isfinite(fps) or fps <= 0
    ):
        raise ValueError("fps must be a finite positive number")
    if checklist is not None and (
        not isinstance(checklist, str) or not checklist.strip()
    ):
        raise ValueError("checklist must be nonempty text or None")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")

    reference_step_ids = set()
    if checklist is not None:
        try:
            checklist_rows = json.loads(checklist)
        except json.JSONDecodeError:
            checklist_rows = None
        if isinstance(checklist_rows, list):
            reference_step_ids = {
                row["step_id"] for row in checklist_rows
                if isinstance(row, dict) and isinstance(row.get("step_id"), str)
                and row["step_id"].strip()
            }

    parts = []
    for label, uri in (("REFERENCE", reference_uri), ("EXECUTION", execution_prefix_uri)):
        parts.extend([
            {"text": label},
            {
                "file_data": {"file_uri": uri, "mime_type": "video/mp4"},
                "video_metadata": {"fps": fps},
                "media_resolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                "media_processing": "STATIC",
            },
        ])
    prompt = Path(__file__).with_name("prompts").joinpath("deviation.txt").read_text(
        encoding="utf-8"
    )
    parts.append({"text": prompt})
    if checklist is not None:
        parts.append({"text": f"CHECKLIST\n{checklist}"})

    request = {
        "model": model,
        "contents": [{"role": "user", "parts": parts}],
        "config": {
            "response_mime_type": "application/json",
            "response_json_schema": PREDICTION_SCHEMA,
        },
    }
    started = perf_counter()
    response = client.models.generate_content(**request)
    duration = perf_counter() - started
    raw_response = response.model_dump(mode="json")

    prediction = None
    validation_error = None
    try:
        candidates = raw_response.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Response must contain exactly one candidate")
        if candidates[0].get("finish_reason") != "STOP":
            raise ValueError(f"Candidate did not complete: {candidates[0].get('finish_reason')}")
        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Response contains no prediction text")
        candidate = json.loads(text)
        Draft202012Validator(PREDICTION_SCHEMA).validate(candidate)
        if (
            candidate["reference_step_id"] is not None
            and candidate["reference_step_id"] not in reference_step_ids
        ):
            raise ValueError("reference_step_id must be a supplied checklist ID or null")
        for field in ("confidence", "reference_time_s"):
            if candidate[field] is not None and not math.isfinite(candidate[field]):
                raise ValueError(f"{field} must be finite")
        if candidate["problem"] == "none":
            if candidate["message"] != "":
                raise ValueError("A prediction with problem=none must have an empty message")
        elif not candidate["message"].strip() or not candidate["evidence"].strip():
            raise ValueError("A problem requires a nonempty message and supporting evidence")
        prediction = candidate
    except (ValueError, ValidationError) as error:
        validation_error = error.message if isinstance(error, ValidationError) else str(error)

    return {
        "prediction": prediction,
        "validation_error": validation_error,
        "request": request,
        "response": raw_response,
        "request_duration_s": duration,
    }


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
