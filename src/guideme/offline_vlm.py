"""Offline Gemini alignment: an annotated reference against a whole unlabelled execution.

The online check (vlm.predict_deviation) sees an execution prefix and reports at
most one current mistake. This module sees the complete execution and reports,
for every reference checklist step, whether and where it happened, plus a list
of deviations. Uploads reuse vlm.upload_video; the request follows the same
response_json_schema pattern.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import cv2
import imageio_ffmpeg
from jsonschema import Draft202012Validator, ValidationError


STEP_STATUSES = ["done", "skipped", "out_of_order", "done_incorrectly", "unclear"]
DEVIATION_TYPES = ["skipped_step", "wrong_order", "execution_error", "extra_action"]

OFFLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference_step_id": {"type": "string"},
                    "status": {"type": "string", "enum": STEP_STATUSES},
                    "execution_start_s": {"type": ["number", "null"], "minimum": 0},
                    "execution_end_s": {"type": ["number", "null"], "minimum": 0},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "reference_step_id", "status", "execution_start_s",
                    "execution_end_s", "evidence", "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "deviations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": DEVIATION_TYPES},
                    "reference_step_id": {"type": ["string", "null"]},
                    "execution_time_s": {"type": ["number", "null"], "minimum": 0},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "message": {"type": "string"},
                },
                "required": [
                    "type", "reference_step_id", "execution_time_s", "evidence",
                    "confidence", "message",
                ],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["steps", "deviations", "summary"],
    "additionalProperties": False,
}

# Statuses that require a located execution interval, and those that forbid one.
_LOCATED = {"done", "out_of_order", "done_incorrectly"}
_UNLOCATED = {"skipped"}


def coin_checklist_rows(info: dict, offset_s: float = 0.0) -> list[dict]:
    """Turn one COIN annotation into reference checklist rows (format v1).

    step_id is s001, s002, ... in COIN order; action is the COIN label; times
    are COIN segment seconds minus offset_s, so they match a clip that starts
    at offset_s in the source video. requirements is left blank: COIN states none.
    """
    rows = []
    for index, step in enumerate(info["annotation"], start=1):
        start, end = (float(value) - offset_s for value in step["segment"])
        rows.append({
            "step_id": f"s{index:03d}",
            "action": step["label"],
            "start_s": round(max(start, 0.0), 3),
            "end_s": round(end, 3),
            "requirements": "",
        })
    return rows


def write_checklist_csv(rows: list[dict], path: str | Path) -> Path:
    """Write rows with the exact header that annotations.load_reference_annotations requires."""
    import csv

    output = Path(path)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["step_id", "action", "start_s", "end_s", "requirements"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def video_duration_s(path: str | Path) -> float:
    """Duration from OpenCV frame count over nominal FPS (the method run_vlm uses)."""
    capture = cv2.VideoCapture(str(path))
    try:
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if not math.isfinite(frames) or frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Cannot read duration of {path}")
    return frames / fps


def cut_segments(source: str | Path, segments: list[tuple[float, float]], output: str | Path) -> Path:
    """Write a video-only MP4 made of source segments [start, end) in the given order.

    Audio is dropped so narration cannot reveal the steps. Re-encodes as H.264
    CRF 18. Never overwrites; a failed cut leaves no file at output.
    """
    if not segments or any(
        not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b) for a, b in segments
    ):
        raise ValueError("segments must be nonempty finite [start, end) pairs")
    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() != ".mp4" or not output.parent.is_dir():
        raise ValueError("output must be a new .mp4 path under an existing directory")
    parts = [
        f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v{i}]"
        for i, (a, b) in enumerate(segments)
    ]
    joined = "".join(f"[v{i}]" for i in range(len(segments)))
    graph = ";".join(parts) + f";{joined}concat=n={len(segments)}:v=1:a=0," \
        "pad=ceil(iw/2)*2:ceil(ih/2)*2[out]"
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".guideme-cut-") as directory:
        prepared = Path(directory) / "cut.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n",
            "-i", str(source), "-filter_complex", graph, "-map", "[out]",
            "-an", "-sn", "-dn", "-map_metadata", "-1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(prepared),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Cut failed: {completed.stderr.strip()[-1000:]}")
        os.link(prepared, output)
    return output


def _validate(candidate: dict, step_ids: list[str], execution_duration_s: float | None) -> None:
    Draft202012Validator(OFFLINE_SCHEMA).validate(candidate)
    returned = [step["reference_step_id"] for step in candidate["steps"]]
    if sorted(returned) != sorted(step_ids) or len(set(returned)) != len(returned):
        raise ValueError("steps must list every checklist step_id exactly once")
    slack = 1.0  # seconds of tolerance past the end for rounding
    for step in candidate["steps"]:
        start, end = step["execution_start_s"], step["execution_end_s"]
        if step["status"] in _LOCATED and (start is None or end is None):
            raise ValueError(f"{step['reference_step_id']}: status {step['status']} needs execution times")
        if step["status"] in _UNLOCATED and (start is not None or end is not None):
            raise ValueError(f"{step['reference_step_id']}: a skipped step cannot have execution times")
        if start is not None and end is not None and start > end:
            raise ValueError(f"{step['reference_step_id']}: execution_start_s is after execution_end_s")
        for value in (start, end):
            if value is not None and execution_duration_s is not None and value > execution_duration_s + slack:
                raise ValueError(f"{step['reference_step_id']}: time is past the end of the execution")
    for deviation in candidate["deviations"]:
        step_id = deviation["reference_step_id"]
        if step_id is not None and step_id not in step_ids:
            raise ValueError("deviation reference_step_id must be a checklist step_id or null")
        if not deviation["message"].strip() or not deviation["evidence"].strip():
            raise ValueError("each deviation needs a nonempty message and evidence")
        t = deviation["execution_time_s"]
        if t is not None and execution_duration_s is not None and t > execution_duration_s + slack:
            raise ValueError("deviation time is past the end of the execution")


def predict_offline_alignment(
    client,
    reference_uri: str,
    execution_uri: str,
    checklist_rows: list[dict],
    *,
    execution_duration_s: float | None = None,
    fps: float = 1,
    model: str = "gemini-3.5-flash-lite",
) -> dict:
    """Ask Gemini for a step-by-step alignment of a whole execution to the reference.

    Both URIs must be ready uploaded MP4s (vlm.upload_video). checklist_rows are
    validated reference rows (annotations.load_reference_annotations) whose times
    are seconds in the reference video. The execution carries no labels.

    Returns prediction (or None with validation_error), the request, the raw
    response and the SDK call duration. API errors raise.
    """
    for name, uri in (("reference_uri", reference_uri), ("execution_uri", execution_uri)):
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"{name} must be a nonempty video URI")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if not checklist_rows:
        raise ValueError("checklist_rows must contain at least one reference step")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    step_ids = [row["step_id"] for row in checklist_rows]

    parts = []
    for label, uri in (("REFERENCE", reference_uri), ("EXECUTION", execution_uri)):
        parts.extend([
            {"text": label},
            {
                "file_data": {"file_uri": uri, "mime_type": "video/mp4"},
                "video_metadata": {"fps": fps},
                "media_resolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                "media_processing": "STATIC",
            },
        ])
    prompt = Path(__file__).with_name("prompts").joinpath("offline_alignment.txt").read_text(
        encoding="utf-8"
    )
    checklist = json.dumps(
        [{k: row[k] for k in ("step_id", "action", "start_s", "end_s", "requirements")}
         for row in checklist_rows],
        ensure_ascii=False, allow_nan=False, indent=2,
    )
    parts.append({"text": prompt})
    parts.append({"text": f"CHECKLIST\n{checklist}"})
    if execution_duration_s is not None:
        parts.append({"text": f"EXECUTION_DURATION_S\n{execution_duration_s:.2f}"})

    request = {
        "model": model,
        "contents": [{"role": "user", "parts": parts}],
        "config": {
            "response_mime_type": "application/json",
            "response_json_schema": OFFLINE_SCHEMA,
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
        _validate(candidate, step_ids, execution_duration_s)
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
