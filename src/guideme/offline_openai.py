"""Offline alignment through an OpenAI-compatible chat API (vLLM, SGLang, Modal Endpoints).

Same prompt, checklist, JSON schema and answer checks as offline_vlm (Gemini). By
default each video is sent as JPEG frames sampled at a fixed rate, each preceded by
its time in seconds from the start of that video. For a model and server that take
video (video_url), the execution can instead be sent whole, shrunk first to keep the
request small; the server then picks its frames. The reference stays as frames,
since servers such as Modal's GLM-5.3-Flash accept one video per request.
"""

import base64
import copy
import json
import math
from pathlib import Path
import re
import subprocess
from time import perf_counter

import cv2
import imageio_ffmpeg
from jsonschema import ValidationError

from src.guideme.offline_vlm import OFFLINE_SCHEMA, _validate, video_duration_s


PROMPT_PATH = Path(__file__).with_name("prompts") / "offline_alignment.txt"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def sample_frames(
    path: str | Path,
    fps: float,
    *,
    max_frames: int | None = None,
    max_side: int = 640,
    jpeg_quality: int = 85,
) -> list[tuple[float, bytes]]:
    """JPEG frames at 0, 1/fps, 2/fps, ... seconds, as (time_s, jpeg bytes).

    A video that would give more than max_frames is sampled at max_frames evenly
    spaced times instead. Frames are shrunk so the longer side is at most max_side.
    Times are the decoded frame's own time, so they can differ from the target by
    up to half a frame.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    duration = video_duration_s(path)
    count = max(1, math.ceil(duration * fps))
    if max_frames is not None and count > max_frames:
        targets = [index * duration / max_frames for index in range(max_frames)]
    else:
        targets = [index / fps for index in range(count)]

    capture = cv2.VideoCapture(str(path))
    try:
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = sorted({min(round(t * native_fps), total - 1) for t in targets})
        frames = []
        index = 0
        for target in wanted:
            while index < target:
                if not capture.grab():
                    break
                index += 1
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            height, width = frame.shape[:2]
            scale = max_side / max(height, width)
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError(f"Cannot encode frame {target} of {path}")
            frames.append((round(target / native_fps, 3), encoded.tobytes()))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def shrink_video(source: str | Path, output: str | Path, max_side: int = 640) -> Path:
    """Write a smaller video-only copy of source: longer side at most max_side, H.264 CRF 28."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    scale = (f"scale=w={max_side}:h={max_side}:force_original_aspect_ratio=decrease,"
             "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n", "-i", str(source),
        "-vf", scale, "-an", "-sn", "-dn", "-map_metadata", "-1", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Shrink failed: {completed.stderr.strip()[-1000:]}")
    return output


def _video_url_parts(label: str, clip: str | Path, max_side: int) -> list[dict]:
    small = shrink_video(clip, Path(clip).with_name(Path(clip).stem + "_small.mp4"), max_side)
    data = base64.b64encode(small.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": label},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}},
    ]


def _video_parts(label: str, frames: list[tuple[float, bytes]]) -> list[dict]:
    spacing = frames[1][0] - frames[0][0] if len(frames) > 1 else 0.0
    parts = [{"type": "text", "text": (
        f"{label}: {len(frames)} frames, one every {spacing:.2f} s. "
        "Each frame follows its time in seconds from the start of this video."
    )}]
    for time_s, jpeg in frames:
        parts.append({"type": "text", "text": f"t={time_s:.1f}s"})
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
        }})
    return parts


def build_request(
    reference_clip: str | Path,
    execution_clip: str | Path,
    checklist_rows: list[dict],
    *,
    model: str,
    execution_duration_s: float | None = None,
    fps: float = 1,
    max_frames: int | None = 256,
    frame_max_side: int = 640,
    max_tokens: int = 16384,
    disable_thinking: bool = False,
    json_schema: bool = True,
    video: bool = False,
) -> tuple[dict, dict]:
    """Chat Completions request in the same order Gemini gets its inputs.

    Order: REFERENCE frames, EXECUTION (frames, or the whole clip when video is
    True), the prompt, CHECKLIST, and EXECUTION_DURATION_S. Returns (request, frame
    times per video); the execution's are empty with video, as the server picks them.
    """
    if not checklist_rows:
        raise ValueError("checklist_rows must contain at least one reference step")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    frames = {"reference": sample_frames(reference_clip, fps, max_frames=max_frames, max_side=frame_max_side)}
    content = _video_parts("REFERENCE", frames["reference"])
    if video:
        frames["execution"] = []
        content += _video_url_parts("EXECUTION", execution_clip, frame_max_side)
    else:
        frames["execution"] = sample_frames(execution_clip, fps, max_frames=max_frames, max_side=frame_max_side)
        content += _video_parts("EXECUTION", frames["execution"])
    checklist = json.dumps(
        [{k: row[k] for k in ("step_id", "action", "start_s", "end_s", "requirements")}
         for row in checklist_rows],
        ensure_ascii=False, allow_nan=False, indent=2,
    )
    content.append({"type": "text", "text": PROMPT_PATH.read_text(encoding="utf-8")})
    content.append({"type": "text", "text": f"CHECKLIST\n{checklist}"})
    if execution_duration_s is not None:
        content.append({"type": "text", "text": f"EXECUTION_DURATION_S\n{execution_duration_s:.2f}"})

    request = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if json_schema:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "offline_alignment", "schema": OFFLINE_SCHEMA},
        }
    if disable_thinking:
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    times = {name: [t for t, _ in video] for name, video in frames.items()}
    return request, times


def redact_images(request: dict) -> dict:
    """Copy of the request with each image or video replaced by its size, for the report."""
    kept = copy.deepcopy(request)
    for message in kept["messages"]:
        for part in message["content"]:
            if part["type"] in ("image_url", "video_url"):
                url = part[part["type"]]["url"]
                part[part["type"]]["url"] = f"<{part['type'][:5]}, {len(url)} base64 chars>"
    return kept


def _answer_text(text: str) -> str:
    """Drop a <think> block and a ```json fence if the server leaves them in."""
    text = _THINK.sub("", text).strip()
    match = _FENCE.match(text)
    return match.group(1) if match else text


def predict_offline_alignment_openai(
    client,
    request: dict,
    step_ids: list[str],
    execution_duration_s: float | None = None,
) -> dict:
    """Send a build_request request and check the answer like the Gemini path.

    Returns prediction (or None with validation_error), warnings for repaired
    timing problems, the request without image data, the raw response and the
    call duration. API errors raise.
    """
    started = perf_counter()
    response = client.chat.completions.create(**request)
    duration = perf_counter() - started
    raw_response = response.model_dump(mode="json")

    prediction = None
    validation_error = None
    warnings: list[str] = []
    try:
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Response must contain exactly one choice")
        if choices[0].get("finish_reason") != "stop":
            raise ValueError(f"Choice did not complete: {choices[0].get('finish_reason')}")
        text = (choices[0].get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Response contains no prediction text")
        candidate = json.loads(_answer_text(text))
        warnings = _validate(candidate, step_ids, execution_duration_s)
        prediction = candidate
    except (ValueError, ValidationError) as error:
        validation_error = error.message if isinstance(error, ValidationError) else str(error)

    return {
        "prediction": prediction,
        "validation_error": validation_error,
        "warnings": warnings,
        "request": redact_images(request),
        "response": raw_response,
        "request_duration_s": duration,
    }
