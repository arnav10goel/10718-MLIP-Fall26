"""Prepare video inputs for causal execution replay."""

import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np


def prepare_execution_prefix(
    source_path: str | Path,
    end_s: float,
    output_path: str | Path,
) -> Path:
    """Save a video-only MP4 with frames strictly before elapsed end_s.

    Time zero is the first frame of the first video stream. An endpoint past
    EOF keeps all available frames; it does not assert execution completion.
    Re-encode as H.264 at CRF 18, preserving frame timing and padding odd image
    dimensions by at most one pixel for yuv420p compatibility. Missing packet
    durations use adjacent timestamps, with the previous interval (or nominal
    source FPS for a single frame) at EOF, capped at the checkpoint. A minimum
    container tick can extend the last display slightly; no later frame is
    included. The source must expose a finite positive nominal FPS.

    The destination parent must exist. Existing files are never overwritten,
    and failed conversions leave no partial destination. The source is read
    only. Returns the absolute destination Path; no upload is performed.
    """
    if (
        isinstance(end_s, bool) or not isinstance(end_s, (int, float))
        or not math.isfinite(end_s) or end_s <= 0
    ):
        raise ValueError("end_s must be a finite positive number")
    requested_output = Path(output_path).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"Output already exists: {requested_output}")
    output = requested_output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_path must have an .mp4 extension")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent does not exist: {output.parent}")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("source_path must point to a nonempty video file")
    source_capture = cv2.VideoCapture(str(source))
    try:
        source_fps = source_capture.get(cv2.CAP_PROP_FPS)
    finally:
        source_capture.release()
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source video must expose a finite positive nominal FPS")

    # libx264 can emit zero durations in passthrough mode. Repair packet
    # durations without resampling frames or letting the MP4 hide the last one.
    duration_expression = (
        f"max(1,min({end_s}/TB-PTS,"
        "if(gt(DURATION,0),DURATION,"
        "if(eq(NEXT_PTS,NOPTS),"
        f"if(gt(PREV_OUTDURATION,0),PREV_OUTDURATION,1/({source_fps}*TB)),"
        "NEXT_PTS-PTS))))"
    )
    duration_filter = "setts=duration=" + duration_expression.replace(",", "\\,")

    # Encode beside the destination, then publish without replacing any file.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".guideme-prefix-") as directory:
        prepared = Path(directory) / "prefix.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin", "-v", "error", "-xerror", "-n",
            "-i", str(source), "-map", "0:v:0",
            "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
            "-vf", (
                f"setpts=PTS-STARTPTS,trim=end={end_s},"
                "pad=ceil(iw/2)*2:ceil(ih/2)*2"
            ),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-bf", "0",
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
            "-bsf:v", duration_filter,
            "-movflags", "+faststart", str(prepared),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Prefix conversion failed: {completed.stderr.strip()[-1000:]}")
        capture = cv2.VideoCapture(str(prepared))
        try:
            decoded, _ = capture.read()
            if not decoded:
                raise RuntimeError("Prefix conversion produced no decodable video frame")
        finally:
            capture.release()
        os.link(prepared, output)
    return output


def sample_video_frames(
    source_path: str | Path,
    start_s: float,
    end_s: float,
    fps: float = 5.0,
    size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    """Return resized BGR frames and their actual elapsed source timestamps.

    Time zero is the first frame of the first video stream. Select the first
    frame in [start_s, end_s), then the earliest frame at least 1/fps seconds
    after each selected frame. No frames are interpolated or duplicated, so
    the achieved sampling rate can be lower than fps. An endpoint beyond EOF
    is allowed, but an interval without frames raises ValueError.

    The source is read only, from its beginning, without an input seek. This
    decodes the video and holds selected frames in memory; it is not optimized
    for live replay. Bicubic resizing to size=(width, height) can change the
    aspect ratio. Outputs have shapes (N, height, width, 3) and (N,), with
    uint8 pixels and float64 seconds relative to the source's first frame.
    Decode failures and inconsistent timestamp metadata raise RuntimeError.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (start_s, end_s, fps)
    ) or not 0 <= start_s < end_s or fps <= 0:
        raise ValueError("Require finite 0 <= start_s < end_s and positive fps")
    if (
        not isinstance(size, (tuple, list)) or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in size)
    ):
        raise ValueError("size must contain two positive integer dimensions")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("source_path must point to a nonempty video file")
    width, height = size
    selection = (
        f"if(gte(t,{start_s})*lt(t,{end_s}),"
        f"if(isnan(prev_selected_pts),1,gte((pts-prev_selected_pts)*TB,{1 / fps})),0)"
    ).replace(",", "\\,")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-hide_banner", "-v", "info", "-xerror",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", (
            f"setpts=PTS-STARTPTS,select={selection},"
            f"scale={width}:{height}:flags=bicubic,format=bgr24,showinfo=checksum=0"
        ),
        "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as error:
        raise RuntimeError(f"Video sampling could not run FFmpeg: {error}") from error
    diagnostics = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"Video sampling failed: {diagnostics.strip()[-1000:]}")

    timestamps = []
    time_base = None
    for line in diagnostics.splitlines():
        if "[Parsed_showinfo_" not in line:
            continue
        if "config in time_base:" in line:
            metadata = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
            if metadata is None or int(metadata[1]) <= 0 or int(metadata[2]) <= 0:
                raise RuntimeError("Video sampling produced invalid timestamp time base")
            time_base = int(metadata[1]) / int(metadata[2])
        elif re.search(r"\bn:\s*", line):
            metadata = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b", line)
            if metadata is None or time_base is None or int(metadata[1]) != len(timestamps):
                raise RuntimeError("Video sampling produced missing or inconsistent timestamps")
            timestamp = int(metadata[2]) * time_base
            if (
                not math.isfinite(timestamp) or not start_s <= timestamp < end_s
                or (timestamps and timestamp <= timestamps[-1])
            ):
                raise RuntimeError("Video sampling produced invalid source timestamps")
            timestamps.append(timestamp)
    frame_bytes = width * height * 3
    if len(completed.stdout) != len(timestamps) * frame_bytes:
        raise RuntimeError("Video sampling frame and timestamp counts do not match")
    if not timestamps:
        raise ValueError("Requested interval contains no video frames")
    frames = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        len(timestamps), height, width, 3
    ).copy()
    return frames, np.asarray(timestamps, dtype=np.float64)
