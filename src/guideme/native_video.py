"""Prepare bounded, silent native-video inputs without changing source media."""

import hashlib
from fractions import Fraction
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import tempfile
from time import monotonic

import imageio_ffmpeg


def _scan_native_frames(path, *, cutoff_s, max_frames, remaining):
    """Decode bounded frame witnesses without piping decoded pixels to Python.

    Integer showinfo PTS and its rational time base retain the RGB decoder's
    clock. The null output still decodes every frame; only the unused PAM pixel
    transfer is removed. A cutoff stops at the first ineligible decoded frame,
    while an unbounded scan requires clean EOF. Zero retains the first valid
    witness solely for the caller's empty-asset metadata.
    """
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)

    def check_identity():
        current = path.stat()
        if identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
            raise RuntimeError("Source video changed during native decoding")

    frames, log = [], bytearray()
    tail = b""
    time_base = previous = None
    video_stream = None
    source_packets = frame_count = 0

    def parse(raw):
        nonlocal time_base, previous, video_stream, source_packets, frame_count
        line = raw.decode("utf-8", errors="replace")
        if re.match(r"^(?:\[[^\]\r\n]+\]\s*)?\[(?:error|fatal|panic)\](?:\s|$)", line):
            raise RuntimeError(f"Video decoder reported an error: {line[-2000:]}")
        stream = re.search(r"Stream #0:(\d+).*?: Video:", line)
        if video_stream is None and stream:
            video_stream = stream[1]
        if "demuxer ->" in line and "type:video" in line:
            packet = re.search(r"ist_index:0:(\d+).*?\bpkt_pts:(\S+)", line)
            if packet is None or video_stream is None:
                raise RuntimeError("Cannot verify source packet timestamps")
            if packet[1] == video_stream:
                if not re.fullmatch(r"-?\d+", packet[2]):
                    raise RuntimeError("Source video has missing presentation timestamps")
                source_packets += 1
        if "[Parsed_showinfo_" not in line:
            return None
        config = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
        if config:
            numerator, denominator = map(int, config.groups())
            if numerator <= 0 or denominator <= 0:
                raise RuntimeError("Invalid presentation time base")
            time_base = Fraction(numerator, denominator)
        if re.search(r"\bn:", line) is None:
            return None
        frame = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b.*?\bfmt:rgb24\b.*?\bs:(\d+)x(\d+)", line)
        if frame is None or time_base is None:
            raise RuntimeError("Missing or malformed decoded frame timing")
        index, pts, width, height = map(int, frame.groups())
        if index != frame_count:
            raise RuntimeError("Decoded frame metadata is out of sequence")
        frame_count += 1
        if not source_packets:
            raise RuntimeError("Source presentation timestamps were not verified")
        if width <= 0 or height <= 0 or width * height * 3 > 128 * 1024 * 1024:
            raise RuntimeError("Decoded frame exceeds supported RGB size")
        timestamp = pts * time_base
        if previous is not None and timestamp <= previous:
            raise RuntimeError("Presentation timestamps must strictly increase")
        if previous is None and timestamp != 0:
            raise ValueError("Native video initially requires source presentation origin zero")
        previous = timestamp
        source_pts = float(timestamp)
        if not math.isfinite(source_pts):
            raise RuntimeError("Nonfinite presentation timestamp")
        check_identity()
        return {"frame_index": index, "source_pts_s": source_pts, "elapsed_s": source_pts,
                "width": width, "height": height}

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-nostats",
        "-v", "level+info", "-debug_ts", "-xerror", "-err_detect", "explode", "-copyts",
        "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "format=rgb24,showinfo=checksum=0", "-fps_mode", "passthrough",
        "-enc_time_base", "1:1000000", "-threads", "1", "-f", "null", "-",
    ]
    environment = os.environ.copy()
    environment.pop("AV_LOG_FORCE_COLOR", None)
    environment["AV_LOG_FORCE_NOCOLOR"] = "1"
    process = None
    selector = selectors.DefaultSelector()
    try:
        remaining()
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                       bufsize=0, env=environment)
        except OSError:
            raise RuntimeError("Cannot start the video decoder") from None
        selector.register(process.stderr, selectors.EVENT_READ)
        while True:
            remaining()
            if b"\n" in log or (log and not selector.get_map()):
                stop = log.find(b"\n")
                if stop < 0:
                    stop = len(log)
                if stop > 65536:
                    raise RuntimeError("Decoder timing log exceeded its limit")
                frame = parse(bytes(log[:stop]))
                del log[:stop + 1]
                if frame is not None:
                    if cutoff_s is not None and frame["elapsed_s"] > cutoff_s:
                        return frames, False
                    if len(frames) >= max_frames:
                        raise RuntimeError("Native video exceeds max_frames")
                    frames.append(frame)
                    if cutoff_s == 0:
                        return frames, False
                continue
            if len(log) > 65536:
                raise RuntimeError("Decoder timing log exceeded its limit")
            if not selector.get_map():
                try:
                    code = process.wait(timeout=remaining())
                except subprocess.TimeoutExpired:
                    raise TimeoutError("Native video decoding timed out") from None
                if code != 0 or not frames:
                    detail = tail.decode("utf-8", errors="replace")
                    raise RuntimeError(f"Video decoding failed (exit {code}): {detail}")
                check_identity()
                return frames, True
            events = selector.select(remaining())
            if not events:
                raise TimeoutError("Native video decoding timed out")
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    tail = (tail + chunk)[-4096:]
                    log.extend(chunk)
    finally:
        try:
            if process is not None:
                try:
                    if process.poll() is None:
                        # This abandoned decoder writes no asset. Avoid waiting
                        # for FFmpeg to flush work beyond the requested cutoff.
                        process.kill()
                finally:
                    process.stderr.close()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            selector.close()


def prepare_native_video(
    path, *, expected_source_sha256, cutoff_s=None, max_side, max_frames,
    max_bytes, timeout_s, data_root=None,
):
    """Return a silent H.264 MP4 and its verified per-frame source mapping.

    None retains the complete reference; a positive cutoff retains the ordered
    source prefix with elapsed PTS <= cutoff. Zero deliberately returns an empty
    asset for the host's no-positive-duration skip. The initial native contract
    requires source origin zero. No FPS filter or temporal thinning is applied.
    Output PTS are verified individually against source PTS at microsecond
    precision; frame_map retains both actual clocks rather than rewriting either.

    Only temporary derived files are written. Relative source paths use explicit
    data_root, GUIDEME_DATA_ROOT, then repository data. Source identity and SHA-256
    are checked before and after preparation. Encoding/decode failures and caps
    raise visibly; no truncated or silently subsampled asset is returned.
    """
    for name, value in (("max_side", max_side), ("max_frames", max_frames), ("max_bytes", max_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_side < 2:
        raise ValueError("max_side must be at least two pixels")
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if cutoff_s is not None and (
        type(cutoff_s) not in (int, float) or not math.isfinite(cutoff_s) or cutoff_s < 0
    ):
        raise ValueError("cutoff_s must be finite and nonnegative, or None")
    if not isinstance(expected_source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256")
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("path must be a nonempty local path")
    source = Path(path).expanduser()
    if not source.is_absolute():
        configured = data_root if data_root is not None else os.environ.get("GUIDEME_DATA_ROOT") or Path(__file__).resolve().parents[2] / "data"
        if not str(configured):
            raise ValueError("data_root cannot be empty")
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("data_root must be absolute")
        source = root / source
    source = source.resolve()
    deadline = monotonic() + timeout_s

    def remaining():
        value = deadline - monotonic()
        if value <= 0:
            raise TimeoutError("Native video preparation timed out")
        return value

    def identity():
        current = source.stat()
        if not stat.S_ISREG(current.st_mode):
            raise ValueError("path must point to a regular video file")
        return current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns

    before = identity()

    def source_digest():
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        remaining()
        return digest

    if source_digest() != expected_source_sha256:
        raise ValueError("Source video SHA-256 does not match the expected source")
    if identity() != before:
        raise RuntimeError("Source video changed during hashing")

    def check_source():
        if identity() != before or source_digest() != expected_source_sha256 or identity() != before:
            raise RuntimeError("Source video changed during native preparation")

    config = {
        "codec": "libx264", "crf": 28, "preset": "veryfast", "pixel_format": "yuv420p",
        "max_side": max_side, "max_frames": max_frames, "max_bytes": max_bytes,
        "temporal_sampling": "all_source_frames", "video_timescale": 1_000_000,
        "timestamp_tolerance_s": 0.000001,
    }
    asset = {
        "schema_version": 1, "status": "empty", "source_sha256": expected_source_sha256,
        "source_size_bytes": before[2], "source_first_pts_s": None, "cutoff_s": cutoff_s,
        "source_eof": False, "mime_type": "video/mp4", "video_bytes": b"",
        "video_sha256": None, "size_bytes": 0, "frame_map": [], "config": config,
    }
    dimensions = None
    frames, asset["source_eof"] = _scan_native_frames(source, cutoff_s=cutoff_s,
        max_frames=max_frames, remaining=remaining)
    for frame in frames:
        if asset["source_first_pts_s"] is None:
            asset["source_first_pts_s"] = frame["source_pts_s"]
            dimensions = frame["width"], frame["height"]
        if cutoff_s == 0:
            break
        if dimensions != (frame["width"], frame["height"]):
            raise ValueError("Native video requires constant source dimensions")
        asset["frame_map"].append({
            "frame_index": frame["frame_index"], "source_pts_s": frame["source_pts_s"],
            "elapsed_s": frame["elapsed_s"], "video_pts_s": None,
        })
    check_source()
    if not asset["frame_map"]:
        if asset["source_first_pts_s"] is None:
            raise RuntimeError("Source video contains no usable frames")
        return asset

    width, height = dimensions
    scale = min(1, max_side / max(width, height))
    output_width, output_height = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
    if min(output_width, output_height) < 2:
        raise ValueError("Source dimensions cannot be encoded without enlargement")
    config["width"], config["height"] = output_width, output_height
    with tempfile.TemporaryDirectory(prefix="guideme-native-") as directory:
        output = Path(directory) / "video.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-nostats",
            "-v", "error", "-xerror", "-err_detect", "explode", "-copyts", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
            "-frames:v", str(len(asset["frame_map"])),
            "-vf", f"scale={output_width}:{output_height}",
            "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough", "-enc_time_base", "1:1000000",
            "-video_track_timescale", "1000000", "-movflags", "+faststart",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-fs", str(max_bytes + 1), str(output),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=remaining())
        except subprocess.TimeoutExpired:
            raise TimeoutError("Native video encoding timed out") from None
        if completed.returncode:
            raise RuntimeError("Native video encoding failed")
        if output.stat().st_size > max_bytes:
            raise RuntimeError("Native video exceeds max_bytes")
        decoded, eof = _scan_native_frames(output, cutoff_s=None,
            max_frames=len(asset["frame_map"]), remaining=remaining)
        if len(decoded) != len(asset["frame_map"]) or not eof:
            raise RuntimeError("Derived video frame count or order differs from the source prefix")
        for row, actual in zip(asset["frame_map"], decoded):
            remaining()
            if actual["frame_index"] != row["frame_index"]:
                raise RuntimeError("Derived video frame count or order differs from the source prefix")
            if (actual["width"], actual["height"]) != (output_width, output_height):
                raise RuntimeError("Derived video dimensions differ from the encoding contract")
            if abs(actual["source_pts_s"] - row["source_pts_s"]) > 0.000001:
                raise RuntimeError("Derived video presentation time differs from its source frame")
            row["video_pts_s"] = actual["source_pts_s"]
        video_bytes = output.read_bytes()
        if len(video_bytes) > max_bytes:
            raise RuntimeError("Native video exceeds max_bytes")
    check_source()
    asset.update(status="ready", video_bytes=video_bytes,
                 video_sha256=hashlib.sha256(video_bytes).hexdigest(), size_bytes=len(video_bytes))
    return asset
