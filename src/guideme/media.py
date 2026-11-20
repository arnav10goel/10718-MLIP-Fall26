"""Read-only technical inspection of local videos, independent of datasets/models."""

import hashlib
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
from fractions import Fraction
from time import monotonic

import cv2
import imageio_ffmpeg
import numpy as np


def inspect_video(
    recording_id: str,
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    timeout_s: float = 300,
) -> dict:
    """Hash a local file and decode its first video stream completely with FFmpeg.

    Relative paths use data_root, GUIDEME_DATA_ROOT, then repository data/.
    Configured roots must be absolute; absolute video paths are used directly.
    Invalid arguments raise ValueError. Missing/unreadable/corrupt files and
    decoder failures return explicit failure records with available facts.

    Timestamp summaries are decoded source presentation times, retaining their
    original origin. duration_s spans the first frame through the last frame's
    declared end; it stays unknown if that final duration is unavailable. Nominal
    FPS and the container duration estimate are reported separately. Failed
    decodes may retain partial frame counts/summaries; only 'ok' means the entire
    selected stream completed without decoder or timestamp errors. Check source
    packet PTS as well as decoded timing: if source PTS are absent/unverified,
    reject the timing and withhold bounds/duration rather than report FFmpeg's
    synthesized times as source evidence.

    Audio presence comes from stream metadata; audio is not decoded or retained.
    No file is written, moved, re-encoded, or uploaded. This does not establish
    semantic identity, correctness, reference compatibility, or gold timing.
    """
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("recording_id must be a nonempty string")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("path must be a nonempty local path")
    source = Path(path).expanduser()
    if not source.is_absolute():
        configured = data_root if data_root is not None else os.environ.get("GUIDEME_DATA_ROOT") or Path(__file__).resolve().parents[2] / "data"
        if str(configured) == "":
            raise ValueError("data_root cannot be empty")
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("data_root must be absolute")
        source = root / source
    source = source.resolve()
    report = {
        "schema_version": 1, "recording_id": recording_id, "local_path": str(source),
        "size_bytes": None, "sha256": None, "width": None, "height": None,
        "native_fps": None, "audio_present": None, "container_duration_s": None,
        "duration_s": None, "decoded_frame_count": 0,
        "first_frame_timestamp_s": None, "last_frame_timestamp_s": None,
        "timestamp_origin": None,
        "timestamp_status": "not_checked", "decoding_status": "failed",
        "errors": [], "warnings": [], "decoder_returncode": None,
        "decoder_message": None,
    }
    errors, warnings = report["errors"], report["warnings"]
    try:
        before = source.stat()
        if not stat.S_ISREG(before.st_mode):
            errors.append("not_regular_file")
            return report
        report["size_bytes"] = before.st_size
        with source.open("rb") as stream:
            report["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        report["decoding_status"] = "missing"
        errors.append("missing_file")
        return report
    except OSError:
        errors.append("file_read_failed")
        return report
    if before.st_size == 0:
        errors.append("empty_file")
        return report

    try:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-nostats",
            "-v", "info", "-debug_ts", "-xerror", "-err_detect", "explode", "-copyts", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "showinfo=checksum=0",
            "-fps_mode", "passthrough", "-f", "null", "-",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        errors.append("decoder_timeout")
        return report
    except (OSError, RuntimeError):
        errors.append("decoder_unavailable")
        return report
    report["decoder_returncode"] = completed.returncode
    if completed.returncode != 0:
        errors.append("decoder_failed")
        report["decoder_message"] = completed.stderr[-2000:]

    time_base = None
    last_duration = None
    timestamp_errors = []
    video_stream = None
    source_packets = 0
    for line in completed.stderr.splitlines():
        if "Input #0," in line:
            report["audio_present"] = False
        if "Stream #0:" in line and ": Audio:" in line:
            report["audio_present"] = True
        stream = re.search(r"Stream #0:(\d+)", line)
        if video_stream is None and stream and ": Video:" in line:
            video_stream = stream[1]
        if "demuxer ->" in line and "type:video" in line:
            packet = re.search(r"ist_index:0:(\d+).*?\bpkt_pts:(\S+)", line)
            if packet and packet[1] == video_stream:
                source_packets += 1
                if not re.fullmatch(r"-?\d+", packet[2]):
                    timestamp_errors.append("missing_source_timestamps")
        duration = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", line)
        if duration:
            hours, minutes, seconds = map(float, duration.groups())
            report["container_duration_s"] = 3600 * hours + 60 * minutes + seconds
        if "[Parsed_showinfo_" not in line:
            continue
        config = re.search(r"config in time_base:\s*(\d+)/(\d+),\s*frame_rate:\s*(\d+)/(\d+)", line)
        if config:
            numerator, denominator, rate_n, rate_d = map(int, config.groups())
            time_base = numerator / denominator if numerator > 0 and denominator > 0 else None
            if report["native_fps"] is None and rate_n > 0 and rate_d > 0:
                report["native_fps"] = rate_n / rate_d
        frame = re.search(r"\bn:\s*\d+\s+pts:\s*(\S+)", line)
        if frame is None:
            continue
        report["decoded_frame_count"] += 1
        dimensions = re.search(r"\bs:(\d+)x(\d+)", line)
        if dimensions:
            width, height = map(int, dimensions.groups())
            if report["width"] is None:
                report["width"], report["height"] = width, height
            elif (width, height) != (report["width"], report["height"]):
                warnings.append("resolution_changed")
        try:
            if time_base is None:
                raise ValueError("No usable time base")
            timestamp = int(frame[1]) * time_base
            if not math.isfinite(timestamp):
                raise ValueError("Nonfinite timestamp")
        except ValueError:
            timestamp_errors.append("missing_frame_timestamp")
            continue
        previous = report["last_frame_timestamp_s"]
        if previous is not None and timestamp <= previous:
            timestamp_errors.append("non_increasing_timestamps")
        if report["first_frame_timestamp_s"] is None:
            report["first_frame_timestamp_s"] = timestamp
        report["last_frame_timestamp_s"] = timestamp
        duration = re.search(r"\bduration:\s*(\d+)", line)
        last_duration = int(duration[1]) * time_base if duration and int(duration[1]) > 0 else None

    if report["decoded_frame_count"] == 0:
        errors.append("no_video_frames")
    if source_packets == 0:
        timestamp_errors.append("source_timestamps_unverified")
    if any(reason in timestamp_errors for reason in ("missing_source_timestamps", "source_timestamps_unverified")):
        report["first_frame_timestamp_s"] = report["last_frame_timestamp_s"] = None
    report["timestamp_status"] = "valid" if report["decoded_frame_count"] and not timestamp_errors else "invalid"
    if report["timestamp_status"] == "valid":
        report["timestamp_origin"] = "source_presentation_time"
    errors.extend(timestamp_errors)
    if report["last_frame_timestamp_s"] is not None and last_duration is not None and not timestamp_errors:
        report["duration_s"] = report["last_frame_timestamp_s"] - report["first_frame_timestamp_s"] + last_duration
    elif report["decoded_frame_count"]:
        warnings.append("decoded_duration_unknown")
    if report["width"] is None or report["height"] is None:
        errors.append("missing_resolution")
    try:
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            errors.append("source_changed_during_inspection")
    except OSError:
        errors.append("source_changed_during_inspection")
    report["errors"] = list(dict.fromkeys(errors))
    report["warnings"] = list(dict.fromkeys(warnings))
    report["decoding_status"] = "failed" if report["errors"] else "ok"
    return report


class FileFrameSource:
    """Sequential RGB frames gated by an increasing content-replay cutoff.

    read(cutoff_s) returns the next eligible frame or None. Equal cutoffs may
    drain multiple frames; a later frame stays private until its elapsed PTS
    is eligible. The first source presentation timestamp maps to elapsed zero.
    Integer PTS/time-base arithmetic preserves nonzero origins and VFR gaps;
    missing source packet timestamps are errors, never estimated from FPS.

    eof distinguishes a clean drained decoder from a frame not yet eligible.
    It is not an announced task finish. Pauses are allowed: timeout_s bounds
    each decoder read, not the replay lifetime (cleanup has a bounded grace).
    Explicit close/context exit releases the process and pipes. After clean
    EOF reads return None; reads after early close fail. Open a new instance
    for a new replay. A decode failure closes the source and leaves eof false.

    Decoder lookahead is private. Availability here is ideal content replay
    at source elapsed time, not measured capture/receipt or delivery latency.
    No audio, labels, paths or final-duration-based sampling enter frame data.
    The caller must stream/retain returned frames under its own memory policy.
    """

    def __init__(self, path: str | Path, *, data_root: str | Path | None = None, timeout_s: float = 30):
        self._closed, self._eof = False, False
        self._process = self._selector = None
        try:
            valid_timeout = type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("timeout_s must be finite and positive")
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
        self._path = source.resolve()
        before = self._path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("path must point to a regular video file")
        self._identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        self._timeout = timeout_s
        self._last_cutoff = -1
        self._pixels, self._log = bytearray(), bytearray()
        self._diagnostic_tail = b""
        self._metadata = []
        self._metadata_count = 0
        self._origin = self._last_pts = self._time_base = None
        self._video_stream = None
        self._source_packets = 0
        self._pending = None
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-nostats",
            "-v", "level+info", "-debug_ts", "-xerror", "-err_detect", "explode", "-copyts",
            "-i", str(self._path), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", "format=rgb24,showinfo=checksum=0", "-fps_mode", "passthrough",
            "-enc_time_base", "1:1000000", "-threads", "1", "-c:v", "pam", "-f", "image2pipe", "pipe:1",
        ]
        try:
            environment = os.environ.copy()
            environment.pop("AV_LOG_FORCE_COLOR", None)
            environment["AV_LOG_FORCE_NOCOLOR"] = "1"
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             bufsize=0, env=environment)
            self._selector = selectors.DefaultSelector()
            self._selector.register(self._process.stdout, selectors.EVENT_READ, "pixels")
            self._selector.register(self._process.stderr, selectors.EVENT_READ, "log")
        except (OSError, RuntimeError):
            self.close()
            raise RuntimeError("Cannot start the video decoder") from None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def eof(self) -> bool:
        return self._eof

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Frame source is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process is not None:
            # Release pipe backpressure before waiting for graceful shutdown.
            # Otherwise an encoder blocked on RGB output can require SIGKILL.
            try:
                if self._process.poll() is None:
                    self._process.terminate()
            finally:
                for pipe in (self._process.stdout, self._process.stderr):
                    if pipe is not None:
                        pipe.close()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._selector is not None:
            self._selector.close()
        self._pixels.clear()
        self._log.clear()
        self._metadata.clear()
        self._pending = None

    def _parse_log(self, raw):
        line = raw.decode("utf-8", errors="replace")
        # Some demuxers log corruption yet exit successfully. Match the actual
        # severity prefix, so an input path containing '[error]' is harmless.
        if re.match(r"^(?:\[[^\]\r\n]+\]\s*)?\[(?:error|fatal|panic)\](?:\s|$)", line):
            raise RuntimeError(f"Video decoder reported an error: {line[-2000:]}")
        stream = re.search(r"Stream #0:(\d+).*?: Video:", line)
        if self._video_stream is None and stream:
            self._video_stream = stream[1]
        if "demuxer ->" in line and "type:video" in line:
            packet = re.search(r"ist_index:0:(\d+).*?\bpkt_pts:(\S+)", line)
            if packet is None or self._video_stream is None:
                raise RuntimeError("Cannot verify source packet timestamps")
            if packet[1] == self._video_stream:
                if not re.fullmatch(r"-?\d+", packet[2]):
                    raise RuntimeError("Source video has missing presentation timestamps")
                self._source_packets += 1
        if "[Parsed_showinfo_" not in line:
            return
        config = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
        if config:
            numerator, denominator = map(int, config.groups())
            if numerator <= 0 or denominator <= 0:
                raise RuntimeError("Invalid presentation time base")
            self._time_base = Fraction(numerator, denominator)
        if re.search(r"\bn:", line) is None:
            return
        frame = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b.*?\bfmt:rgb24\b.*?\bs:(\d+)x(\d+)", line)
        if frame is None or self._time_base is None:
            raise RuntimeError("Missing or malformed decoded frame timing")
        index, pts, width, height = map(int, frame.groups())
        if index != self._metadata_count:
            raise RuntimeError("Decoded frame metadata is out of sequence")
        self._metadata_count += 1
        self._metadata.append((index, pts * self._time_base, width, height))
        if len(self._metadata) > 1024:
            raise RuntimeError("Decoder metadata buffer exceeded its limit")

    def _take_frame(self):
        header_end = self._pixels.find(b"ENDHDR\n")
        if header_end < 0:
            if len(self._pixels) > 4096:
                raise RuntimeError("Malformed RGB frame header")
            return None
        header_end += len(b"ENDHDR\n")
        lines = bytes(self._pixels[:header_end]).splitlines()
        if not lines or lines[0] != b"P7" or header_end > 4096:
            raise RuntimeError("Invalid RGB frame header")
        fields = dict(line.split(maxsplit=1) for line in lines[1:-1])
        if fields.get(b"DEPTH") != b"3" or fields.get(b"MAXVAL") != b"255" or fields.get(b"TUPLTYPE") != b"RGB":
            raise RuntimeError("Decoder did not produce RGB888 pixels")
        width, height = int(fields[b"WIDTH"]), int(fields[b"HEIGHT"])
        size = width * height * 3
        if width <= 0 or height <= 0 or size > 128 * 1024 * 1024:
            raise RuntimeError("Decoded frame exceeds supported RGB size")
        if len(self._pixels) < header_end + size or not self._metadata:
            return None
        if not self._source_packets:
            raise RuntimeError("Source presentation timestamps were not verified")
        index, timestamp, expected_width, expected_height = self._metadata.pop(0)
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError("Frame pixels and timestamp metadata disagree")
        if self._last_pts is not None and timestamp <= self._last_pts:
            raise RuntimeError("Presentation timestamps must strictly increase")
        if self._origin is None:
            self._origin = timestamp
        source_pts, elapsed = float(timestamp), float(timestamp - self._origin)
        if not math.isfinite(source_pts) or not math.isfinite(elapsed):
            raise RuntimeError("Nonfinite presentation timestamp")
        frame = {"frame_index": index, "source_pts_s": source_pts, "elapsed_s": elapsed,
                 "width": width, "height": height,
                 "rgb_bytes": bytes(self._pixels[header_end:header_end + size])}
        del self._pixels[:header_end + size]
        self._last_pts = timestamp
        return frame

    def _decode_next(self):
        deadline = monotonic() + self._timeout
        while True:
            if monotonic() >= deadline:
                raise TimeoutError("Timed out reading the next video frame")
            frame = self._take_frame()
            if frame is not None:
                return frame
            if not self._selector.get_map():
                if self._log:
                    self._parse_log(bytes(self._log))
                    self._log.clear()
                if self._pixels or self._metadata:
                    raise RuntimeError("Decoder ended with incomplete pixels or timing")
                try:
                    code = self._process.wait(timeout=max(0.001, deadline - monotonic()))
                except subprocess.TimeoutExpired:
                    raise TimeoutError("Timed out waiting for the video decoder") from None
                if code != 0 or self._metadata_count == 0:
                    detail = self._diagnostic_tail.decode("utf-8", errors="replace")
                    raise RuntimeError(f"Video decoding failed (exit {code}): {detail}")
                self._eof = True
                self.close()
                return None
            events = self._selector.select(max(0, deadline - monotonic()))
            if not events:
                raise TimeoutError("Timed out reading the next video frame")
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    self._selector.unregister(key.fileobj)
                    continue
                if key.data == "pixels":
                    self._pixels.extend(chunk)
                    if len(self._pixels) > 128 * 1024 * 1024 + 65536 + 4096:
                        raise RuntimeError("Decoder pixel buffer exceeded its limit")
                else:
                    self._diagnostic_tail = (self._diagnostic_tail + chunk)[-4096:]
                    self._log.extend(chunk)
                    while b"\n" in self._log:
                        stop = self._log.index(b"\n")
                        self._parse_log(bytes(self._log[:stop]))
                        del self._log[:stop + 1]
                    if len(self._log) > 65536:
                        raise RuntimeError("Decoder timing log exceeded its limit")

    def _check_source(self):
        current = self._path.stat()
        if self._identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
            raise RuntimeError("Source video changed during replay")

    def read(self, cutoff_s: float) -> dict | None:
        if type(cutoff_s) not in (int, float):
            raise ValueError("cutoff_s must be finite and nonnegative")
        try:
            valid = math.isfinite(cutoff_s) and cutoff_s >= 0
        except OverflowError:
            valid = False
        if not valid or cutoff_s < self._last_cutoff:
            raise ValueError("cutoff_s must be finite, nonnegative and nondecreasing")
        if self._eof:
            self._last_cutoff = cutoff_s
            return None
        if self._closed:
            raise RuntimeError("Frame source is closed")
        self._last_cutoff = cutoff_s
        try:
            self._check_source()
            if self._pending is None:
                self._pending = self._decode_next()
            self._check_source()
            if self._pending is None or self._pending["elapsed_s"] > cutoff_s:
                return None
            frame, self._pending = self._pending, None
            return frame
        except (OSError, ValueError, KeyError, OverflowError, RuntimeError) as exc:
            self._eof = False
            self.close()
            if isinstance(exc, (RuntimeError, TimeoutError)):
                raise
            raise RuntimeError("Video frame decoding failed") from exc


class TimestampFrameSampler:
    """Select fixed-cadence samples from an already received frame stream.

    consider(frame, cutoff_s) returns None or a selection containing a canonical
    frame copy, target_s and skipped_targets. Targets are integer index / fps,
    compared directly with supplied elapsed seconds, including equality. No
    epsilon, timestamp rounding or repeated interval addition is used.

    Select the first real frame at/after the next target. If it crosses several
    targets, emit it once, record the remaining skipped targets, and advance
    past its actual elapsed time. Historical selections never change. fps and
    the cumulative selected-frame budget are explicit and fixed per instance.

    Valid unselected frames advance stream-order state. Invalid/future frames,
    unrepresentable grids and budget exhaustion leave ALL state unchanged.
    Exceeding the budget raises RuntimeError; never thin or truncate silently.
    Full-reference budget adaptation is a separate preparation policy.

    This class retains counters/timestamps only, performs no I/O or resizing,
    and cannot authenticate receipt times. The caller supplies received frames
    in order and owns source cleanup, retained images and failure logging.
    """

    @staticmethod
    def _number(value, name, *, nonnegative=False):
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be an ordinary finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (nonnegative and value < 0):
            raise ValueError(f"{name} must be finite{' and nonnegative' if nonnegative else ''}")
        return value

    def __init__(self, *, fps: float, max_frames: int):
        fps = self._number(fps, "fps", nonnegative=True)
        if fps == 0 or not math.isfinite(1 / fps):
            raise ValueError("fps must define a finite positive sampling interval")
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self._fps, self._max_frames = float(fps), max_frames
        self._sampled_count = self._next_index = 0
        self._last_index = self._last_elapsed = self._last_cutoff = -1
        self._last_source_pts = None

    @property
    def fps(self):
        return self._fps

    @property
    def max_frames(self):
        return self._max_frames

    @property
    def sampled_count(self):
        return self._sampled_count

    @property
    def next_target_s(self):
        return self._next_index / self._fps

    def consider(self, frame: dict, cutoff_s: float) -> dict | None:
        cutoff = self._number(cutoff_s, "cutoff_s", nonnegative=True)
        if not isinstance(frame, dict):
            raise ValueError("frame must be a timestamped RGB packet")
        index = frame.get("frame_index")
        if type(index) is not int or index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        elapsed = self._number(frame.get("elapsed_s"), "elapsed_s", nonnegative=True)
        source_pts = self._number(frame.get("source_pts_s"), "source_pts_s")
        width, height, pixels = frame.get("width"), frame.get("height"), frame.get("rgb_bytes")
        if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
                or not isinstance(pixels, bytes) or len(pixels) != width * height * 3):
            raise ValueError("frame must contain correctly sized immutable RGB888 bytes")
        if (index <= self._last_index or elapsed <= self._last_elapsed
                or (self._last_source_pts is not None and source_pts <= self._last_source_pts)
                or cutoff < self._last_cutoff or elapsed > cutoff):
            raise ValueError("Frames must increase in order and be available by a nondecreasing cutoff")

        target = self.next_target_s
        result = None
        next_index = self._next_index
        if elapsed >= target:
            if self._sampled_count >= self._max_frames:
                raise RuntimeError("Cumulative frame sample budget exceeded")
            try:
                scaled = elapsed * self._fps
                if not math.isfinite(scaled):
                    raise ValueError("Sampling grid exceeds numeric precision")
                next_index = max(self._next_index + 1, math.floor(scaled) + 1)
                # Multiplication can round across a grid boundary. Correct by
                # division, with bounded adjustments rather than a gap-sized loop.
                if next_index > self._next_index + 1 and (next_index - 1) / self._fps > elapsed:
                    next_index -= 1
                if next_index / self._fps <= elapsed:
                    next_index += 1
                next_target = next_index / self._fps
                if (not math.isfinite(next_target) or next_target <= elapsed
                        or (next_index - 1) / self._fps > elapsed):
                    raise ValueError("Sampling grid exceeds numeric precision")
            except OverflowError:
                raise ValueError("Sampling grid exceeds numeric precision") from None
            result = {
                "frame": {key: frame[key] for key in (
                    "frame_index", "source_pts_s", "elapsed_s", "width", "height", "rgb_bytes")},
                "target_s": target,
                "skipped_targets": next_index - self._next_index - 1,
            }

        # Commit only after validation, capacity checks and output construction.
        self._last_index, self._last_elapsed = index, elapsed
        self._last_source_pts, self._last_cutoff = source_pts, cutoff
        if result is not None:
            self._next_index = next_index
            self._sampled_count += 1
        return result


def encode_frame_jpeg(frame: dict, *, max_side: int = 640, jpeg_quality: int = 85) -> dict:
    """Prepare one timestamped RGB packet for provider-independent visual input.

    Shrink proportionally with area interpolation, rounding dimensions to the
    nearest integer (minimum one pixel). Never upscale, crop, pad or rotate.
    Preserve source geometry, index and timestamps; only whitelisted fields
    enter the output. Inputs remain unchanged and no file or network is used.

    JPEG is lossy. The defaults are starting settings, not verified sufficient
    detail for every task. Record actual dimensions, settings, encoder version
    and the exact returned payload's SHA-256; byte identity across OpenCV/JPEG
    builds is not promised. Sampling, cutoff checks and receipt availability
    belong to the caller; this helper only transforms the supplied frame.
    """
    if type(max_side) is not int or max_side <= 0:
        raise ValueError("max_side must be a positive integer")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer from 1 to 100")
    if not isinstance(frame, dict):
        raise ValueError("frame must be a timestamped RGB packet")
    index = frame.get("frame_index")
    if type(index) is not int or index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    for name in ("source_pts_s", "elapsed_s"):
        value = frame.get(name)
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid or (name == "elapsed_s" and value < 0):
            raise ValueError(f"{name} must be finite{' and nonnegative' if name == 'elapsed_s' else ''}")
    width, height, pixels = frame.get("width"), frame.get("height"), frame.get("rgb_bytes")
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or not isinstance(pixels, bytes) or len(pixels) != width * height * 3):
        raise ValueError("frame must contain correctly sized immutable RGB888 bytes")

    resized = max(width, height) > max_side
    try:
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if resized:
            scale = max_side / max(width, height)
            output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success or encoded is None or encoded.size == 0:
            raise RuntimeError("JPEG encoding returned no image")
        payload = encoded.tobytes()
    except cv2.error as exc:
        raise RuntimeError("JPEG image preparation failed") from exc
    output_height, output_width = image.shape[:2]
    return {
        "frame_index": index, "source_pts_s": frame["source_pts_s"], "elapsed_s": frame["elapsed_s"],
        "source_width": width, "source_height": height,
        "width": output_width, "height": output_height,
        "mime_type": "image/jpeg", "jpeg_bytes": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "encoding": {"max_side": max_side, "jpeg_quality": jpeg_quality,
                     "resize_interpolation": "area" if resized else "none",
                     "encoder": "opencv", "encoder_version": cv2.__version__},
    }
