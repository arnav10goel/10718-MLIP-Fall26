"""Encode received RGB frames once and remux immutable causal MP4 prefixes."""

import hashlib
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import tempfile
from threading import Lock
from time import monotonic

import imageio_ffmpeg

from .native_video import _scan_native_frames


class NativeVideoBuffer:
    """Single-writer, explicitly sealed native-video buffer.

    Source identity is caller-supplied provenance, not authentication of the RGB
    packets. Only appended, available frames may be encoded. Snapshots stop at a
    completed chunk boundary, and can coexist with later append/seal operations.
    The changed chunk encoding profile must be qualified separately from the
    whole-source encoder. Byte identity is not promised; visual quality
    equivalence is unverified.

    Raw pending pixels are bounded by max_pending_bytes, retained chunks by
    max_bytes, and each encode/remux output by max_bytes. At most one encoder and
    one snapshot run concurrently. Temporary PPM inputs duplicate only pending
    pixels plus bounded per-frame headers. No ingestion thread is created.
    """

    def __init__(self, *, source_sha256, source_size_bytes, max_side, max_frames,
                 max_bytes, max_pending_bytes, timeout_s):
        if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        for name, value in (("source_size_bytes", source_size_bytes), ("max_side", max_side),
                            ("max_frames", max_frames), ("max_bytes", max_bytes),
                            ("max_pending_bytes", max_pending_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_side < 2:
            raise ValueError("max_side must be at least two pixels")
        self._number(timeout_s, "timeout_s")
        if timeout_s == 0:
            raise ValueError("timeout_s must be positive")
        self._source_sha256, self._source_size_bytes = source_sha256, source_size_bytes
        self._max_side, self._max_frames = max_side, max_frames
        self._max_bytes, self._max_pending_bytes = max_bytes, max_pending_bytes
        self._timeout = timeout_s
        self._state_lock, self._writer_lock, self._snapshot_lock = Lock(), Lock(), Lock()
        self._closed = self._closing = self._source_eof = False
        self._pending, self._chunks = [], ()
        self._pending_bytes = self._encoded_bytes = self._frame_count = 0
        self._last_pts = self._last_tick = self._watermark = self._dimensions = None
        self._temporary = tempfile.TemporaryDirectory(prefix="guideme-buffer-")
        self._root = Path(self._temporary.name)

    @staticmethod
    def _number(value, name):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be finite and nonnegative")

    def _open(self):
        if self._closed or self._closing:
            raise RuntimeError("Native video buffer is closed")

    def append(self, frame, available_through_s):
        """Accept one contiguous original RGB packet, or leave state unchanged."""
        with self._writer_lock, self._state_lock:
            self._open()
            if self._source_eof:
                raise RuntimeError("Native video source was finalized")
            self._number(available_through_s, "available_through_s")
            if not isinstance(frame, dict) or set(frame) != {
                    "frame_index", "source_pts_s", "elapsed_s", "width", "height", "rgb_bytes"}:
                raise ValueError("frame must be an original RGB frame packet")
            index, pts = frame["frame_index"], frame["source_pts_s"]
            self._number(pts, "source_pts_s")
            self._number(frame["elapsed_s"], "elapsed_s")
            if (type(index) is not int or index != self._frame_count
                    or frame["elapsed_s"] != pts or pts > available_through_s
                    or (self._watermark is not None and available_through_s < self._watermark)
                    or (self._last_pts is None and pts != 0)
                    or (self._last_pts is not None and pts <= self._last_pts)):
                raise ValueError("Frame index, source clock or availability is inconsistent")
            try:
                tick = round(pts * 1_000_000)
            except (OverflowError, ValueError):
                raise ValueError("Source timestamp exceeds the MP4 clock") from None
            if (tick > 2**63 - 1 or (self._last_tick is not None and tick <= self._last_tick)
                    or abs(tick / 1_000_000 - pts) > 0.000001):
                raise ValueError("Source timestamps cannot retain distinct microsecond PTS")
            width, height, pixels = frame["width"], frame["height"], frame["rgb_bytes"]
            if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
                    or type(pixels) is not bytes or len(pixels) != width * height * 3
                    or len(pixels) > 128 * 1024 * 1024
                    or (self._dimensions is not None and self._dimensions != (width, height))):
                raise ValueError("Frame dimensions or immutable RGB bytes are inconsistent")
            scale = min(1, self._max_side / max(width, height))
            if min(int(width * scale) // 2 * 2, int(height * scale) // 2 * 2) < 2:
                raise ValueError("Frame dimensions cannot be encoded without enlargement")
            if self._frame_count >= self._max_frames:
                raise RuntimeError("Native video buffer exceeds max_frames")
            if self._pending_bytes + len(pixels) > self._max_pending_bytes:
                raise RuntimeError("Native video buffer exceeds max_pending_bytes; seal first")
            self._pending.append((dict(frame), tick))
            self._pending_bytes += len(pixels)
            self._frame_count += 1
            self._last_pts, self._last_tick, self._watermark = pts, tick, available_through_s
            self._dimensions = width, height

    def _remaining(self, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Native video buffer operation timed out")
        return remaining

    def _run(self, command, deadline):
        """Bound subprocess time and diagnostic memory, and always reap it."""
        process = None
        selector = selectors.DefaultSelector()
        diagnostic = bytearray()
        try:
            self._remaining(deadline)
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            selector.register(process.stderr, selectors.EVENT_READ)
            while selector.get_map():
                events = selector.select(self._remaining(deadline))
                if not events:
                    raise TimeoutError("Native video encoding/remux timed out")
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        diagnostic.extend(chunk)
                        if len(diagnostic) > 65536:
                            raise RuntimeError("Native video diagnostic output exceeds its limit")
            try:
                code = process.wait(timeout=self._remaining(deadline))
            except subprocess.TimeoutExpired:
                raise TimeoutError("Native video encoding/remux timed out") from None
            if code != 0 or diagnostic:
                raise RuntimeError("Native video encoding/remux failed")
        finally:
            try:
                if process is not None:
                    try:
                        if process.poll() is None:
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

    def _command(self):
        return [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-nostats",
                "-v", "error", "-xerror"]

    def _scan(self, path, count, deadline):
        rows, eof = _scan_native_frames(path, cutoff_s=None, max_frames=count,
            remaining=lambda: self._remaining(deadline))
        if not eof or len(rows) != count:
            raise RuntimeError("Native video frame count differs from accepted frames")
        return rows

    def seal(self, source_eof=False):
        """Encode pending frames once; a pause seal does not announce EOF."""
        if type(source_eof) is not bool:
            raise ValueError("source_eof must be boolean")
        admitted = False
        try:
            with self._writer_lock:
                with self._state_lock:
                    self._open()
                    if self._source_eof:
                        raise RuntimeError("Native video source was finalized")
                    admitted = True
                    pending = tuple(self._pending)
                    if not pending:
                        self._source_eof = source_eof
                        return
                    dimensions = self._dimensions
                    chunk_index, encoded_bytes = len(self._chunks), self._encoded_bytes
                deadline = monotonic() + self._timeout
                width, height = dimensions
                scale = min(1, self._max_side / max(width, height))
                output_width, output_height = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
                with tempfile.TemporaryDirectory(prefix="encode-", dir=self._root) as directory:
                    work = Path(directory)
                    entries = []
                    for index, (frame, tick) in enumerate(pending):
                        self._remaining(deadline)
                        image = work / f"frame-{index:08d}.ppm"
                        with image.open("wb") as stream:
                            stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                            stream.write(frame["rgb_bytes"])
                        entries.extend([f"file '{image.name}'", "option framerate 1000000"])
                        if index + 1 < len(pending):
                            delta = pending[index + 1][1] - tick
                            entries.append(f"duration {delta // 1_000_000}.{delta % 1_000_000:06d}")
                    listing = work / "frames.txt"
                    listing.write_text("ffconcat version 1.0\n" + "\n".join(entries) + "\n", encoding="ascii")
                    output = work / "chunk.mp4"
                    self._run(self._command() + ["-f", "concat", "-safe", "0", "-i", str(listing),
                        "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
                        "-map_chapters", "-1", "-vf", f"scale={output_width}:{output_height}",
                        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-tune", "zerolatency",
                        "-bf", "0", "-g", "60", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
                        "-fps_mode", "passthrough", "-enc_time_base", "1:1000000",
                        "-video_track_timescale", "1000000", "-movflags", "+faststart",
                        "-fflags", "+bitexact", "-flags:v", "+bitexact",
                        "-fs", str(self._max_bytes + 1), str(output)], deadline)
                    size = output.stat().st_size
                    if size > self._max_bytes or encoded_bytes + size > self._max_bytes:
                        raise RuntimeError("Encoded chunks exceed max_bytes")
                    actual = self._scan(output, len(pending), deadline)
                    start_tick = pending[0][1]
                    for row, (_, tick) in zip(actual, pending):
                        if ((row["width"], row["height"]) != (output_width, output_height)
                                or abs(row["source_pts_s"] - (tick - start_tick) / 1_000_000) > 0.000001):
                            raise RuntimeError("Encoded chunk changed accepted frame PTS or dimensions")
                    destination = self._root / f"chunk-{chunk_index:08d}.mp4"
                    output.replace(destination)
                    witnesses = tuple((frame["frame_index"], frame["source_pts_s"], tick)
                                      for frame, tick in pending)
                    with self._state_lock:
                        self._chunks += ((destination, witnesses),)
                        self._encoded_bytes += size
                        self._pending.clear()
                        self._pending_bytes = 0
                        self._source_eof = source_eof
        except Exception:
            if admitted:
                self.close()
            raise

    def snapshot(self, max_cutoff_s):
        """Return a detached full prefix at a committed cutoff, or None."""
        self._number(max_cutoff_s, "max_cutoff_s")
        admitted = False
        try:
            with self._snapshot_lock:
                with self._state_lock:
                    self._open()
                    admitted = True
                    chunks = tuple(chunk for chunk in self._chunks if chunk[1][-1][1] <= max_cutoff_s)
                    if not chunks:
                        return None
                    eof = self._source_eof and len(chunks) == len(self._chunks)
                    width, height = self._dimensions
                deadline = monotonic() + self._timeout
                witnesses = tuple(row for _, rows in chunks for row in rows)
                with tempfile.TemporaryDirectory(prefix="snapshot-", dir=self._root) as directory:
                    work = Path(directory)
                    entries = []
                    for index, (path, rows) in enumerate(chunks):
                        # Internal generated names contain no quoting characters.
                        entries.append(f"file '{path}'")
                        if index + 1 < len(chunks):
                            delta = chunks[index + 1][1][0][2] - rows[0][2]
                            entries.append(f"duration {delta // 1_000_000}.{delta % 1_000_000:06d}")
                    listing = work / "chunks.txt"
                    listing.write_text("ffconcat version 1.0\n" + "\n".join(entries) + "\n", encoding="ascii")
                    output = work / "prefix.mp4"
                    self._run(self._command() + ["-copyts", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
                        "-map_metadata", "-1", "-map_chapters", "-1", "-avoid_negative_ts", "disabled",
                        "-video_track_timescale", "1000000", "-movflags", "+faststart",
                        "-fflags", "+bitexact", "-fs", str(self._max_bytes + 1), str(output)], deadline)
                    if output.stat().st_size > self._max_bytes:
                        raise RuntimeError("Native video snapshot exceeds max_bytes")
                    actual = self._scan(output, len(witnesses), deadline)
                    scale = min(1, self._max_side / max(width, height))
                    output_width, output_height = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
                    frame_map = []
                    for row, (index, pts, tick) in zip(actual, witnesses):
                        if ((row["width"], row["height"]) != (output_width, output_height)
                                or abs(row["source_pts_s"] - pts) > 0.000001):
                            raise RuntimeError("Native snapshot changed accepted frame PTS or dimensions")
                        frame_map.append({"frame_index": index, "source_pts_s": pts,
                            "elapsed_s": pts, "video_pts_s": row["source_pts_s"]})
                    video = output.read_bytes()
                    self._remaining(deadline)
                return {"schema_version": 1, "status": "ready", "source_sha256": self._source_sha256,
                    "source_size_bytes": self._source_size_bytes, "source_first_pts_s": 0,
                    "cutoff_s": witnesses[-1][1], "source_eof": eof, "mime_type": "video/mp4",
                    "video_bytes": video, "video_sha256": hashlib.sha256(video).hexdigest(),
                    "size_bytes": len(video), "frame_map": frame_map,
                    "config": {"codec": "libx264", "crf": 28, "preset": "veryfast",
                        "tune": "zerolatency", "b_frames": 0, "gop_frames": 60, "scene_threshold": 0,
                        "pixel_format": "yuv420p", "max_side": self._max_side,
                        "max_frames": self._max_frames, "max_bytes": self._max_bytes,
                        "temporal_sampling": "all_source_frames", "video_timescale": 1_000_000,
                        "timestamp_tolerance_s": 0.000001, "width": output_width, "height": output_height,
                        "preparation": "received_frame_chunks",
                        "chunk_frame_counts": [len(rows) for _, rows in chunks],
                        "max_pending_bytes": self._max_pending_bytes,
                        "absolute_pts_quantization": "nearest_microsecond"}}
        except Exception:
            if admitted:
                self.close()
            raise

    def close(self):
        """Wait for active users before deleting chunks; returned bytes survive."""
        with self._state_lock:
            self._closing = True
        with self._writer_lock, self._snapshot_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                self._pending.clear()
                self._chunks = ()
            self._temporary.cleanup()

    def __enter__(self):
        with self._state_lock:
            self._open()
        return self

    def __exit__(self, *_):
        self.close()
