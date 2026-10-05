"""Serve a local browser UI for annotating the recorded SIM-card videos.

The viewer writes temporal step annotations to:

    data/prepared/recorded_annotations.json

It uses only the Python standard library. Run it from the repository root:

    .venv/bin/python scripts/annotate_recordings.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from classical_pairwise_dtw import STEP_LABELS
from coin_tasks import PREPARED_DIR, REPO_ROOT
DEFAULT_VIDEO_DIR = REPO_ROOT / "data/videos/ReplaceSIMCard/recorded"
DEFAULT_OUTPUT = PREPARED_DIR / "recorded_annotations.json"
VIEWER_HTML = Path(__file__).with_name("recording_annotation_viewer.html")
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def default_video(filename: str) -> dict:
    incorrect = filename.lower().startswith("incorrect")
    return {
        "file": filename,
        "outcome": "incorrect" if incorrect else "correct",
        "deviation_step": 2 if incorrect else None,
        "duration": None,
        "steps": [
            {
                "index": index,
                "label": label,
                "start": None,
                "end": None,
            }
            for index, label in enumerate(STEP_LABELS, start=1)
        ],
    }


def load_state(video_names: list[str], output_path: Path) -> dict:
    existing_by_name: dict[str, dict] = {}
    if output_path.exists():
        with output_path.open() as file:
            payload = json.load(file)
        existing_by_name = {
            entry["file"]: entry
            for entry in payload.get("videos", [])
            if isinstance(entry, dict) and isinstance(entry.get("file"), str)
        }

    videos = []
    for name in video_names:
        entry = default_video(name)
        existing = existing_by_name.get(name)
        if existing:
            entry["outcome"] = existing.get("outcome", entry["outcome"])
            entry["deviation_step"] = existing.get(
                "deviation_step", entry["deviation_step"]
            )
            entry["duration"] = existing.get("duration")
            old_steps = {
                step.get("index"): step
                for step in existing.get("steps", [])
                if isinstance(step, dict)
            }
            for step in entry["steps"]:
                old = old_steps.get(step["index"])
                if old:
                    step["start"] = old.get("start")
                    step["end"] = old.get("end")
        videos.append(entry)

    try:
        output_display = str(output_path.relative_to(REPO_ROOT))
    except ValueError:
        output_display = str(output_path)
    return {
        "schema_version": 1,
        "task": "ReplaceSIMCard",
        "step_labels": list(STEP_LABELS),
        "output_path": output_display,
        "videos": videos,
    }


def optional_time(value: object, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number or null")
    result = round(float(value), 3)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite, non-negative number")
    return result


def normalize_payload(payload: object, video_names: list[str]) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
        raise ValueError("payload must contain a videos list")

    posted: dict[str, dict] = {}
    for entry in payload["videos"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("every video entry must have a filename")
        posted[entry["file"]] = entry

    if set(posted) != set(video_names):
        raise ValueError("video list does not match the recordings folder")

    videos = []
    for filename in video_names:
        source = posted[filename]
        outcome = source.get("outcome")
        if outcome not in {"correct", "incorrect"}:
            raise ValueError(f"{filename}: outcome must be correct or incorrect")

        deviation_step = source.get("deviation_step")
        if deviation_step in ("", None):
            deviation_step = None
        elif (
            isinstance(deviation_step, bool)
            or not isinstance(deviation_step, int)
            or deviation_step not in range(1, 4)
        ):
            raise ValueError(f"{filename}: deviation_step must be 1, 2, 3, or null")
        if outcome == "correct":
            deviation_step = None

        duration = optional_time(source.get("duration"), f"{filename}.duration")
        source_steps = source.get("steps")
        if not isinstance(source_steps, list):
            raise ValueError(f"{filename}: steps must be a list")
        source_by_index = {
            step.get("index"): step
            for step in source_steps
            if isinstance(step, dict)
        }

        steps = []
        for index, label in enumerate(STEP_LABELS, start=1):
            source_step = source_by_index.get(index, {})
            start = optional_time(
                source_step.get("start"), f"{filename}.step{index}.start"
            )
            end = optional_time(
                source_step.get("end"), f"{filename}.step{index}.end"
            )
            if start is not None and end is not None and end <= start:
                raise ValueError(
                    f"{filename}: step {index} end must be later than its start"
                )
            if duration is not None:
                if start is not None and start > duration + 0.05:
                    raise ValueError(f"{filename}: step {index} starts after the video")
                if end is not None and end > duration + 0.05:
                    raise ValueError(f"{filename}: step {index} ends after the video")
            steps.append(
                {
                    "index": index,
                    "label": label,
                    "start": start,
                    "end": end,
                }
            )

        videos.append(
            {
                "file": filename,
                "outcome": outcome,
                "deviation_step": deviation_step,
                "duration": duration,
                "steps": steps,
            }
        )

    return {
        "schema_version": 1,
        "task": "ReplaceSIMCard",
        "step_labels": list(STEP_LABELS),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "videos": videos,
    }


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        video_dir: Path,
        output_path: Path,
        video_names: list[str],
    ):
        super().__init__(address, AnnotationHandler)
        self.video_dir = video_dir
        self.output_path = output_path
        self.video_names = video_names


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def log_message(self, format_string: str, *args: object) -> None:
        if not self.path.startswith("/videos/"):
            super().log_message(format_string, *args)

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(
        self, payload: object, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload).encode()
        self.send_bytes(body, "application/json; charset=utf-8", status)

    def do_HEAD(self) -> None:  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        if path.startswith("/videos/"):
            self.serve_video(path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            try:
                body = VIEWER_HTML.read_bytes()
            except OSError as error:
                self.send_json(
                    {"error": f"could not read viewer: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self.send_bytes(body, "text/html; charset=utf-8")
            return
        if path == "/api/state":
            try:
                state = load_state(self.server.video_names, self.server.output_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.send_json(
                    {"error": f"could not load annotations: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self.send_json(state)
            return
        if path.startswith("/videos/"):
            self.serve_video(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        if path != "/api/annotations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            normalized = normalize_payload(payload, self.server.video_names)
            self.server.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.server.output_path.with_suffix(
                self.server.output_path.suffix + ".tmp"
            )
            with temporary.open("w") as file:
                json.dump(normalized, file, indent=2)
                file.write("\n")
            temporary.replace(self.server.output_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"ok": True, "updated_at": normalized["updated_at"]})

    def serve_video(self, request_path: str) -> None:
        filename = request_path.removeprefix("/videos/")
        if (
            not filename
            or Path(filename).name != filename
            or filename not in self.server.video_names
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        video_path = self.server.video_dir / filename
        size = video_path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = RANGE_RE.fullmatch(range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start_text, end_text = match.groups()
            if not start_text:
                suffix_length = int(end_text)
                start = max(0, size - suffix_length)
            else:
                start = int(start_text)
                if end_text:
                    end = min(int(end_text), size - 1)
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return

        try:
            with video_path.open("rb") as file:
                file.seek(start)
                remaining = length
                while remaining:
                    chunk = file.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a local viewer to annotate the three SIM-card steps."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser automatically"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    output_path = args.output.resolve()
    video_names = sorted(path.name for path in video_dir.glob("*.mp4"))
    if not video_names:
        raise SystemExit(f"no .mp4 files found in {video_dir}")
    if not VIEWER_HTML.is_file():
        raise SystemExit(f"viewer file not found: {VIEWER_HTML}")

    server = AnnotationServer(
        (args.host, args.port), video_dir, output_path, video_names
    )
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Annotating {len(video_names)} videos at {url}")
    print(f"Annotations will be saved to {output_path}")
    print("Press Ctrl-C when finished.")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped annotation viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
