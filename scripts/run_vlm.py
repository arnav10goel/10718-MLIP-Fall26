"""Run one causal VLM check on a tutorial and execution prefix."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform

import cv2
from dotenv import dotenv_values
from google import genai
import imageio_ffmpeg

from scripts.coin_tasks import resolve_data_root
from src.guideme.annotations import load_reference_annotations
from src.guideme.video import prepare_execution_prefix
from src.guideme.vlm import predict_deviation, upload_video


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Prepare/upload two videos, request a prediction, and save a new report.

    Relative media paths use --data-root, GUIDEME_DATA_ROOT, or repo/data.
    CSV checklists are validated using a reference metadata duration estimate,
    then sent as JSON text; plain-text checklists are also accepted.
    Report/checklist paths use the current working directory. Credentials come
    from GEMINI_API_KEY in the process environment, then the selected .env file.
    Uploaded files remain available for reuse; this command does not delete
    them. It performs one checkpoint, without alert filtering or evaluation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Complete tutorial MP4")
    parser.add_argument("--execution", required=True, help="Full execution source video")
    parser.add_argument("--end-s", required=True, type=float, help="Execution checkpoint in seconds")
    parser.add_argument("--prefix-output", required=True, help="New video-only prefix MP4")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--data-root", help="Absolute media root; overrides GUIDEME_DATA_ROOT")
    parser.add_argument("--checklist", help="Reference-only checklist CSV or text file")
    parser.add_argument("--fps", type=float, default=2, help="Gemini static sampling FPS (default: 2)")
    parser.add_argument(
        "--model", default="gemini-3.5-flash-lite",
        help="Gemini model ID (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"), help="Local credentials file")
    args = parser.parse_args(argv)
    if not args.model.strip():
        raise ValueError("model must be a nonempty string")

    requested_output = Path(args.output).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"Report already exists: {requested_output}")
    output = requested_output.resolve()
    if output.suffix.lower() != ".json" or not output.parent.is_dir():
        raise ValueError("Report must be a new .json path under an existing directory")
    for name, value in (("end_s", args.end_s), ("fps", args.fps)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")

    root = resolve_data_root(args.data_root)
    paths = {}
    for name, supplied in (
        ("reference", args.reference), ("execution", args.execution),
        ("prefix", args.prefix_output),
    ):
        requested = Path(supplied).expanduser()
        requested = requested if requested.is_absolute() else root / requested
        if name == "prefix" and (requested.exists() or requested.is_symlink()):
            raise FileExistsError(f"Prefix already exists: {requested}")
        paths[name] = requested.resolve()
    if paths["reference"].suffix.lower() != ".mp4" or paths["prefix"].suffix.lower() != ".mp4":
        raise ValueError("Reference and prefix paths must have .mp4 extensions")
    for name in ("reference", "execution"):
        if not paths[name].is_file() or paths[name].stat().st_size == 0:
            raise ValueError(f"{name} must point to a nonempty video file")

    checklist_path = Path(args.checklist).expanduser().resolve() if args.checklist else None
    checklist = None
    checklist_metadata = {}
    if checklist_path:
        if checklist_path.suffix.lower() == ".csv":
            capture = cv2.VideoCapture(str(paths["reference"]))
            try:
                if not capture.isOpened():
                    raise ValueError("Cannot read reference video metadata for CSV validation")
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                reference_fps = capture.get(cv2.CAP_PROP_FPS)
            finally:
                capture.release()
            if (
                not math.isfinite(frame_count) or frame_count <= 0
                or not math.isfinite(reference_fps) or reference_fps <= 0
            ):
                raise ValueError("Reference metadata must have a finite positive frame count and FPS")
            duration = frame_count / reference_fps
            rows = load_reference_annotations(checklist_path, duration)
            if not rows:
                raise ValueError("CSV checklist must contain at least one reference step")
            checklist = json.dumps(rows, ensure_ascii=False, allow_nan=False, indent=2)
            checklist_metadata = {
                "format": "csv", "format_version": 1, "step_count": len(rows),
                "reference_duration_s": duration,
                "duration_method": "opencv_frame_count_over_fps",
                "reference_frame_count": frame_count, "reference_fps": reference_fps,
            }
        else:
            checklist = checklist_path.read_text(encoding="utf-8")
            checklist_metadata = {"format": "text"}
    if checklist is not None and not checklist.strip():
        raise ValueError("Checklist must contain reference-only notes")
    env_values = dotenv_values(Path(args.env_file).expanduser(), interpolate=False)
    api_key = os.environ.get("GEMINI_API_KEY") or env_values.get("GEMINI_API_KEY")
    if not api_key or not api_key.strip() or api_key.strip() in ("<>", "your_actual_key"):
        raise ValueError("Set GEMINI_API_KEY locally before running this command")

    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "checkpoint_s": args.end_s,
        "requested_model": args.model,
        "inputs": {},
        "checklist": None,
        "runtime": {
            "python": platform.python_version(),
            "google_genai": version("google-genai"),
            "python_dotenv": version("python-dotenv"),
            "imageio_ffmpeg": version("imageio-ffmpeg"),
            "ffmpeg": imageio_ffmpeg.get_ffmpeg_version(),
            "request_timeout_ms": 120000,
            "request_attempts": 1,
            "execution_audio": False,
            "fps": args.fps,
        },
        "implementation_sha256": {},
        "uploads": {},
        "result": None,
    }
    for relative in (
        "scripts/run_vlm.py", "src/guideme/vlm.py", "src/guideme/video.py",
        "src/guideme/annotations.py",
    ):
        report["implementation_sha256"][relative] = hashlib.sha256(
            (REPO_ROOT / relative).read_bytes()
        ).hexdigest()
    for name in ("reference", "execution"):
        with paths[name].open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["inputs"][name] = {"path": str(paths[name]), "sha256": digest}
    if checklist_path:
        report["checklist"] = {
            "path": str(checklist_path),
            "sha256": hashlib.sha256(checklist_path.read_bytes()).hexdigest(),
            **checklist_metadata,
        }

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 120000, "retry_options": {"attempts": 1}},
    )
    stage = "prepare_prefix"
    try:
        paths["prefix"].parent.mkdir(parents=True, exist_ok=True)
        prefix = prepare_execution_prefix(paths["execution"], args.end_s, paths["prefix"])
        with prefix.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["inputs"]["prefix"] = {"path": str(prefix), "sha256": digest}
        stage = "upload_reference"
        report["uploads"]["reference"] = upload_video(client, paths["reference"])
        stage = "upload_execution"
        report["uploads"]["execution"] = upload_video(client, prefix)
        stage = "predict"
        report["result"] = predict_deviation(
            client, report["uploads"]["reference"]["uri"],
            report["uploads"]["execution"]["uri"], checklist=checklist, fps=args.fps,
            model=args.model,
        )
        report["status"] = "ok" if report["result"]["prediction"] is not None else "invalid_output"
    except Exception as error:
        # Avoid copying exception messages that could contain credentials.
        report["status"] = "error"
        code = getattr(error, "code", None)
        report["error"] = {
            "stage": stage, "type": type(error).__name__,
            "code": code if isinstance(code, int) else None,
        }
    finally:
        try:
            client.close()
        except Exception as error:
            report["client_close_error"] = type(error).__name__

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(f"VLM check: {report['status']}. Report: {output}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
