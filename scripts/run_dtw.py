"""Save one causal classical or CLIP DTW checkpoint with provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import cv2
import imageio_ffmpeg

from scripts.coin_tasks import resolve_data_root
from src.guideme.baselines.dtw import run_classical_dtw_checkpoint, run_dtw_checkpoint
from src.guideme.features import load_clip_encoder


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Resolve inputs, call the existing checkpoint core, and save a new report.

    Relative reference, execution, checklist and weight paths use --data-root,
    GUIDEME_DATA_ROOT, or repo/data. Relative report paths use the current
    directory. An optional supplied reference duration bypasses metadata;
    otherwise frame_count/FPS estimates the reference sampling bound.
    Classical remains the default. CLIP requires explicit local weights and
    defaults to CPU with batches of 32. No weights are downloaded. Both use
    subsequence matching with free tutorial start/end; results remain raw
    visual matches, including candidate or unknown steps. --warp-penalty
    optionally discourages repeated frames; zero preserves prior behavior.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Complete reference video")
    parser.add_argument("--execution", required=True, help="Execution source video")
    parser.add_argument("--checklist", required=True, help="Reference-only checklist CSV")
    parser.add_argument("--end-s", required=True, type=float, help="Execution checkpoint in seconds")
    parser.add_argument("--window-s", required=True, type=float, help="Execution window in seconds")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--fps", type=float, default=5.0, help="Sampling FPS (default: 5)")
    parser.add_argument(
        "--warp-penalty", type=float, default=0.0,
        help="Nonnegative additive cost for each repeated-frame transition (default: 0)",
    )
    parser.add_argument("--data-root", help="Absolute input root; overrides GUIDEME_DATA_ROOT")
    parser.add_argument("--encoder", choices=("classical", "clip"), default="classical")
    parser.add_argument("--clip-weights", help="Local CLIP state-dict or safetensors checkpoint")
    parser.add_argument("--device", help="CLIP device (default: cpu)")
    parser.add_argument("--batch-size", type=int, help="CLIP frame batch size (default: 32)")
    parser.add_argument(
        "--reference-duration-s", type=float,
        help="Supplied reference sampling bound in seconds; bypasses metadata",
    )
    args = parser.parse_args(argv)

    for name, value in (
        ("end_s", args.end_s), ("window_s", args.window_s), ("fps", args.fps),
        ("reference_duration_s", args.reference_duration_s),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be a finite positive number")
    if not math.isfinite(args.warp_penalty) or args.warp_penalty < 0:
        parser.error("--warp-penalty must be a finite nonnegative number")
    alignment_options = {"warp_penalty": args.warp_penalty} if args.warp_penalty > 0 else {}
    requested_output = Path(args.output).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        parser.error(f"Report already exists: {requested_output}")
    output = requested_output.resolve()
    if output.suffix.lower() != ".json" or not output.parent.is_dir():
        parser.error("Report must be a new .json path under an existing directory")
    if args.encoder == "clip":
        if not args.clip_weights:
            parser.error("--encoder clip requires --clip-weights")
        if args.device is not None and not args.device.strip():
            parser.error("--device must be a nonempty Torch device string")
        if args.batch_size is not None and args.batch_size <= 0:
            parser.error("--batch-size must be a positive integer")
    elif any(value is not None for value in (args.clip_weights, args.device, args.batch_size)):
        parser.error("--clip-weights, --device and --batch-size require --encoder clip")

    root = resolve_data_root(args.data_root)
    paths = {}
    supplied_paths = [
        ("reference", args.reference), ("execution", args.execution),
        ("checklist", args.checklist),
    ]
    if args.encoder == "clip":
        supplied_paths.append(("clip_weights", args.clip_weights))
    for role, supplied in supplied_paths:
        requested = Path(supplied).expanduser()
        paths[role] = (requested if requested.is_absolute() else root / requested).resolve()

    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "checkpoint": {"end_s": args.end_s, "window_s": args.window_s, "fps": args.fps},
        "inputs": {role: {"path": str(path), "sha256": None} for role, path in paths.items()},
        "reference_duration": None,
        "runtime": {},
        "implementation_sha256": {},
        "result": None,
    }
    report["checkpoint"].update(alignment_options)
    stage = "input_hashes"
    try:
        for role, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{role} must point to a nonempty regular file")
            with path.open("rb") as stream:
                report["inputs"][role]["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()

        stage = "implementation_hashes"
        for relative in (
            "scripts/run_dtw.py", "src/guideme/baselines/dtw.py", "src/guideme/video.py",
            "src/guideme/alignment.py", "src/guideme/annotations.py", "src/guideme/features.py",
            "scripts/classical_pairwise_dtw.py", "scripts/coin_tasks.py", "uv.lock",
        ):
            with (REPO_ROOT / relative).open("rb") as stream:
                report["implementation_sha256"][relative] = hashlib.file_digest(stream, "sha256").hexdigest()

        stage = "runtime"
        report["runtime"] = {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "opencv": version("opencv-python-headless"),
            "imageio_ffmpeg": version("imageio-ffmpeg"),
            "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
            "numba": version("numba"),
        }

        stage = "reference_metadata"
        duration = args.reference_duration_s
        if duration is not None:
            report["reference_duration"] = {"method": "provided", "value_s": duration}
        else:
            capture = cv2.VideoCapture(str(paths["reference"]))
            try:
                if not capture.isOpened():
                    raise ValueError("Cannot open reference video metadata")
                frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                reference_fps = capture.get(cv2.CAP_PROP_FPS)
            finally:
                capture.release()
            if (
                not math.isfinite(frame_count) or frame_count <= 0
                or not math.isfinite(reference_fps) or reference_fps <= 0
            ):
                raise ValueError("Reference metadata must have positive finite frame count and FPS")
            duration = frame_count / reference_fps
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Reference metadata must yield a positive finite duration")
            report["reference_duration"] = {
                "method": "opencv_frame_count_over_fps", "value_s": duration,
                "frame_count": frame_count, "fps": reference_fps,
            }

        if args.encoder == "clip":
            stage = "load_encoder"
            model, device, model_metadata = load_clip_encoder(
                paths["clip_weights"], device=args.device or "cpu",
            )
            if model_metadata["checkpoint"]["sha256"] != report["inputs"]["clip_weights"]["sha256"]:
                raise ValueError("Checkpoint hash changed before loading")
            batch_size = args.batch_size if args.batch_size is not None else 32
            report["encoder"] = {"name": "clip", "batch_size": batch_size, "model": model_metadata}
            stage = "run_checkpoint"
            report["result"] = run_dtw_checkpoint(
                paths["reference"], paths["execution"], paths["checklist"],
                reference_duration_s=duration, checkpoint_s=args.end_s,
                window_s=args.window_s, fps=args.fps, encoder="clip",
                clip_model=model, device=device, batch_size=batch_size,
                **alignment_options,
            )
        else:
            stage = "run_checkpoint"
            runner = run_dtw_checkpoint if args.warp_penalty > 0 else run_classical_dtw_checkpoint
            report["result"] = runner(
                paths["reference"], paths["execution"], paths["checklist"],
                reference_duration_s=duration, checkpoint_s=args.end_s,
                window_s=args.window_s, fps=args.fps, **alignment_options,
            )
        report["status"] = "ok"
    except Exception as error:
        # Persist the failure without copying potentially sensitive exception text.
        report["status"] = "error"
        report["error"] = {"stage": stage, "type": type(error).__name__}

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(f"DTW checkpoint: {report['status']}. Report: {output}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
