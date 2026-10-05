"""Offline VLM alignment: annotated reference vs whole unlabelled execution.

Examples (repository root):

    # COIN reference: its annotation becomes the checklist; both videos are cut
    # to their COIN task section and their audio is removed.
    python -m scripts.run_offline_vlm \
        --reference videos/MakePaperWindMill/4ufBW5Cfpgw.mp4 \
        --coin-reference-id 4ufBW5Cfpgw \
        --execution videos/MakePaperWindMill/TULczOB5joI.mp4 --coin-execution-id TULczOB5joI \
        --output runs/windmill_offline.json

    # Own recording: a reference CSV checklist (docs/annotation-guide.md) and a raw execution.
    python -m scripts.run_offline_vlm --reference ref.mp4 --checklist ref.csv \
        --execution my_attempt.mp4 --output runs/attempt.json

Add --dry-run to prepare the clips and the request without calling Gemini.
Credentials come from GEMINI_API_KEY in the environment, then --env-file.
Uploaded files are not deleted. The execution's own labels are never sent.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform

from scripts.coin_tasks import load_database, resolve_data_root
from src.guideme.annotations import load_reference_annotations
from src.guideme.offline_vlm import (
    coin_checklist_rows, cut_segments, predict_offline_alignment, video_duration_s,
    write_checklist_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Reference tutorial video")
    parser.add_argument("--execution", required=True, help="Execution video (no labels)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checklist", help="Reference checklist CSV (format v1, reference-video seconds)")
    source.add_argument("--coin-reference-id", help="Build the checklist from this COIN video's annotation")
    parser.add_argument("--coin-execution-id",
                        help="Cut the execution to this COIN video's task section (its labels are not used)")
    parser.add_argument("--execution-start-s", type=float, help="Cut the execution from here")
    parser.add_argument("--execution-end-s", type=float, help="Cut the execution up to here")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--clips-dir", help="Where prepared clips go (default: <output stem>_clips/)")
    parser.add_argument("--data-root", help="Absolute media root; overrides GUIDEME_DATA_ROOT")
    parser.add_argument("--fps", type=float, default=1, help="Gemini sampling FPS (default: 1)")
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Gemini model ID")
    parser.add_argument("--media-resolution", default="high", choices=("low", "medium", "high"),
                        help="Gemini media resolution per frame (default: high)")
    parser.add_argument("--timeout-s", type=float, default=900, help="Network timeout per request")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"), help="Local credentials file")
    parser.add_argument("--dry-run", action="store_true", help="Prepare clips and request only")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report already exists: {output}")
    output = output.resolve()
    if output.suffix.lower() != ".json" or not output.parent.is_dir():
        raise ValueError("Report must be a new .json path under an existing directory")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("fps must be a finite positive number")
    if args.coin_execution_id and (args.execution_start_s is not None or args.execution_end_s is not None):
        raise ValueError("Use either --coin-execution-id or explicit execution times, not both")

    root = resolve_data_root(args.data_root)
    paths = {}
    for name in ("reference", "execution"):
        p = Path(getattr(args, name)).expanduser()
        paths[name] = (p if p.is_absolute() else root / p).resolve()
        if not paths[name].is_file() or paths[name].stat().st_size == 0:
            raise ValueError(f"{name} must point to a nonempty video file: {paths[name]}")
    clips = Path(args.clips_dir).expanduser().resolve() if args.clips_dir else output.with_name(output.stem + "_clips")
    clips.mkdir(parents=True, exist_ok=False)

    database = load_database(root / "raw" / "COIN.json") if (args.coin_reference_id or args.coin_execution_id) else {}

    # Reference: cut to the COIN task section (or keep whole) and drop audio.
    if args.coin_reference_id:
        info = database[args.coin_reference_id]
        ref_start, ref_end = float(info["start"]), float(info["end"])
        rows = coin_checklist_rows(info, offset_s=ref_start)
        checklist_source = {"coin_id": args.coin_reference_id, "offset_s": ref_start}
    else:
        ref_start, ref_end = 0.0, video_duration_s(paths["reference"])
        checklist_source = {"path": str(Path(args.checklist).resolve()),
                            "sha256": _sha256(Path(args.checklist).resolve())}
    reference_clip = cut_segments(paths["reference"], [(ref_start, ref_end)], clips / "reference.mp4")
    ref_duration = video_duration_s(reference_clip)
    if args.coin_reference_id:
        for row in rows:
            row["end_s"] = min(row["end_s"], round(ref_duration, 3))
        checklist_path = write_checklist_csv(rows, clips / "reference_checklist.csv")
    else:
        checklist_path = Path(args.checklist).resolve()
    # A CSV checklist uses the original video's seconds; a re-encoded clip can be a few ms shorter.
    rows = load_reference_annotations(
        checklist_path, ref_duration if args.coin_reference_id else max(ref_duration, ref_end)
    )
    if not rows:
        raise ValueError("Checklist must contain at least one step")

    # Execution: cut (optional) and drop audio so narration cannot reveal the steps.
    if args.coin_execution_id:
        info = database[args.coin_execution_id]
        exe_start, exe_end = float(info["start"]), float(info["end"])
    else:
        exe_start = args.execution_start_s or 0.0
        exe_end = args.execution_end_s or video_duration_s(paths["execution"])
    execution_clip = cut_segments(paths["execution"], [(exe_start, exe_end)], clips / "execution.mp4")
    exe_duration = video_duration_s(execution_clip)

    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": args.model,
        "fps": args.fps,
        "media_resolution": args.media_resolution,
        "timeout_s": args.timeout_s,
        "inputs": {
            "reference": {"path": str(paths["reference"]), "sha256": _sha256(paths["reference"]),
                          "cut_s": [ref_start, ref_end]},
            "execution": {"path": str(paths["execution"]), "sha256": _sha256(paths["execution"]),
                          "cut_s": [exe_start, exe_end]},
            "reference_clip": {"path": str(reference_clip), "duration_s": ref_duration},
            "execution_clip": {"path": str(execution_clip), "duration_s": exe_duration, "audio": False},
        },
        "checklist": {"source": checklist_source, "path": str(checklist_path), "rows": rows},
        "runtime": {"python": platform.python_version()},
        "uploads": {},
        "result": None,
    }

    if args.dry_run:
        report["status"] = "dry_run"
    else:
        from dotenv import dotenv_values
        from google import genai

        from src.guideme.vlm import upload_video

        env_values = dotenv_values(Path(args.env_file).expanduser(), interpolate=False)
        api_key = os.environ.get("GEMINI_API_KEY") or env_values.get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("Set GEMINI_API_KEY locally before running this command")
        client = genai.Client(api_key=api_key, http_options={"timeout": int(args.timeout_s * 1000), "retry_options": {"attempts": 1}})
        stage = "upload_reference"
        try:
            report["uploads"]["reference"] = upload_video(client, reference_clip, processing_timeout_s=300)
            stage = "upload_execution"
            report["uploads"]["execution"] = upload_video(client, execution_clip, processing_timeout_s=300)
            stage = "predict"
            report["result"] = predict_offline_alignment(
                client, report["uploads"]["reference"]["uri"], report["uploads"]["execution"]["uri"],
                rows, execution_duration_s=exe_duration, fps=args.fps, model=args.model,
                media_resolution=f"MEDIA_RESOLUTION_{args.media_resolution.upper()}",
            )
            report["status"] = "ok" if report["result"]["prediction"] is not None else "invalid_output"
        except Exception as error:
            # Do not copy exception text: it could contain credentials.
            code = getattr(error, "code", None)
            report["status"] = "error"
            report["error"] = {"stage": stage, "type": type(error).__name__,
                               "code": code if isinstance(code, int) else None}
        finally:
            try:
                client.close()
            except Exception as error:
                report["client_close_error"] = type(error).__name__

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Offline VLM: {report['status']}. Report: {output}", flush=True)
    return 0 if report["status"] in ("ok", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
