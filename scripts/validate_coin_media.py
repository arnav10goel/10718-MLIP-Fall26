"""Check whether one local COIN video candidate is usable for scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import imageio_ffmpeg

from scripts.coin_tasks import index_task_media


def validate_media_candidate(
    path: Path, expected_sha256: str | None = None
) -> dict:
    """Hash and fully decode one candidate; do not infer its COIN identity."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"Media path must be absolute: {path}")

    result = {
        "path": str(path),
        "size_bytes": None,
        "sha256": None,
        "decoded_frame_count": None,
        "usable": False,
        "issues": [],
    }
    issues = result["issues"]
    if path.is_symlink() or (path.exists() and not path.is_file()):
        issues.append("not_file")
        return result
    if not path.exists():
        issues.append("missing_media")
        return result

    try:
        result["size_bytes"] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    except OSError as exc:
        issues.append("read_error")
        result["error_detail"] = str(exc)
        return result

    if expected_sha256 is not None and result["sha256"] != expected_sha256.lower():
        issues.append("checksum_mismatch")
    if result["size_bytes"] == 0:
        issues.append("empty_media")
        return result

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-v", "error", "-xerror",
        "-i", str(path),
        "-map", "0:v:0", "-an", "-f", "null", "-",
        "-progress", "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    frames = 0
    for line in completed.stdout.splitlines():
        if line.startswith("frame="):
            try:
                frames = int(line.partition("=")[2].strip())
            except ValueError:
                pass
    result["decoded_frame_count"] = frames
    if completed.returncode != 0:
        issues.append("decode_error")
        result["error_detail"] = completed.stderr.strip()[-500:]
    elif frames == 0:
        issues.append("no_video_frames")
    result["usable"] = not issues
    return result


def inventory_task_media(
    database: dict[str, dict],
    task: str,
    data_root: str | Path | None = None,
    expected_sha256_by_id: dict[str, str] | None = None,
) -> dict:
    """Report every task ID and return paths whose media passed validation."""
    indexed = index_task_media(database, task, data_root)
    candidates = indexed["candidate_paths"]
    index_issues = {issue["video_id"]: issue for issue in indexed["issues"]}
    expected_hashes = expected_sha256_by_id or {}
    usable_paths: dict[str, Path] = {}
    records: list[dict] = []

    for video_id, info in sorted(database.items()):
        if info.get("class") != task:
            continue
        expected_hash = expected_hashes.get(video_id)
        record = {
            "video_id": video_id,
            "subset": info.get("subset"),
            "source_url": info.get("video_url"),
            "expected_sha256": expected_hash,
            "path": None,
            "size_bytes": None,
            "sha256": None,
            "decoded_frame_count": None,
            "usable": False,
            "issues": [],
        }
        if video_id in candidates:
            path = candidates[video_id]
            record.update(validate_media_candidate(path, expected_hash))
            if record["usable"]:
                usable_paths[video_id] = path
        else:
            issue = index_issues[video_id]
            record["issues"] = [issue["reason"]]
            if "paths" in issue:
                record["paths"] = issue["paths"]
        records.append(record)

    return {"task": task, "usable_paths": usable_paths, "records": records}
