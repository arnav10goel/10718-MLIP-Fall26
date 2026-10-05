"""Shared paths and the four COIN tasks we keep."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
COIN_JSON = DATA_DIR / "raw" / "COIN.json"
SELECTED_JSON = DATA_DIR / "raw" / "selected.json"
VIDEOS_DIR = DATA_DIR / "videos"
PREPARED_DIR = DATA_DIR / "prepared"

# Official COIN task names (the "class" field in COIN.json).
TASKS = (
    "ReplaceSIMCard",
    "BoilNoodles",
    "WashDish",
    "CleanToilet",
)

COIN_JSON_URL = (
    "https://raw.githubusercontent.com/coin-dataset/annotations/master/COIN.json"
)
HF_REPO_ID = "ttyue/COIN_Dataset"


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Choose an absolute data root without creating or checking the directory."""
    configured = os.environ.get("GUIDEME_DATA_ROOT") if data_root is None else data_root
    if configured is None or configured == "":
        if data_root is not None:
            raise ValueError("Data root cannot be empty")
        configured = REPO_ROOT / "data"
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ValueError(f"Data root must be absolute: {root}")
    return root.resolve()


def load_database(path: Path) -> dict:
    with path.open() as f:
        payload = json.load(f)
    database = payload["database"]
    if not isinstance(database, dict):
        raise ValueError(f"{path} has no 'database' object")
    return database


def select_videos(database: dict, tasks: tuple[str, ...] = TASKS) -> dict[str, dict]:
    """Return COIN entries whose task is in `tasks`, keyed by YouTube id."""
    wanted = set(tasks)
    selected = {
        youtube_id: info
        for youtube_id, info in database.items()
        if info.get("class") in wanted
    }
    found = {info["class"] for info in selected.values()}
    missing = wanted - found
    if missing:
        raise ValueError(f"Tasks not found in COIN.json: {sorted(missing)}")
    return selected


def index_task_media(
    database: dict[str, dict], task: str, data_root: str | Path | None = None
) -> dict:
    """Find local COIN media candidates without claiming they are validated."""
    root = resolve_data_root(data_root)
    if not root.is_dir():
        raise ValueError(f"Data root is not a directory: {root}")
    video_ids = sorted(
        video_id for video_id, info in database.items() if info.get("class") == task
    )
    if not video_ids:
        raise ValueError(f"Task not found in COIN database: {task}")

    task_dir = root / "videos" / task
    if (root / "videos").is_symlink() or task_dir.is_symlink():
        raise ValueError(f"Video layout contains a symlink: {task_dir}")
    candidate_paths: dict[str, Path] = {}
    issues: list[dict] = []
    for video_id in video_ids:
        possible = [task_dir / f"{video_id}.{ext}" for ext in ("mp4", "webm")]
        present = [path for path in possible if path.exists() or path.is_symlink()]
        if not present:
            issues.append({"video_id": video_id, "reason": "missing_media"})
            continue
        if len(present) > 1:
            reason = "duplicate_media"
        elif present[0].is_symlink() or not present[0].is_file():
            reason = "not_file"
        elif present[0].stat().st_size == 0:
            reason = "empty_media"
        else:
            candidate_paths[video_id] = present[0]
            continue
        issues.append({
            "video_id": video_id,
            "reason": reason,
            "paths": [path.as_posix() for path in present],
        })
    return {"candidate_paths": candidate_paths, "issues": issues}


def select_matching_cohort(
    database: dict[str, dict],
    task: str,
    reference_id: str,
    available_ids: set[str],
) -> dict:
    """Group available task videos by exact ordered COIN step IDs."""
    reference = database.get(reference_id)
    if (
        reference is None
        or reference.get("class") != task
        or reference.get("subset") != "training"
        or not reference.get("annotation")
        or reference_id not in available_ids
    ):
        raise ValueError(f"{reference_id} is not an available training reference for {task}")

    step_ids = [str(step["id"]) for step in reference["annotation"]]
    training_ids: list[str] = []
    testing_ids: list[str] = []
    excluded: list[dict[str, str]] = []

    for video_id, info in sorted(database.items()):
        if info.get("class") != task:
            continue
        ids = [str(step["id"]) for step in info.get("annotation", [])]
        if ids != step_ids:
            reason = "different_steps"
        elif video_id not in available_ids:
            reason = "unavailable_media"
        elif info.get("subset") == "training":
            training_ids.append(video_id)
            continue
        elif info.get("subset") == "testing":
            testing_ids.append(video_id)
            continue
        else:
            reason = "unexpected_subset"
        excluded.append({"video_id": video_id, "reason": reason})

    return {
        "task": task,
        "reference_id": reference_id,
        "step_ids": step_ids,
        "training_ids": training_ids,
        "testing_ids": testing_ids,
        "excluded": excluded,
    }


def video_repo_path(info: dict, youtube_id: str) -> str:
    """Path inside the Hugging Face repo. Those folders are numeric task ids."""
    return f"videos/{info['recipe_type']}/{youtube_id}.mp4"


def video_local_path(info: dict, youtube_id: str) -> Path:
    return VIDEOS_DIR / info["class"] / f"{youtube_id}.mp4"


def recipe_to_task(selected: dict[str, dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for info in selected.values():
        recipe_id = str(info["recipe_type"])
        task = info["class"]
        previous = mapping.setdefault(recipe_id, task)
        if previous != task:
            raise ValueError(f"recipe_type {recipe_id} maps to both {previous} and {task}")
    return mapping


def rename_recipe_folders(selected: dict[str, dict]) -> list[tuple[str, str]]:
    """Move data/videos/{recipe_type}/ to data/videos/{task}/."""
    renamed: list[tuple[str, str]] = []
    for recipe_id, task in recipe_to_task(selected).items():
        src = VIDEOS_DIR / recipe_id
        dest = VIDEOS_DIR / task
        if not src.is_dir():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for video in src.glob("*.mp4"):
            target = dest / video.name
            if target.exists():
                continue
            video.rename(target)
        if not any(src.iterdir()):
            src.rmdir()
        renamed.append((recipe_id, task))
    return renamed
