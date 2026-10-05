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
