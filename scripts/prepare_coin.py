"""Build a local manifest for the downloaded COIN tasks.

Reads data/raw/COIN.json and the mp4s under data/videos/. Writes:

    data/prepared/manifest.jsonl   every video that is on disk
    data/prepared/train.jsonl      official training subset of those videos
    data/prepared/test.jsonl       official testing subset of those videos
    data/prepared/summary.json     counts, including videos missing from the mirror

The official COIN field is "subset", with values "training" and "testing".
Run scripts/download_coin.py first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coin_tasks import (  # noqa: E402
    COIN_JSON,
    PREPARED_DIR,
    REPO_ROOT,
    TASKS,
    VIDEOS_DIR,
    load_database,
    select_videos,
    video_local_path,
)


def step_record(step: dict) -> dict:
    start, end = step["segment"]
    return {
        "id": step["id"],
        "label": step["label"],
        "start": start,
        "end": end,
    }


def manifest_row(youtube_id: str, info: dict, video_path: Path) -> dict:
    return {
        "youtube_id": youtube_id,
        "task": info["class"],
        "recipe_type": info["recipe_type"],
        "subset": info["subset"],
        "duration": info["duration"],
        "roi_start": info["start"],
        "roi_end": info["end"],
        "video_path": video_path.relative_to(REPO_ROOT).as_posix(),
        "steps": [step_record(step) for step in info["annotation"]],
    }


def find_video(youtube_id: str, info: dict) -> Path | None:
    """The task folder itself, then its sorting subfolders such as used/ and unused/."""
    path = video_local_path(info, youtube_id)
    candidates = [path] + sorted(
        candidate
        for candidate in path.parent.glob(f"*/{youtube_id}.mp4")
        if candidate.parent.name != "recorded"
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    if not COIN_JSON.exists():
        sys.exit(f"Missing {COIN_JSON}. Run scripts/download_coin.py first.")

    selected = select_videos(load_database(COIN_JSON), TASKS)
    rows: list[dict] = []
    missing: dict[str, list[str]] = {task: [] for task in TASKS}

    for youtube_id, info in sorted(selected.items(), key=lambda item: (item[1]["class"], item[0])):
        path = find_video(youtube_id, info)
        if path is not None:
            rows.append(manifest_row(youtube_id, info, path))
        else:
            missing[info["class"]].append(youtube_id)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = [row for row in rows if row["subset"] == "training"]
    test_rows = [row for row in rows if row["subset"] == "testing"]
    other = [row["youtube_id"] for row in rows if row["subset"] not in {"training", "testing"}]

    write_jsonl(PREPARED_DIR / "manifest.jsonl", rows)
    write_jsonl(PREPARED_DIR / "train.jsonl", train_rows)
    write_jsonl(PREPARED_DIR / "test.jsonl", test_rows)

    summary_tasks = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        official = sum(1 for info in selected.values() if info["class"] == task)
        recipe_type = next(
            info["recipe_type"] for info in selected.values() if info["class"] == task
        )
        summary_tasks[task] = {
            "recipe_type": recipe_type,
            "official": official,
            "downloaded": len(task_rows),
            "missing": len(missing[task]),
            "training": sum(1 for row in task_rows if row["subset"] == "training"),
            "testing": sum(1 for row in task_rows if row["subset"] == "testing"),
            "missing_ids": missing[task],
        }

    summary = {
        "videos_dir": VIDEOS_DIR.relative_to(REPO_ROOT).as_posix(),
        "downloaded_total": len(rows),
        "training": len(train_rows),
        "testing": len(test_rows),
        "unexpected_subset": other,
        "tasks": summary_tasks,
    }
    with (PREPARED_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"videos on disk: {len(rows)}")
    print(f"  training: {len(train_rows)}")
    print(f"  testing:  {len(test_rows)}")
    for task in TASKS:
        stats = summary_tasks[task]
        print(
            f"  {task}: {stats['downloaded']}/{stats['official']} "
            f"(train {stats['training']}, test {stats['testing']}, "
            f"missing {stats['missing']})"
        )
    print(f"wrote {PREPARED_DIR.relative_to(REPO_ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
