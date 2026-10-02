"""Download COIN annotations and the Hugging Face videos for four tasks.

Annotations come from https://github.com/coin-dataset/annotations (COIN.json).
Videos come from https://huggingface.co/datasets/ttyue/COIN_Dataset, which is a
partial mirror. Files are saved as data/videos/{task}/{youtube_id}.mp4.

Setup (from the repo root):

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python scripts/download_coin.py
    python3 scripts/prepare_coin.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coin_tasks import (  # noqa: E402
    COIN_JSON,
    COIN_JSON_URL,
    DATA_DIR,
    HF_REPO_ID,
    SELECTED_JSON,
    TASKS,
    load_database,
    rename_recipe_folders,
    select_videos,
    video_local_path,
    video_repo_path,
)


def download_annotations() -> None:
    COIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    if COIN_JSON.exists() and COIN_JSON.stat().st_size > 0:
        print(f"annotations already present: {COIN_JSON.relative_to(DATA_DIR.parent)}")
        return
    print(f"downloading {COIN_JSON_URL}")
    request = urllib.request.Request(
        COIN_JSON_URL, headers={"User-Agent": "coin-subset-download"}
    )
    with urllib.request.urlopen(request) as response:
        COIN_JSON.write_bytes(response.read())
    print(f"wrote {COIN_JSON.relative_to(DATA_DIR.parent)}")


def write_selected(selected: dict) -> None:
    SELECTED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SELECTED_JSON.open("w") as f:
        json.dump({"database": selected}, f)
    print(
        f"wrote {SELECTED_JSON.relative_to(DATA_DIR.parent)} "
        f"({len(selected)} videos)"
    )


def fetch_one(filename: str, dest: Path) -> str:
    from huggingface_hub import hf_hub_download

    if dest.is_file() and dest.stat().st_size > 0:
        return str(dest)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=DATA_DIR,
    )
    src = DATA_DIR / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        src.replace(dest)
        if src.parent.is_dir() and not any(src.parent.iterdir()):
            src.parent.rmdir()
    return str(dest)


def available_on_hub(recipe_ids: set[int]) -> dict[str, int]:
    from huggingface_hub import HfApi

    api = HfApi()
    available: dict[str, int] = {}
    for recipe_id in sorted(recipe_ids):
        for item in api.list_repo_tree(
            HF_REPO_ID,
            repo_type="dataset",
            path_in_repo=f"videos/{recipe_id}",
            recursive=True,
        ):
            path = getattr(item, "path", "")
            if path.endswith(".mp4"):
                available[path] = int(getattr(item, "size", 0) or 0)
    return available


def download_videos(
    jobs: list[tuple[str, Path]], workers: int
) -> tuple[int, list[str]]:
    ok = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one, filename, dest): filename for filename, dest in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 — keep going so one failure does not stop the rest
                failed.append(filename)
                print(f"[{index}/{len(jobs)}] FAILED {filename}: {exc}")
            else:
                ok += 1
                print(f"[{index}/{len(jobs)}] {filename}")
    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download COIN.json and write the filtered subset, then stop.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel Hugging Face downloads (default: 4).",
    )
    args = parser.parse_args()

    download_annotations()
    selected = select_videos(load_database(COIN_JSON), TASKS)
    write_selected(selected)

    by_task: dict[str, int] = {}
    for info in selected.values():
        by_task[info["class"]] = by_task.get(info["class"], 0) + 1
    for task in TASKS:
        print(f"  {task}: {by_task[task]} videos in COIN.json")

    if args.annotations_only:
        return

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. From the repo root:\n"
            "  python3 -m venv .venv\n"
            "  .venv/bin/pip install -r requirements.txt\n"
            "  .venv/bin/python scripts/download_coin.py"
        )

    print(f"checking which videos are on {HF_REPO_ID}")
    recipe_ids = {int(info["recipe_type"]) for info in selected.values()}
    on_hub = available_on_hub(recipe_ids)
    present = []
    absent = []
    for youtube_id, info in sorted(selected.items()):
        filename = video_repo_path(info, youtube_id)
        if filename in on_hub:
            present.append((filename, video_local_path(info, youtube_id)))
        else:
            absent.append(filename)
    total_bytes = sum(on_hub[filename] for filename, _dest in present)
    print(f"on Hugging Face: {len(present)} ({total_bytes / 1e9:.1f} GB)")
    print(f"not on the mirror: {len(absent)}")
    if not present:
        sys.exit("No matching videos found on the Hugging Face mirror.")

    print(f"downloading {len(present)} videos")
    ok, failed = download_videos(present, args.workers)
    renamed = rename_recipe_folders(selected)
    for recipe_id, task in renamed:
        print(f"renamed videos/{recipe_id} -> videos/{task}")
    print(f"finished: {ok} downloaded, {len(failed)} failed, {len(absent)} absent from the mirror")
    if failed:
        failure_path = DATA_DIR / "raw" / "download_failures.txt"
        failure_path.write_text("\n".join(failed) + "\n")
        print(f"failed paths written to {failure_path.relative_to(DATA_DIR.parent)}")
        sys.exit(1)
    print("next: python3 scripts/prepare_coin.py")


if __name__ == "__main__":
    main()
