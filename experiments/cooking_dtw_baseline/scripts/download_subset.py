"""Download a few chosen COIN cooking videos from the Hugging Face mirror.

Saves each video as data/videos/<Task>/<youtube_id>.mp4, the layout that
scripts.run_coin_reference expects. Skips files that are already present.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
HF_REPO_ID = "ttyue/COIN_Dataset"

# The 8 smallest mirror videos in each task's most common step sequence.
VIDEOS = {
    "BoilNoodles": [
        "krNVdmW0Nto", "1BrN3RGFXDs", "TiI8UcSKnqs", "78u2RZpv_o4",
        "4JwUCS5JjfY", "V5B-nTO2Jl4", "2BJnbSPb-V0", "bFu3hfS6GNQ",
    ],
    "MakeBurger": [
        "aogu4SX7ULg", "avEBVTvYePk", "DbTTRH-iJfk", "6yLbrPvcxDs",
        "pwRFmEYGbyc", "gnlUBK-cvfc", "ZZlBqkVbRaQ", "goEDc2eOZsw",
    ],
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def main() -> None:
    database = json.loads((DATA_DIR / "raw" / "COIN.json").read_text())["database"]
    staging = DATA_DIR / "_hf_staging"
    for task, ids in VIDEOS.items():
        for youtube_id in ids:
            info = database[youtube_id]
            if info["class"] != task:
                raise ValueError(f"{youtube_id} is {info['class']}, not {task}")
            dest = DATA_DIR / "videos" / task / f"{youtube_id}.mp4"
            if dest.is_file() and dest.stat().st_size > 0:
                log(f"skip {task}/{youtube_id} (present)")
                continue
            filename = f"videos/{info['recipe_type']}/{youtube_id}.mp4"
            src = Path(hf_hub_download(
                repo_id=HF_REPO_ID, repo_type="dataset",
                filename=filename, local_dir=staging,
            ))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dest)
            log(f"got {task}/{youtube_id} {info['subset']} "
                f"{dest.stat().st_size / 1e6:.1f} MB")
    shutil.rmtree(staging, ignore_errors=True)
    log("done")


if __name__ == "__main__":
    main()
