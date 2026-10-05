"""Download every matching video for the three candidate tasks into data/videos/<task>/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deviation_detection" / "scripts"))

from common import download, load_database, load_mirror, log, task_cohort  # noqa: E402

TASKS = ("UseRiceCookerToCookRice", "MakePaperWindMill", "PutOnQuiltCover")


def main() -> None:
    database, mirror = load_database(), load_mirror()
    for task in TASKS:
        cohort = task_cohort(database, task, mirror)
        annotated = sum(1 for info in database.values() if info["class"] == task)
        log(f"{task}: {len(cohort['correct'])} of {annotated} annotated share {cohort['canon_labels']}")
        for index, vid in enumerate(cohort["correct"], 1):
            try:
                download(task, vid, database[vid])
                log(f"[{index}/{len(cohort['correct'])}] {vid}")
            except Exception as exc:  # noqa: BLE001 - keep going
                log(f"[{index}/{len(cohort['correct'])}] FAILED {vid}: {exc}")


if __name__ == "__main__":
    main()
