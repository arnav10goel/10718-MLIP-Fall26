"""Download every matching ReplaceSIMCard and MakeStrawberrySmoothie video into data/videos/<task>/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deviation_detection" / "scripts"))

from common import download, load_database, load_mirror, log, task_cohort  # noqa: E402

for task in ("ReplaceSIMCard", "MakeStrawberrySmoothie"):
    database, mirror = load_database(), load_mirror()
    cohort = task_cohort(database, task, mirror)
    for index, vid in enumerate(cohort["correct"], 1):
        download(task, vid, database[vid])
        log(f"{task} [{index}/{len(cohort['correct'])}] {vid}")
