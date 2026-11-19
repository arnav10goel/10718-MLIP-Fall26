"""ReplaceBatteryOnTVControl settings for the shared DTW runner.

The canonical videos are the 13 downloaded COIN clips that open the cover,
remove the old battery, put a battery in, and close the cover, once each.
One more COIN clip has that annotation but is not on the Hugging Face mirror.
From the repo root:

    .venv/bin/python scripts/replace_battery.py cutoff
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_spec import TaskSpec  # noqa: E402

SPEC = TaskSpec(
    task="ReplaceBatteryOnTVControl",
    step_labels=(
        "open cover",
        "remove battery",
        "put battery in",
        "close cover",
    ),
    expected_videos=13,
    artifact_stem="battery",
    recording_count=6,
    annotations_name="battery_recorded_annotations.json",
    annotated_output_name="battery_recorded_similarity.json",
    unlabeled_output_name="battery_recorded_similarity_unlabeled.json",
    recording_classical_cache_name="battery_recorded_classical_features.npz",
    recording_clip_cache_name="battery_recorded_clip_features.npz",
)


def main(argv: list[str] | None = None) -> int:
    from task_runner import run

    return run(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
