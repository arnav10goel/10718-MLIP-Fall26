"""CleanToilet settings for the shared DTW runner.

The canonical videos are the COIN clips whose annotation is detergent, scrub,
then flush, once each. From the repo root:

    .venv/bin/python scripts/clean_toilet.py cutoff
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_spec import TaskSpec  # noqa: E402

SPEC = TaskSpec(
    task="CleanToilet",
    step_labels=(
        "apply detergent to the intertior of the toilet",
        "scrub the toilet interior",
        "flush and wash the interior",
    ),
    expected_videos=3,
    artifact_stem="toilet",
    recording_count=6,
    annotations_name="toilet_recorded_annotations.json",
    annotated_output_name="toilet_recorded_similarity.json",
    unlabeled_output_name="toilet_recorded_similarity_unlabeled.json",
    recording_classical_cache_name="toilet_recorded_classical_features.npz",
    recording_clip_cache_name="toilet_recorded_clip_features.npz",
)


def main(argv: list[str] | None = None) -> int:
    from task_runner import run

    return run(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
