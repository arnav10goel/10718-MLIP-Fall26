"""MakeStrawberrySmoothie settings for the shared DTW runner.

The canonical videos are the COIN smoothie videos on disk with the three COIN
steps. From the repo root:

    .venv/bin/python scripts/make_strawberry_smoothie.py cutoff
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_spec import TaskSpec  # noqa: E402

SPEC = TaskSpec(
    task="MakeStrawberrySmoothie",
    step_labels=(
        "put strawberries and other fruits into the juicer",
        "put yogurt, honey and other ingredients into the juicer",
        "shake and juice",
    ),
    expected_videos=17,
    artifact_stem="smoothie",
    recording_count=6,
    annotations_name="smoothie_recorded_annotations.json",
    annotated_output_name="smoothie_recorded_similarity.json",
    unlabeled_output_name="smoothie_recorded_similarity_unlabeled.json",
    recording_classical_cache_name="smoothie_recorded_classical_features.npz",
    recording_clip_cache_name="smoothie_recorded_clip_features.npz",
)


def main(argv: list[str] | None = None) -> int:
    from task_runner import run

    return run(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
