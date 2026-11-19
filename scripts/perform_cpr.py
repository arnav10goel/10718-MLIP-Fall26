"""PerformCPR settings for the recording annotator.

The phone recordings are labeled against reference Bk-DzEUeTaY. That clip is
the only COIN video with this step list, so this task has no pairwise cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_spec import TaskSpec  # noqa: E402

SPEC = TaskSpec(
    task="PerformCPR",
    step_labels=(
        "check the foreign matter in the mouth",
        "check the breath of the patient",
        "check the pulse of the patient",
        "press the chest of the patient",
    ),
    expected_videos=1,
    artifact_stem="cpr",
    recording_count=6,
    annotations_name="cpr_recorded_annotations.json",
    annotated_output_name="cpr_recorded_similarity.json",
    unlabeled_output_name="cpr_recorded_similarity_unlabeled.json",
    recording_classical_cache_name="cpr_recorded_classical_features.npz",
    recording_clip_cache_name="cpr_recorded_clip_features.npz",
)

# False-positive checks on the correct recordings. One video per step, assigned
# by shuffling breath, chest press, and pulse with random.Random(0).
CORRECT_FP_CASES = (
    {"file": "cpr-correct1.mp4", "step": 2, "check": "breath"},
    {"file": "cpr-correct2.mp4", "step": 4, "check": "chest press"},
    {"file": "cpr-correct3.mp4", "step": 3, "check": "pulse"},
)


def main(argv: list[str] | None = None) -> int:
    raise SystemExit(
        "PerformCPR has no pairwise cutoff. Annotate with "
        "scripts/annotate_recordings.py --task cpr"
    )


if __name__ == "__main__":
    raise SystemExit(main())
