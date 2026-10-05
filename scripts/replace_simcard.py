"""ReplaceSIMCard settings for the shared DTW runner.

Another task is a copy of this file with a different TaskSpec. From the repo root:

    .venv/bin/python scripts/replace_simcard.py cutoff
    .venv/bin/python scripts/replace_simcard.py score
    .venv/bin/python scripts/replace_simcard.py score --no-recording-annotations
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_spec import TaskSpec  # noqa: E402

SPEC = TaskSpec(
    task="ReplaceSIMCard",
    step_labels=(
        "use the needle to open the SIM card slot",
        "put the SIM card into the SIM card slot",
        "press the SIM card slot back",
    ),
    expected_videos=48,
    artifact_stem="sim",
    recording_count=6,
)


def main(argv: list[str] | None = None) -> int:
    from task_runner import run

    return run(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
