"""Read a manually authored reference checklist without edits."""

import csv
import math
from pathlib import Path


def load_reference_annotations(
    path: str | Path,
    video_duration_s: float,
) -> list[dict]:
    """Load format-v1 rows, converting start_s/end_s to floats.

    The caller supplies the CSV path and the corresponding video's duration.
    Require the exact five-column header, complete rows, nonempty unique IDs,
    nonempty actions and finite intervals within the video. Requirements may
    be blank. Preserve authored text and row order; never fill unlabelled gaps,
    renumber IDs, sort rows or infer whether a step was performed correctly.

    A header-only template returns []. Invalid content raises ValueError with
    a line number when available; filesystem and decoding errors propagate.
    This function does not decode the video or verify its identity or labels.
    """
    if (
        isinstance(video_duration_s, bool)
        or not isinstance(video_duration_s, (int, float))
        or not math.isfinite(video_duration_s)
        or video_duration_s <= 0
    ):
        raise ValueError("video_duration_s must be a finite positive number")

    columns = ["step_id", "action", "start_s", "end_s", "requirements"]
    rows = []
    seen_ids = set()
    try:
        with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != columns:
                raise ValueError(f"CSV header must be {','.join(columns)}")
            for row in reader:
                location = f"CSV line {reader.line_num}"
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{location}: expected exactly five fields")
                if not row["step_id"].strip() or row["step_id"] in seen_ids:
                    raise ValueError(f"{location}: step_id must be nonempty and unique")
                if not row["action"].strip():
                    raise ValueError(f"{location}: action must be nonempty")
                try:
                    start = float(row["start_s"])
                    end = float(row["end_s"])
                except ValueError as error:
                    raise ValueError(f"{location}: timestamps must be numeric seconds") from error
                if (
                    not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= video_duration_s
                ):
                    raise ValueError(
                        f"{location}: require finite 0 <= start_s < end_s <= video_duration_s"
                    )
                rows.append({**row, "start_s": start, "end_s": end})
                seen_ids.add(row["step_id"])
    except csv.Error as error:
        raise ValueError(f"Invalid CSV: {error}") from error
    return rows
