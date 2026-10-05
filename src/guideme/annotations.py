"""Read manually authored reference and execution annotations without edits."""

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


def load_execution_annotations(
    path: str | Path,
    video_duration_s: float,
    reference_rows: list[dict],
) -> list[dict]:
    """Load private execution labels against an already validated reference.

    The caller supplies the execution duration and its paired reference rows
    from load_reference_annotations. Require the exact seven-column header,
    complete rows, unique nonblank occurrence IDs, nonblank actions, known
    reference-step IDs or empty matches, and finite intervals within the video.
    Convert empty reference matches to None and timestamps to floats; preserve
    all other text and row order. Repeated reference matches, overlapping
    intervals and gaps are retained without choosing a current step.

    Evidence must be visible_action or result. A result interval locates
    visible evidence, not when an unseen action occurred. Notes may be empty.
    A header-only file returns []; invalid content raises ValueError, with a
    line number when available. Filesystem and decoding errors propagate.
    Does not verify video identity, label correctness or missed steps. Keep
    these private rows out of model inputs; they are only for evaluation.
    """
    if (
        isinstance(video_duration_s, bool)
        or not isinstance(video_duration_s, (int, float))
        or not math.isfinite(video_duration_s)
        or video_duration_s <= 0
    ):
        raise ValueError("video_duration_s must be a finite positive number")

    columns = [
        "occurrence_id", "reference_step_id", "action", "start_s", "end_s",
        "evidence_type", "notes",
    ]
    reference_ids = {row["step_id"] for row in reference_rows}
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
                    raise ValueError(f"{location}: expected exactly seven fields")
                if not row["occurrence_id"].strip() or row["occurrence_id"] in seen_ids:
                    raise ValueError(f"{location}: occurrence_id must be nonempty and unique")
                if not row["action"].strip():
                    raise ValueError(f"{location}: action must be nonempty")
                reference_step_id = row["reference_step_id"] or None
                if reference_step_id is not None and reference_step_id not in reference_ids:
                    raise ValueError(f"{location}: reference_step_id must match the paired reference")
                if row["evidence_type"] not in ("visible_action", "result"):
                    raise ValueError(f"{location}: evidence_type must be visible_action or result")
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
                rows.append({
                    **row, "reference_step_id": reference_step_id,
                    "start_s": start, "end_s": end,
                })
                seen_ids.add(row["occurrence_id"])
    except csv.Error as error:
        raise ValueError(f"Invalid CSV: {error}") from error
    return rows


def reference_step_at_time(rows: list[dict], time_s: float | None) -> dict | None:
    """Look up one timestamp in already validated reference checklist rows.

    Intervals include their start and exclude their end. Return a copy of the
    sole matching row, or None for an unknown time, gap or overlapping labels.
    Preserve IDs and gaps; do not choose the nearest step. This labels a
    reference timestamp, not execution correctness or DTW match confidence.
    """
    if time_s is None:
        return None
    if (
        isinstance(time_s, bool) or not isinstance(time_s, (int, float))
        or (isinstance(time_s, float) and not math.isfinite(time_s)) or time_s < 0
    ):
        raise ValueError("time_s must be None or a finite nonnegative number")
    matches = [row for row in rows if row["start_s"] <= time_s < row["end_s"]]
    return matches[0].copy() if len(matches) == 1 else None


def execution_step_at_time(rows: list[dict], time_s: float | None) -> dict | None:
    """Return the sole matched visible-action label at an execution timestamp.

    Rows must already be validated by load_execution_annotations. Intervals
    include their start and exclude their end. Return a copy of the sole active
    visible_action row when it has a reference-step match; otherwise None.
    Unknown times, gaps, unmatched actions and overlapping visible actions are
    unknown, even when overlapping actions share a step ID. Result evidence
    does not establish an action's timing and is ignored. Do not carry a step
    forward or pick its nearest interval.

    Invalid timestamps raise ValueError. This reads private evaluation labels,
    not video frames; it does not recognize progress or detect a deviation.
    """
    if time_s is None:
        return None
    if (
        isinstance(time_s, bool) or not isinstance(time_s, (int, float))
        or (isinstance(time_s, float) and not math.isfinite(time_s)) or time_s < 0
    ):
        raise ValueError("time_s must be None or a finite nonnegative number")
    matches = [
        row for row in rows
        if row["evidence_type"] == "visible_action"
        and row["start_s"] <= time_s < row["end_s"]
    ]
    if len(matches) != 1 or matches[0]["reference_step_id"] is None:
        return None
    return matches[0].copy()
