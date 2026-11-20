"""Pure source-time selection of compact reference and execution JPEG pools."""

import math


def select_compact_frames(reference_rows, execution_rows, *, source_cutoff_s,
                          coarse_interval_s=5, recent_window_s=12):
    """Return original ordinals and a detached host-side selection audit.

    Each metadata row has exactly frame_index, elapsed_s and selection_reason.
    Reference rows are cadence samples plus an optional final_frame at the end.
    Execution rows are cadence samples plus an optional current_endpoint at the
    end; old transient endpoints must never enter the persistent cadence pool.
    Cadence frames supply the first available row at/after each global source
    grid target. Reference first/final frames and execution's actual last frame
    are always retained; execution also keeps cadence samples in the recent
    source-time window, including its lower boundary.

    Ordinals refer to these exact input pools, without resampling or rewriting
    frame metadata. The caller must preserve their alignment with the ready
    observation. Source elapsed time is separate from its session capture and
    receipt clocks; observation construction owns those availability checks.
    This function reads no images, clocks, task annotations or model output.
    """
    def number(value, *, positive=False):
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid or value < 0 or positive and value == 0:
            raise ValueError("Expected a finite source-time setting within its bound")
        return value

    try:
        cutoff = number(source_cutoff_s)
        interval = number(coarse_interval_s, positive=True)
        recent = number(recent_window_s)
        selected, audit_rows = {}, {}
        reason_order = ("first_frame", "coarse_grid", "recent_window",
                        "current_endpoint", "final_frame")
        for role, rows in (("reference", reference_rows), ("execution", execution_rows)):
            if not isinstance(rows, list) or role == "reference" and not rows:
                raise ValueError("Expected a nonempty reference and an execution metadata list")
            marks, previous, previous_grid = {}, None, -1
            endpoint = "final_frame" if role == "reference" else "current_endpoint"
            for ordinal, row in enumerate(rows):
                if not isinstance(row, dict) or set(row) != {"frame_index", "elapsed_s", "selection_reason"}:
                    raise ValueError("Frame metadata has missing or unsupported fields")
                index, elapsed = row["frame_index"], number(row["elapsed_s"])
                reason = row["selection_reason"]
                if (type(index) is not int or index < 0
                        or reason not in ("cadence", endpoint)
                        or reason == endpoint and ordinal != len(rows) - 1
                        or role == "execution" and elapsed > cutoff
                        or previous is not None and (index <= previous[0] or elapsed <= previous[1])):
                    raise ValueError("Frame order, source cutoff or endpoint metadata disagrees")
                previous = index, elapsed
                if reason == "cadence":
                    grid = math.floor(elapsed / interval)
                    # Division can round onto a grid index whose multiplied
                    # target still follows this frame. Check actual boundaries,
                    # without iterating through every target in a source gap.
                    if grid * interval > elapsed:
                        grid -= 1
                    if (grid + 1) * interval <= elapsed:
                        grid += 1
                    if not grid * interval <= elapsed < (grid + 1) * interval:
                        raise ValueError("Source grid is not resolvable at this numeric precision")
                    if grid > previous_grid:
                        marks.setdefault(ordinal, set()).add("coarse_grid")
                    previous_grid = grid
                    if role == "execution" and elapsed >= max(0, cutoff - recent):
                        marks.setdefault(ordinal, set()).add("recent_window")
            if rows:
                if role == "reference":
                    marks.setdefault(0, set()).add("first_frame")
                marks.setdefault(len(rows) - 1, set()).add(endpoint)
            indices = sorted(marks)
            selected[role] = indices
            audit_rows[role] = [{"ordinal": ordinal,
                "frame_index": rows[ordinal]["frame_index"],
                "elapsed_s": rows[ordinal]["elapsed_s"],
                "reasons": [reason for reason in reason_order if reason in marks[ordinal]]}
                for ordinal in indices]
        return {"selected_indices": selected, "audit": {
            "schema_version": 1, "policy": "source_grid_recent_window",
            "source_cutoff_s": cutoff, "coarse_interval_s": interval,
            "recent_window_s": recent, **audit_rows}}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid compact frame metadata or source-time settings") from exc
