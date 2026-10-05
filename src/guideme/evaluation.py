"""Score supplied predictions against private reviewer labels."""

import math

from .annotations import execution_step_at_time


def score_current_steps(records: list[dict]) -> dict:
    """Return current-step accuracy averaged equally across labeled steps.

    Each row pairs reference_id, true_step_id and predicted_step_id at one
    checkpoint. IDs are exact, nonblank strings; step IDs may be None for
    unknown. Unknown truth is excluded; unknown or wrong predictions count
    as misses when truth is known. Extra fields are ignored, inputs unchanged.

    Group by (reference_id, true_step_id), then average group accuracies.
    Return None overall when no rows are evaluable, with explicit evaluated
    and excluded counts and per-step results sorted by reference and step ID.
    Missing fields or invalid types raise ValueError rather than silently
    becoming unknown labels.

    The caller must supply the same checkpoints for every method, including
    unknown predictions, and keep reviewer labels out of inference. This
    function does not load videos, map free text to IDs, validate labels or
    measure deviation warnings, delay, or step completion.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list of dictionaries")

    fields = ("reference_id", "true_step_id", "predicted_step_id")
    groups = {}
    evaluated_count = 0
    excluded_count = 0
    for index, row in enumerate(records):
        if not isinstance(row, dict) or any(field not in row for field in fields):
            raise ValueError(f"record {index}: require {', '.join(fields)}")
        for field in fields:
            value = row[field]
            if value is None and field != "reference_id":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"record {index}: invalid {field}")

        if row["true_step_id"] is None:
            excluded_count += 1
            continue
        key = (row["reference_id"], row["true_step_id"])
        group = groups.setdefault(key, {
            "reference_id": key[0], "step_id": key[1],
            "correct_count": 0, "evaluated_count": 0,
        })
        group["evaluated_count"] += 1
        group["correct_count"] += int(row["predicted_step_id"] == row["true_step_id"])
        evaluated_count += 1

    per_step = [
        {**groups[key], "accuracy": groups[key]["correct_count"] / groups[key]["evaluated_count"]}
        for key in sorted(groups)
    ]
    return {
        "balanced_step_accuracy": (
            sum(group["accuracy"] for group in per_step) / len(per_step)
            if per_step else None
        ),
        "evaluated_count": evaluated_count,
        "excluded_count": excluded_count,
        "per_step": per_step,
    }


def evaluate_current_steps(
    reference_id: str,
    predictions: list[dict],
    execution_rows: list[dict],
) -> dict:
    """Pair checkpoint predictions with private labels and score current steps.

    Each prediction must supply checkpoint_s (nonnegative finite seconds on
    the execution clock) and predicted_step_id (a nonblank string or explicit
    None). Execution rows must already be validated by the execution loader.
    Use execution_step_at_time for truth and score_current_steps for accuracy.
    Return scores and paired checkpoints, preserving prediction order and IDs
    without changing inputs. Extra prediction fields are ignored.

    Reject malformed inputs, unknown checkpoint times and duplicate times
    with ValueError. Unknown predictions remain misses for known truth, and
    unknown truth is excluded. Empty predictions produce null accuracy.
    The caller must supply every shared evaluation checkpoint, use the correct
    paired execution labels and keep them out of inference. This function
    cannot detect omitted checkpoints, verify causal video inputs, read model
    reports or map free-text predictions to IDs. It measures step recognition,
    not warning correctness or delay.
    """
    if not isinstance(reference_id, str) or not reference_id.strip():
        raise ValueError("reference_id must be a nonblank string")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list of dictionaries")

    checkpoints = []
    seen_times = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or any(
            field not in prediction for field in ("checkpoint_s", "predicted_step_id")
        ):
            raise ValueError(f"prediction {index}: require checkpoint_s and predicted_step_id")
        time_s = prediction["checkpoint_s"]
        if time_s is None:
            raise ValueError(f"prediction {index}: checkpoint_s must be known")
        label = execution_step_at_time(execution_rows, time_s)
        if time_s in seen_times:
            raise ValueError(f"prediction {index}: duplicate checkpoint_s")
        seen_times.add(time_s)
        checkpoints.append({
            "reference_id": reference_id,
            "checkpoint_s": time_s,
            "true_step_id": label["reference_step_id"] if label is not None else None,
            "predicted_step_id": prediction["predicted_step_id"],
        })

    return {"scores": score_current_steps(checkpoints), "checkpoints": checkpoints}


def dtw_prediction_from_report(report: dict) -> dict:
    """Extract one checkpoint prediction from a parsed run_dtw report.

    Accept schema-version 1 reports with status ok or error and a positive
    finite checkpoint.end_s. Successful classical/CLIP subsequence-DTW results
    must have a matching checkpoint_s and an explicit candidate_step. Preserve
    the exact candidate step ID; null candidates stay unknown. Recorded errors
    must have a null result and an error stage/type; keep their checkpoint as
    an unknown prediction so failures are not dropped from evaluation.

    Return checkpoint_s and predicted_step_id without changing the report.
    Malformed essential fields raise ValueError. Other metadata is ignored;
    this is not full report-schema, source-identity or causal-input validation.
    The caller must retain original reports, verify the paired inputs and
    supply all shared checkpoints to evaluate_current_steps. No file reading,
    video processing, model call or reference-timestamp fallback occurs here.
    """
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("Expected a schema-version 1 DTW report")
    status = report.get("status")
    if status not in ("ok", "error") or "result" not in report:
        raise ValueError("DTW report must have a final status and explicit result")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("DTW report must have a checkpoint dictionary")
    time_s = checkpoint.get("end_s")
    if (
        isinstance(time_s, bool) or not isinstance(time_s, (int, float))
        or (isinstance(time_s, float) and not math.isfinite(time_s)) or time_s <= 0
    ):
        raise ValueError("checkpoint.end_s must be a finite positive number")

    result = report["result"]
    if status == "error":
        error = report.get("error")
        if result is not None or not isinstance(error, dict) or any(
            not isinstance(error.get(field), str) or not error[field].strip()
            for field in ("stage", "type")
        ):
            raise ValueError("Failed DTW report must have a null result and error stage/type")
        return {"checkpoint_s": time_s, "predicted_step_id": None}

    if not isinstance(result, dict) or result.get("method") not in (
        "classical_subsequence_dtw", "clip_subsequence_dtw",
    ):
        raise ValueError("Successful report must contain a classical/CLIP subsequence-DTW result")
    result_time = result.get("checkpoint_s")
    if (
        isinstance(result_time, bool) or not isinstance(result_time, (int, float))
        or result_time != time_s
    ):
        raise ValueError("Result checkpoint_s must match checkpoint.end_s")
    if "candidate_step" not in result:
        raise ValueError("Successful result must have an explicit candidate_step")
    candidate = result["candidate_step"]
    step_id = None
    if candidate is not None:
        if (
            not isinstance(candidate, dict) or not isinstance(candidate.get("step_id"), str)
            or not candidate["step_id"].strip()
        ):
            raise ValueError("candidate_step must be null or contain a nonblank step_id")
        step_id = candidate["step_id"]
    return {"checkpoint_s": time_s, "predicted_step_id": step_id}
