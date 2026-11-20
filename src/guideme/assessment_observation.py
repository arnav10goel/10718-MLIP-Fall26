"""Decode a model's current visual observations without creating issue state."""

import json


def decode_observations(raw_text, *, max_response_bytes):
    """Validate a version-1 observation reply; claims are not verified facts.

    Observation IDs are local to this reply. This function does not bind video,
    associate issues, mutate session state, or infer correctness from silence.
    """
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("Response-byte limit must be a positive integer")
    if not isinstance(raw_text, str):
        raise ValueError("A raw UTF-8 JSON response string is required")
    try:
        size = len(raw_text.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("Response must contain valid UTF-8") from exc
    if size > max_response_bytes:
        raise RuntimeError("Observation response-byte limit exceeded")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("Nonstandard JSON constants are unsupported")

    try:
        result = json.loads(raw_text, object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
        if not isinstance(result, dict) or set(result) != {"schema_version", "status", "observations"}:
            raise ValueError("Observation reply has missing or unsupported fields")
        if type(result["schema_version"]) is not int or result["schema_version"] != 1:
            raise ValueError("Expected observation schema version 1")
        if result["status"] not in ("ok", "insufficient_evidence"):
            raise ValueError("Unsupported observation status")
        observations = result["observations"]
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list")
        if result["status"] == "insufficient_evidence" and observations:
            raise ValueError("Insufficient evidence cannot support observations")
        seen = set()
        for observation in observations:
            if (not isinstance(observation, dict)
                    or not {"id", "description"} <= observation.keys()
                    or not observation.keys() <= {"id", "description", "resolution_condition"}):
                raise ValueError("Observation has missing or unsupported fields")
            identifier = observation["id"]
            if type(identifier) is not int or identifier < 0 or identifier in seen:
                raise ValueError("Observation IDs must be unique nonnegative integers")
            seen.add(identifier)
            for key in ("description", "resolution_condition"):
                if key in observation:
                    value = observation[key]
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("Observation text must be nonblank")
                    value.encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid observation completion") from exc
    return result
