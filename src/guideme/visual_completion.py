"""Strict visual-completion decoding without output repair or semantic inference."""

from copy import deepcopy
import json
import math

from .issue_updates import validate_issue_updates
from .mock_predictor import mock_predict


VISUAL_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "candidate_issues", "candidate_links", "issue_updates",
                 "new_issue_candidate_indices"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "insufficient_evidence"]},
        "candidate_issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["description", "execution_evidence_s", "reference_evidence_s"],
            "properties": {
                "description": {"type": "string", "minLength": 1},
                "execution_evidence_s": {"type": "array", "minItems": 1,
                    "items": {"type": "number", "minimum": 0}},
                "reference_evidence_s": {"type": "array",
                    "items": {"type": "number", "minimum": 0}},
                "confidence": {"anyOf": [
                    {"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}]},
                "issue_key": {"anyOf": [
                    {"type": "string", "minLength": 1}, {"type": "null"}]},
            },
        }},
        "candidate_links": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["candidate_index", "issue_id"],
            "properties": {
                "candidate_index": {"type": "integer", "minimum": 0},
                "issue_id": {"anyOf": [{"type": "string", "minLength": 1},
                                         {"type": "null"}]},
            },
        }},
        "issue_updates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["issue_id", "state", "execution_evidence_s", "reference_evidence_s"],
            "properties": {
                "issue_id": {"type": "string", "minLength": 1},
                "state": {"type": "string", "enum": ["resolved", "uncertain"]},
                "execution_evidence_s": {"type": "array", "minItems": 1,
                    "items": {"type": "number", "minimum": 0}},
                "reference_evidence_s": {"type": "array",
                    "items": {"type": "number", "minimum": 0}},
            },
        }},
        "new_issue_candidate_indices": {"type": "array",
            "items": {"type": "integer", "minimum": 0}},
    },
}


def decode_visual_completion(raw_text, *, model_input, prior_issues,
                             dispatched_at_s, max_response_bytes):
    """Bind and validate exactly one model JSON object, returning replay fields.

    The caller must persist the raw provider response before calling this pure
    function. There is no JSON extraction, fence removal, output repair, retry,
    identity minting, inferred association or implicit resolution. Malformed
    values raise ValueError and an exceeded UTF-8 response cap raises RuntimeError.
    The cap bounds response bytes, not all parsing/copying heap memory.

    prior_issues is the causally available model-derived context: each issue has
    its identity/state/last cutoff plus description and execution/reference
    evidence. Old execution evidence may no longer be in a current visual window;
    it must precede that issue's last cutoff. Reference evidence must identify the
    fixed supplied reference. Its identity projection is passed to the existing
    update validator. Input integrity and semantic correctness remain caller and
    model responsibilities, respectively; validation does not establish either.

    Only ok/insufficient_evidence are model statuses. Provider failures are
    host-owned. Schema/session/observation/reference IDs and error=None are bound
    from trusted request context rather than echoed model text.
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
        raise RuntimeError("Visual response-byte limit exceeded")

    def keys(record, fields):
        if not isinstance(record, dict) or set(record) != set(fields):
            raise ValueError("Record has missing or unsupported fields")

    def reject(message):
        raise ValueError(message)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def finite_json(value):
        if isinstance(value, dict):
            for key, child in value.items():
                key.encode("utf-8")
                finite_json(child)
        elif isinstance(value, list):
            for child in value:
                finite_json(child)
        elif isinstance(value, str):
            value.encode("utf-8")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")

    def number(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Evidence and cutoffs must be finite nonnegative numbers")
        return value

    try:
        raw = json.loads(raw_text, object_pairs_hook=unique_object, parse_constant=reject)
        finite_json(raw)
        keys(raw, ("status", "candidate_issues", "candidate_links", "issue_updates",
                   "new_issue_candidate_indices"))
        if raw["status"] not in ("ok", "insufficient_evidence"):
            raise ValueError("Model status must be ok or insufficient_evidence")
        # Validate the current visual input before deriving its frame-time index.
        response = mock_predict(model_input, response={
            "schema_version": 1, "session_id": model_input["session_id"],
            "observation_id": model_input["observation_id"], "status": raw["status"],
            "candidate_issues": raw["candidate_issues"], "error": None,
        })
        keys(prior_issues, ("schema_version", "session_id", "reference_id",
                            "available_at_s", "issues"))
        if not isinstance(prior_issues["issues"], list):
            raise ValueError("Prior issues must be a list")
        prior = {name: prior_issues[name] for name in (
            "schema_version", "session_id", "reference_id", "available_at_s")}
        prior["issues"] = []
        reference_times = {frame["timestamp_s"] for frame in model_input["reference_frames"]}
        for issue in prior_issues["issues"]:
            keys(issue, ("issue_id", "state", "last_observation_cutoff_s", "description",
                         "execution_evidence_s", "reference_evidence_s"))
            description = issue["description"]
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Prior descriptions must be nonblank strings")
            description.encode("utf-8")
            cutoff = number(issue["last_observation_cutoff_s"])
            for name in ("execution_evidence_s", "reference_evidence_s"):
                evidence = issue[name]
                if not isinstance(evidence, list) or (name == "execution_evidence_s" and not evidence):
                    raise ValueError("Prior evidence must be lists with nonempty execution support")
                for timestamp in evidence:
                    number(timestamp)
                    if name == "execution_evidence_s" and timestamp > cutoff:
                        raise ValueError("Prior execution evidence follows its issue cutoff")
                    if name == "reference_evidence_s" and timestamp not in reference_times:
                        raise ValueError("Prior reference evidence must identify a supplied frame")
            prior["issues"].append({name: issue[name] for name in (
                "issue_id", "state", "last_observation_cutoff_s")})
        updates = {"schema_version": 1, "session_id": model_input["session_id"],
            "observation_id": model_input["observation_id"],
            "reference_id": prior["reference_id"], "candidate_links": raw["candidate_links"],
            "issue_updates": raw["issue_updates"]}
        observation = {"schema_version": 1, "status": "ready",
            "session_id": model_input["session_id"], "observation_id": model_input["observation_id"],
            "reference_id": prior["reference_id"], "cutoff_s": model_input["cutoff_s"],
            "model_input": model_input}
        validate_issue_updates(response, updates, observation=observation,
            prior_issues=prior, dispatched_at_s=dispatched_at_s)
        admissions = raw["new_issue_candidate_indices"]
        if not isinstance(admissions, list):
            raise ValueError("New-issue admissions must be an explicit list")
        targets = {link["candidate_index"]: link["issue_id"] for link in updates["candidate_links"]}
        seen = set()
        for index in admissions:
            if (type(index) is not int or index in seen
                    or index not in targets or targets[index] is not None):
                raise ValueError("Admit unique null-linked candidate indices only")
            seen.add(index)
        return {"response": response, "updates": deepcopy(updates),
                "new_issue_candidate_indices": deepcopy(admissions)}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid visual completion or request context") from exc
