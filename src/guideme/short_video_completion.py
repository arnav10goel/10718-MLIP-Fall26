"""Translate compact video replies without inventing evidence or issue state."""

import json

from .visual_completion import decode_visual_completion


SHORT_VIDEO_SYSTEM_INSTRUCTION = (
    "The reference video shows the correct procedure. The execution should follow "
    "the same steps in the same order. Compare the execution so far with the "
    "reference. Briefly describe any visible mistakes. If you cannot tell, say so. "
    "Compare object relationships, allowing for viewpoint changes and temporary hand occlusion."
)

SHORT_VIDEO_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "issues", "updates"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "insufficient_evidence"],
            "description": "Use insufficient_evidence with no issues when the view cannot support an assessment."},
        "issues": {"type": "array",
            "description": "One currently relevant visible discrepancy per item, without progress reports or coaching.",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["description", "link", "execution_evidence_s", "reference_evidence_s"],
                "properties": {
                    "description": {"type": "string", "minLength": 1},
                    "link": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
                        "description": "Use new only for a clearly new issue, an open prior issue ID for a continuing issue, or null if association is unknown."},
                    "execution_evidence_s": {"type": "array", "minItems": 1,
                        "items": {"type": "number", "minimum": 0},
                        "description": "Exact supplied EVIDENCE_ANCHORS execution times supporting this issue, not its onset."},
                    "reference_evidence_s": {"type": "array",
                        "items": {"type": "number", "minimum": 0},
                        "description": "Exact supplied EVIDENCE_ANCHORS reference times; may be empty."},
                },
            }},
        "updates": {"type": "array",
            "description": "Explicit changes to known open issues. Resolve only after visible correction; an empty issue list or video ending never resolves an issue.",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["issue_id", "state", "execution_evidence_s", "reference_evidence_s"],
                "properties": {
                    "issue_id": {"type": "string", "minLength": 1},
                    "state": {"type": "string", "enum": ["resolved", "uncertain"]},
                    "execution_evidence_s": {"type": "array", "minItems": 1,
                        "items": {"type": "number", "minimum": 0},
                        "description": "Exact supplied EVIDENCE_ANCHORS execution times supporting this update."},
                    "reference_evidence_s": {"type": "array",
                        "items": {"type": "number", "minimum": 0},
                        "description": "Exact supplied EVIDENCE_ANCHORS reference times; may be empty."},
                },
            }},
    },
}


def decode_short_video_completion(raw_text, *, model_input, prior_issues,
                                  dispatched_at_s, max_response_bytes):
    """Expand explicit compact fields into the existing validated replay result.

    The byte cap applies to received short UTF-8 text before parsing. Host-added
    field names and link arrays can enlarge its deterministic serialization;
    that internal envelope is validated using its exact serialized byte length.
    No timestamp rounding, identity minting, output repair, inferred association
    or implicit resolution occurs. Evidence/context/state validation belongs to
    decode_visual_completion, and does not establish semantic correctness.
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
        raise RuntimeError("Short-video response-byte limit exceeded")

    def reject(message):
        raise ValueError(message)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def keys(record, fields):
        if not isinstance(record, dict) or set(record) != set(fields):
            raise ValueError("Record has missing or unsupported fields")

    try:
        raw = json.loads(raw_text, object_pairs_hook=unique_object, parse_constant=reject)
        keys(raw, ("status", "issues", "updates"))
        if not isinstance(raw["issues"], list) or not isinstance(raw["updates"], list):
            raise ValueError("Issues and updates must be explicit lists")
        candidates, links, admissions = [], [], []
        for index, issue in enumerate(raw["issues"]):
            keys(issue, ("description", "link", "execution_evidence_s", "reference_evidence_s"))
            link = issue["link"]
            if link is not None and (not isinstance(link, str) or not link.strip()):
                raise ValueError("Issue link must be new, an open issue ID, or null")
            if link == "new":
                admissions.append(index)
            links.append({"candidate_index": index, "issue_id": None if link == "new" else link})
            candidates.append({key: issue[key] for key in (
                "description", "execution_evidence_s", "reference_evidence_s")})
        expanded = json.dumps({"status": raw["status"], "candidate_issues": candidates,
            "candidate_links": links, "issue_updates": raw["updates"],
            "new_issue_candidate_indices": admissions}, ensure_ascii=False, allow_nan=False)
        return decode_visual_completion(expanded, model_input=model_input, prior_issues=prior_issues,
            dispatched_at_s=dispatched_at_s, max_response_bytes=len(expanded.encode("utf-8")))
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid short-video completion or request context") from exc
