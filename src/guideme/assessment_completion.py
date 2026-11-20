"""Validate assessment core output independently of optional localization."""

import json
import math
from uuid import UUID


def decode_assessment_completion(raw_text, *, known_open_issue_ids,
                                 frame_registry, max_response_bytes):
    """Return contract-2 core claims and separate citation diagnostics.

    The caller supplies open runtime IDs and actual supplied-frame times. Their
    registration is checked here, not visual correctness or causal availability.
    Missing or invalid citations never repair, discard or resolve core claims.
    Invalid citation values remain only in the caller's raw-response trace.

    No host observation binding, clocks, identity minting, admission decision or
    state mutation occurs. This pure decoder does not accept legacy wire output.
    """
    def keys(record, required, optional=()):
        if (not isinstance(record, dict) or not set(required) <= record.keys()
                or not record.keys() <= set(required) | set(optional)):
            raise ValueError("Record has missing or unsupported fields")

    def timestamp(value):
        return (type(value) is int and value >= 0
                or type(value) is float and math.isfinite(value) and value >= 0)

    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("Response-byte limit must be a positive integer")
    if type(known_open_issue_ids) not in (list, tuple, set, frozenset):
        raise ValueError("Open issue IDs must be an explicit collection")
    known = set()
    for identifier in known_open_issue_ids:
        try:
            valid = isinstance(identifier, str) and str(UUID(identifier)) == identifier
        except ValueError:
            valid = False
        if not valid or identifier in known:
            raise ValueError("Open issue IDs must be unique canonical UUID strings")
        known.add(identifier)
    roles = ("execution_s", "reference_s")
    keys(frame_registry, roles)
    registered = {}
    for role in roles:
        times = frame_registry[role]
        if not isinstance(times, list) or not times:
            raise ValueError("Each host frame registry must be a nonempty list")
        previous = None
        for value in times:
            if not timestamp(value) or (previous is not None and value <= previous):
                raise ValueError("Host frame times must be finite, nonnegative and strictly increasing")
            previous = value
        registered[role] = set(times)
    if not isinstance(raw_text, str):
        raise ValueError("A raw UTF-8 JSON response string is required")
    try:
        size = len(raw_text.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("Response must contain valid UTF-8") from exc
    if size > max_response_bytes:
        raise RuntimeError("Assessment response-byte limit exceeded")

    def reject_constant(_):
        raise ValueError("Nonstandard JSON constants are unsupported")

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate JSON object key")
            result[name] = value
        return result

    def citations(claim, target, index):
        diagnostic = {"target": target, "index": index, "status": "absent",
                      "reason": None, "citations": None}
        if "citations" not in claim:
            return diagnostic
        value = claim["citations"]
        if (not isinstance(value, dict) or set(value) != set(roles)
                or any(not isinstance(value[role], list) for role in roles)):
            reason = "invalid_shape"
        elif any(not timestamp(time) for role in roles for time in value[role]):
            reason = "invalid_timestamp"
        elif any(time not in registered[role] for role in roles for time in value[role]):
            reason = "unregistered_timestamp"
        elif any(value[role] for role in roles):
            diagnostic.update(status="valid", citations={role: list(value[role]) for role in roles})
            return diagnostic
        else:
            return diagnostic
        diagnostic.update(status="invalid", reason=reason)
        return diagnostic

    try:
        raw = json.loads(raw_text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        keys(raw, ("status", "issues", "updates"))
        status = raw["status"]
        if status not in ("ok", "insufficient_evidence"):
            raise ValueError("Unsupported assessment status")
        if not isinstance(raw["issues"], list) or not isinstance(raw["updates"], list):
            raise ValueError("Issues and updates must be explicit lists")
        if status == "insufficient_evidence" and raw["issues"]:
            raise ValueError("Insufficient evidence cannot support positive issues")
        issues, updates, diagnostics, claimed_ids = [], [], [], set()
        for index, issue in enumerate(raw["issues"]):
            keys(issue, ("description", "link"), ("citations", "resolution_condition"))
            description, link = issue["description"], issue["link"]
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Issue descriptions must be nonblank strings")
            description.encode("utf-8")
            if "resolution_condition" in issue:
                condition = issue["resolution_condition"]
                if not isinstance(condition, str) or not condition.strip():
                    raise ValueError("Resolution conditions must be nonblank strings")
                condition.encode("utf-8")
            if link is not None:
                if not isinstance(link, str) or (link != "new" and link not in known):
                    raise ValueError("Issue link must be new, a known open issue ID, or null")
                if link != "new":
                    if link in claimed_ids:
                        raise ValueError("Conflicting claims for one open issue")
                    claimed_ids.add(link)
            decoded_issue = {"description": description, "link": link}
            if "resolution_condition" in issue:
                decoded_issue["resolution_condition"] = condition
            issues.append(decoded_issue)
            diagnostics.append(citations(issue, "issues", index))
        for index, update in enumerate(raw["updates"]):
            keys(update, ("issue_id", "state"), ("citations",))
            identifier, state = update["issue_id"], update["state"]
            if not isinstance(identifier, str) or identifier not in known:
                raise ValueError("State updates require a known open issue ID")
            if state not in ("resolved", "uncertain"):
                raise ValueError("Explicit updates must be resolved or uncertain")
            if status == "insufficient_evidence" and state != "uncertain":
                raise ValueError("Insufficient evidence cannot resolve an issue")
            if identifier in claimed_ids:
                raise ValueError("Conflicting claims for one open issue")
            claimed_ids.add(identifier)
            updates.append({"issue_id": identifier, "state": state})
            diagnostics.append(citations(update, "updates", index))
        return {"assessment_contract_version": 2, "status": status,
                "issues": issues, "updates": updates,
                "citation_diagnostics": diagnostics}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid assessment completion") from exc
