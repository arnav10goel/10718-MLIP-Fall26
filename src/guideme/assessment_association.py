"""Validate two-stage association proposals without changing session state."""

import json
from uuid import UUID


def adapt_association(observations, raw_text, *, open_issue_ids,
                      pending_observations, max_response_bytes):
    """Normalize association decisions and describe pending operations.

    Inputs are detached, model-derived proposals and host-supplied identity
    registries. This function cannot establish visual correctness, frame
    provenance, freshness, budget, or transactional application.
    """
    def keys(value, required, optional=()):
        if not isinstance(value, dict) or not set(required) <= value.keys() or not value.keys() <= set(required) | set(optional):
            raise ValueError("Record has missing or unsupported fields")

    def uuid_string(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except (ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("Expected a canonical UUID")
        return value

    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected nonblank text")
        value.encode("utf-8")
        return value

    def reject_constant(_):
        raise ValueError("Nonstandard JSON constant")

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate JSON object key")
            result[name] = value
        return result

    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("Response-byte limit must be a positive integer")
    if not isinstance(raw_text, str):
        raise ValueError("Response must be a JSON string")
    try:
        if len(raw_text.encode("utf-8")) > max_response_bytes:
            raise RuntimeError("Association response-byte limit exceeded")
        keys(observations, ("schema_version", "status", "observations"))
        if (type(observations["schema_version"]) is not int or observations["schema_version"] != 1
                or observations["status"] not in ("ok", "insufficient_evidence")
                or not isinstance(observations["observations"], list)
                or observations["status"] == "insufficient_evidence" and observations["observations"]):
            raise ValueError("Invalid stage-one observation response")
        current = {}
        for item in observations["observations"]:
            keys(item, ("id", "description"), ("resolution_condition",))
            identifier = item["id"]
            if type(identifier) is not int or identifier < 0 or identifier in current:
                raise ValueError("Current IDs must be unique nonnegative integers")
            text(item["description"])
            if "resolution_condition" in item:
                text(item["resolution_condition"])
            current[identifier] = item
        if type(open_issue_ids) not in (list, tuple, set, frozenset):
            raise ValueError("Open issue IDs must be an explicit collection")
        opened = set()
        for identifier in open_issue_ids:
            uuid_string(identifier)
            if identifier in opened:
                raise ValueError("Duplicate open issue ID")
            opened.add(identifier)
        if not isinstance(pending_observations, dict):
            raise ValueError("Pending observations must be a mapping")
        for identifier, pending in pending_observations.items():
            uuid_string(identifier)
            if identifier in opened or not isinstance(pending, dict):
                raise ValueError("Pending identities and records must be distinct")
            text(pending.get("description"))
            if "resolution_condition" in pending:
                text(pending["resolution_condition"])

        raw = json.loads(raw_text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        keys(raw, ("schema_version", "status", "decisions", "updates"))
        if (type(raw["schema_version"]) is not int or raw["schema_version"] != 1
                or raw["status"] not in ("ok", "insufficient_evidence")
                or not isinstance(raw["decisions"], list) or not isinstance(raw["updates"], list)):
            raise ValueError("Invalid stage-two association response")
        insufficient = raw["status"] == "insufficient_evidence" or observations["status"] == "insufficient_evidence"
        used_current, used_pending, claimed_issues = set(), set(), set()
        issues, updates, pending_ops, clear_sightings = [], [], [], {}
        for decision in raw["decisions"]:
            keys(decision, ("current_id", "pending_id", "match", "issue_id"))
            cid, pid, match, issue_id = (decision[name] for name in ("current_id", "pending_id", "match", "issue_id"))
            if cid is None and pid is None:
                raise ValueError("Decision needs a candidate")
            if cid is not None:
                if type(cid) is not int or cid not in current or cid in used_current:
                    raise ValueError("Unknown or repeated current ID")
                used_current.add(cid)
            if pid is not None:
                if not isinstance(pid, str) or pid not in pending_observations or pid in used_pending:
                    raise ValueError("Unknown or repeated pending ID")
                used_pending.add(pid)
            if match not in ("same", "distinct", "uncertain") or insufficient and match != "uncertain":
                raise ValueError("Invalid association judgment")
            if match == "same":
                if not isinstance(issue_id, str) or issue_id not in opened or issue_id in claimed_issues:
                    raise ValueError("Unknown or conflicting linked issue")
                claimed_issues.add(issue_id)
            elif issue_id is not None:
                raise ValueError("Only same may name an issue")
            if match in ("same", "distinct"):
                if cid is None:
                    raise ValueError("Positive association needs current evidence")
                source = current[cid]
                issue = {"description": source["description"], "link": issue_id if match == "same" else "new"}
                if "resolution_condition" in source:
                    issue["resolution_condition"] = source["resolution_condition"]
                issues.append(issue)
                if pid is not None:
                    pending_ops.append({"action": "clear", "pending_id": pid})
                    clear_sightings[pid] = {"current_id": cid, "issue_index": len(issues) - 1}
            elif cid is not None and pid is not None:
                pending_ops.append({"action": "refresh", "pending_id": pid, "current_id": cid})
            elif cid is not None:
                pending_ops.append({"action": "create", "current_id": cid})
        if used_current != set(current) or used_pending != set(pending_observations):
            raise ValueError("Association must cover all candidates exactly once")
        for update in raw["updates"]:
            keys(update, ("issue_id", "state"))
            identifier, state = update["issue_id"], update["state"]
            if (not isinstance(identifier, str) or identifier not in opened or identifier in claimed_issues
                    or state not in ("resolved", "uncertain") or insufficient and state != "uncertain"):
                raise ValueError("Invalid or conflicting issue update")
            claimed_issues.add(identifier)
            updates.append({"issue_id": identifier, "state": state})
        diagnostics = [{"target": target, "index": index, "status": "absent", "reason": None, "citations": None}
                       for target, rows in (("issues", issues), ("updates", updates))
                       for index in range(len(rows))]
        return {"assessment": {"assessment_contract_version": 2,
                               "status": "insufficient_evidence" if insufficient else "ok",
                               "issues": issues, "updates": updates, "citation_diagnostics": diagnostics},
                "pending_operations": pending_ops, "pending_clear_sightings": clear_sightings}
    except (KeyError, TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid association completion or context") from exc
