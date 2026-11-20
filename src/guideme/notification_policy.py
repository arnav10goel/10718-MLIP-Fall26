"""Pure selection of one fresh, unreported runtime issue."""

from copy import deepcopy
import math
import re
from uuid import UUID


def choose_notification(tracker_snapshot, *, delivered_issue_ids, pending_issue_ids,
                        at_s, max_evidence_age_s, issue_priority=None):
    """Return one proposal and dispositions without reserving or displaying it.

    Consume an IssueTracker snapshot and caller-scoped pending/delivered IDs.
    The caller must reserve a proposal atomically before sending, acknowledge
    actual delivery, and handle ambiguous acknowledgments. Identical inputs may
    propose the same issue repeatedly; this pure function does not guarantee
    physical exactly-once delivery. It changes neither tracker nor delivery history.

    Require support in the latest applied observation and freshness measured
    from execution evidence, not dispatch/cutoff. A newer empty/failed batch
    withholds old alerts without declaring those issues repaired. Fresh candidate
    records are eligible immediately, without a confidence gate or second check.
    The explicit freshness limit is configuration, not a calibrated default.

    Upstream components own actual frame membership, semantic issue identity and
    correctness of the candidate description. This policy selects one runtime ID
    and its exact text; it does not prove that the text describes one true error.
    Clocks and delivery history are caller-declared. No I/O or model call occurs.

    Optional issue_priority maps every known issue ID to a distinct nonnegative
    integer discovery rank. It breaks ties at the same first observation cutoff;
    omitted priority preserves the historical oldest-cutoff/UUID ordering. The
    coordinator owns discovery/candidate order; ranks are not physical onsets.
    """
    def number(value):
        if type(value) not in (int, float):
            raise ValueError("Expected a finite nonnegative number, not a boolean")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ValueError("Expected a finite nonnegative number")
        return value

    def uuid_string(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Expected a canonical UUID string")

    def evidence(record, cutoff):
        if not isinstance(record, dict):
            raise ValueError("Expected evidence metadata")
        for name in ("execution_evidence_s", "reference_evidence_s"):
            values = record.get(name)
            if not isinstance(values, list) or (name == "execution_evidence_s" and not values):
                raise ValueError("Evidence must be lists with nonempty execution evidence")
            for timestamp in values:
                if number(timestamp) > cutoff and name == "execution_evidence_s":
                    raise ValueError("Execution evidence follows its observation cutoff")

    if not isinstance(tracker_snapshot, dict):
        raise ValueError("Expected a tracker snapshot")
    if type(tracker_snapshot.get("schema_version")) is not int or tracker_snapshot["schema_version"] != 1:
        raise ValueError("Expected tracker snapshot schema version 1")
    session_id, reference_id = tracker_snapshot.get("session_id"), tracker_snapshot.get("reference_id")
    uuid_string(session_id)
    if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
        raise ValueError("Expected a lowercase reference SHA-256 identity")
    snapshot_time = number(tracker_snapshot.get("at_s"))
    now = number(at_s)
    limit = number(max_evidence_age_s)
    if now < snapshot_time or limit == 0:
        raise ValueError("Decision time cannot precede snapshot; freshness limit must be positive")
    issues, ledger = tracker_snapshot.get("issues"), tracker_snapshot.get("processed_observations")
    if not isinstance(issues, list) or not isinstance(ledger, list):
        raise ValueError("Snapshot issues and processed observations must be lists")
    observations, previous_cutoff, previous_completion = {}, None, None
    for record in ledger:
        if not isinstance(record, dict):
            raise ValueError("Expected processed-observation metadata")
        identifier = record.get("observation_id")
        uuid_string(identifier)
        cutoff = number(record.get("observation_cutoff_s"))
        dispatch = number(record.get("dispatched_at_s"))
        completion = number(record.get("completed_at_s"))
        if (identifier in observations or cutoff > dispatch or dispatch > completion
                or completion > snapshot_time
                or (previous_cutoff is not None and cutoff <= previous_cutoff)
                or (previous_completion is not None and completion < previous_completion)):
            raise ValueError("Inconsistent processed-observation identity, order or availability")
        observations[identifier] = (cutoff, completion)
        previous_cutoff, previous_completion = cutoff, completion
    latest_cutoff = tracker_snapshot.get("latest_observation_cutoff_s")
    if ledger:
        if number(latest_cutoff) != previous_cutoff:
            raise ValueError("Latest cutoff disagrees with the processed-observation ledger")
        latest_id = ledger[-1]["observation_id"]
    else:
        if latest_cutoff is not None or issues:
            raise ValueError("Issues require a processed observation and latest cutoff")
        latest_id = None

    known = {}
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError("Expected issue metadata")
        identifier = issue.get("issue_id")
        uuid_string(identifier)
        if identifier in known:
            raise ValueError("Issue IDs must be unique")
        state = issue.get("state")
        if state not in ("candidate", "active", "uncertain", "resolved"):
            raise ValueError("Unsupported runtime issue state")
        first_cutoff = number(issue.get("first_observation_cutoff_s"))
        last_cutoff = number(issue.get("last_observation_cutoff_s"))
        first_available = number(issue.get("first_available_at_s"))
        last_available = number(issue.get("last_available_at_s"))
        if (first_cutoff > last_cutoff or last_cutoff > latest_cutoff
                or first_cutoff > first_available or last_cutoff > last_available
                or first_available > last_available or last_available > snapshot_time):
            raise ValueError("Inconsistent issue timing or future availability")
        for prefix, times in (("first", (first_cutoff, first_available)),
                              ("last", (last_cutoff, last_available))):
            observation_id = issue.get(f"{prefix}_observation_id")
            uuid_string(observation_id)
            if observations.get(observation_id) != times:
                raise ValueError("Issue observation binding disagrees with the ledger")
        candidate, update = issue.get("latest_candidate"), issue.get("latest_update")
        evidence(candidate, last_cutoff)
        if not isinstance(candidate.get("description"), str) or not candidate["description"].strip():
            raise ValueError("Candidate description must be nonblank")
        evidence(update, last_cutoff)
        if (update.get("state") != state
                or update.get("observation_id") != issue["last_observation_id"]
                or number(update.get("observation_cutoff_s")) != last_cutoff
                or number(update.get("available_at_s")) != last_available):
            raise ValueError("Latest update disagrees with the issue record")
        if state in ("candidate", "active") and any(
                update[name] != candidate[name] for name in ("execution_evidence_s", "reference_evidence_s")):
            raise ValueError("Current issue support disagrees with its latest update")
        known[identifier] = issue

    if issue_priority is not None:
        if (not isinstance(issue_priority, dict) or set(issue_priority) != set(known)
                or any(type(rank) is not int or rank < 0 for rank in issue_priority.values())
                or len(set(issue_priority.values())) != len(issue_priority)):
            raise ValueError("Issue priority must rank every known issue once with distinct integers")

    def history_ids(values):
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise ValueError("Delivery history must be a collection of issue IDs")
        result = set()
        for identifier in values:
            uuid_string(identifier)
            if identifier not in known:
                raise ValueError("Delivery history refers to an unknown issue")
            result.add(identifier)
        return result

    delivered, pending = history_ids(delivered_issue_ids), history_ids(pending_issue_ids)
    decisions, eligible = [], []
    for identifier in sorted(known):
        issue = known[identifier]
        age = now - max(issue["latest_candidate"]["execution_evidence_s"])
        if issue["state"] == "resolved":
            reason = "resolved"
        elif issue["state"] == "uncertain":
            reason = "uncertain"
        elif identifier in delivered:
            reason = "already_delivered"
        elif identifier in pending:
            reason = "pending_delivery"
        elif (issue["last_observation_cutoff_s"] != latest_cutoff
              or issue["last_observation_id"] != latest_id):
            reason = "not_reconfirmed"
        elif age > limit:
            reason = "stale_evidence"
        else:
            reason = "lower_priority"
        decision = {"issue_id": identifier,
                    "disposition": "deferred" if reason == "lower_priority" else "suppressed",
                    "reason": reason, "evidence_age_s": age}
        decisions.append(decision)
        if reason == "lower_priority":
            eligible.append((issue["first_observation_cutoff_s"], identifier, decision))
    notification = None
    if eligible:
        _, identifier, decision = min(eligible, key=lambda item: (
            item[0], item[1] if issue_priority is None else issue_priority[item[1]]))
        decision.update(disposition="selected", reason="selected")
        issue = known[identifier]
        candidate = issue["latest_candidate"]
        notification = {"issue_id": identifier, "observation_id": issue["last_observation_id"],
                        "text": candidate["description"],
                        "execution_evidence_s": deepcopy(candidate["execution_evidence_s"]),
                        "reference_evidence_s": deepcopy(candidate["reference_evidence_s"])}
    return {"schema_version": 1, "session_id": session_id, "reference_id": reference_id,
            "at_s": at_s, "max_evidence_age_s": max_evidence_age_s,
            "notification": notification, "decisions": decisions}
