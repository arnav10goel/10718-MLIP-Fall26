"""Select one unreported issue using its own observation capture clock."""

from copy import deepcopy
import math

from .frame_validation import validate_assessment_frame
from .frame_binding import validate_host_replay_passage
from .validation import require_keys as keys, require_sha256 as digest, require_uuid as uuid_string


def choose_assessment_notification(tracker_snapshot, *, delivered_issue_ids,
                                   pending_issue_ids, at_s, clock_id, max_input_age_s):
    """Propose one contract-2 or contract-3 notification without displaying it.

    Consume an already validated tracker snapshot and caller-owned delivery
    histories. Check the identities/timing used here, not the complete ledger,
    metadata accounting or physical truth of model claims and clock mappings.
    A newer unrelated or empty assessment neither refreshes nor disqualifies an
    older issue. Browser passage bounds use the oldest possible input age, with
    no fabricated scalar capture point. Optional citations never determine age.

    The caller must reserve/recheck a proposal before publication. Identical
    inputs may select the same issue again; this function changes no state,
    reads no clock and makes no delivery or exactly-once guarantee.
    """
    def number(value):
        try:
            valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative number")
        return value

    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected nonblank UTF-8 text")
        value.encode("utf-8")

    def binding_version(record):
        if type(record["assessment_contract_version"]) is not int or record["assessment_contract_version"] != 2:
            raise ValueError("Expected assessment contract 2")

    try:
        now, limit = number(at_s), number(max_input_age_s)
        snapshot = tracker_snapshot
        contract = snapshot["assessment_contract_version"]
        if type(contract) is not int or contract not in (2, 3):
            raise ValueError("Unsupported assessment contract")
        keys(snapshot, ("assessment_contract_version", "session_id", "reference_id", "execution_source_sha256",
            "clock_id", "timing_basis", "revision", "context_available_at_s", "latest_applied_source_cutoff_s",
            "last_applied_at_s", "issues", "processed_observations", "metadata_bytes", "limits")
            + (("pending_observations", "pending_repairs") if contract == 3 else ()))
        if contract == 3 and (not isinstance(snapshot["pending_observations"], dict)
                              or not isinstance(snapshot["pending_repairs"], dict)):
            raise ValueError("Pending state must be explicit mappings")
        uuid_string(snapshot["session_id"])
        digest(snapshot["reference_id"])
        digest(snapshot["execution_source_sha256"])
        text(clock_id)
        text(snapshot["clock_id"])
        if clock_id != snapshot["clock_id"] or snapshot["timing_basis"] not in ("ideal_content_cutoff", "session_monotonic"):
            raise ValueError("Decision clock or timing basis disagrees")
        if type(snapshot["revision"]) is not int or snapshot["revision"] < 0 or not isinstance(snapshot["issues"], list):
            raise ValueError("Invalid tracker revision or issue list")
        last_applied, context = snapshot["last_applied_at_s"], snapshot["context_available_at_s"]
        latest_cutoff = snapshot["latest_applied_source_cutoff_s"]
        if last_applied is None:
            if latest_cutoff is not None or context is not None or snapshot["issues"] or snapshot["revision"] != 0:
                raise ValueError("Unapplied tracker contains unavailable state")
        else:
            if number(last_applied) > now:
                raise ValueError("Tracker was not available at decision time")
            number(latest_cutoff)
        if context is not None:
            if last_applied is None or number(context) > last_applied:
                raise ValueError("Context availability disagrees with application time")
        if (snapshot["revision"] == 0) != (context is None):
            raise ValueError("Context availability disagrees with tracker revision")
        known, ages, age_bounds = {}, {}, {}
        for issue in snapshot["issues"]:
            keys(issue, ("issue_id", "state", "initial_description", "latest_description", "first_binding", "latest_binding",
                         "first_available_at_s", "latest_available_at_s", "latest_citation_diagnostic")
                 + (("terminal_outcome", "recurrence_of") if contract == 3 else ()))
            identifier = issue["issue_id"]
            uuid_string(identifier)
            if identifier in known or issue["state"] not in ("candidate", "active", "uncertain", "resolved"):
                raise ValueError("Duplicate issue identity or unsupported issue state")
            if contract == 3:
                terminal = issue["terminal_outcome"]
                if ((issue["state"] == "resolved" and terminal not in ("corrected", "retracted"))
                        or (issue["state"] != "resolved" and terminal is not None)):
                    raise ValueError("Issue terminal outcome disagrees with its state")
                if issue["recurrence_of"] is not None:
                    uuid_string(issue["recurrence_of"])
            text(issue["latest_description"])
            first, latest = number(issue["first_available_at_s"]), number(issue["latest_available_at_s"])
            if (last_applied is None or context is None or not first <= latest <= context <= last_applied <= now):
                raise ValueError("Issue was not available in the declared tracker state")
            binding = issue["latest_binding"]
            playback = isinstance(binding, dict) and "playback_mapping" in binding
            fields = ("assessment_contract_version", "session_id", "observation_id", "reference_id",
                "observation_manifest_sha256", "input_manifest_sha256", "request_sha256", "observation_cutoff_s",
                "latest_execution_frame", "clock_id", "timing_basis", "dispatched_at_s", "completed_at_s",
                "timing_status", "timing_reason", "latest_execution_captured_at_s", "latest_execution_received_at_s")
            keys(binding, fields + (("playback_mapping",) if playback else ()))
            binding_version(binding)
            uuid_string(binding["observation_id"])
            if any(binding[name] != snapshot[name] for name in ("session_id", "reference_id", "clock_id", "timing_basis")):
                raise ValueError("Issue binding belongs to another session, reference or clock")
            for name in ("observation_manifest_sha256", "input_manifest_sha256", "request_sha256"):
                digest(binding[name])
            cutoff = number(binding["observation_cutoff_s"])
            if cutoff > latest_cutoff:
                raise ValueError("Issue binding follows the applied source cutoff")
            elapsed = validate_assessment_frame(binding["latest_execution_frame"],
                snapshot["execution_source_sha256"], cutoff)
            dispatch, completion = number(binding["dispatched_at_s"]), number(binding["completed_at_s"])
            if not dispatch <= completion <= latest:
                raise ValueError("Issue response was unavailable at application time")
            captured, received = binding["latest_execution_captured_at_s"], binding["latest_execution_received_at_s"]
            age = None
            if playback:
                mapping = binding["playback_mapping"]
                keys(mapping, ("schema_version", "session_id", "execution_source_sha256", "clock_id",
                    "client_epoch_id", "segment_id", "kind", "status", "reason", "host_interval_s",
                    "support", "received_at_s"))
                if (type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
                        or snapshot["timing_basis"] != "session_monotonic" or mapping["kind"] not in ("replay_passage", "host_replay_passage")
                        or any(mapping[name] != snapshot[name] for name in
                               ("session_id", "execution_source_sha256", "clock_id"))
                        or captured is not None or received is not None):
                    raise ValueError("Playback mapping identity or scalar timing disagrees")
                uuid_string(mapping["client_epoch_id"]); uuid_string(mapping["segment_id"])
                if mapping["kind"] == "host_replay_passage":
                    mapped = validate_host_replay_passage(mapping, elapsed, dispatch)
                    if ((mapped and (binding["timing_status"] != "bounded" or binding["timing_reason"] is not None))
                            or (not mapped and (binding["timing_status"] != "unavailable" or binding["timing_reason"] != "missing_clock_mapping"))):
                        raise ValueError("Host replay timing status disagrees")
                    interval = mapping["host_interval_s"]
                    age_bounds[identifier] = ([number(now - interval[1]), number(now - interval[0])] if mapped else None)
                    known[identifier], ages[identifier] = issue, age
                    continue
                support = mapping["support"]
                keys(support, ("source_elapsed_s", "sample_ids"))
                ids = support["sample_ids"]
                if number(support["source_elapsed_s"]) != elapsed or not isinstance(ids, list):
                    raise ValueError("Playback support identifies another native tail")
                for value in ids:
                    uuid_string(value)
                bounds = None
                if mapping["status"] == "mapped":
                    interval = mapping["host_interval_s"]
                    if (not isinstance(interval, list) or len(interval) != 2 or len(ids) != 2 or ids[0] == ids[1]
                            or mapping["reason"] is not None or binding["timing_status"] != "bounded"
                            or binding["timing_reason"] is not None
                            or not number(interval[0]) <= number(interval[1]) == number(mapping["received_at_s"]) <= dispatch):
                        raise ValueError("Playback bounds or availability disagree")
                    bounds = [number(now - interval[1]), number(now - interval[0])]
                elif (mapping["status"] != "unavailable" or mapping["reason"] not in
                      ("missing_lower_sample", "missing_upper_sample") or ids
                      or mapping["host_interval_s"] is not None or mapping["received_at_s"] is not None
                      or binding["timing_status"] != "unavailable" or binding["timing_reason"] != "missing_clock_mapping"):
                    raise ValueError("Unavailable playback mapping disagrees")
                age_bounds[identifier] = bounds
            elif binding["timing_status"] == "mapped":
                if binding["timing_reason"] is not None or not number(captured) <= number(received) <= dispatch:
                    raise ValueError("Mapped capture, receipt and dispatch chronology disagrees")
                age = now - captured
                if age < 0:
                    raise ValueError("Input capture follows decision time")
            elif (binding["timing_status"] != "unavailable" or binding["timing_reason"] != "missing_clock_mapping"
                    or captured is not None or received is not None):
                raise ValueError("Unavailable timing must not invent capture or receipt")
            known[identifier], ages[identifier] = issue, age

        def history(values):
            if type(values) not in (list, tuple, set, frozenset):
                raise ValueError("Delivery history must be an explicit collection")
            result = set()
            for identifier in values:
                uuid_string(identifier)
                if identifier not in known or identifier in result:
                    raise ValueError("Delivery history contains an unknown or duplicate issue")
                result.add(identifier)
            return result

        repairs = set()
        if contract == 3:
            for identifier in snapshot["pending_observations"]:
                uuid_string(identifier)
                if identifier in known:
                    raise ValueError("Pending observation collides with an issue")
            for identifier in snapshot["pending_repairs"]:
                uuid_string(identifier)
                if identifier not in known or known[identifier]["state"] not in ("candidate", "active"):
                    raise ValueError("Pending repair names an unavailable issue")
                repairs.add(identifier)
        delivered, pending = history(delivered_issue_ids), history(pending_issue_ids)
        if delivered & pending:
            raise ValueError("Delivery histories cannot overlap")
        decisions, eligible = [], []
        for identifier, issue in known.items():
            age = ages[identifier]
            if issue["state"] == "resolved":
                reason = "resolved"
            elif issue["state"] == "uncertain":
                reason = "uncertain"
            elif identifier in delivered:
                reason = "already_delivered"
            elif identifier in pending:
                reason = "pending_delivery"
            elif identifier in repairs:
                reason = "pending_repair"
            elif age is None and age_bounds.get(identifier) is None:
                reason = "timing_unavailable"
            elif (age_bounds[identifier][1] if identifier in age_bounds else age) > limit:
                reason = "stale_input"
            else:
                reason = "lower_priority"
            decision = {"issue_id": identifier, "disposition": "deferred" if reason == "lower_priority" else "suppressed",
                        "reason": reason, "input_age_s": age}
            if identifier in age_bounds:
                decision["input_age_bounds_s"] = deepcopy(age_bounds[identifier])
            decisions.append(decision)
            if reason == "lower_priority":
                eligible.append((issue, decision))
        notification = None
        if eligible:
            issue, decision = eligible[0]
            decision.update(disposition="selected", reason="selected")
            notification = {"issue_id": issue["issue_id"], "text": issue["latest_description"],
                            "binding": deepcopy(issue["latest_binding"])}
        return {"assessment_contract_version": contract, "session_id": snapshot["session_id"],
                "reference_id": snapshot["reference_id"], "execution_source_sha256": snapshot["execution_source_sha256"],
                "clock_id": clock_id, "timing_basis": snapshot["timing_basis"], "tracker_revision": snapshot["revision"],
                "at_s": at_s, "max_input_age_s": max_input_age_s, "notification": notification, "decisions": decisions}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid assessment notification metadata") from exc
