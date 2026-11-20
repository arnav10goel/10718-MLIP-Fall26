"""Pure projection of recorded content-replay delivery evidence."""

from copy import deepcopy
import hashlib
import json
import math
import re
from uuid import UUID


def project_content_trace(trace, *, run_id, session_id, reference_id):
    """Project a bounded ``read_run_trace`` result without repairing its evidence.

    Only acknowledged sends become evaluator notifications. Pending and unknown
    sends remain unresolved; proposals and definite send failures are not displays.
    Timings are the runner's ideal content cutoffs, never measured live latency.
    A damaged suffix retains its valid prefix and diagnostic, but cannot establish
    completion. ``run_complete`` reports valid recorded termination, not monitoring
    health, writer acknowledgment, semantic correctness or a scoring horizon.

    The caller owns bounded reading and gold/adjudication. This function performs
    no I/O, infers no gold coverage, and does not combine provider/runtime failures
    into a count: both records may describe the same failure. Unsupported runtime
    modes, conflicting identities and inconsistent send histories raise ValueError.
    Consumed runtime failures require an active run; unrelated driver preamble
    and postlude records are ignored.
    """
    def number(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative time")
        return value

    def uuid_string(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Expected a canonical UUID identity")

    def sha256(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Expected a lowercase SHA-256 identity")

    def identities(record, *, include_run=False):
        expected = {"session_id": session_id, "reference_id": reference_id}
        if include_run:
            expected["run_id"] = run_id
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Trace or envelope belongs to another run, session or reference")

    uuid_string(run_id)
    uuid_string(session_id)
    sha256(reference_id)
    if (not isinstance(trace, dict) or type(trace.get("schema_version")) is not int
            or trace["schema_version"] != 1 or trace.get("status") not in ("valid", "damaged")
            or not isinstance(trace.get("records"), list)):
        raise ValueError("Expected a decoded run-trace result")
    problem = trace.get("problem")
    if ((trace["status"] == "valid" and problem is not None)
            or (trace["status"] == "damaged" and not isinstance(problem, dict))):
        raise ValueError("Trace status and damage diagnostic disagree")
    identities(trace, include_run=True)
    attempts, displayed, failures = {}, [], []
    displayed_ids = set()
    started = False
    contract_version = 1
    assessment_identity, pending_issues, delivered_issues, last_not_sent = {}, set(), set(), {}
    finished = None
    last_at = None
    outcome_fields = {"status", "completed_at_s", "delivered_at_s", "error_code"}
    envelope_fields = {"schema_version", "session_id", "reference_id", "delivery_mode",
        "notification_id", "attempt_id", "issue_id", "observation_id", "text",
        "execution_evidence_s", "reference_evidence_s", "policy_at_s", "reserved_at_s",
        "max_evidence_age_s"}

    def assessment_attempt(payload, at_s):
        # These are recorded host bindings, not localized model evidence. Validate
        # their consistency without reconstructing an observation or repairing it.
        fields = {"assessment_contract_version", "session_id", "reference_id", "execution_source_sha256",
            "clock_id", "timing_basis", "delivery_mode", "notification_id", "attempt_id", "issue_id", "text",
            "binding", "policy_at_s", "reserved_at_s", "max_input_age_s", "tracker_revision"}
        outcomes = {"status", "first_unknown_at_s", "last_outcome_at_s", "ack_received_at_s", "delivered_at_s",
            "input_age_s_at_delivery", "input_fresh_at_delivery", "error_code"}
        def frozen_envelope(record):
            return json.dumps({name: record[name] for name in fields}, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if not isinstance(payload, dict) or set(payload) != fields | outcomes:
            raise ValueError("Assessment attempt has missing or unsupported fields")
        identities(payload)
        if (type(payload["assessment_contract_version"]) is not int or payload["assessment_contract_version"] != contract_version
                or payload["delivery_mode"] != "simulated" or payload["timing_basis"] != "ideal_content_cutoff"):
            raise ValueError("Require matching assessment content/simulated delivery")
        sha256(payload["execution_source_sha256"])
        clock = payload["clock_id"]
        if not isinstance(clock, str) or not clock.strip():
            raise ValueError("A declared content clock is required")
        for name in ("execution_source_sha256", "clock_id"):
            if name in assessment_identity and assessment_identity[name] != payload[name]:
                raise ValueError("Assessment attempt changed execution source or clock")
        for name in ("attempt_id", "issue_id"):
            uuid_string(payload[name])
        identity = {name: payload[name] for name in ("assessment_contract_version", "session_id", "reference_id",
                                                     "execution_source_sha256", "issue_id")}
        logical_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        if payload["notification_id"] != logical_id:
            raise ValueError("Notification identity disagrees with its issue")
        if not isinstance(payload["text"], str) or not payload["text"].strip():
            raise ValueError("Notification text must be nonblank")
        revision = payload["tracker_revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("Expected a nonnegative tracker revision")
        policy, reserved, age_limit = (number(payload[name]) for name in
                                       ("policy_at_s", "reserved_at_s", "max_input_age_s"))
        if not policy <= reserved <= at_s:
            raise ValueError("Policy, reservation and trace chronology disagrees")
        binding = payload["binding"]
        binding_fields = {"assessment_contract_version", "session_id", "observation_id", "reference_id",
            "observation_manifest_sha256", "input_manifest_sha256", "request_sha256", "observation_cutoff_s",
            "latest_execution_frame", "clock_id", "timing_basis", "dispatched_at_s", "completed_at_s",
            "timing_status", "timing_reason", "latest_execution_captured_at_s", "latest_execution_received_at_s"}
        if not isinstance(binding, dict) or set(binding) != binding_fields:
            raise ValueError("Expected a complete assessment binding")
        if any(binding[name] != payload[name] for name in
               ("session_id", "reference_id", "clock_id", "timing_basis")):
            raise ValueError("Binding and notification context disagree")
        if type(binding["assessment_contract_version"]) is not int or binding["assessment_contract_version"] != 2:
            raise ValueError("Expected assessment contract 2")
        uuid_string(binding["observation_id"])
        for name in ("observation_manifest_sha256", "input_manifest_sha256", "request_sha256"):
            sha256(binding[name])
        cutoff = number(binding["observation_cutoff_s"])
        latest = binding["latest_execution_frame"]
        native_fields = {"source_sha256", "frame_index", "source_pts_s", "elapsed_s", "video_pts_s"}
        jpeg_fields = {"media_type", "source_sha256", "frame_index", "source_pts_s", "elapsed_s", "jpeg_sha256", "jpeg_size_bytes"}
        if not isinstance(latest, dict) or set(latest) not in (native_fields, jpeg_fields):
            raise ValueError("Expected the latest supplied native or JPEG frame")
        elapsed = number(latest["elapsed_s"])
        pts = latest["source_pts_s"]
        try:
            valid_pts = type(pts) in (int, float) and math.isfinite(pts)
        except OverflowError:
            valid_pts = False
        if not valid_pts:
            raise ValueError("Expected a finite source timestamp")
        if set(latest) == native_fields:
            video_pts = number(latest["video_pts_s"])
            if pts != elapsed or abs(video_pts - elapsed) > 0.000001:
                raise ValueError("Native frame timestamps disagree")
        else:
            if contract_version != 3 or latest["media_type"] != "image/jpeg":
                raise ValueError("Unsupported JPEG binding")
            sha256(latest["jpeg_sha256"])
            if type(latest["jpeg_size_bytes"]) is not int or latest["jpeg_size_bytes"] <= 0:
                raise ValueError("Expected a positive JPEG size")
        if (latest["source_sha256"] != payload["execution_source_sha256"] or type(latest["frame_index"]) is not int
                or latest["frame_index"] < 0 or elapsed > cutoff):
            raise ValueError("Latest frame disagrees with the execution source or cutoff")
        captured, received, dispatched, completed = (number(binding[name]) for name in
            ("latest_execution_captured_at_s", "latest_execution_received_at_s", "dispatched_at_s", "completed_at_s"))
        # Only this ideal-content mode equates source elapsed time with capture.
        # A future session-clock reader must use its explicit mapping instead.
        if (binding["timing_status"] != "mapped" or binding["timing_reason"] is not None
                or captured != elapsed or cutoff > dispatched
                or not captured <= received <= dispatched <= completed <= policy
                or reserved - captured > age_limit):
            raise ValueError("Binding chronology or reservation freshness disagrees")
        status, first_unknown = payload["status"], payload["first_unknown_at_s"]
        if first_unknown is not None and not reserved <= number(first_unknown) <= at_s:
            raise ValueError("Unknown outcome time disagrees with reservation")
        error = payload["error_code"]
        if error is not None and (status != "unknown" or error != "malformed_outcome"):
            raise ValueError("Unsupported assessment delivery error")
        if status == "pending":
            if any(payload[name] is not None for name in outcomes - {"status"}):
                raise ValueError("Pending sends cannot have an outcome")
        elif status in ("unknown", "not_sent", "acknowledged"):
            if number(payload["last_outcome_at_s"]) != at_s:
                raise ValueError("Outcome receipt must equal its trace time")
            if status == "acknowledged":
                delivered = number(payload["delivered_at_s"])
                age = number(payload["input_age_s_at_delivery"])
                if (number(payload["ack_received_at_s"]) != at_s or not reserved <= delivered <= at_s
                        or age != delivered - captured or type(payload["input_fresh_at_delivery"]) is not bool
                        or payload["input_fresh_at_delivery"] != (age <= age_limit)):
                    raise ValueError("Acknowledgment timing or input freshness disagrees")
            elif any(payload[name] is not None for name in
                     ("ack_received_at_s", "delivered_at_s", "input_age_s_at_delivery", "input_fresh_at_delivery")):
                raise ValueError("Only acknowledgments establish delivery")
            if status == "unknown" and first_unknown is None:
                raise ValueError("Unknown outcome requires its first receipt time")
        else:
            raise ValueError("Unsupported assessment delivery outcome")
        previous = attempts.get(payload["attempt_id"])
        if previous is None:
            if (status != "pending" or reserved != at_s or payload["issue_id"] in pending_issues
                    or payload["issue_id"] in delivered_issues
                    or policy < last_not_sent.get(payload["issue_id"], 0)):
                raise ValueError("Require one initial pending reservation for an unreported issue")
            pending_issues.add(payload["issue_id"])
        else:
            if (previous["status"] not in ("pending", "unknown") or status == "pending"
                    or frozen_envelope(previous) != frozen_envelope(payload)):
                raise ValueError("Delivery history changes an envelope or repeats a terminal outcome")
            expected_unknown = previous["first_unknown_at_s"]
            if status == "unknown" and expected_unknown is None:
                expected_unknown = at_s
            if first_unknown != expected_unknown:
                raise ValueError("First unknown receipt cannot be rewritten")
        attempts[payload["attempt_id"]] = deepcopy(payload)
        assessment_identity.update({name: payload[name] for name in ("execution_source_sha256", "clock_id")})
        if status in ("acknowledged", "not_sent"):
            pending_issues.remove(payload["issue_id"])
        if status == "not_sent":
            last_not_sent[payload["issue_id"]] = at_s
        if status == "acknowledged":
            delivered_issues.add(payload["issue_id"])
            displayed_ids.add(payload["notification_id"])
            displayed.append({"notification_id": payload["notification_id"], "text": payload["text"],
                              "arrival_s": payload["delivered_at_s"]})

    for record in trace["records"]:
        if not isinstance(record, dict):
            raise ValueError("Trace records must be decoded mappings")
        identities(record, include_run=True)
        at_s = number(record.get("at_s"))
        if last_at is not None and at_s < last_at:
            raise ValueError("Trace recording time cannot move backward")
        last_at = at_s
        kind, payload = record.get("event_type"), record.get("payload")
        if started and kind in ("prediction", "issue_state", "policy_decision", "notification_attempt", "run_finished"):
            version = payload.get("assessment_contract_version", 1) if isinstance(payload, dict) else None
            if type(version) is not int or version != contract_version:
                raise ValueError("Trace mixes assessment contracts")
        if kind == "run_started":
            if (started or not isinstance(payload, dict) or payload.get("mode") != "content"
                    or payload.get("delivery_mode") != "simulated"):
                raise ValueError("Require one content/simulated run start")
            contract_version = payload.get("assessment_contract_version", 1)
            if (type(contract_version) is not int or contract_version not in (1, 2, 3)
                    or contract_version in (2, 3) and payload.get("timing_basis") != "ideal_content_cutoff"):
                raise ValueError("Unsupported assessment contract or content timing")
            started = True
        elif kind == "run_finished":
            if not started or finished is not None or not isinstance(payload, dict):
                raise ValueError("Run finish requires one preceding start and no prior finish")
            identities(payload, include_run=True)
            if payload.get("status") not in ("complete", "incomplete"):
                raise ValueError("Unsupported recorded run status")
            finished = deepcopy(payload)
        elif kind in ("failure", "provider_failure"):
            if not started or finished is not None:
                raise ValueError("Runtime failures require an unfinished content run")
            if not isinstance(payload, dict):
                raise ValueError("Failure payloads must be metadata mappings")
            failures.append({"event_type": kind, "at_s": at_s, "payload": deepcopy(payload)})
        elif kind == "notification_attempt":
            if not started or finished is not None:
                raise ValueError("Delivery attempts require an unfinished content run")
            if contract_version in (2, 3):
                assessment_attempt(payload, at_s)
                continue
            if not isinstance(payload, dict) or set(payload) != envelope_fields | outcome_fields:
                raise ValueError("Notification attempt has missing or unsupported fields")
            identities(payload)
            if (type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                    or payload["delivery_mode"] != "simulated"):
                raise ValueError("Notification attempt must use simulated schema version 1")
            for field in ("attempt_id", "issue_id", "observation_id"):
                uuid_string(payload[field])
            sha256(payload["notification_id"])
            if not isinstance(payload["text"], str) or not payload["text"].strip():
                raise ValueError("Notification text must be nonblank")
            if (number(payload["policy_at_s"]) != at_s
                    or number(payload["reserved_at_s"]) != at_s):
                raise ValueError("Content send times must equal the recording cutoff")
            age = number(payload["max_evidence_age_s"])
            if age <= 0:
                raise ValueError("Freshness limit must be positive")
            for field in ("execution_evidence_s", "reference_evidence_s"):
                values = payload[field]
                if not isinstance(values, list) or (field == "execution_evidence_s" and not values):
                    raise ValueError("Require evidence lists and nonempty execution evidence")
                for value in values:
                    if number(value) > at_s and field == "execution_evidence_s":
                        raise ValueError("Execution evidence cannot follow the policy cutoff")
            if at_s - max(payload["execution_evidence_s"]) > age:
                raise ValueError("Notification support is stale at reservation")
            status = payload["status"]
            completed, delivered = payload["completed_at_s"], payload["delivered_at_s"]
            error = payload["error_code"]
            if error is not None and (not isinstance(error, str) or not error.strip()):
                raise ValueError("Error code must be null or nonblank text")
            if status == "pending":
                if any(value is not None for value in (completed, delivered, error)):
                    raise ValueError("Pending sends cannot have an outcome")
            elif status in ("acknowledged", "not_sent", "unknown"):
                if completed is None:
                    if status != "unknown":
                        raise ValueError("Definite outcomes require a completion time")
                elif number(completed) != at_s:
                    raise ValueError("Content completion must equal its recording cutoff")
                if status == "acknowledged":
                    if number(delivered) != at_s or error is not None:
                        raise ValueError("Acknowledgment requires a matching content arrival")
                elif delivered is not None:
                    raise ValueError("Only acknowledged sends have a display time")
                if status == "not_sent" and error is not None:
                    raise ValueError("A definite unsent outcome cannot have an ambiguous error")
            else:
                raise ValueError("Unsupported delivery outcome")
            identifier = payload["attempt_id"]
            previous = attempts.get(identifier)
            if previous is None:
                if status != "pending":
                    raise ValueError("A final outcome requires its pending send record")
            elif (previous["status"] != "pending" or status == "pending"
                    or any(previous[field] != payload[field] for field in envelope_fields)):
                raise ValueError("Send history repeats an outcome or changes its envelope")
            attempts[identifier] = deepcopy(payload)
            if status == "acknowledged":
                notification_id = payload["notification_id"]
                if notification_id in displayed_ids:
                    raise ValueError("Notification identity was acknowledged more than once")
                displayed_ids.add(notification_id)
                displayed.append({"notification_id": notification_id,
                    "text": payload["text"], "arrival_s": delivered})
    if not started:
        raise ValueError("Trace has no supported run start")
    latest = list(attempts.values())
    return {"schema_version": 1, **({"assessment_contract_version": contract_version} if contract_version in (2, 3) else {}),
        "run_id": run_id, "session_id": session_id,
        "reference_id": reference_id, "mode": "content", "delivery_mode": "simulated",
        "timing_basis": "ideal_content_cutoff", "trace_status": trace["status"],
        "trace_problem": deepcopy(problem),
        "run_complete": trace["status"] == "valid" and finished is not None
                        and finished["status"] == "complete",
        "run_finished": finished, "displayed_notifications": displayed,
        "delivery_attempts": latest,
        "unresolved_attempts": [deepcopy(row) for row in latest
                                if row["status"] in ("pending", "unknown")],
        "failures": failures}
