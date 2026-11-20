"""Bounded notification reservations and correlated assessment acknowledgments."""

from copy import deepcopy
import hashlib
import json
import math
import re
from uuid import uuid4

from .notification_selection import choose_assessment_notification
from .validation import require_keys, require_uuid


class AssessmentNotificationSink:
    """Keep one coordinator's reservations and explicit delivery outcomes.

    No sender, clock callback, persistence, timeout or automatic retry is used.
    The caller publishes reserved envelopes and traces returned receipts. It
    owns receiver truth and clock mapping; simulated ACKs prove no real display.
    Unknown outcomes remain pending and may later receive a matched outcome.
    A valid ACK records delivery even when the frozen input has become stale.
    A browser render report retains intervals and its actual HTTP receipt time,
    separately from processing time; it does not establish physical display.

    Reservation rechecks current state, but cannot prevent changes between
    reservation and actual publication. Transport/receiver integration must
    handle that race and receiver idempotency; this is not physical exactly-once.
    """

    _ENVELOPE_FIELDS = ("assessment_contract_version", "session_id", "reference_id", "execution_source_sha256",
        "clock_id", "timing_basis", "delivery_mode", "notification_id", "attempt_id", "issue_id", "text", "binding",
        "policy_at_s", "reserved_at_s", "max_input_age_s", "tracker_revision")

    def __init__(self, session_id, reference_id, execution_source_sha256, *, clock_id,
                 timing_basis, delivery_mode, max_attempts, max_metadata_bytes,
                 assessment_contract_version=2):
        if type(assessment_contract_version) is not int or assessment_contract_version not in (2, 3):
            raise ValueError("Unsupported assessment contract version")
        self._uuid(session_id)
        for value in (reference_id, execution_source_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Expected lowercase SHA-256 identities")
        if not isinstance(clock_id, str) or not clock_id.strip():
            raise ValueError("A nonblank UTF-8 clock identity is required")
        clock_id.encode("utf-8")
        if timing_basis not in ("ideal_content_cutoff", "session_monotonic"):
            raise ValueError("Unsupported timing basis")
        if delivery_mode not in ("simulated", "display_ack") or delivery_mode == "display_ack" and timing_basis != "session_monotonic":
            raise ValueError("Display acknowledgments require a session-monotonic clock")
        if any(type(value) is not int or value <= 0 for value in (max_attempts, max_metadata_bytes)):
            raise ValueError("Capacity limits must be positive integers")
        self._identity = {"assessment_contract_version": assessment_contract_version, "session_id": session_id,
            "reference_id": reference_id, "execution_source_sha256": execution_source_sha256,
            "clock_id": clock_id, "timing_basis": timing_basis}
        self._mode = delivery_mode
        self._limits = {"max_attempts": max_attempts, "max_metadata_bytes": max_metadata_bytes}
        self._attempts, self._reservations = {}, {}
        self._pending, self._delivered = set(), set()
        self._reserved_bytes, self._last_event = 0, None

    _keys = staticmethod(require_keys)
    _uuid = staticmethod(require_uuid)

    @staticmethod
    def _number(value):
        try:
            valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative IEEE-representable number")
        return value

    @staticmethod
    def _canonical(value):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Notification metadata must be finite UTF-8 JSON") from exc

    def _pinned(self, record):
        if (not isinstance(record, dict) or type(record.get("assessment_contract_version")) is not int
                or any(record.get(name) != value for name, value in self._identity.items())):
            raise ValueError("Notification context belongs to another contract, identity or clock")

    def _event_time(self, at_s):
        now = self._number(at_s)
        if self._last_event is not None and now < self._last_event:
            raise ValueError("Event clock cannot reverse")
        return now

    def reserve(self, proposal, current_tracker_snapshot, *, at_s):
        """Recheck a selected proposal and reserve its full outcome capacity."""
        now = self._event_time(at_s)
        self._keys(proposal, ("assessment_contract_version", "session_id", "reference_id", "execution_source_sha256",
            "clock_id", "timing_basis", "tracker_revision", "at_s", "max_input_age_s", "notification", "decisions"))
        self._pinned(proposal)
        self._pinned(current_tracker_snapshot)
        policy_at, age_limit = self._number(proposal["at_s"]), self._number(proposal["max_input_age_s"])
        if (policy_at > now or type(proposal["tracker_revision"]) is not int or proposal["tracker_revision"] < 0
                or not isinstance(proposal["decisions"], list)):
            raise ValueError("Invalid policy time, revision or decision list")
        current = choose_assessment_notification(current_tracker_snapshot,
            delivered_issue_ids=self._delivered, pending_issue_ids=self._pending,
            at_s=now, clock_id=self._identity["clock_id"], max_input_age_s=age_limit)
        if proposal["tracker_revision"] > current["tracker_revision"]:
            raise ValueError("Policy revision is ahead of current state")
        notification = proposal["notification"]
        if notification is None:
            return None
        self._keys(notification, ("issue_id", "text", "binding"))
        if current["notification"] is None or self._canonical(notification) != self._canonical(current["notification"]):
            raise ValueError("Selected notification is no longer current or eligible")
        issue_id = notification["issue_id"]
        previous = next((row for row in reversed(list(self._attempts.values())) if row["issue_id"] == issue_id), None)
        if previous is not None and previous["status"] == "not_sent" and policy_at < previous["last_outcome_at_s"]:
            raise ValueError("Retry requires a fresh policy proposal after definite failure")
        if len(self._attempts) >= self._limits["max_attempts"]:
            raise RuntimeError("Notification attempt capacity exceeded")
        identity = {name: self._identity[name] for name in (
            "assessment_contract_version", "session_id", "reference_id", "execution_source_sha256")}
        identity["issue_id"] = issue_id
        attempt_id = str(uuid4())
        if attempt_id in self._attempts:
            raise RuntimeError("Notification attempt identity collision")
        envelope = {**self._identity, "delivery_mode": self._mode,
            "notification_id": hashlib.sha256(self._canonical(identity)).hexdigest(), "attempt_id": attempt_id,
            **deepcopy(notification), "policy_at_s": policy_at, "reserved_at_s": now,
            "max_input_age_s": age_limit, "tracker_revision": current["tracker_revision"]}
        record = {**envelope, "status": "pending", "first_unknown_at_s": None, "last_outcome_at_s": None,
            "ack_received_at_s": None, "delivered_at_s": None, "input_age_s_at_delivery": None,
            "input_fresh_at_delivery": None, "error_code": None}
        playback = notification["binding"].get("playback_mapping")
        if playback is not None:
            record.update(render_mapping=None, input_age_bounds_s_at_delivery=None)
        # Every permitted nonnegative number needs at most 309 JSON digits.
        # Strings also reserve two quotes, conservatively covering all five slots.
        worst = dict(record, status="acknowledged", error_code="malformed_outcome", input_fresh_at_delivery=False)
        for name in ("first_unknown_at_s", "last_outcome_at_s", "ack_received_at_s", "delivered_at_s", "input_age_s_at_delivery"):
            worst[name] = "9" * 309
        if playback is not None:
            # Freeze enough room for a full future correlated render report,
            # including its largest permitted numeric bounds, before publication.
            worst["render_mapping"] = {**deepcopy(playback), "kind": "reported_render",
                "status": "mapped", "reason": None, "host_interval_s": ["9" * 309] * 2,
                "received_at_s": "9" * 309, "support": {
                    "notification_id": envelope["notification_id"], "attempt_id": attempt_id,
                    "marker_id": "00000000-0000-0000-0000-000000000000"}}
            worst["input_age_bounds_s_at_delivery"] = ["9" * 309] * 2
        reservation = len(self._canonical(worst))
        if self._reserved_bytes + reservation > self._limits["max_metadata_bytes"]:
            raise RuntimeError("Notification outcome reservation capacity exceeded")
        receipt = deepcopy(envelope)
        pending = self._pending | {issue_id}
        self._attempts[attempt_id], self._reservations[attempt_id] = record, reservation
        self._pending, self._reserved_bytes, self._last_event = pending, self._reserved_bytes + reservation, now
        return receipt

    def complete(self, envelope, outcome, *, at_s):
        """Record a correlated outcome; expiry never erases a valid display ACK."""
        now = self._event_time(at_s)
        self._keys(envelope, self._ENVELOPE_FIELDS)
        self._uuid(envelope["attempt_id"])
        record = self._attempts.get(envelope["attempt_id"])
        if record is None or self._canonical(envelope) != self._canonical({name: record[name] for name in self._ENVELOPE_FIELDS}):
            raise ValueError("Outcome envelope does not match a reserved attempt")
        if record["status"] not in ("pending", "unknown") or now < record["reserved_at_s"]:
            raise ValueError("Attempt is terminal or its outcome precedes reservation")
        status, delivered, error = "unknown", None, None
        render_mapping = bounds = None
        playback = record["binding"].get("playback_mapping")
        try:
            if not isinstance(outcome, dict):
                raise ValueError("Expected a structured outcome")
            status = outcome.get("status")
            if status == "acknowledged":
                if playback is not None:
                    self._keys(outcome, ("status", "clock_id", "render_mapping"))
                    render_mapping = outcome["render_mapping"]
                    self._keys(render_mapping, ("schema_version", "session_id", "execution_source_sha256", "clock_id",
                        "client_epoch_id", "segment_id", "kind", "status", "reason", "host_interval_s",
                        "support", "received_at_s"))
                    if (type(render_mapping["schema_version"]) is not int or render_mapping["schema_version"] != 1
                            or render_mapping["kind"] != "reported_render" or render_mapping["status"] != "mapped"
                            or render_mapping["reason"] is not None or outcome["clock_id"] != self._identity["clock_id"]
                            or any(render_mapping[name] != playback[name] for name in
                                   ("session_id", "execution_source_sha256", "clock_id", "client_epoch_id", "segment_id"))):
                        raise ValueError("Render mapping identity or kind disagrees")
                    support = render_mapping["support"]
                    self._keys(support, ("notification_id", "attempt_id", "marker_id"))
                    self._uuid(support["marker_id"])
                    if any(support[name] != record[name] for name in ("notification_id", "attempt_id")):
                        raise ValueError("Render mapping belongs to another attempt")
                    interval = render_mapping["host_interval_s"]
                    if (not isinstance(interval, list) or len(interval) != 2
                            or not record["reserved_at_s"] <= self._number(interval[0]) <= self._number(interval[1])
                            == self._number(render_mapping["received_at_s"]) <= now):
                        raise ValueError("Render interval chronology disagrees")
                    source_interval = playback["host_interval_s"]
                    bounds = [self._number(interval[0] - source_interval[1]),
                              self._number(interval[1] - source_interval[0])]
                else:
                    self._keys(outcome, ("status", "clock_id", "delivered_at_s"))
                    delivered = self._number(outcome["delivered_at_s"])
                    if outcome["clock_id"] != self._identity["clock_id"] or not record["reserved_at_s"] <= delivered <= now:
                        raise ValueError("Display acknowledgment clock or chronology disagrees")
            elif status in ("unknown", "not_sent"):
                self._keys(outcome, ("status",))
            else:
                raise ValueError("Unsupported outcome status")
        except (KeyError, TypeError, ValueError, OverflowError):
            status, delivered, error = "unknown", None, "malformed_outcome"
            render_mapping = bounds = None
        proposed = deepcopy(record)
        proposed.update(status=status, last_outcome_at_s=now, ack_received_at_s=None, delivered_at_s=None,
            input_age_s_at_delivery=None, input_fresh_at_delivery=None, error_code=error)
        pending, acknowledged = set(self._pending), set(self._delivered)
        if status == "acknowledged":
            if playback is not None:
                limit = record["max_input_age_s"]
                fresh = True if bounds[1] <= limit else False if bounds[0] > limit else None
                proposed.update(render_mapping=deepcopy(render_mapping), input_age_bounds_s_at_delivery=bounds,
                    ack_received_at_s=render_mapping["received_at_s"], input_fresh_at_delivery=fresh)
            else:
                age = self._number(delivered - record["binding"]["latest_execution_captured_at_s"])
                proposed.update(ack_received_at_s=now, delivered_at_s=delivered,
                    input_age_s_at_delivery=age, input_fresh_at_delivery=age <= record["max_input_age_s"])
            pending.remove(record["issue_id"])
            acknowledged.add(record["issue_id"])
        elif status == "not_sent":
            pending.remove(record["issue_id"])
        elif proposed["first_unknown_at_s"] is None:
            proposed["first_unknown_at_s"] = now
        if len(self._canonical(proposed)) > self._reservations[record["attempt_id"]]:
            raise RuntimeError("Outcome exceeded its pre-reserved metadata capacity")
        receipt = deepcopy(proposed)
        self._attempts[record["attempt_id"]] = proposed
        self._pending, self._delivered, self._last_event = pending, acknowledged, now
        return receipt

    def snapshot(self):
        """Copy bounded current records without advancing the event clock."""
        return deepcopy({**self._identity, "delivery_mode": self._mode,
            "pending_issue_ids": sorted(self._pending), "delivered_issue_ids": sorted(self._delivered),
            "attempts": list(self._attempts.values()),
            "metadata_bytes": sum(len(self._canonical(record)) for record in self._attempts.values()),
            "reserved_metadata_bytes": self._reserved_bytes, "limits": self._limits, "last_event_at_s": self._last_event})
