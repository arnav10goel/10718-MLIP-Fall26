"""Bounded notification reservations and explicit delivery acknowledgments."""

from copy import deepcopy
import hashlib
import json
import math
import re
from uuid import UUID, uuid4


class NotificationSink:
    """Reserve an episode before invoking an injected sender and session clock.

    Use one serialized session coordinator. reserve/complete separate publication
    from its outcome; deliver retains the synchronous sender/clock adapter.
    Reservation also blocks duplicate callback reentry.
    Acknowledgments in simulated mode establish simulated delivery only. Display
    acknowledgments require an actual receiver confirmation and mapped timestamp;
    server acceptance or file writes alone do not establish display delivery.

    No transport, UI, timeout, persistence, restart recovery or automatic retry is
    implemented here. Unknown delivery stays pending indefinitely. The runner must
    recheck current issue state before sending; this sink rechecks payload age but
    cannot detect a repair that occurred after the policy snapshot. Physical
    exactly-once behavior additionally requires receiver-side idempotency.
    """

    def __init__(self, session_id, reference_id, *, sender, clock, delivery_mode,
                 max_attempts, max_metadata_bytes):
        self._uuid(session_id)
        if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
            raise ValueError("Reference ID must be a lowercase SHA-256 identity")
        if not callable(sender) or not callable(clock):
            raise ValueError("Sender and session clock must be callable")
        if delivery_mode not in ("simulated", "display_ack"):
            raise ValueError("Delivery mode must be simulated or display_ack")
        for value in (max_attempts, max_metadata_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Attempt and reserved metadata limits must be positive integers")
        self._session_id = session_id
        self._reference_id = reference_id
        self._sender, self._clock = sender, clock
        self._delivery_mode = delivery_mode
        self._limits = {"max_attempts": max_attempts, "max_metadata_bytes": max_metadata_bytes}
        self._attempts = []
        self._open_attempts = {}
        self._pending, self._delivered = set(), set()
        self._reserved_bytes = 0
        self._last_clock_s = None

    @staticmethod
    def _uuid(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Expected a canonical UUID string")

    @staticmethod
    def _number(value):
        if type(value) not in (int, float):
            raise ValueError("Expected a finite nonnegative number, not a boolean")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ValueError("Expected a finite nonnegative number")
        return value

    @staticmethod
    def _keys(record, fields):
        if not isinstance(record, dict) or set(record) != set(fields):
            raise ValueError("Record has missing or unsupported fields")

    @staticmethod
    def _canonical(value):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Notification metadata must be finite UTF-8 JSON") from exc

    def _read_clock(self):
        try:
            value = self._number(self._clock())
        except Exception as exc:
            raise ValueError("Session clock failed or returned an invalid value") from exc
        if self._last_clock_s is not None and value < self._last_clock_s:
            raise ValueError("Session clock cannot move backward")
        return value

    def reserve(self, policy_decision):
        """Reserve one issue and return its detached envelope without sending.

        Invalid/duplicate/stale payloads raise ValueError and capacity failures
        raise RuntimeError with no retained state changes. Publication is not an
        acknowledgment: the reservation remains pending until complete is called.
        The caller owns publication and must recheck current issue state.
        """
        self._keys(policy_decision, ("schema_version", "session_id", "reference_id", "at_s",
                                    "max_evidence_age_s", "notification", "decisions"))
        if type(policy_decision["schema_version"]) is not int or policy_decision["schema_version"] != 1:
            raise ValueError("Expected policy schema version 1")
        if (policy_decision["session_id"] != self._session_id
                or policy_decision["reference_id"] != self._reference_id):
            raise ValueError("Policy decision belongs to another session or reference")
        policy_time = self._number(policy_decision["at_s"])
        age_limit = self._number(policy_decision["max_evidence_age_s"])
        if age_limit == 0:
            raise ValueError("Freshness limit must be positive")
        decisions = policy_decision["decisions"]
        if not isinstance(decisions, list) or any(not isinstance(row, dict) for row in decisions):
            raise ValueError("Policy decisions must be metadata records")
        selected = [row for row in decisions if row.get("disposition") == "selected"]
        notification = policy_decision["notification"]
        if notification is None:
            if selected:
                raise ValueError("Selected policy issue has no notification")
            return None
        self._keys(notification, ("issue_id", "observation_id", "text",
                                  "execution_evidence_s", "reference_evidence_s"))
        issue_id = notification["issue_id"]
        self._uuid(issue_id)
        self._uuid(notification["observation_id"])
        if (len(selected) != 1 or selected[0].get("issue_id") != issue_id
                or selected[0].get("reason") != "selected"):
            raise ValueError("Notification must identify the single selected policy issue")
        if not isinstance(notification["text"], str) or not notification["text"].strip():
            raise ValueError("Notification text must be nonblank")
        for name in ("execution_evidence_s", "reference_evidence_s"):
            evidence = notification[name]
            if not isinstance(evidence, list) or (name == "execution_evidence_s" and not evidence):
                raise ValueError("Evidence must be lists with nonempty execution support")
            for timestamp in evidence:
                if self._number(timestamp) > policy_time and name == "execution_evidence_s":
                    raise ValueError("Execution evidence follows the policy decision")
        if issue_id in self._pending or issue_id in self._delivered:
            raise ValueError("Issue delivery is already pending or acknowledged")
        previous = next((row for row in reversed(self._attempts) if row["issue_id"] == issue_id), None)
        if previous is not None and policy_time < previous["completed_at_s"]:
            raise ValueError("Retry requires a policy decision at or after the known send failure")
        if len(self._attempts) >= self._limits["max_attempts"]:
            raise RuntimeError("Notification attempt capacity exceeded")
        reserved_at = self._read_clock()
        if reserved_at < policy_time:
            raise ValueError("Sending cannot precede its policy decision")
        if reserved_at - max(notification["execution_evidence_s"]) > age_limit:
            raise ValueError("Notification support became stale before sending")
        identity = {"schema_version": 1, "session_id": self._session_id,
                    "reference_id": self._reference_id, "issue_id": issue_id}
        envelope = {"schema_version": 1, "session_id": self._session_id,
                    "reference_id": self._reference_id, "delivery_mode": self._delivery_mode,
                    "notification_id": hashlib.sha256(self._canonical(identity)).hexdigest(),
                    "attempt_id": str(uuid4()), **deepcopy(notification),
                    "policy_at_s": policy_time, "reserved_at_s": reserved_at,
                    "max_evidence_age_s": age_limit}
        record = {**envelope, "status": "pending", "completed_at_s": None,
                  "delivered_at_s": None, "error_code": None}
        # Every accepted ordinary finite nonnegative int/float serializes in at
        # most 309 numeric characters. These slots and longest status/error code
        # reserve room for any allowed result before irreversible sender effects.
        worst_case = dict(record, status="acknowledged", completed_at_s=10**308,
                          delivered_at_s=10**308, error_code="malformed_outcome")
        reservation = len(self._canonical(worst_case))
        if self._reserved_bytes + reservation > self._limits["max_metadata_bytes"]:
            raise RuntimeError("Notification outcome reservation capacity exceeded")
        self._attempts.append(record)
        self._open_attempts[envelope["attempt_id"]] = record
        self._pending.add(issue_id)
        self._reserved_bytes += reservation
        self._last_clock_s = reserved_at
        return deepcopy(envelope)

    def _claim(self, envelope):
        if not isinstance(envelope, dict):
            raise ValueError("Completion requires the reserved notification envelope")
        identifier = envelope.get("attempt_id")
        self._uuid(identifier)
        record = self._open_attempts.get(identifier)
        if record is None:
            raise ValueError("Notification attempt is unknown or already completed")
        expected = {key: value for key, value in record.items()
                    if key not in {"status", "completed_at_s", "delivered_at_s", "error_code"}}
        if self._canonical(envelope) != self._canonical(expected):
            raise ValueError("Completion envelope differs from its reserved notification")
        del self._open_attempts[identifier]
        return record

    def complete(self, envelope, outcome):
        """Finish one matching reservation; only a valid fresh ACK is delivered.

        Completion consumes the attempt even when its outcome is unknown or
        malformed. An unknown issue stays pending and cannot be retried or
        reopened by a later acknowledgment through this API. Foreign, modified
        and repeated completion envelopes are rejected before clock/state changes.
        """
        return self._finish(self._claim(envelope), outcome)

    def _finish(self, record, outcome, error_code=None):
        issue_id, reserved_at = record["issue_id"], record["reserved_at_s"]
        try:
            try:
                completed_at = self._read_clock()
            except Exception:
                record.update(status="unknown", error_code="clock_error")
                return deepcopy(record)
            self._last_clock_s = completed_at
            record["completed_at_s"] = completed_at
            status, delivered_at = "unknown", None
            if error_code is None:
                try:
                    if not isinstance(outcome, dict):
                        raise ValueError("Expected a structured sender outcome")
                    status = outcome.get("status")
                    if status == "acknowledged":
                        self._keys(outcome, ("status", "delivered_at_s"))
                        delivered_at = self._number(outcome["delivered_at_s"])
                        if not reserved_at <= delivered_at <= completed_at:
                            raise ValueError("Acknowledgment timestamp is outside the send interval")
                        if (delivered_at - max(record["execution_evidence_s"])
                                > record["max_evidence_age_s"]):
                            raise ValueError("Notification support became stale before display")
                    elif status in ("not_sent", "unknown"):
                        self._keys(outcome, ("status",))
                    else:
                        raise ValueError("Unsupported sender outcome")
                except Exception:
                    status, delivered_at, error_code = "unknown", None, "malformed_outcome"
            record.update(status=status, delivered_at_s=delivered_at, error_code=error_code)
            if status == "acknowledged":
                self._pending.remove(issue_id)
                self._delivered.add(issue_id)
            elif status == "not_sent":
                self._pending.remove(issue_id)
            return deepcopy(record)
        except BaseException:
            # Preserve the ambiguous reservation when a caller interrupts a
            # potentially completed send. Never convert it into a retryable fail.
            record.update(status="unknown", completed_at_s=None,
                          delivered_at_s=None, error_code="interrupted")
            self._pending.add(issue_id)
            self._delivered.discard(issue_id)
            raise

    def deliver(self, policy_decision):
        """Synchronous compatibility wrapper around reserve, sender and completion.

        Sender exceptions/malformed results become unknown receipts because
        delivery may have happened. BaseException interruptions keep an unknown
        pending reservation and propagate. No automatic retry is performed.
        """
        envelope = self.reserve(policy_decision)
        if envelope is None:
            return None
        try:
            try:
                outcome = self._sender(deepcopy(envelope))
                error_code = None
            except Exception:
                outcome, error_code = None, "sender_exception"
            return self._finish(self._claim(envelope), outcome, error_code)
        except BaseException:
            record = self._open_attempts.pop(envelope["attempt_id"], None)
            if record is not None:
                record.update(status="unknown", completed_at_s=None,
                              delivered_at_s=None, error_code="interrupted")
            raise

    def snapshot(self):
        """Copy current records/history without advancing or reading the clock.

        Reserved JSON bytes, not actual Python memory, are charged to capacity;
        reservations are retained for all attempts, including definite failures.
        Raw exception text and arbitrary sender-returned details are not stored.
        This in-memory history cannot survive a process restart.
        """
        return {"schema_version": 1, "session_id": self._session_id,
                "reference_id": self._reference_id, "delivery_mode": self._delivery_mode,
                "pending_issue_ids": sorted(self._pending),
                "delivered_issue_ids": sorted(self._delivered),
                "attempts": deepcopy(self._attempts),
                "metadata_bytes": sum(len(self._canonical(record)) for record in self._attempts),
                "reserved_metadata_bytes": self._reserved_bytes, "limits": dict(self._limits)}
