"""Browser playback pacing and bounded, receiver-acknowledged notifications."""

from copy import deepcopy
import hashlib
import json
import math
import re
from threading import Condition
import time
from uuid import UUID, uuid4

from .playback_clock import PlaybackClockMap


class MonitoringSession:
    """Coordinate one runner worker with local browser HTTP handlers.

    The runner owns model and issue state. This class exposes a bounded
    read-only projection of open tracked issues, the newest playback report and one
    notification mailbox; it never calls a model or fabricates acknowledgments.
    """

    def __init__(self, *, stop_at_s=None, ack_timeout_s=5,
                 heartbeat_timeout_s=10, clock=time.monotonic, assessment_contract_version=1,
                 replay_clock_mode="browser"):
        if type(assessment_contract_version) is not int or assessment_contract_version not in (1, 2, 3):
            raise ValueError("Assessment contract must be 1, 2 or 3")
        if replay_clock_mode not in ("browser", "host") or replay_clock_mode == "host" and assessment_contract_version != 3:
            raise ValueError("Host replay requires assessment contract 3")
        if stop_at_s is not None:
            self._number(stop_at_s, positive=True)
        self._number(ack_timeout_s, positive=True)
        self._number(heartbeat_timeout_s, positive=True)
        if not callable(clock):
            raise ValueError("A monotonic clock is required")
        self.session_id = str(uuid4())
        self._clock = clock
        self._started = self._number(clock())
        self._last_clock = self._started
        self._condition = Condition()
        self._stop_at_s = stop_at_s
        self._ack_timeout_s = ack_timeout_s
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._replay_clock_mode = replay_clock_mode
        self._host_duration_s = None
        self._host_segments = []
        self._source_time_s = 0
        self._sequence = -1
        self._playing = self._ended = self._ready = False
        self._active = True
        self._status = "preparing"
        self._error_code = None
        self._last_heartbeat = None
        self._notification = None
        self._ack = None
        self._binding = None
        self._delivery = None
        self._delivery_observed = False
        self._ack_received_at = None
        self._notification_deadline = None
        self._dismissed = self._retired = self._legacy_sender = False
        self._assessment_version = assessment_contract_version
        self._served_viewer_sha256 = None
        self._v2_config = self._v2_mapper = self._v2_sample_marker = None
        self._v2_claim = None
        self._v2_sample_digests, self._v2_attempts = {}, {}
        self._v2_reserved_bytes = 0
        self._v2_closed = None
        self._tracked_issues = []
        self._tracked_revision = -1

    @staticmethod
    def _number(value, *, positive=False):
        try:
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and value >= 0 and (not positive or value > 0))
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative number")
        return value

    @staticmethod
    def _keys(value, expected):
        if type(value) is not dict or set(value) != set(expected):
            raise ValueError("Missing or unsupported session fields")

    def _now(self):
        now = self._number(self._clock())
        if now < self._last_clock:
            raise ValueError("Session clock cannot move backward")
        self._last_clock = now
        return now

    def _invalidate(self, code=None, *, status="failed"):
        self._active = self._playing = False
        self._status, self._error_code = status, code
        self._notification = None
        self._retired = True
        if self._binding is not None and self._delivery is None:
            self._delivery = {"status": "unknown"}
        if self._assessment_version in (2, 3):
            self._v2_unknown_all(self._now() - self._started)
        self._condition.notify_all()

    def _expire(self):
        now = self._now()
        if self._replay_clock_mode == "host" and self._active and self._playing:
            segment = self._host_segments[-1]
            source = min(self._host_duration_s, segment["source"] + now - segment["absolute_host"])
            self._source_time_s = source
            if source >= self._host_duration_s:
                segment["end_source"] = source
                self._playing, self._ended, self._status = False, True, "ended"
                self._condition.notify_all()
        if (self._active and self._playing and self._last_heartbeat is not None
                and self._replay_clock_mode != "host"
                and now - self._last_heartbeat > self._heartbeat_timeout_s):
            self._invalidate("playback_disconnected")
        return now

    def source_time(self):
        with self._condition:
            if self._replay_clock_mode == "host":
                self._expire()
            return self._source_time_s

    def elapsed_s(self):
        with self._condition:
            return self._now() - self._started

    def active(self):
        with self._condition:
            self._expire()
            return self._active

    def run_if_active(self, callback):
        """Run a short local action atomically with respect to external Stop.

        The callback must not wait for transport or another thread. The
        condition is reentrant; a callback that itself stops this session must
        recheck activity before any subsequent state-changing action.
        """
        if not callable(callback):
            raise ValueError("An active-session action must be callable")
        with self._condition:
            self._expire()
            if not self._active:
                return False, None
            return True, callback()

    def stop(self):
        """Stop host admissions without forging a browser position or receipt.

        Previously reported attempts remain correlatable until receipt closure;
        stopping does not manufacture a new sample or resolve an issue.
        """
        with self._condition:
            status = self._status if self._status in ("stopped", "failed", "ended") else "stopped"
            self._invalidate(self._error_code, status=status)
            return self.snapshot()

    def ready(self):
        with self._condition:
            if self._active:
                self._ready = True
                self._status = "monitoring" if self._playing else "ready"
                self._condition.notify_all()

    def playback(self, body):
        if self._assessment_version in (2, 3):
            return self._v2_playback(body)
        self._keys(body, ("session_id", "sequence", "source_time_s", "state"))
        sequence, source, state = body["sequence"], body["source_time_s"], body["state"]
        self._number(source)
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Playback sequence must be a nonnegative integer")
        if state not in ("playing", "paused", "ended", "stopped"):
            raise ValueError("Unsupported playback state")
        with self._condition:
            self._expire()
            if body["session_id"] != self.session_id:
                raise ValueError("Playback belongs to another session")
            if not self._active and state != "stopped":
                raise ValueError("Session has stopped")
            if sequence <= self._sequence or source < self._source_time_s:
                raise ValueError("Playback sequence and source time cannot move backward")
            if self._stop_at_s is not None and source > self._stop_at_s:
                raise ValueError("Playback exceeds the configured horizon")
            if self._ended and state == "playing":
                raise ValueError("Ended playback cannot restart this session")
            if state == "playing" and not self._ready:
                raise ValueError("Reference preparation is not ready")
            self._sequence, self._source_time_s = sequence, source
            self._last_heartbeat = self._now()
            if state == "stopped":
                self._invalidate(status="stopped")
            else:
                self._ended = (self._ended or state == "ended"
                               or self._stop_at_s is not None and source == self._stop_at_s)
                self._playing = state == "playing" and not self._ended
                if self._ended:
                    self._status = "ended"
                elif self._ready:
                    self._status = "monitoring" if self._playing else "paused"
                self._condition.notify_all()
            return self.snapshot()

    def opportunity_state(self):
        """Return pacing state without waiting for playback to advance."""
        with self._condition:
            self._expire()
            return {"active": self._active, "ready": self._ready,
                    "playing": self._playing, "ended": self._ended,
                    "source_time_s": self._source_time_s}

    def poll_cutoff(self, previous_cutoff_s, interval_s):
        """Return the newest eligible cutoff or None, without blocking completions."""
        self._number(interval_s, positive=True)
        if previous_cutoff_s is not None:
            self._number(previous_cutoff_s)
        with self._condition:
            self._expire()
            if not self._active:
                return None
            source = self._source_time_s
            previous = 0 if previous_cutoff_s is None else previous_cutoff_s
            final = self._ended and source > previous
            due = (source >= previous + interval_s or math.isclose(
                source, previous + interval_s, rel_tol=1e-12, abs_tol=1e-9))
            if self._ready and source > previous and (final or self._playing and due):
                self._status = "processing"
                return source
            return None

    def next_cutoff(self, previous_cutoff_s, interval_s):
        """Synchronous compatibility gate; async coordinators use poll_cutoff."""
        with self._condition:
            while True:
                cutoff = self.poll_cutoff(previous_cutoff_s, interval_s)
                if cutoff is not None or not self._active or self._ended:
                    return cutoff
                if self._ready:
                    self._status = "monitoring" if self._playing else "paused"
                self._condition.wait(timeout=min(self._heartbeat_timeout_s, 0.5))

    def _validate_envelope(self, envelope):
        required = ("session_id", "notification_id", "attempt_id", "text",
                    "reserved_at_s", "max_evidence_age_s", "execution_evidence_s")
        if type(envelope) is not dict or any(name not in envelope for name in required):
            raise ValueError("Notification envelope is incomplete")
        if (envelope["session_id"] != self.session_id
                or type(envelope["text"]) is not str
                or not envelope["text"].strip()
                or len(envelope["text"].encode("utf-8")) > 8192
                or any(type(envelope[name]) is not str or not envelope[name]
                       or len(envelope[name]) > 128 for name in ("notification_id", "attempt_id"))):
            raise ValueError("Notification identity or text is invalid")
        self._number(envelope["reserved_at_s"])
        self._number(envelope["max_evidence_age_s"], positive=True)
        evidence = envelope["execution_evidence_s"]
        if type(evidence) is not list or not evidence:
            raise ValueError("Notification requires execution support")
        for timestamp in evidence:
            if self._number(timestamp) > envelope["reserved_at_s"]:
                raise ValueError("Notification support follows the send reservation")
        return required

    def _match_notification(self, envelope):
        if (type(envelope) is not dict or self._binding is None
                or tuple(envelope.get(key) for key in (
                    "session_id", "notification_id", "attempt_id")) != self._binding):
            raise ValueError("Notification belongs to another mailbox attempt")

    def _retire_unknown(self, code=None):
        self._notification = None
        self._retired = True
        if self._delivery is None:
            self._delivery = {"status": "unknown"}
        if self._legacy_sender:
            self._invalidate(code)
        self._condition.notify_all()

    def notification_ready(self, min_visible_s=0):
        """Permit replacement after the prior outcome and wall-clock reading time."""
        self._number(min_visible_s)
        if self._assessment_version in (2, 3):
            return self._v2_notification_ready(min_visible_s)
        with self._condition:
            now = self._expire()
            if not self._active:
                return False
            if self._binding is None:
                return True
            if not self._delivery_observed:
                return False
            if self._retired or self._dismissed:
                return True
            if self._ack is None:
                return False
            if (self._notification is not None and self._source_time_s
                    - max(self._notification["execution_evidence_s"])
                    > self._notification["max_evidence_age_s"]):
                self._notification = None
                self._retired = True
                return True
            return now - self._ack_received_at >= min_visible_s

    def publish(self, envelope, *, min_visible_s=0):
        """Publish one fresh popup without waiting for its render acknowledgment."""
        if self._assessment_version in (2, 3):
            return self._v2_publish(envelope, min_visible_s)
        required = self._validate_envelope(envelope)
        self._number(min_visible_s)
        with self._condition:
            now = self._expire()
            if not self._active:
                return {"status": "not_sent"}
            if not self.notification_ready(min_visible_s):
                raise RuntimeError("The previous notification is not ready for replacement")
            if (envelope["reserved_at_s"] > self._source_time_s
                    or self._source_time_s - max(envelope["execution_evidence_s"])
                    > envelope["max_evidence_age_s"]):
                raise ValueError("Notification is not fresh at publication")
            self._notification = {key: deepcopy(envelope[key]) for key in required}
            self._binding = tuple(envelope[key] for key in (
                "session_id", "notification_id", "attempt_id"))
            self._ack = None
            self._delivery = None
            self._delivery_observed = False
            self._ack_received_at = None
            self._dismissed = self._retired = self._legacy_sender = False
            self._notification_deadline = now + self._ack_timeout_s
            self._condition.notify_all()
            return None

    def delivery_outcome(self, envelope, *, after_revision=0):
        """Poll a bound publication outcome, including after Stop clears its popup."""
        with self._condition:
            if self._assessment_version in (2, 3):
                return self._v2_delivery_outcome(envelope, after_revision)
            self._match_notification(envelope)
            now = self._expire()
            if self._delivery is None:
                if not self._active:
                    self._retire_unknown()
                elif (self._notification is not None and self._source_time_s
                        - max(self._notification["execution_evidence_s"])
                        > self._notification["max_evidence_age_s"]):
                    self._retire_unknown("notification_expired")
                elif now >= self._notification_deadline:
                    self._retire_unknown("display_ack_timeout")
            if self._delivery is not None:
                self._delivery_observed = True
            return deepcopy(self._delivery)

    def retire_notification(self, envelope):
        """Hide stale/resolved UI while preserving the actual or unknown send outcome."""
        with self._condition:
            if self._assessment_version in (2, 3):
                record = self._v2_match(envelope)
                record["retired"] = True
                self._v2_transition(record, {"status": "unknown"}, self._now() - self._started)
                if self._notification is not None and self._notification["attempt_id"] == envelope["attempt_id"]:
                    self._notification = None
                self._condition.notify_all()
                return
            self._match_notification(envelope)
            self._notification = None
            self._retired = True
            if self._delivery is None:
                self._delivery = {"status": "unknown"}
            self._condition.notify_all()

    def sender(self, envelope):
        """Synchronous compatibility publication, waiting for its actual render ACK."""
        result = self.publish(envelope)
        if result is not None:
            return result
        with self._condition:
            self._legacy_sender = True
            while True:
                outcome = self.delivery_outcome(envelope)
                if outcome is not None:
                    return outcome
                self._condition.wait(timeout=min(self._notification_deadline - self._now(), 0.5))

    def acknowledge(self, body):
        if self._assessment_version in (2, 3):
            return self._v2_acknowledge(body)
        self._keys(body, ("session_id", "notification_id", "attempt_id",
                          "source_time_s", "rendered_at_ms"))
        self._number(body["source_time_s"])
        self._number(body["rendered_at_ms"])
        with self._condition:
            now = self._expire()
            self._match_notification(body)
            if not self._active:
                raise ValueError("No active notification awaits this acknowledgment")
            if self._ack is not None:
                if body != self._ack:
                    raise ValueError("Acknowledgment conflicts with its recorded render")
                return self.snapshot()
            popup = self._notification
            if popup is None or self._delivery is not None:
                raise ValueError("No active notification awaits this acknowledgment")
            if now >= self._notification_deadline:
                self._retire_unknown("display_ack_timeout")
                raise ValueError("Notification acknowledgment arrived after its deadline")
            if not popup["reserved_at_s"] <= body["source_time_s"] <= self._source_time_s:
                raise ValueError("Render time is outside the observed send interval")
            if (body["source_time_s"] - max(popup["execution_evidence_s"])
                    > popup["max_evidence_age_s"]):
                self._retire_unknown("notification_expired")
                raise ValueError("Notification support expired before rendering")
            self._ack = deepcopy(body)
            self._ack_received_at = self._now()
            self._delivery = {"status": "acknowledged", "delivered_at_s": body["source_time_s"]}
            self._condition.notify_all()
            return self.snapshot()

    def dismiss(self, body):
        """Dismiss the current acknowledged card without resolving its issue."""
        self._keys(body, ("session_id", "notification_id", "attempt_id"))
        with self._condition:
            if self._assessment_version in (2, 3):
                record = self._v2_match_ids(body)
                if not record["transitions"] or record["transitions"][-1]["outcome"]["status"] != "acknowledged":
                    raise ValueError("Only a reported notification can be dismissed")
                record["dismissed"] = record["retired"] = True
                if self._notification is not None and self._notification["attempt_id"] == body["attempt_id"]:
                    self._notification = None
                self._condition.notify_all()
                return self.snapshot()
            self._expire()
            self._match_notification(body)
            if not self._active or self._ack is None:
                raise ValueError("Only the current acknowledged notification can be dismissed")
            self._dismissed = self._retired = True
            self._notification = None
            self._condition.notify_all()
            return self.snapshot()

    def emit(self, kind, payload):
        """Observe sanitized runner events without retaining prediction bodies."""
        with self._condition:
            if not self._active:
                return
            if kind in ("failure", "provider_failure"):
                code = payload.get("code", "monitoring_failed") if type(payload) is dict else "monitoring_failed"
                if type(code) is not str or not code or len(code) > 128:
                    code = "monitoring_failed"
                self._invalidate(code)
            elif kind == "provider_request":
                self._status = "processing"

    def finish(self, summary):
        with self._condition:
            if not self._active and not (self._assessment_version in (2, 3) and self._v2_closed is not None
                                         and self._status not in ("stopped", "failed")):
                return
            report = summary.get("run_report") if type(summary) is dict else None
            abort = summary.get("abort") if type(summary) is dict else None
            if abort is not None or report is None:
                self._invalidate("monitoring_failed")
            elif report.get("failures", 0):
                self._invalidate("monitoring_failed")
            elif report.get("reason") in ("budget_reservation_unavailable",
                                          "generation_call_limit_reached",
                                          "provider_admission_refused"):
                self._invalidate(report["reason"])
            else:
                self._active = self._playing = False
                self._status = "limit_reached" if report.get("reason") == "max_observations" else "ended"
                self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            if self._assessment_version == 1 or self._replay_clock_mode == "host":
                self._expire()
            value = {"session_id": self.session_id, "status": self._status,
                    "active": self._active,
                    "source_time_s": self._source_time_s, "playing": self._playing,
                    "stop_at_s": self._stop_at_s, "notification": deepcopy(self._notification),
                    "sequence": self._sequence, "error_code": self._error_code}
            if self._assessment_version in (2, 3):
                configured = self._v2_config or {}
                identity = self._v2_claim or {}
                value.update(assessment_contract_version=self._assessment_version, clock_id=self.clock_id,
                    timing_basis="session_monotonic", reference_id=configured.get("reference_id"),
                    execution_source_sha256=configured.get("execution_source_sha256"),
                    timing_limits=deepcopy(configured.get("timing_limits")),
                    receipt_limits=deepcopy(configured.get("receipt_limits")),
                    playback_claimed=self._v2_mapper is not None, baseline_received=self._sequence >= 0,
                    client_epoch_id=identity.get("client_epoch_id"), segment_id=identity.get("segment_id"),
                    sample_marker=deepcopy(self._v2_sample_marker), receipts_closed=self._v2_closed is not None)
                if self._replay_clock_mode == "host":
                    value.update(replay_clock_mode="host", replay_duration_s=self._host_duration_s)
                if self._assessment_version == 3:
                    value["tracked_issues"] = deepcopy(self._tracked_issues)
            return value

    def update_tracked_issues(self, snapshot):
        """Expose a bounded read-only projection of the runner's tracked issue state.

        This projection is independent of the delivery ledger: an issue shown
        here is not evidence that its notification was sent or seen.
        """
        if self._assessment_version != 3 or not isinstance(snapshot, dict):
            raise ValueError("A tracker snapshot is required")
        revision, issues = snapshot.get("revision"), snapshot.get("issues")
        if (type(snapshot.get("assessment_contract_version")) is not int
                or snapshot["assessment_contract_version"] != 3
                or type(revision) is not int or revision < 0
                or not isinstance(issues, list) or len(issues) > 4096):
            raise ValueError("Invalid tracker revision or issue list")
        projected, seen = [], set()
        for issue in issues:
            if not isinstance(issue, dict):
                raise ValueError("Invalid tracked issue")
            issue_id, state, description = (issue.get(key) for key in
                                            ("issue_id", "state", "latest_description"))
            self._uuid(issue_id)
            if (issue_id in seen or state not in ("candidate", "active", "uncertain", "resolved")
                    or not isinstance(description, str) or not description.strip()):
                raise ValueError("Invalid tracked issue")
            seen.add(issue_id)
            if state != "resolved":
                projected.append({"issue_id": issue_id, "state": state,
                                  "latest_description": description})
        if len(self._canonical(projected)) > 4 * 1024 * 1024:
            raise ValueError("Tracked issue projection exceeds capacity")
        with self._condition:
            if self._v2_config is None or any(snapshot.get(key) != value for key, value in (
                    ("session_id", self.session_id), ("reference_id", self._v2_config["reference_id"]),
                    ("execution_source_sha256", self._v2_config["execution_source_sha256"]))):
                raise ValueError("Tracker identity disagrees with session")
            if not self._active:
                return False
            if revision < self._tracked_revision or (revision == self._tracked_revision
                    and projected != self._tracked_issues):
                raise ValueError("Tracker projection is stale or changed at the same revision")
            self._tracked_issues = projected
            self._tracked_revision = revision
            self._condition.notify_all()
            return True

    @property
    def clock_id(self):
        return "session:" + self.session_id

    @property
    def served_viewer_sha256(self):
        return self._served_viewer_sha256

    def bind_viewer_bytes(self, data):
        """Bind the actual immutable bytes loaded by the local HTTP server once."""
        if type(data) is not bytes:
            raise ValueError("Served viewer must be immutable bytes")
        digest = hashlib.sha256(data).hexdigest()
        with self._condition:
            if self._served_viewer_sha256 not in (None, digest):
                raise ValueError("The served viewer cannot change within a session")
            self._served_viewer_sha256 = digest

    @staticmethod
    def _canonical(value):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Receiver metadata must be finite UTF-8 JSON") from exc

    @staticmethod
    def _uuid(value):
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError("Expected a canonical UUID")

    @staticmethod
    def _digest(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Expected a lowercase SHA-256 identity")

    def configure_assessment(self, *, assessment_contract_version, reference_id,
                             execution_source_sha256, timing_limits, receipt_limits):
        """Configure the selected assessment contract before client claim; no browser event is invented."""
        if self._assessment_version not in (2, 3) or type(assessment_contract_version) is not int or assessment_contract_version != self._assessment_version:
            raise ValueError("This runtime must explicitly select its assessment contract")
        self._digest(reference_id)
        self._digest(execution_source_sha256)
        self._keys(timing_limits, ("max_markers", "max_samples", "max_metadata_bytes"))
        self._keys(receipt_limits, ("max_attempts", "max_metadata_bytes"))
        if any(type(value) is not int or value <= 0 for group in (timing_limits, receipt_limits) for value in group.values()):
            raise ValueError("Timing and receipt limits must be positive integers")
        configuration = {"reference_id": reference_id, "execution_source_sha256": execution_source_sha256,
                         "timing_limits": deepcopy(timing_limits), "receipt_limits": deepcopy(receipt_limits)}
        with self._condition:
            if self._v2_config is not None and self._v2_config != configuration:
                raise ValueError("Assessment configuration cannot change within a session")
            self._v2_config = configuration

    def _v2_playback(self, body):
        if not isinstance(body, dict) or type(body.get("assessment_contract_version")) is not int or body["assessment_contract_version"] != self._assessment_version:
            raise ValueError("Versioned playback is required")
        with self._condition:
            if self._v2_config is None or body.get("session_id") != self.session_id or self._v2_closed is not None:
                raise ValueError("Playback is unconfigured, foreign or closed")
            now = self._expire() - self._started
            if body.get("action") == "claim":
                fields = ("assessment_contract_version", "session_id", "action", "client_epoch_id", "segment_id")
                self._keys(body, fields + (("duration_s",) if self._replay_clock_mode == "host" else ()))
                if self._replay_clock_mode == "host":
                    duration = self._number(body["duration_s"], positive=True)
                    duration = min(duration, self._stop_at_s) if self._stop_at_s is not None else duration
                    if self._host_duration_s is not None and self._host_duration_s != duration:
                        raise ValueError("Replay duration cannot change")
                for name in ("client_epoch_id", "segment_id"):
                    self._uuid(body[name])
                claim = {name: body[name] for name in ("client_epoch_id", "segment_id")}
                if self._v2_claim is not None:
                    if claim != self._v2_claim:
                        self._invalidate("playback_segment_changed")
                        raise ValueError("Reload or seek requires a new session")
                    return self.snapshot()
                if not self._active:
                    raise ValueError("Session has stopped")
                mapper = PlaybackClockMap(self.session_id, self._v2_config["execution_source_sha256"],
                    clock_id=self.clock_id, **claim, **self._v2_config["timing_limits"])
                marker = mapper.register_marker(str(uuid4()), issued_at_s=now)
                self._v2_mapper, self._v2_claim, self._v2_sample_marker = mapper, claim, marker
                if self._replay_clock_mode == "host":
                    self._host_duration_s = duration
                    self._sequence = 0
                self._last_heartbeat = self._now()
                return self.snapshot()
            if self._replay_clock_mode == "host" and body.get("action") == "control":
                self._keys(body, ("assessment_contract_version", "session_id", "action", "command"))
                command = body["command"]
                if command not in ("play", "pause", "stop") or self._v2_claim is None or not self._active:
                    raise ValueError("Invalid replay control")
                if command == "stop":
                    if self._playing:
                        self._host_segments[-1]["end_source"] = self._source_time_s
                    self._invalidate(status="stopped")
                elif command == "pause":
                    if self._playing:
                        self._host_segments[-1]["end_source"] = self._source_time_s
                    self._playing = False
                    self._status = "paused" if self._ready else "preparing"
                elif not self._playing:
                    if not self._ready or self._ended:
                        raise ValueError("Replay is not ready to play")
                    segment = {"source": self._source_time_s,
                        "absolute_host": self._started + now, "end_source": None}
                    limits = self._v2_config["timing_limits"]
                    if (len(self._host_segments) >= limits["max_samples"]
                            or len(self._canonical(self._host_segments + [segment])) > limits["max_metadata_bytes"]):
                        self._invalidate("timing_capacity_exceeded")
                        raise RuntimeError("Host replay timing capacity exceeded")
                    self._host_segments.append(segment)
                    self._playing, self._status = True, "monitoring"
                self._condition.notify_all()
                return self.snapshot()
            if self._replay_clock_mode == "host":
                raise ValueError("Host replay does not accept browser time samples")
            self._keys(body, ("assessment_contract_version", "session_id", "action", "marker_id", "event"))
            if body["action"] != "sample" or self._v2_mapper is None:
                raise ValueError("An initial client claim is required")
            event = body["event"]
            if not isinstance(event, dict):
                raise ValueError("Playback sample must be a record")
            sample_id = event.get("sample_id")
            self._uuid(sample_id)
            digest = hashlib.sha256(self._canonical(body)).hexdigest()
            if sample_id in self._v2_sample_digests:
                if digest != self._v2_sample_digests[sample_id]:
                    raise ValueError("Sample retry conflicts with its original report")
                return self.snapshot()
            if any(event.get(name) != self._v2_claim[name] for name in self._v2_claim):
                self._invalidate("playback_segment_changed")
                raise ValueError("Reload or seek requires a new session")
            source = self._number(event.get("source_elapsed_s"))
            state = event.get("state")
            if not self._active or self._ended and state == "playing":
                raise ValueError("Session playback is no longer active")
            if self._stop_at_s is not None and source > self._stop_at_s:
                raise ValueError("Playback exceeds the configured horizon")
            if source < self._source_time_s:
                self._invalidate("playback_segment_changed")
                raise ValueError("Backward playback requires a new session")
            if state == "playing" and (not self._ready or self._sequence < 0):
                raise ValueError("Playback needs readiness and an accepted baseline sample")
            self._v2_mapper.record_sample(body["marker_id"], event, received_at_s=now)
            self._v2_sample_digests[sample_id] = digest
            self._sequence, self._source_time_s = event["sequence"], source
            self._last_heartbeat = self._now()
            if state == "stopped":
                self._invalidate(status="stopped")
            else:
                self._ended = self._ended or state == "ended" or self._stop_at_s is not None and source == self._stop_at_s
                self._playing = state == "playing" and not self._ended
                self._status = "ended" if self._ended else "monitoring" if self._playing else "paused"
            try:
                self._v2_sample_marker = self._v2_mapper.register_marker(str(uuid4()), issued_at_s=now)
            except RuntimeError:
                self._v2_sample_marker = None
                self._invalidate("timing_capacity_exceeded")
            self._condition.notify_all()
            return self.snapshot()

    def source_passage_interval(self, source_elapsed_s, *, at_s):
        with self._condition:
            if self._assessment_version not in (2, 3) or self._v2_mapper is None:
                raise ValueError("No client playback clock has been claimed")
            if self._replay_clock_mode == "host":
                self._expire()
                source = self._number(source_elapsed_s)
                now = self._number(at_s)
                support = {"source_elapsed_s": source, "anchor_source_s": None, "anchor_host_s": None}
                mapping = {"schema_version": 1, "session_id": self.session_id,
                    "execution_source_sha256": self._v2_config["execution_source_sha256"],
                    "clock_id": self.clock_id, **self._v2_claim, "kind": "host_replay_passage",
                    "status": "unavailable", "reason": "missing_host_passage", "host_interval_s": None,
                    "support": support, "received_at_s": None}
                # Use first passage so pausing and resuming cannot make the
                # same already-seen frame appear newly captured.
                for segment in self._host_segments:
                    end = self._source_time_s if segment["end_source"] is None else segment["end_source"]
                    if segment["source"] <= source <= end:
                        anchor = segment["absolute_host"] - self._started
                        passage = anchor + source - segment["source"]
                        if passage <= now:
                            support.update(anchor_source_s=segment["source"], anchor_host_s=anchor)
                            mapping.update(status="mapped", reason=None,
                                host_interval_s=[passage, passage], received_at_s=passage)
                        break
                return mapping
            return self._v2_mapper.map_source_frame(source_elapsed_s, at_s=at_s)

    def _v2_match_ids(self, identities):
        if not isinstance(identities, dict) or identities.get("session_id") != self.session_id:
            raise ValueError("Notification belongs to another session")
        record = self._v2_attempts.get(identities.get("attempt_id"))
        if record is None or identities.get("notification_id") != record["envelope"]["notification_id"]:
            raise ValueError("Notification attempt is unknown")
        return record

    def _v2_match(self, envelope):
        record = self._v2_match_ids(envelope)
        if self._canonical(envelope) != self._canonical(record["envelope"]):
            raise ValueError("Notification envelope changed after publication")
        return record

    def _v2_transition(self, record, outcome, available):
        transitions = record["transitions"]
        if transitions and (transitions[-1]["outcome"]["status"] != "unknown" or outcome["status"] == "unknown"):
            return
        transitions.append({"assessment_contract_version": self._assessment_version, "attempt_id": record["envelope"]["attempt_id"],
            "revision": len(transitions) + 1, "available_at_s": available, "outcome": deepcopy(outcome)})

    def _v2_unknown_all(self, now):
        for record in self._v2_attempts.values():
            self._v2_transition(record, {"status": "unknown"}, now)
            record["retired"] = True

    def _v2_expire_attempts(self, now):
        for record in self._v2_attempts.values():
            if not record["transitions"] and now >= record["deadline_at_s"]:
                self._v2_transition(record, {"status": "unknown"}, now)
                # A backgrounded host replay may not have had a chance to
                # render yet. Keep the same uncertain attempt available for a
                # later real receipt; the runner still retires it when stale,
                # resolved, superseded, or stopped. Never create a second send.
                if self._replay_clock_mode != "host":
                    record["retired"] = True
                    if self._notification is not None and self._notification["attempt_id"] == record["envelope"]["attempt_id"]:
                        self._notification = None

    def _v2_notification_ready(self, min_visible_s):
        self._number(min_visible_s)
        with self._condition:
            now = self._expire() - self._started
            self._v2_expire_attempts(now)
            if not self._active or self._v2_closed is not None:
                return False
            if self._notification is None:
                return True
            record = self._v2_attempts[self._notification["attempt_id"]]
            if record["retired"] or record["dismissed"]:
                return True
            if not record["transitions"]:
                return False
            terminal = record["transitions"][-1]
            return terminal["outcome"]["status"] == "acknowledged" and now - terminal["available_at_s"] >= min_visible_s

    def _v2_publish(self, envelope, min_visible_s):
        self._number(min_visible_s)
        with self._condition:
            now = self._expire() - self._started
            if self._v2_mapper is None or self._v2_config is None or self._v2_closed is not None:
                raise ValueError("Playback receiver is unclaimed or closed")
            if not self._active:
                raise ValueError("Session has stopped before publication")
            if not self._v2_notification_ready(min_visible_s):
                raise RuntimeError("The previous card is not ready for replacement")
            required = ("assessment_contract_version", "session_id", "reference_id", "execution_source_sha256",
                "clock_id", "timing_basis", "delivery_mode", "notification_id", "attempt_id", "issue_id", "text", "binding",
                "policy_at_s", "reserved_at_s", "max_input_age_s", "tracker_revision")
            self._keys(envelope, required)
            if (type(envelope["assessment_contract_version"]) is not int or envelope["assessment_contract_version"] != self._assessment_version
                    or envelope["session_id"] != self.session_id or envelope["clock_id"] != self.clock_id
                    or envelope["reference_id"] != self._v2_config["reference_id"]
                    or envelope["execution_source_sha256"] != self._v2_config["execution_source_sha256"]
                    or envelope["timing_basis"] != "session_monotonic" or envelope["delivery_mode"] != "display_ack"):
                raise ValueError("Notification identity or timing contract disagrees")
            for name in ("attempt_id", "issue_id"):
                self._uuid(envelope[name])
            self._digest(envelope["notification_id"])
            if envelope["attempt_id"] in self._v2_attempts:
                raise ValueError("Attempt was already published")
            if not isinstance(envelope["text"], str) or not envelope["text"].strip() or len(envelope["text"].encode("utf-8")) > 8192:
                raise ValueError("Notification text is invalid")
            if not self._number(envelope["policy_at_s"]) <= self._number(envelope["reserved_at_s"]) <= now:
                raise ValueError("Notification reservation follows publication")
            limit = self._number(envelope["max_input_age_s"])
            binding = envelope["binding"]
            mapping = binding.get("playback_mapping") if isinstance(binding, dict) else None
            if (not isinstance(mapping, dict) or binding.get("timing_status") != "bounded"
                    or mapping.get("status") != "mapped" or mapping.get("kind") != (
                        "host_replay_passage" if self._replay_clock_mode == "host" else "replay_passage")
                    or any(mapping.get(name) != value for name, value in {
                        "session_id": self.session_id, "execution_source_sha256": self._v2_config["execution_source_sha256"],
                        "clock_id": self.clock_id, **self._v2_claim}.items())):
                raise ValueError("Notification needs this playback passage mapping")
            interval = mapping.get("host_interval_s")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Playback input interval is missing")
            lower, upper = (self._number(value) for value in interval)
            if not lower <= upper <= now or now - lower > limit:
                raise ValueError("Notification input is not definitely fresh")
            if any(other["envelope"]["issue_id"] == envelope["issue_id"] and
                   (not other["transitions"] or other["transitions"][-1]["outcome"]["status"] != "not_sent")
                   for other in self._v2_attempts.values()):
                raise ValueError("Issue already has a pending or reported delivery")
            limits = self._v2_config["receipt_limits"]
            if len(self._v2_attempts) >= limits["max_attempts"]:
                raise RuntimeError("Receiver attempt capacity exceeded")
            staged_clock = deepcopy(self._v2_mapper)
            marker = staged_clock.register_marker(str(uuid4()), issued_at_s=now)
            record = {"envelope": deepcopy(envelope), "render_marker": marker, "receipt": None,
                "transitions": [], "deadline_at_s": now + self._ack_timeout_s,
                "retired": False, "dismissed": False}
            # Reserve two outcome slots and a full canonical client report before
            # exposure. Finite IEEE timestamps need at most 309 JSON digits.
            number = "9" * 309
            event = {**self._v2_claim, "marker_received_at_s": number, "event_started_at_s": number,
                "event_finished_at_s": number, "report_sent_at_s": number,
                "notification_id": envelope["notification_id"], "attempt_id": envelope["attempt_id"]}
            render = {"schema_version": 1, "session_id": self.session_id,
                "execution_source_sha256": self._v2_config["execution_source_sha256"], "clock_id": self.clock_id,
                **self._v2_claim, "kind": "reported_render", "status": "mapped", "reason": None,
                "host_interval_s": [number, number], "support": {"notification_id": envelope["notification_id"],
                    "attempt_id": envelope["attempt_id"], "marker_id": marker["marker_id"]}, "received_at_s": number}
            worst = deepcopy(record)
            worst["receipt"] = {"assessment_contract_version": self._assessment_version, "session_id": self.session_id,
                "notification_id": envelope["notification_id"], "attempt_id": envelope["attempt_id"],
                "receipt": {"status": "acknowledged", "marker_id": marker["marker_id"], "event": event}}
            worst["transitions"] = [{"assessment_contract_version": self._assessment_version, "attempt_id": envelope["attempt_id"],
                "revision": 1, "available_at_s": number, "outcome": {"status": "unknown"}},
                {"assessment_contract_version": self._assessment_version, "attempt_id": envelope["attempt_id"], "revision": 2,
                 "available_at_s": number, "outcome": {"status": "acknowledged", "clock_id": self.clock_id, "render_mapping": render}}]
            reservation = len(self._canonical(worst))
            if self._v2_reserved_bytes + reservation > limits["max_metadata_bytes"]:
                raise RuntimeError("Receiver future-outcome capacity exceeded")
            self._v2_mapper = staged_clock
            self._v2_attempts[envelope["attempt_id"]] = record
            self._v2_reserved_bytes += reservation
            self._notification = {**deepcopy(envelope), "render_marker": deepcopy(marker)}
            self._condition.notify_all()
            return None

    def _v2_delivery_outcome(self, envelope, after_revision):
        if type(after_revision) is not int or after_revision < 0:
            raise ValueError("Outcome cursor must be a nonnegative integer")
        record = self._v2_match(envelope)
        now = self._expire() - self._started
        if self._v2_closed is None:
            self._v2_expire_attempts(now)
        return deepcopy(next((row for row in record["transitions"] if row["revision"] > after_revision), None))

    def _v2_acknowledge(self, body):
        self._keys(body, ("assessment_contract_version", "session_id", "notification_id", "attempt_id", "receipt"))
        if type(body["assessment_contract_version"]) is not int or body["assessment_contract_version"] != self._assessment_version:
            raise ValueError("Versioned receiver receipt is required")
        with self._condition:
            record = self._v2_match_ids(body)
            canonical = self._canonical(body)
            if record["receipt"] is not None:
                if canonical != self._canonical(record["receipt"]):
                    raise ValueError("Receipt conflicts with its first accepted report")
                return self.snapshot()
            if self._v2_closed is not None:
                raise ValueError("Receipt coverage has closed")
            now = self._expire() - self._started
            receipt = body["receipt"]
            if not isinstance(receipt, dict):
                raise ValueError("Receipt must be a record")
            if receipt.get("status") == "acknowledged":
                self._keys(receipt, ("status", "marker_id", "event"))
                if receipt["marker_id"] != record["render_marker"]["marker_id"]:
                    raise ValueError("Receipt does not use this attempt's render marker")
                event = receipt["event"]
                if not isinstance(event, dict) or any(event.get(name) != body[name] for name in ("notification_id", "attempt_id")):
                    raise ValueError("Render report identities disagree")
                mapping = self._v2_mapper.map_render_receipt(receipt["marker_id"], event, received_at_s=now)
                outcome = {"status": "acknowledged", "clock_id": self.clock_id, "render_mapping": mapping}
            elif receipt.get("status") == "not_sent":
                self._keys(receipt, ("status", "client_epoch_id", "segment_id", "reason"))
                if any(receipt[name] != self._v2_claim[name] for name in self._v2_claim) or receipt["reason"] not in (
                        "stale_input", "timing_unavailable", "stopped_before_insertion"):
                    raise ValueError("Receiver decline context or reason disagrees")
                outcome = {"status": "not_sent"}
            else:
                raise ValueError("Unsupported receiver receipt")
            record["receipt"] = deepcopy(body)
            self._v2_transition(record, outcome, now)
            self._condition.notify_all()
            return self.snapshot()

    def close_receipts(self):
        """Close receipt and timing mutations atomically before the final drain."""
        with self._condition:
            if self._assessment_version not in (2, 3):
                raise ValueError("Receipt closure belongs to a versioned assessment contract")
            if self._v2_closed is None:
                self._v2_closed = self._now() - self._started
                self._v2_unknown_all(self._v2_closed)
                self._active = self._playing = False
                self._notification = None
                self._condition.notify_all()
            return {"assessment_contract_version": self._assessment_version, "session_id": self.session_id,
                    "clock_id": self.clock_id, "closed_at_s": self._v2_closed}

    def timing_snapshot(self):
        """Return host-only supporting records; never forward this on UI polls."""
        with self._condition:
            return deepcopy({"assessment_contract_version": self._assessment_version if self._assessment_version in (2, 3) else 2, "session_id": self.session_id, "clock_id": self.clock_id,
                "served_viewer_sha256": self._served_viewer_sha256,
                "playback_clock": None if self._v2_mapper is None else self._v2_mapper.snapshot(),
                "attempts": list(self._v2_attempts.values()), "reserved_metadata_bytes": self._v2_reserved_bytes,
                "receipts_closed_at_s": self._v2_closed})
