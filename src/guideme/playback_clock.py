"""Bound reported file-playback events by host-issued causal markers."""

from copy import deepcopy
import json
import math
from .validation import require_keys, require_sha256, require_uuid


class PlaybackClockMap:
    """Retain bounded source reports through one serialized coordinator.

    Host and client clocks remain separate. A client event reported after
    receiving a known marker lies between host marker issuance and host receipt,
    provided the caller's causal report is truthful. No offset, midpoint, clock
    calibration, interpolation or real-camera capture time is inferred.

    Use a new instance for reloads/seeks. The caller must report those boundaries;
    a source jump alone cannot distinguish a seek from dropped video frames.
    Reported render opportunities are not proof of physical display. Their marker
    must precede insertion/event, not merely the later ACK. Queries never mutate
    retained history or expire late receipts. No clock, transport or UI is used.
    """

    _CLIENT_FIELDS = ("client_epoch_id", "segment_id", "marker_received_at_s",
                      "event_started_at_s", "event_finished_at_s", "report_sent_at_s")

    def __init__(self, session_id, execution_source_sha256, *, clock_id, client_epoch_id,
                 segment_id, max_markers, max_samples, max_metadata_bytes):
        for identifier in (session_id, client_epoch_id, segment_id):
            self._uuid(identifier)
        self._digest(execution_source_sha256)
        if not isinstance(clock_id, str) or not clock_id.strip():
            raise ValueError("A nonblank UTF-8 clock identity is required")
        clock_id.encode("utf-8")
        limits = {"max_markers": max_markers, "max_samples": max_samples,
                  "max_metadata_bytes": max_metadata_bytes}
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("Capacity limits must be positive integers")
        self._identity = {"schema_version": 1, "session_id": session_id,
            "execution_source_sha256": execution_source_sha256, "clock_id": clock_id,
            "client_epoch_id": client_epoch_id, "segment_id": segment_id}
        self._limits, self._markers, self._samples, self._sample_ids = limits, {}, [], set()
        # Logical bytes charge the canonical identity plus each retained record,
        # excluding containers, counters, snapshots and Python heap overhead.
        self._metadata_bytes = len(self._canonical(self._identity))
        if self._metadata_bytes > max_metadata_bytes:
            raise RuntimeError("Clock identity exceeds its metadata capacity")
        self._last_event = None

    _keys = staticmethod(require_keys)
    _uuid = staticmethod(require_uuid)
    _digest = staticmethod(require_sha256)

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
            raise ValueError("Clock metadata must be finite UTF-8 JSON") from exc

    def _mutation_time(self, value):
        now = self._number(value)
        if self._last_event is not None and now < self._last_event:
            raise ValueError("Host mutation time cannot reverse")
        return now

    def _client_event(self, marker_id, event, extra_fields, received):
        self._uuid(marker_id)
        if marker_id not in self._markers:
            raise ValueError("Client event names an unknown host marker")
        self._keys(event, (*self._CLIENT_FIELDS, *extra_fields))
        if any(event[name] != self._identity[name] for name in ("client_epoch_id", "segment_id")):
            raise ValueError("Client event belongs to another epoch or playback segment")
        times = [self._number(event[name]) for name in self._CLIENT_FIELDS[2:]]
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("Client event must follow marker receipt and precede report sending")
        marker = self._markers[marker_id]
        if marker["issued_at_s"] > received:
            raise ValueError("Host receipt precedes marker issuance")
        return marker

    def register_marker(self, marker_id, *, issued_at_s):
        """Retain a unique marker before the caller sends it to the client."""
        issued = self._mutation_time(issued_at_s)
        self._uuid(marker_id)
        if marker_id in self._markers:
            raise ValueError("Host marker was already registered")
        marker = {"marker_id": marker_id, "issued_at_s": issued}
        size = len(self._canonical(marker))
        if (len(self._markers) >= self._limits["max_markers"]
                or self._metadata_bytes + size > self._limits["max_metadata_bytes"]):
            raise RuntimeError("Marker or metadata capacity exceeded")
        receipt = deepcopy(marker)
        self._markers[marker_id] = marker
        self._metadata_bytes, self._last_event = self._metadata_bytes + size, issued
        return receipt

    def record_sample(self, marker_id, event, *, received_at_s):
        """Retain one ordered source read; repeats never replace older samples."""
        received = self._mutation_time(received_at_s)
        self._client_event(marker_id, event, ("sample_id", "sequence", "source_elapsed_s", "state"), received)
        self._uuid(event["sample_id"])
        source = self._number(event["source_elapsed_s"])
        sequence = event["sequence"]
        if (type(sequence) is not int or sequence < 0 or event["sample_id"] in self._sample_ids
                or event["state"] not in ("playing", "paused", "ended", "stopped")):
            raise ValueError("Invalid sample identity, sequence or playback state")
        self._number(sequence)
        if self._samples:
            previous = self._samples[-1]
            if (sequence <= previous["sequence"] or source < previous["source_elapsed_s"]
                    or event["event_started_at_s"] < previous["event_finished_at_s"]):
                raise ValueError("Sample source, sequence or client read windows reverse")
        sample = {**event, "marker_id": marker_id, "received_at_s": received}
        size = len(self._canonical(sample))
        if (len(self._samples) >= self._limits["max_samples"]
                or self._metadata_bytes + size > self._limits["max_metadata_bytes"]):
            raise RuntimeError("Sample or metadata capacity exceeded")
        receipt = deepcopy(sample)
        self._samples.append(sample)
        self._sample_ids.add(event["sample_id"])
        self._metadata_bytes, self._last_event = self._metadata_bytes + size, received
        return receipt

    def _result(self, kind, support, *, interval=None, reason=None, received=None):
        return {**self._identity, "kind": kind, "status": "unavailable" if interval is None else "mapped",
                "reason": reason, "host_interval_s": interval, "support": support, "received_at_s": received}

    def map_source_frame(self, source_elapsed_s, *, at_s):
        """Bracket the first reported passage of a source point, including skipped frames."""
        source, now = self._number(source_elapsed_s), self._number(at_s)
        lower = upper = None
        for sample in self._samples:
            if sample["received_at_s"] > now:
                break
            if sample["source_elapsed_s"] >= source:
                upper = sample
                break
            lower = sample
        support = {"source_elapsed_s": source, "sample_ids": []}
        if upper is None:
            return self._result("replay_passage", support, reason="missing_upper_sample")
        if lower is None:
            return self._result("replay_passage", support, reason="missing_lower_sample")
        support["sample_ids"] = [lower["sample_id"], upper["sample_id"]]
        # The source may cross while the lower report is in transit. Its host
        # receipt is therefore not a valid lower bound; marker issuance is.
        interval = [self._markers[lower["marker_id"]]["issued_at_s"], upper["received_at_s"]]
        return self._result("replay_passage", support, interval=interval, received=upper["received_at_s"])

    def map_render_receipt(self, marker_id, event, *, received_at_s):
        """Bound a claimed render opportunity without retaining or expiring it."""
        received = self._number(received_at_s)
        marker = self._client_event(marker_id, event, ("notification_id", "attempt_id"), received)
        self._digest(event["notification_id"])
        self._uuid(event["attempt_id"])
        support = {"notification_id": event["notification_id"], "attempt_id": event["attempt_id"], "marker_id": marker_id}
        return self._result("reported_render", support, interval=[marker["issued_at_s"], received], received=received)

    def snapshot(self):
        """Copy retained history without advancing any host or client clock."""
        return deepcopy({**self._identity, "markers": list(self._markers.values()), "samples": self._samples,
            "limits": self._limits, "metadata_bytes": self._metadata_bytes, "last_event_at_s": self._last_event})
