"""Bounded records of completed predictions, gated by session time."""

from copy import deepcopy
import json
import math
import re
from uuid import UUID


class SessionState:
    """Preserve reported candidates without inferring issue identity or repair.

    Times share one caller-declared session clock. A snapshot exposes a record
    only at/after completion, even when replay preloads future completions.
    The caller supplies predictions already checked against their actual model
    input; this register checks metadata shape and cutoff bounds, not image
    evidence membership. Neither caller clocks nor model claims are authenticated.

    Capacity covers all stored records, including not-yet-visible completions.
    Metadata bytes mean the summed canonical UTF-8 JSON sizes of records, not
    Python heap usage or snapshot overhead. Overflow never evicts old records.
    Create a new instance for a new session/reference. This in-memory class is
    not a durable trace, issue association policy, or predictor-memory adapter.
    """

    def __init__(self, session_id, reference_id, *, max_records, max_metadata_bytes):
        self._uuid(session_id)
        if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
            raise ValueError("Reference ID must be a lowercase SHA-256 digest")
        for value in (max_records, max_metadata_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Record and metadata limits must be positive integers")
        self._session_id = session_id
        self._reference_id = reference_id
        self._max_records = max_records
        self._max_metadata_bytes = max_metadata_bytes
        self._records = []
        self._metadata_bytes = 0
        self._last_snapshot_s = None

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
    def _keys(record, required, optional=()):
        if (not isinstance(record, dict) or not set(required) <= record.keys()
                or not record.keys() <= set(required) | set(optional)):
            raise ValueError("Metadata record has missing or unsupported fields")

    @staticmethod
    def _text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected a nonblank string")

    def _validate_prediction(self, response, cutoff):
        self._keys(response, ("schema_version", "session_id", "observation_id",
                              "status", "candidate_issues", "error"))
        if type(response["schema_version"]) is not int or response["schema_version"] != 1:
            raise ValueError("Expected prediction schema version 1")
        if response["session_id"] != self._session_id:
            raise ValueError("Prediction belongs to another session")
        self._uuid(response["observation_id"])
        status, candidates = response["status"], response["candidate_issues"]
        if status not in ("ok", "insufficient_evidence", "failed"):
            raise ValueError("Unsupported prediction status")
        if not isinstance(candidates, list) or (status != "ok" and candidates):
            raise ValueError("Only ok predictions may contain candidate issues")
        if status == "failed":
            self._keys(response["error"], ("type", "message"))
            self._text(response["error"]["type"])
            self._text(response["error"]["message"])
        elif response["error"] is not None:
            raise ValueError("Only failed predictions may contain an error")
        for issue in candidates:
            self._keys(issue, ("description", "execution_evidence_s", "reference_evidence_s"),
                       ("confidence", "issue_key"))
            self._text(issue["description"])
            for field in ("execution_evidence_s", "reference_evidence_s"):
                evidence = issue[field]
                if not isinstance(evidence, list) or (field == "execution_evidence_s" and not evidence):
                    raise ValueError("Evidence must be lists with nonempty execution evidence")
                for timestamp in evidence:
                    self._number(timestamp)
                    if field == "execution_evidence_s" and timestamp > cutoff:
                        raise ValueError("Execution evidence cannot follow the observation cutoff")
            if issue.get("confidence") is not None and self._number(issue["confidence"]) > 1:
                raise ValueError("Confidence must be between zero and one")
            if issue.get("issue_key") is not None:
                self._text(issue["issue_key"])

    def record_prediction(self, response, *, reference_id, observation_cutoff_s,
                          dispatched_at_s, completed_at_s):
        """Atomically record one normalized prediction, retaining no images.

        Completion may equal the last snapshot time, but cannot precede it:
        already-published state cannot acquire retrospectively available data.
        Each observation contributes at most one normalized result. Recording
        individual retry attempts belongs to the future runtime trace.
        """
        if reference_id != self._reference_id:
            raise ValueError("Prediction belongs to another reference")
        cutoff = self._number(observation_cutoff_s)
        dispatch = self._number(dispatched_at_s)
        completion = self._number(completed_at_s)
        if cutoff > dispatch or dispatch > completion:
            raise ValueError("Expected observation cutoff <= dispatch <= completion")
        if self._last_snapshot_s is not None and completion < self._last_snapshot_s:
            raise ValueError("Completion cannot precede the last published snapshot")
        self._validate_prediction(response, cutoff)
        if any(record["prediction"]["observation_id"] == response["observation_id"]
               for record, _ in self._records):
            raise ValueError("Observation ID has already been recorded")
        record = {"reference_id": reference_id, "observation_cutoff_s": observation_cutoff_s,
                  "dispatched_at_s": dispatched_at_s, "completed_at_s": completed_at_s,
                  "prediction": deepcopy(response)}
        try:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Prediction metadata must be finite UTF-8 JSON") from exc
        size = len(encoded)
        if (len(self._records) >= self._max_records
                or self._metadata_bytes + size > self._max_metadata_bytes):
            raise RuntimeError("Session prediction record or metadata budget exceeded")
        self._records.append((record, size))
        self._metadata_bytes += size

    def snapshot(self, at_s):
        """Return independent metadata copies available by a nondecreasing time.

        Candidate identity here is only (observation ID, candidate index).
        An absent mention, failure or insufficient evidence never erases prior
        candidates or declares them resolved. The latest observation is chosen
        by source cutoff, then dispatch/completion/ID to break equal-cutoff ties.
        """
        self._number(at_s)
        if self._last_snapshot_s is not None and at_s < self._last_snapshot_s:
            raise ValueError("Snapshot time cannot move backward")
        visible = sorted((entry for entry in self._records if entry[0]["completed_at_s"] <= at_s),
                         key=lambda entry: (entry[0]["completed_at_s"], entry[0]["observation_cutoff_s"],
                                            entry[0]["prediction"]["observation_id"]))
        predictions = [deepcopy(record) for record, _ in visible]
        candidates = []
        for record in predictions:
            for index, candidate in enumerate(record["prediction"]["candidate_issues"]):
                candidates.append({"observation_id": record["prediction"]["observation_id"],
                                   "candidate_index": index, "candidate": deepcopy(candidate),
                                   "observation_cutoff_s": record["observation_cutoff_s"],
                                   "completed_at_s": record["completed_at_s"]})
        latest = max(predictions, key=lambda record: (
            record["observation_cutoff_s"], record["dispatched_at_s"],
            record["completed_at_s"], record["prediction"]["observation_id"]), default=None)
        result = {"schema_version": 1, "session_id": self._session_id,
                  "reference_id": self._reference_id, "at_s": at_s,
                  "predictions": predictions, "candidate_records": candidates,
                  "latest_observation_id": latest["prediction"]["observation_id"] if latest else None,
                  "record_count": len(visible), "metadata_bytes": sum(size for _, size in visible),
                  "limits": {"max_records": self._max_records, "max_metadata_bytes": self._max_metadata_bytes}}
        self._last_snapshot_s = at_s
        return result
