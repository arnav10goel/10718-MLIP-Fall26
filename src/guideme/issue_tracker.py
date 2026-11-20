"""Bounded, explicit issue lifecycle bookkeeping for one runtime session."""

from copy import deepcopy
import json
import math
import re
from uuid import UUID, uuid4


class IssueTracker:
    """Apply completed runtime decisions without inferring visual correctness.

    Inputs are normalized predictions and validate_issue_updates() results.
    Upstream validation owns actual frame membership and semantic association;
    this class rechecks metadata, binding and lifecycle consistency. Admissions
    are explicit runtime decisions: a null link alone never creates an issue.

    Completion and snapshot times share a trusted caller-declared session clock.
    State advances forward only; snapshots cannot reconstruct historical state.
    Strictly newer applied observation cutoffs reject stale/equal-cutoff batches,
    including new admissions, rather than merging late evidence into current
    state. Rejected raw results belong in the separate runtime trace.

    Capacity includes closed episodes and the processed-observation ledger.
    Byte limits count canonical UTF-8 JSON records, not Python heap/snapshot
    overhead. Nothing is silently evicted. This class performs no model calls,
    notifications or persistent I/O, and does not wire memory into predictions.
    """

    def __init__(self, session_id, reference_id, *, max_issues, max_observations, max_metadata_bytes):
        self._uuid(session_id)
        if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
            raise ValueError("Reference ID must be a lowercase SHA-256 identity")
        for value in (max_issues, max_observations, max_metadata_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Tracker limits must be positive integers")
        self._session_id = session_id
        self._reference_id = reference_id
        self._limits = {"max_issues": max_issues, "max_observations": max_observations,
                        "max_metadata_bytes": max_metadata_bytes}
        self._issues = {}
        self._observations = []
        self._clock_s = None
        self._latest_cutoff_s = None
        self._metadata_bytes = 0

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

    def _evidence(self, record, cutoff):
        for field in ("execution_evidence_s", "reference_evidence_s"):
            evidence = record[field]
            if not isinstance(evidence, list) or (field == "execution_evidence_s" and not evidence):
                raise ValueError("Evidence must be lists with nonempty execution evidence")
            for timestamp in evidence:
                self._number(timestamp)
                if field == "execution_evidence_s" and timestamp > cutoff:
                    raise ValueError("Execution evidence follows the observation cutoff")

    def apply_completed(self, response, validated_updates, *, new_issue_candidate_indices, completed_at_s):
        """Atomically apply one completed observation and return a copied receipt.

        Original candidates/evidence and their clock mapping are already checked
        upstream. UUID4 runtime IDs are generated only for selected null links;
        optional model issue keys never determine identity. No implicit changes
        are inferred for unmentioned records. All checks and prospective capacity
        calculations precede commit; failed batches consume no retained IDs,
        observation entries, budgets or clock advancement.
        """
        self._keys(response, ("schema_version", "session_id", "observation_id", "status",
                              "candidate_issues", "error"))
        self._keys(validated_updates, ("schema_version", "session_id", "observation_id", "reference_id",
                                      "observation_cutoff_s", "dispatched_at_s", "prior_available_at_s",
                                      "candidate_links", "issue_updates"))
        for record in (response, validated_updates):
            if type(record["schema_version"]) is not int or record["schema_version"] != 1:
                raise ValueError("Expected schema version 1")
            if record["session_id"] != self._session_id:
                raise ValueError("Record belongs to another session")
        observation_id = response["observation_id"]
        self._uuid(observation_id)
        if (validated_updates["observation_id"] != observation_id
                or validated_updates["reference_id"] != self._reference_id):
            raise ValueError("Update binding does not match the prediction and tracker")
        if any(record["observation_id"] == observation_id for record in self._observations):
            raise ValueError("Observation has already been applied")
        cutoff = self._number(validated_updates["observation_cutoff_s"])
        dispatch = self._number(validated_updates["dispatched_at_s"])
        prior_available = self._number(validated_updates["prior_available_at_s"])
        completion = self._number(completed_at_s)
        if cutoff > dispatch or prior_available > dispatch or dispatch > completion:
            raise ValueError("Invalid observation, prior availability, dispatch or completion timing")
        if self._clock_s is not None and completion < self._clock_s:
            raise ValueError("Completion cannot precede the tracker clock")
        if self._latest_cutoff_s is not None and cutoff <= self._latest_cutoff_s:
            raise ValueError("Applied observation cutoffs must strictly increase")

        status, candidates = response["status"], response["candidate_issues"]
        if status not in ("ok", "insufficient_evidence", "failed"):
            raise ValueError("Unsupported prediction status")
        if not isinstance(candidates, list) or (status != "ok" and candidates):
            raise ValueError("Only ok predictions may contain candidates")
        if status == "failed":
            self._keys(response["error"], ("type", "message"))
            for value in response["error"].values():
                self._text(value)
        elif response["error"] is not None:
            raise ValueError("Only failed predictions may contain an error")
        for candidate in candidates:
            self._keys(candidate, ("description", "execution_evidence_s", "reference_evidence_s"),
                       ("confidence", "issue_key"))
            self._text(candidate["description"])
            self._evidence(candidate, cutoff)
            if candidate.get("confidence") is not None and self._number(candidate["confidence"]) > 1:
                raise ValueError("Confidence must be between zero and one")
            if candidate.get("issue_key") is not None:
                self._text(candidate["issue_key"])

        def open_issue(identifier):
            self._uuid(identifier)
            if identifier not in self._issues or self._issues[identifier]["state"] == "resolved":
                raise ValueError("Issue is unknown or its episode is closed")
            if self._issues[identifier]["first_available_at_s"] > prior_available:
                raise ValueError("Issue identity was unavailable in the declared prior context")

        links, transitions = validated_updates["candidate_links"], validated_updates["issue_updates"]
        if not isinstance(links, list) or not isinstance(transitions, list):
            raise ValueError("Links and issue updates must be lists")
        if status == "failed" and (links or transitions):
            raise ValueError("Failed predictions cannot change issues")
        targets, linked_ids = {}, set()
        for link in links:
            self._keys(link, ("candidate_index", "issue_id"))
            index, identifier = link["candidate_index"], link["issue_id"]
            if type(index) is not int or not 0 <= index < len(candidates) or index in targets:
                raise ValueError("Candidate indices must be unique and in range")
            if identifier is not None:
                open_issue(identifier)
                if identifier in linked_ids:
                    raise ValueError("Multiple candidates cannot claim one issue identity")
                linked_ids.add(identifier)
            targets[index] = identifier
        if len(targets) != len(candidates):
            raise ValueError("Every candidate must have an explicit link or null association")
        updated_ids = set()
        for transition in transitions:
            self._keys(transition, ("issue_id", "state", "execution_evidence_s", "reference_evidence_s"))
            identifier, state = transition["issue_id"], transition["state"]
            open_issue(identifier)
            if identifier in linked_ids or identifier in updated_ids:
                raise ValueError("Duplicate or conflicting issue updates")
            updated_ids.add(identifier)
            if state not in ("resolved", "uncertain"):
                raise ValueError("Explicit transitions must be resolved or uncertain")
            if status == "insufficient_evidence" and state != "uncertain":
                raise ValueError("Insufficient evidence cannot resolve issues")
            self._evidence(transition, cutoff)
        if not isinstance(new_issue_candidate_indices, list):
            raise ValueError("New-issue admissions must be an explicit list")
        admitted = set()
        for index in new_issue_candidate_indices:
            if (type(index) is not int or index not in targets or index in admitted
                    or targets[index] is not None):
                raise ValueError("Admit only unique null-linked candidate indices")
            admitted.add(index)
        if (len(self._issues) + len(admitted) > self._limits["max_issues"]
                or len(self._observations) >= self._limits["max_observations"]):
            raise RuntimeError("Issue or processed-observation capacity exceeded")

        def stamp(state, evidence):
            return {"state": state, "observation_id": observation_id,
                    "observation_cutoff_s": cutoff, "available_at_s": completion,
                    "execution_evidence_s": deepcopy(evidence["execution_evidence_s"]),
                    "reference_evidence_s": deepcopy(evidence["reference_evidence_s"])}

        # Work on detached records so even a late encoding/budget failure leaves
        # the live issue book, duplicate ledger and both clocks unchanged.
        proposed = deepcopy(self._issues)
        created = []
        for index in new_issue_candidate_indices:
            identifier = str(uuid4())
            if identifier in proposed:
                raise RuntimeError("Runtime issue ID collision")
            candidate = candidates[index]
            proposed[identifier] = {
                "issue_id": identifier, "state": "candidate",
                "first_observation_id": observation_id, "first_observation_cutoff_s": cutoff,
                "first_available_at_s": completion, "last_observation_id": observation_id,
                "last_observation_cutoff_s": cutoff, "last_available_at_s": completion,
                "initial_candidate": deepcopy(candidate), "latest_candidate": deepcopy(candidate),
                "latest_update": stamp("candidate", candidate),
            }
            created.append({"candidate_index": index, "issue_id": identifier})
        for index, identifier in targets.items():
            if identifier is not None:
                issue = proposed[identifier]
                issue.update(state="active", last_observation_id=observation_id,
                             last_observation_cutoff_s=cutoff, last_available_at_s=completion,
                             latest_candidate=deepcopy(candidates[index]),
                             latest_update=stamp("active", candidates[index]))
        for transition in transitions:
            issue = proposed[transition["issue_id"]]
            issue.update(state=transition["state"], last_observation_id=observation_id,
                         last_observation_cutoff_s=cutoff, last_available_at_s=completion,
                         latest_update=stamp(transition["state"], transition))
        ledger = {"observation_id": observation_id, "observation_cutoff_s": cutoff,
                  "dispatched_at_s": dispatch, "completed_at_s": completion,
                  "created_issues": created,
                  "unassociated_candidate_indices": [link["candidate_index"] for link in links
                      if link["issue_id"] is None and link["candidate_index"] not in admitted]}
        observations = self._observations + [ledger]
        try:
            size = sum(len(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8"))
                       for record in list(proposed.values()) + observations)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Retained metadata must be finite UTF-8 JSON") from exc
        if size > self._limits["max_metadata_bytes"]:
            raise RuntimeError("Issue tracker metadata capacity exceeded")
        receipt = {"schema_version": 1, "session_id": self._session_id,
                   "reference_id": self._reference_id, **deepcopy(ledger)}
        self._issues, self._observations = proposed, observations
        self._metadata_bytes = size
        self._clock_s, self._latest_cutoff_s = completion, cutoff
        return receipt

    def snapshot(self, at_s):
        """Return detached current records and a validator-compatible identity index.

        first_observation_cutoff_s is a model observation boundary, not a gold
        physical-error onset. Closed issues and unmentioned issues stay retained;
        candidate/active/uncertain/resolved are runtime beliefs only. Full event
        history, notification history and past-time reconstruction remain outside
        this bounded register.
        """
        self._number(at_s)
        if self._clock_s is not None and at_s < self._clock_s:
            raise ValueError("Snapshot cannot precede the tracker clock")
        issues = [deepcopy(self._issues[identifier]) for identifier in sorted(self._issues)]
        prior = {"schema_version": 1, "session_id": self._session_id,
                 "reference_id": self._reference_id, "available_at_s": at_s,
                 "issues": [{"issue_id": issue["issue_id"], "state": issue["state"],
                             "last_observation_cutoff_s": issue["last_observation_cutoff_s"]}
                            for issue in issues]}
        result = {"schema_version": 1, "session_id": self._session_id,
                  "reference_id": self._reference_id, "at_s": at_s, "issues": issues,
                  "processed_observations": deepcopy(self._observations), "prior_issues": prior,
                  "latest_observation_cutoff_s": self._latest_cutoff_s,
                  "metadata_bytes": self._metadata_bytes, "limits": dict(self._limits)}
        self._clock_s = at_s
        return result
