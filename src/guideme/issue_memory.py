"""Track bounded issue episodes for host-bound assessments."""

from copy import deepcopy
import json
from uuid import uuid4

from .frame_binding import validate_tracker_binding
from .validation import require_keys, require_nonnegative_number, require_sha256, require_uuid


class AssessmentIssueTracker:
    """Apply explicit model proposals through one serialized coordinator.

    The caller pairs already validated decoder/binder outputs from the same
    frozen completion; the core reply has no observation identity of its own.
    Boundary checks here establish consistency, not visual association, truthful
    provenance or citation membership. No media, provider or notification work
    occurs. Select one tracker per session rather than writing both contracts.

    Source cutoffs and the declared application clock remain separate. Context
    becomes available at explicit application time, never HTTP completion time.
    Capacity includes closed issues and every accepted observation. Nothing is
    evicted; snapshots are detached current records, not historical queries.

    Optional two-step repair keeps the existing issue unchanged after its first
    explicit resolution proposal. Only another explicit proposal on a newer sent
    source frame closes it. Pending witnesses consume metadata capacity and are
    audited in observation receipts; repeated agreement is not visual proof.
    """

    def __init__(self, session_id, reference_id, execution_source_sha256, *,
                 clock_id, timing_basis, max_issues, max_observations, max_metadata_bytes,
                 repair_confirmation_count=1, max_pending_observations=128):
        self._uuid(session_id)
        self._digest(reference_id)
        self._digest(execution_source_sha256)
        self._text(clock_id)
        if timing_basis not in ("ideal_content_cutoff", "session_monotonic"):
            raise ValueError("Unsupported timing basis")
        if type(repair_confirmation_count) is not int or repair_confirmation_count not in (1, 2):
            raise ValueError("Repair confirmation count must be 1 or 2")
        if type(max_pending_observations) is not int or max_pending_observations <= 0:
            raise ValueError("Pending observation capacity must be a positive integer")
        limits = {"max_issues": max_issues, "max_observations": max_observations,
                  "max_metadata_bytes": max_metadata_bytes}
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("Capacity limits must be positive integers")
        self._session_id, self._reference_id = session_id, reference_id
        self._source, self._clock_id, self._basis = execution_source_sha256, clock_id, timing_basis
        self._limits, self._issues, self._observations = limits, {}, []
        self._revision, self._metadata_bytes = 0, 0
        self._context_available = self._latest_cutoff = self._last_applied = None
        self._playback_identity = None
        self._repair_confirmation_count = repair_confirmation_count
        self._pending_repairs = {}
        self._resolution_conditions = {}
        self._pending_observations = {}
        self._max_pending_observations = max_pending_observations
        self._contract_mode = None

    _keys = staticmethod(require_keys)

    _number = staticmethod(require_nonnegative_number)

    _uuid = staticmethod(require_uuid)
    _digest = staticmethod(require_sha256)

    @staticmethod
    def _text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected nonblank UTF-8 text")
        value.encode("utf-8")

    @staticmethod
    def _version(record):
        if type(record["assessment_contract_version"]) is not int or record["assessment_contract_version"] != 2:
            raise ValueError("Expected assessment contract 2")

    def _validate_citations(self, diagnostics, targets):
        """Validate one optional localization diagnostic per model claim."""
        if not isinstance(diagnostics, list) or len(diagnostics) != len(targets):
            raise ValueError("Every claim requires one citation diagnostic")
        for diagnostic, (target, index) in zip(diagnostics, targets):
            self._keys(diagnostic, ("target", "index", "status", "reason", "citations"))
            if diagnostic["target"] != target or type(diagnostic["index"]) is not int or diagnostic["index"] != index:
                raise ValueError("Citation diagnostic addresses the wrong claim")
            status, reason, citations = diagnostic["status"], diagnostic["reason"], diagnostic["citations"]
            if status == "absent" and reason is None and citations is None:
                continue
            if status == "invalid" and reason in ("invalid_shape", "invalid_timestamp", "unregistered_timestamp") and citations is None:
                continue
            if status != "valid" or reason is not None:
                raise ValueError("Invalid citation diagnostic")
            self._keys(citations, ("execution_s", "reference_s"))
            if any(not isinstance(values, list) for values in citations.values()) or not any(citations.values()):
                raise ValueError("Valid localization requires nonempty numeric lists")
            for values in citations.values():
                for value in values:
                    self._number(value)

    def _binding(self, binding, applied):
        return validate_tracker_binding(binding, applied, session_id=self._session_id,
            reference_id=self._reference_id, execution_source_sha256=self._source,
            clock_id=self._clock_id, timing_basis=self._basis, playback_identity=self._playback_identity)

    def _assessment(self, assessment):
        self._keys(assessment, ("assessment_contract_version", "status", "issues", "updates", "citation_diagnostics"))
        self._version(assessment)
        status, issues, updates = assessment["status"], assessment["issues"], assessment["updates"]
        if status not in ("ok", "insufficient_evidence") or not isinstance(issues, list) or not isinstance(updates, list):
            raise ValueError("Unsupported assessment status or claim lists")
        if status == "insufficient_evidence" and issues:
            raise ValueError("Insufficient evidence cannot support positive issues")
        claimed = set()

        def claim(identifier):
            self._uuid(identifier)
            if identifier not in self._issues or self._issues[identifier]["state"] == "resolved" or identifier in claimed:
                raise ValueError("Issue identity is unknown, closed or claimed more than once")
            claimed.add(identifier)

        for issue in issues:
            fields = ("description", "link")
            self._keys(issue, fields + (("resolution_condition",) if isinstance(issue, dict)
                                      and "resolution_condition" in issue else ()))
            self._text(issue["description"])
            if "resolution_condition" in issue:
                self._text(issue["resolution_condition"])
            if issue["link"] is not None and issue["link"] != "new":
                claim(issue["link"])
        for update in updates:
            self._keys(update, ("issue_id", "state"))
            claim(update["issue_id"])
            if update["state"] not in ("resolved", "uncertain") or status == "insufficient_evidence" and update["state"] != "uncertain":
                raise ValueError("Unsupported explicit issue transition")
        targets = [(target, index) for target, rows in (("issues", issues), ("updates", updates))
                   for index in range(len(rows))]
        self._validate_citations(assessment["citation_diagnostics"], targets)
        return claimed

    def apply_completed(self, assessment, binding, *, prior_revision,
                        new_issue_candidate_indices, applied_at_s,
                        pending_operations=None, current_observations=None,
                        pending_clear_sightings=None):
        """Atomically apply explicit proposals; silence never refreshes an issue."""
        if self._contract_mode == 3:
            raise ValueError("Cannot mix assessment contracts in one tracker")
        applied = self._number(applied_at_s)
        playback_identity = self._binding(binding, applied)
        claimed = self._assessment(assessment)
        if type(prior_revision) is not int or prior_revision != self._revision:
            raise ValueError("Dispatch context revision is stale or invalid")
        if self._context_available is not None and self._context_available > binding["dispatched_at_s"]:
            raise ValueError("Issue context was unavailable at dispatch")
        if self._last_applied is not None and applied < self._last_applied:
            raise ValueError("Application clock cannot reverse")
        cutoff, observation_id = binding["observation_cutoff_s"], binding["observation_id"]
        if self._latest_cutoff is not None and cutoff <= self._latest_cutoff:
            raise ValueError("Applied source cutoffs must strictly increase")
        if any(row["observation_id"] == observation_id for row in self._observations):
            raise ValueError("Observation was already applied")
        issues, updates = assessment["issues"], assessment["updates"]
        if not isinstance(new_issue_candidate_indices, list):
            raise ValueError("Admissions must be an explicit list")
        admitted = set()
        for index in new_issue_candidate_indices:
            if (type(index) is not int or not 0 <= index < len(issues) or index in admitted
                    or issues[index]["link"] != "new"):
                raise ValueError("Admit only unique indices explicitly proposed as new")
            admitted.add(index)
        if (len(self._issues) + len(admitted) > self._limits["max_issues"]
                or len(self._observations) >= self._limits["max_observations"]):
            raise RuntimeError("Issue or observation capacity exceeded")
        proposed, created = deepcopy(self._issues), []
        pending, repair_decisions = deepcopy(self._pending_repairs), []
        conditions = self._resolution_conditions.copy()
        candidates = deepcopy(self._pending_observations)
        pending_decisions = []
        if (pending_operations is None) != (current_observations is None):
            raise ValueError("Pending operations and current observations must be supplied together")
        if pending_operations is None and pending_clear_sightings is not None:
            raise ValueError("Clear sightings require pending operations")
        if pending_operations is not None:
            self._keys(current_observations, ("schema_version", "status", "observations"))
            if (type(current_observations["schema_version"]) is not int
                    or current_observations["schema_version"] != 1
                    or current_observations["status"] not in ("ok", "insufficient_evidence")
                    or not isinstance(current_observations["observations"], list)
                    or current_observations["status"] == "insufficient_evidence"
                    and current_observations["observations"]):
                raise ValueError("Invalid current observations")
            if (current_observations["status"] == "insufficient_evidence"
                    and assessment["status"] != "insufficient_evidence"):
                raise ValueError("Insufficient observations cannot support a positive assessment")
            current = {}
            for row in current_observations["observations"]:
                fields = ("id", "description")
                self._keys(row, fields + (("resolution_condition",) if isinstance(row, dict)
                                          and "resolution_condition" in row else ()))
                identifier = row["id"]
                if type(identifier) is not int or identifier < 0 or identifier in current:
                    raise ValueError("Current observation IDs must be unique nonnegative integers")
                self._text(row["description"])
                if "resolution_condition" in row:
                    self._text(row["resolution_condition"])
                current[identifier] = row
            if not isinstance(pending_operations, list):
                raise ValueError("Pending operations must be a list")
            sightings = {} if pending_clear_sightings is None else pending_clear_sightings
            if not isinstance(sightings, dict):
                raise ValueError("Clear sightings must be an explicit mapping")
            clear_ids = [op.get("pending_id") for op in pending_operations
                         if isinstance(op, dict) and op.get("action") == "clear"]
            if (any(not isinstance(identifier, str) for identifier in clear_ids)
                    or len(clear_ids) != len(set(clear_ids)) or set(sightings) != set(clear_ids)):
                raise ValueError("Every clear requires one matching current sighting")
            seen_current, seen_pending, seen_issue_indices = set(), set(), set()
            for operation in pending_operations:
                if not isinstance(operation, dict):
                    raise ValueError("Invalid pending operation")
                action = operation.get("action")
                if action == "create":
                    self._keys(operation, ("action", "current_id"))
                elif action == "refresh":
                    self._keys(operation, ("action", "current_id", "pending_id"))
                elif action == "clear":
                    self._keys(operation, ("action", "pending_id"))
                else:
                    raise ValueError("Unknown pending operation")
                if action in ("create", "refresh"):
                    current_id = operation["current_id"]
                    if type(current_id) is not int or current_id not in current or current_id in seen_current:
                        raise ValueError("Unknown or duplicate current observation")
                    seen_current.add(current_id)
                if action in ("refresh", "clear"):
                    identifier = operation["pending_id"]
                    if not isinstance(identifier, str) or identifier not in self._pending_observations or identifier in seen_pending:
                        raise ValueError("Unknown or duplicate pending observation")
                    seen_pending.add(identifier)
                if action == "clear":
                    sighting = sightings[identifier]
                    self._keys(sighting, ("current_id", "issue_index"))
                    current_id, issue_index = sighting["current_id"], sighting["issue_index"]
                    if (current_observations["status"] != "ok" or assessment["status"] != "ok"
                            or type(current_id) is not int or current_id not in current or current_id in seen_current
                            or type(issue_index) is not int or not 0 <= issue_index < len(issues)
                            or issue_index in seen_issue_indices):
                        raise ValueError("Clear requires unique current evidence and issue claim")
                    claim, source = issues[issue_index], current[current_id]
                    if (claim["link"] is None or claim["link"] == "new" and issue_index not in admitted
                            or claim["description"] != source["description"]
                            or ("resolution_condition" in claim) != ("resolution_condition" in source)
                            or "resolution_condition" in source and claim["resolution_condition"] != source["resolution_condition"]):
                        raise ValueError("Clear sighting must match an applied issue claim")
                    seen_current.add(current_id)
                    seen_issue_indices.add(issue_index)
                if action == "create":
                    identifier = str(uuid4())
                    if identifier in candidates or identifier in self._pending_observations or identifier in self._issues:
                        raise RuntimeError("Pending observation identity collision")
                    source = current[current_id]
                    candidates[identifier] = {"description": source["description"],
                        "latest_description": source["description"],
                        "first_binding": deepcopy(binding), "latest_binding": deepcopy(binding),
                        "first_available_at_s": applied, "latest_available_at_s": applied}
                    if "resolution_condition" in source:
                        candidates[identifier]["resolution_condition"] = source["resolution_condition"]
                    pending_decisions.append({"action": "create", "pending_id": identifier, "current_id": current_id})
                elif action == "refresh":
                    candidates[identifier].update(latest_description=current[current_id]["description"],
                        latest_binding=deepcopy(binding), latest_available_at_s=applied)
                    pending_decisions.append(deepcopy(operation))
                else:
                    del candidates[identifier]
                    pending_decisions.append({**deepcopy(operation), **deepcopy(sightings[identifier])})
            if len(candidates) > self._max_pending_observations:
                raise RuntimeError("Pending observation capacity exceeded")

        def repair_decision(identifier, decision, proposal):
            repair_decisions.append({"issue_id": identifier, "decision": decision,
                                     "proposal": deepcopy(proposal)})

        diagnostics = assessment["citation_diagnostics"]
        for index in sorted(admitted):
            identifier = str(uuid4())
            if identifier in proposed or identifier in candidates or identifier in self._pending_observations:
                raise RuntimeError("Runtime issue identity collision")
            description = issues[index]["description"]
            if "resolution_condition" in issues[index]:
                conditions[identifier] = issues[index]["resolution_condition"]
            proposed[identifier] = {"issue_id": identifier, "state": "candidate",
                "initial_description": description, "latest_description": description,
                "first_binding": deepcopy(binding), "latest_binding": deepcopy(binding),
                "first_available_at_s": applied, "latest_available_at_s": applied,
                "latest_citation_diagnostic": deepcopy(diagnostics[index])}
            created.append({"candidate_index": index, "issue_id": identifier})
        for index, issue in enumerate(issues):
            identifier = issue["link"]
            if identifier is not None and identifier != "new":
                previous = pending.pop(identifier, None)
                if previous is not None:
                    repair_decision(identifier, "cancelled_by_continuation", previous)
                proposed[identifier].update(state="active", latest_description=issue["description"],
                    latest_binding=deepcopy(binding), latest_available_at_s=applied,
                    latest_citation_diagnostic=deepcopy(diagnostics[index]))
        for index, update in enumerate(updates):
            identifier = update["issue_id"]
            if self._repair_confirmation_count == 2:
                previous = pending.get(identifier)
                if update["state"] == "resolved":
                    frame = binding["latest_execution_frame"]
                    if previous is None:
                        proposal = {"observation_id": observation_id, "request_sha256": binding["request_sha256"],
                            "observation_cutoff_s": cutoff, "frame_index": frame["frame_index"],
                            "elapsed_s": frame["elapsed_s"]}
                        pending[identifier] = proposal
                        repair_decision(identifier, "pending", proposal)
                        continue
                    if (cutoff <= previous["observation_cutoff_s"]
                            or frame["frame_index"] <= previous["frame_index"]
                            or frame["elapsed_s"] <= previous["elapsed_s"]):
                        repair_decision(identifier, "waiting_for_new_frame", previous)
                        continue
                    del pending[identifier]
                    repair_decision(identifier, "confirmed", previous)
                elif previous is not None:
                    del pending[identifier]
                    repair_decision(identifier, "cancelled_by_uncertainty", previous)
            proposed[identifier].update(state=update["state"], latest_binding=deepcopy(binding),
                latest_available_at_s=applied,
                latest_citation_diagnostic=deepcopy(diagnostics[len(issues) + index]))
        changed = bool(admitted or claimed or pending_decisions)
        revision = self._revision + int(changed)
        ledger = {"observation_id": observation_id, "observation_cutoff_s": cutoff,
            "dispatched_at_s": binding["dispatched_at_s"], "completed_at_s": binding["completed_at_s"],
            "applied_at_s": applied, "prior_revision": prior_revision, "revision": revision,
            "created_issues": created, "unadmitted_candidate_indices": [index for index, issue in enumerate(issues)
                if issue["link"] is None or issue["link"] == "new" and index not in admitted]}
        if self._repair_confirmation_count == 2:
            ledger["repair_decisions"] = repair_decisions
        if pending_operations is not None:
            ledger["pending_decisions"] = pending_decisions
        observations = self._observations + [ledger]
        try:
            size = sum(len(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8"))
                       for record in list(proposed.values()) + observations
                       + ([pending] if self._repair_confirmation_count == 2 else [])
                       + ([conditions] if conditions else [])
                       + ([candidates] if candidates else []))
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Retained metadata must be finite UTF-8 JSON") from exc
        if size > self._limits["max_metadata_bytes"]:
            raise RuntimeError("Metadata capacity exceeded")
        receipt = {"assessment_contract_version": 2, **deepcopy(ledger)}
        self._issues, self._observations, self._metadata_bytes = proposed, observations, size
        self._pending_repairs = pending
        self._resolution_conditions = conditions
        self._pending_observations = candidates
        self._revision, self._latest_cutoff, self._last_applied = revision, cutoff, applied
        self._contract_mode = 2
        if playback_identity is not None:
            self._playback_identity = playback_identity
        if changed:
            self._context_available = applied
        return receipt

    def resolution_conditions(self):
        """Copy model-proposed correction conditions, pinned when issues are admitted."""
        return self._resolution_conditions.copy()

    def issue_context(self, *, observation_id, manifest_hash, source_cutoff_s,
                   max_resolved_history, max_context_bytes):
        """Project this tracker's retained state into a bounded assessment request."""
        from .assessment import build_issue_context
        if self._contract_mode == 2:
            raise ValueError("Cannot mix assessment contracts in one tracker")
        opened, history = [], []
        for issue in self._issues.values():
            first = issue["first_binding"]["observation_cutoff_s"]
            last = issue["latest_binding"]["observation_cutoff_s"]
            if issue["state"] == "resolved":
                history.append({"issue_id": issue["issue_id"], "description": issue["latest_description"],
                    "outcome": issue["terminal_outcome"], "first_assessment_cutoff_s": first,
                    "terminal_assessment_cutoff_s": last})
            else:
                opened.append({"issue_id": issue["issue_id"], "state": issue["state"],
                    "initial_description": issue["initial_description"], "latest_description": issue["latest_description"],
                    "first_assessment_cutoff_s": first, "latest_assessment_cutoff_s": last,
                    "pending_repair": issue["issue_id"] in self._pending_repairs,
                    "resolution_condition": self._resolution_conditions.get(issue["issue_id"])})
        pending = [{"pending_id": identifier, "description": row["latest_description"],
                    "first_assessment_cutoff_s": row["first_binding"]["observation_cutoff_s"],
                    "latest_assessment_cutoff_s": row["latest_binding"]["observation_cutoff_s"],
                    "recurrence_of": row.get("recurrence_of")}
                   for identifier, row in self._pending_observations.items()]
        return build_issue_context(session_id=self._session_id, reference_id=self._reference_id,
            observation_id=observation_id, manifest_hash=manifest_hash, source_cutoff_s=source_cutoff_s,
            context_revision=self._revision, open_issues=opened, resolved_history=history,
            pending_observations=pending, max_open_issues=self._limits["max_issues"],
            max_pending_observations=self._max_pending_observations,
            max_resolved_history=max_resolved_history, max_context_bytes=max_context_bytes)

    def issue_snapshot(self):
        """Copy host records and their clocks without changing state."""
        if self._contract_mode == 2:
            raise ValueError("Cannot reinterpret contract 2 state as contract 3")
        return deepcopy({"assessment_contract_version": 3, "session_id": self._session_id,
            "reference_id": self._reference_id, "execution_source_sha256": self._source,
            "clock_id": self._clock_id, "timing_basis": self._basis, "revision": self._revision,
            "context_available_at_s": self._context_available, "latest_applied_source_cutoff_s": self._latest_cutoff,
            "last_applied_at_s": self._last_applied, "issues": list(self._issues.values()),
            "processed_observations": self._observations, "metadata_bytes": self._metadata_bytes,
            "limits": self._limits, "pending_observations": self._pending_observations,
            "pending_repairs": self._pending_repairs})

    def apply_assessment(self, assessment, binding, *, context, applied_at_s):
        """Apply a decoded assessment atomically against its frozen host context.

        The caller must supply outputs of the existing binder and assessment decoder for
        the same request. Checks here cannot authenticate the caller's provenance
        or establish whether a visual judgment is true.
        """
        from .assessment import decode_assessment
        if self._contract_mode == 2:
            raise ValueError("Cannot mix assessment contracts in one tracker")
        applied = self._number(applied_at_s)
        playback_identity = self._binding(binding, applied)
        if not isinstance(context, dict):
            raise ValueError("Missing frozen context")
        try:
            expected = self.issue_context(observation_id=binding["observation_id"],
                manifest_hash=binding["observation_manifest_sha256"], source_cutoff_s=binding["observation_cutoff_s"],
                max_resolved_history=len(context["resolved_history"]), max_context_bytes=self._limits["max_metadata_bytes"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid frozen context") from exc
        try:
            actual_json = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
            expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Invalid frozen context") from exc
        if actual_json != expected_json:
            raise ValueError("Frozen context does not match tracker and binding")
        if self._context_available is not None and self._context_available > binding["dispatched_at_s"]:
            raise ValueError("Issue context was unavailable at dispatch")
        cutoff, observation_id = binding["observation_cutoff_s"], binding["observation_id"]
        if ((self._latest_cutoff is not None and cutoff <= self._latest_cutoff)
                or (self._last_applied is not None and applied < self._last_applied)
                or any(row["observation_id"] == observation_id for row in self._observations)):
            raise ValueError("Stale or repeated observation")
        self._keys(assessment, ("assessment_contract_version", "status", "observations", "updates",
                                "pending_updates", "citation_diagnostics"))
        core = {k: deepcopy(v) for k, v in assessment.items() if k != "citation_diagnostics"}
        try:
            decoded = decode_assessment(json.dumps(core, allow_nan=False), context=context,
                frame_registry={"execution_s": [binding["latest_execution_frame"]["elapsed_s"]],
                                "reference_s": [0]}, max_response_bytes=self._limits["max_metadata_bytes"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid assessment") from exc
        if any(decoded[k] != core[k] for k in core):
            raise ValueError("Invalid decoded assessment")
        diagnostics = assessment["citation_diagnostics"]
        targets = [(name, i) for name in ("observations", "updates") for i in range(len(core[name]))]
        self._validate_citations(diagnostics, targets)
        issues, pending, repairs = deepcopy(self._issues), deepcopy(self._pending_observations), deepcopy(self._pending_repairs)
        conditions = self._resolution_conditions.copy()
        created, pending_decisions, repair_decisions, deferred = [], [], [], []
        frame = binding["latest_execution_frame"]
        frame_time = frame["elapsed_s"]

        def new_id():
            identifier = str(uuid4())
            if identifier in issues or identifier in pending:
                raise RuntimeError("Host identity collision")
            return identifier

        def refresh_pending(identifier, claim, recurrence_of):
            if identifier is None:
                identifier = new_id()
                pending[identifier] = {"description": claim["description"], "first_binding": deepcopy(binding),
                                       "first_available_at_s": applied}
                action = "create"
            else:
                action = "refresh"
            pending[identifier].update(latest_description=claim["description"], latest_binding=deepcopy(binding),
                                       latest_available_at_s=applied, recurrence_of=recurrence_of)
            pending_decisions.append({"action": action, "pending_id": identifier})

        for index, claim in enumerate(core["observations"]):
            association = claim["association"]
            kind, identifier, pending_id = (association[k] for k in ("kind", "issue_id", "pending_id"))
            if kind == "uncertain":
                lineage = pending[pending_id].get("recurrence_of") if pending_id is not None else None
                refresh_pending(pending_id, claim, lineage)
                continue
            if kind == "recurrence" and frame_time <= issues[identifier]["latest_binding"]["observation_cutoff_s"]:
                refresh_pending(pending_id, claim, identifier)
                pending_decisions[-1]["reason"] = "recurrence_needs_newer_frame"
                continue
            if pending_id is not None:
                prior_pending = pending.pop(pending_id)
                pending_decisions.append({"action": "adopt", "pending_id": pending_id,
                                          "prior_pending": prior_pending})
            if kind == "same_open":
                repairs.pop(identifier, None)
                issues[identifier].update(state="active", latest_description=claim["description"],
                    latest_binding=deepcopy(binding), latest_available_at_s=applied,
                    latest_citation_diagnostic=deepcopy(diagnostics[index]))
            else:
                prior = identifier if kind == "recurrence" else None
                identifier = new_id()
                issues[identifier] = {"issue_id": identifier, "state": "candidate",
                    "initial_description": claim["description"], "latest_description": claim["description"],
                    "first_binding": deepcopy(binding), "latest_binding": deepcopy(binding),
                    "first_available_at_s": applied, "latest_available_at_s": applied,
                    "latest_citation_diagnostic": deepcopy(diagnostics[index]),
                    "terminal_outcome": None, "recurrence_of": prior}
                if "resolution_condition" in claim:
                    conditions[identifier] = claim["resolution_condition"]
                created.append({"observation_index": index, "issue_id": identifier})
        for index, update in enumerate(core["updates"]):
            identifier, action = update["issue_id"], update["action"]
            issue = issues[identifier]
            if action in ("corrected", "retract") and frame_time <= issue["latest_binding"]["observation_cutoff_s"]:
                deferred.append({"issue_id": identifier, "action": action, "reason": "needs_newer_frame"})
                continue
            if action == "corrected" and self._repair_confirmation_count == 2:
                previous = repairs.get(identifier)
                if previous is None:
                    repairs[identifier] = {"observation_id": observation_id, "request_sha256": binding["request_sha256"],
                        "observation_cutoff_s": cutoff, "frame_index": frame["frame_index"], "elapsed_s": frame_time}
                    repair_decisions.append({"issue_id": identifier, "decision": "pending"})
                    continue
                if cutoff <= previous["observation_cutoff_s"] or frame_time <= previous["elapsed_s"] or frame["frame_index"] <= previous["frame_index"]:
                    repair_decisions.append({"issue_id": identifier, "decision": "waiting_for_new_frame"})
                    continue
                repair_decisions.append({"issue_id": identifier, "decision": "confirmed"})
            repairs.pop(identifier, None)
            issue.update(state="uncertain" if action == "uncertain" else "resolved",
                terminal_outcome=None if action == "uncertain" else ("corrected" if action == "corrected" else "retracted"),
                latest_binding=deepcopy(binding), latest_available_at_s=applied,
                latest_citation_diagnostic=deepcopy(diagnostics[len(core["observations"]) + index]))
        for update in core["pending_updates"]:
            identifier = update["pending_id"]
            if frame_time <= pending[identifier]["latest_binding"]["observation_cutoff_s"]:
                pending_decisions.append({"action": "defer_discard", "pending_id": identifier})
            else:
                prior_pending = pending.pop(identifier)
                pending_decisions.append({**deepcopy(update), "prior_pending": prior_pending})
        if len(issues) > self._limits["max_issues"] or len(pending) > self._max_pending_observations or len(self._observations) >= self._limits["max_observations"]:
            raise RuntimeError("Issue or observation capacity exceeded")
        changed = issues != self._issues or pending != self._pending_observations or repairs != self._pending_repairs
        revision = self._revision + int(changed)
        ledger = {"observation_id": observation_id, "observation_cutoff_s": cutoff,
            "dispatched_at_s": binding["dispatched_at_s"], "completed_at_s": binding["completed_at_s"],
            "applied_at_s": applied, "prior_revision": self._revision, "revision": revision,
            "created_issues": created, "pending_decisions": pending_decisions,
            "repair_decisions": repair_decisions, "deferred_updates": deferred}
        observations = self._observations + [ledger]
        try:
            size = sum(len(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                      allow_nan=False).encode("utf-8"))
                       for record in list(issues.values()) + observations + [pending, repairs, conditions])
        except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
            raise ValueError("Retained metadata must be finite UTF-8 JSON") from exc
        if size > self._limits["max_metadata_bytes"]:
            raise RuntimeError("Metadata capacity exceeded")
        self._issues, self._pending_observations, self._pending_repairs = issues, pending, repairs
        self._resolution_conditions, self._observations, self._metadata_bytes = conditions, observations, size
        self._revision, self._latest_cutoff, self._last_applied = revision, cutoff, applied
        self._contract_mode = 3
        if playback_identity is not None:
            self._playback_identity = playback_identity
        if changed:
            self._context_available = applied
        return {"assessment_contract_version": 3, **deepcopy(ledger)}

    def pending_observations(self):
        """Return detached, bounded observations awaiting association."""
        return deepcopy(self._pending_observations)

    def pending_repairs(self):
        """Copy retained first repair proposals without changing state or clocks.

        These bounded witnesses record model judgments, not established visible
        corrections. The coordinator may expose pending membership separately
        from the issue state; callers cannot mutate the retained proposals.
        """
        return deepcopy(self._pending_repairs)

    def snapshot(self):
        """Copy current state without reading or advancing any clock."""
        if self._contract_mode == 3:
            raise ValueError("Cannot reinterpret contract 3 state as contract 2")
        return deepcopy({"assessment_contract_version": 2, "session_id": self._session_id,
            "reference_id": self._reference_id, "execution_source_sha256": self._source,
            "clock_id": self._clock_id, "timing_basis": self._basis, "revision": self._revision,
            "context_available_at_s": self._context_available, "latest_applied_source_cutoff_s": self._latest_cutoff,
            "last_applied_at_s": self._last_applied, "issues": list(self._issues.values()),
            "processed_observations": self._observations, "metadata_bytes": self._metadata_bytes, "limits": self._limits})
