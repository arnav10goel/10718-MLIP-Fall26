"""Serialize one two-stage visual assessment and apply it atomically."""

from copy import deepcopy
import math
from threading import Lock

from .frame_binding import build_assessment_binding


class AssessmentPairCoordinator:
    """Coordinate a single session's stages using one provider and tracker.

    The supplied clock and clock context must use the tracker's timing basis.
    One owner must exclusively coordinate admissions through this provider and
    tracker; the local lock prevents overlapping calls through this instance.
    Request/provenance checks establish input identity, not visual correctness.
    Notifications, retries, replay scheduling and session-end handling belong
    to the caller. Transport/accounting failures never apply stage-one claims.
    """

    def __init__(self, provider, tracker, *, clock):
        if not callable(clock):
            raise ValueError("A coordinator clock is required")
        self._provider, self._tracker, self._clock = provider, tracker, clock
        self._lock = Lock()
        self._split = None

    @property
    def stage(self):
        return "idle" if self._split is None else self._split["stage"]

    def _end_split(self):
        self._split = None
        self._lock.release()

    def begin(self, observation, selected_indices, *, clock_context, prepared=None, expected_revision=None):
        """Freeze a pair on its owner thread, without admitting transport."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("An assessment pair is already in progress")
        try:
            observation, selection, context = deepcopy(observation), deepcopy(selected_indices), deepcopy(clock_context)
            snapshot = self._tracker.snapshot()
            revision = snapshot["revision"]
            if expected_revision is not None and (type(expected_revision) is not int or expected_revision != revision):
                raise RuntimeError("Issue context changed before pair")
            repairs, conditions, pending = (self._tracker.pending_repairs(),
                self._tracker.resolution_conditions(), self._tracker.pending_observations())
            if self._tracker.snapshot()["revision"] != revision:
                raise RuntimeError("Issue context changed while freezing the pair")
            issues = [{"issue_id": row["issue_id"], "state": row["state"],
                "initial_description": row["initial_description"], "latest_description": row["latest_description"],
                "repair_pending": row["issue_id"] in repairs,
                **({"resolution_condition": conditions[row["issue_id"]]} if row["issue_id"] in conditions else {})}
                for row in snapshot["issues"] if row["state"] != "resolved"]
            compact_pending = {}
            for identifier, row in pending.items():
                first, latest = row["first_binding"], row["latest_binding"]
                compact_pending[identifier] = {"description": row["description"],
                    "first_observation_id": first["observation_id"], "first_cutoff_s": first["observation_cutoff_s"],
                    "latest_observation_id": latest["observation_id"], "latest_cutoff_s": latest["observation_cutoff_s"],
                    **({"resolution_condition": row["resolution_condition"]} if "resolution_condition" in row else {})}
            expected = self._provider.prepare(observation["model_input"], None, observation=observation,
                selected_indices=selection, stage="observation")
            descriptor = self._provider.describe_prepared(expected)
            if prepared is not None:
                if self._provider.describe_prepared(prepared) != descriptor:
                    raise ValueError("Prepared observation differs from frozen evidence")
            else:
                prepared = expected
            now = self._clock()
            binding = build_assessment_binding(observation, jpeg_provenance=descriptor["jpeg_provenance"],
                request_sha256=descriptor["request_sha256"], dispatched_at_s=now, completed_at_s=now,
                clock_context=context)
            if (binding["session_id"] != snapshot["session_id"] or binding["reference_id"] != snapshot["reference_id"]
                    or binding["clock_id"] != snapshot["clock_id"] or binding["timing_basis"] != snapshot["timing_basis"]
                    or binding["latest_execution_frame"]["source_sha256"] != snapshot["execution_source_sha256"]
                    or snapshot["context_available_at_s"] is not None and snapshot["context_available_at_s"] > now):
                raise ValueError("Observation, tracker and dispatch clock disagree")
            self._split = {"stage": "observation", "observation": observation, "selection": selection,
                "clock_context": context, "snapshot": snapshot, "revision": revision, "issues": issues,
                "pending": compact_pending, "prepared": prepared, "descriptor": descriptor,
                "first_descriptor": deepcopy(descriptor), "ticket": None, "first": None, "last_dispatch": None}
            return deepcopy(descriptor)
        except BaseException:
            self._lock.release()
            raise

    def admit(self):
        """Admit current stage on the owner; caller transports the returned ticket."""
        state = self._split
        if state is None or state["ticket"] is not None:
            raise RuntimeError("Pair is not ready for admission")
        try:
            if self._tracker.snapshot()["revision"] != state["revision"]:
                raise RuntimeError("Issue context changed before dispatch")
            dispatched = self._clock()
            if (type(dispatched) not in (int, float) or not math.isfinite(dispatched) or dispatched < 0
                    or state["last_dispatch"] is not None and dispatched < state["last_dispatch"]):
                raise ValueError("Coordinator clock reversed before dispatch")
            ticket = self._provider.admit(state["prepared"])
            state["ticket"], state["last_dispatch"] = ticket, dispatched
            return ticket
        except BaseException:
            self._end_split()
            raise

    def complete(self, ticket, outcome, *, allow_continue=True, allow_apply=True, is_active=None):
        """Settle transport and advance or apply while the session stays active.

        Browser callers must hold their session guard across this method; the
        callback also catches a stop invoked reentrantly during local work.
        """
        if is_active is not None and not callable(is_active):
            raise ValueError("Active-session check must be callable")
        state = self._split
        if state is None or state["ticket"] is not ticket:
            raise ValueError("Ticket does not identify the current pair")
        try:
            result = self._provider.complete(ticket, outcome)
            state["ticket"] = None
            descriptor = state["descriptor"]
            if (result["outcome"] != "completed" or result["error"] is not None):
                raise RuntimeError("Assessment stage failed")
            if (result["observation_id"] != descriptor["observation_id"]
                    or result["request_sha256"] != descriptor["request_sha256"]
                    or result["jpeg_provenance"] != descriptor["jpeg_provenance"]):
                raise ValueError("Completed request differs from prepared evidence")
            if state["stage"] == "observation":
                if not allow_continue or is_active is not None and not is_active():
                    self._end_split()
                    return {"status": "discarded"}
                if self._tracker.snapshot()["revision"] != state["revision"]:
                    raise RuntimeError("Issue context changed before association")
                context = {"assessment_contract_version": 2, "session_id": state["snapshot"]["session_id"],
                    "reference_id": state["snapshot"]["reference_id"], "issues": state["issues"],
                    "pending_observations": state["pending"], "observations": result["assessment"]}
                prepared = self._provider.prepare(state["observation"]["model_input"], context,
                    observation=state["observation"], selected_indices=state["selection"], stage="association")
                second = self._provider.describe_prepared(prepared)
                if (second["observation_id"] != state["first_descriptor"]["observation_id"]
                        or second["jpeg_provenance"] != state["first_descriptor"]["jpeg_provenance"]):
                    raise ValueError("Assessment stages must use the same selected frames")
                if is_active is not None and not is_active():
                    self._end_split()
                    return {"status": "discarded"}
                state.update(stage="association", prepared=prepared, descriptor=second, first=result)
                return {"status": "association_ready", "first": deepcopy(result)}
            if not allow_apply or is_active is not None and not is_active():
                self._end_split()
                return {"status": "discarded"}
            completed = self._clock()
            if self._tracker.snapshot()["revision"] != state["revision"]:
                raise RuntimeError("Issue context changed before application")
            binding = build_assessment_binding(state["observation"], jpeg_provenance=result["jpeg_provenance"],
                request_sha256=result["request_sha256"], dispatched_at_s=state["last_dispatch"],
                completed_at_s=completed, clock_context=state["clock_context"])
            combined, first = result["assessment"], state["first"]
            assessment = combined["assessment"]
            admissions = [index for index, row in enumerate(assessment["issues"]) if row["link"] == "new"]
            if is_active is not None and not is_active():
                self._end_split()
                return {"status": "discarded"}
            receipt = self._tracker.apply_completed(assessment, binding, prior_revision=state["revision"],
                new_issue_candidate_indices=admissions, applied_at_s=completed,
                pending_operations=combined["pending_operations"], current_observations=first["assessment"],
                pending_clear_sightings=combined["pending_clear_sightings"])
            output = {"first": deepcopy(first), "second": deepcopy(result), "binding": binding, "receipt": receipt}
            self._end_split()
            return {"status": "applied", "result": output}
        except BaseException:
            if self._split is state:
                self._end_split()
            raise

    def abort(self):
        """Drop an unsent pair; an admitted ticket must first be settled."""
        if self._split is None:
            return
        if self._split["ticket"] is not None:
            raise RuntimeError("Cannot abort an in-flight request")
        self._end_split()

    def transport_failed(self, ticket):
        """Release a provably unsent ticket; otherwise retain the provider hold."""
        if self._split is None or self._split["ticket"] is not ticket:
            raise ValueError("Ticket does not identify the current pair")
        try:
            return self._provider.cancel_unsent(ticket)
        finally:
            self._end_split()

    def run(self, observation, selected_indices, *, clock_context):
        """Assess one frozen JPEG observation and atomically apply the result."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("An assessment pair is already in progress")
        try:
            observation, selection = deepcopy(observation), deepcopy(selected_indices)
            clock_context = deepcopy(clock_context)
            snapshot = self._tracker.snapshot()
            revision = snapshot["revision"]
            repairs = self._tracker.pending_repairs()
            conditions = self._tracker.resolution_conditions()
            pending = self._tracker.pending_observations()
            if self._tracker.snapshot()["revision"] != revision:
                raise RuntimeError("Issue context changed while freezing the pair")
            issues = [{"issue_id": row["issue_id"], "state": row["state"],
                       "initial_description": row["initial_description"],
                       "latest_description": row["latest_description"],
                       "repair_pending": row["issue_id"] in repairs,
                       **({"resolution_condition": conditions[row["issue_id"]]}
                          if row["issue_id"] in conditions else {})}
                      for row in snapshot["issues"] if row["state"] != "resolved"]
            compact_pending = {}
            for identifier, row in pending.items():
                first, latest = row["first_binding"], row["latest_binding"]
                compact_pending[identifier] = {
                    "description": row["description"],
                    "first_observation_id": first["observation_id"],
                    "first_cutoff_s": first["observation_cutoff_s"],
                    "latest_observation_id": latest["observation_id"],
                    "latest_cutoff_s": latest["observation_cutoff_s"],
                    **({"resolution_condition": row["resolution_condition"]}
                       if "resolution_condition" in row else {})}

            def prepare(stage, prior):
                prepared = self._provider.prepare(observation["model_input"], prior,
                    observation=observation, selected_indices=selection, stage=stage)
                descriptor = self._provider.describe_prepared(prepared)
                if descriptor["observation_id"] != observation["observation_id"]:
                    raise ValueError("Prepared request identifies another observation")
                return prepared, descriptor

            def perform(prepared, descriptor):
                ticket = self._provider.admit(prepared)
                # Once admitted, transport owns the ticket. Complete every
                # returned outcome so usage or unknown holds are retained.
                outcome = self._provider.perform(ticket)
                result = self._provider.complete(ticket, outcome)
                if result["outcome"] != "completed" or result["error"] is not None:
                    raise RuntimeError("Assessment stage failed")
                if (result["observation_id"] != descriptor["observation_id"]
                        or result["request_sha256"] != descriptor["request_sha256"]
                        or result["jpeg_provenance"] != descriptor["jpeg_provenance"]):
                    raise ValueError("Completed request differs from prepared evidence")
                return result

            first_prepared, first_descriptor = prepare("observation", None)
            first_dispatched = self._clock()
            first_binding = build_assessment_binding(observation,
                jpeg_provenance=first_descriptor["jpeg_provenance"],
                request_sha256=first_descriptor["request_sha256"], dispatched_at_s=first_dispatched,
                completed_at_s=first_dispatched, clock_context=clock_context)
            if (first_binding["session_id"] != snapshot["session_id"]
                    or first_binding["reference_id"] != snapshot["reference_id"]
                    or first_binding["clock_id"] != snapshot["clock_id"]
                    or first_binding["timing_basis"] != snapshot["timing_basis"]
                    or first_binding["latest_execution_frame"]["source_sha256"] != snapshot["execution_source_sha256"]
                    or snapshot["context_available_at_s"] is not None
                    and snapshot["context_available_at_s"] > first_dispatched):
                raise ValueError("Observation, tracker and dispatch clock disagree")
            first = perform(first_prepared, first_descriptor)
            if self._tracker.snapshot()["revision"] != revision:
                raise RuntimeError("Issue context changed before association")
            context = {"assessment_contract_version": 2, "session_id": snapshot["session_id"],
                "reference_id": snapshot["reference_id"], "issues": issues,
                "pending_observations": compact_pending, "observations": first["assessment"]}
            second_prepared, second_descriptor = prepare("association", context)
            if (second_descriptor["observation_id"] != first_descriptor["observation_id"]
                    or second_descriptor["jpeg_provenance"] != first_descriptor["jpeg_provenance"]):
                raise ValueError("Assessment stages must use the same selected frames")
            if self._tracker.snapshot()["revision"] != revision:
                raise RuntimeError("Issue context changed before association dispatch")
            second_dispatched = self._clock()
            if (type(second_dispatched) not in (int, float) or not math.isfinite(second_dispatched)
                    or second_dispatched < first_dispatched):
                raise ValueError("Coordinator clock reversed before association")
            second = perform(second_prepared, second_descriptor)
            completed = self._clock()
            if self._tracker.snapshot()["revision"] != revision:
                raise RuntimeError("Issue context changed before application")
            binding = build_assessment_binding(observation, jpeg_provenance=second["jpeg_provenance"],
                request_sha256=second["request_sha256"], dispatched_at_s=second_dispatched,
                completed_at_s=completed, clock_context=clock_context)
            combined = second["assessment"]
            assessment = combined["assessment"]
            admissions = [index for index, row in enumerate(assessment["issues"]) if row["link"] == "new"]
            receipt = self._tracker.apply_completed(assessment, binding, prior_revision=revision,
                new_issue_candidate_indices=admissions, applied_at_s=completed,
                pending_operations=combined["pending_operations"], current_observations=first["assessment"],
                pending_clear_sightings=combined["pending_clear_sightings"])
            return {"first": deepcopy(first), "second": deepcopy(second),
                    "binding": binding, "receipt": receipt}
        finally:
            self._lock.release()
