"""Validate explicit runtime issue links and evidence-backed update records."""

from copy import deepcopy
import math
import re
from uuid import UUID


def validate_issue_updates(response, updates, *, observation, prior_issues, dispatched_at_s):
    """Check a proposed update envelope without inferring or applying changes.

    response is an upstream-validated normalized prediction. observation is the
    ready builder result; prior_issues is a caller-supplied runtime identity/
    causality index, not evaluator labels or a human-authored procedure. Their
    integrity and declared clock/availability mapping are caller responsibilities.
    Candidate evidence was already validated upstream; new transition evidence
    must identify actual supplied frames. Semantic correctness is not certified.

    This function neither supplies memory to the predictor nor associates issues
    visually. It mints no IDs, changes no state, displays nothing and performs no
    I/O. Null candidate links remain unassociated. Absence creates no implicit
    resolution. Invalid envelopes remain untouched for caller inspection/logging.
    """
    def keys(record, required, optional=()):
        if (not isinstance(record, dict) or not set(required) <= record.keys()
                or not record.keys() <= set(required) | set(optional)):
            raise ValueError("Record has missing or unsupported fields")

    def version(record):
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("Expected schema version 1")

    def number(value):
        if type(value) not in (int, float):
            raise ValueError("Expected a finite nonnegative number, not a boolean")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ValueError("Expected a finite nonnegative number")
        return value

    def uuid_string(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Expected a canonical UUID string")

    if not isinstance(observation, dict) or observation.get("status") != "ready":
        raise ValueError("Issue updates require a ready observation")
    version(observation)
    model = observation.get("model_input")
    keys(model, ("schema_version", "session_id", "observation_id", "cutoff_s",
                 "reference_frames", "execution_frames"))
    version(model)
    for name in ("session_id", "observation_id"):
        uuid_string(model[name])
        if observation.get(name) != model[name]:
            raise ValueError("Observation wrapper and model input identities disagree")
    session_id, observation_id = model["session_id"], model["observation_id"]
    reference_id = observation.get("reference_id")
    if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
        raise ValueError("Expected a lowercase reference SHA-256 identity")
    cutoff = number(model["cutoff_s"])
    if number(observation.get("cutoff_s")) != cutoff:
        raise ValueError("Observation wrapper and model input cutoffs disagree")
    dispatch = number(dispatched_at_s)
    if dispatch < cutoff:
        raise ValueError("Dispatch cannot precede the observation cutoff")

    frame_times = {}
    for name in ("reference_frames", "execution_frames"):
        frames = model[name]
        if not isinstance(frames, list) or not frames:
            raise ValueError("Both visual frame lists must be nonempty")
        times, previous = set(), None
        for frame in frames:
            keys(frame, ("timestamp_s", "mime_type", "jpeg_bytes"))
            timestamp = number(frame["timestamp_s"])
            if previous is not None and timestamp <= previous:
                raise ValueError("Visual timestamps must strictly increase")
            if name == "execution_frames" and timestamp > cutoff:
                raise ValueError("Execution frame follows the observation cutoff")
            if frame["mime_type"] != "image/jpeg" or type(frame["jpeg_bytes"]) is not bytes or not frame["jpeg_bytes"]:
                raise ValueError("Expected model-ready immutable JPEG frames")
            times.add(timestamp)
            previous = timestamp
        frame_times[name] = times

    keys(response, ("schema_version", "session_id", "observation_id", "status",
                    "candidate_issues", "error"))
    keys(updates, ("schema_version", "session_id", "observation_id", "reference_id",
                   "candidate_links", "issue_updates"))
    keys(prior_issues, ("schema_version", "session_id", "reference_id", "available_at_s", "issues"))
    for record in (response, updates, prior_issues):
        version(record)
        if record["session_id"] != session_id:
            raise ValueError("Record belongs to another session")
    for record in (response, updates):
        if record["observation_id"] != observation_id:
            raise ValueError("Record belongs to another observation")
    for record in (updates, prior_issues):
        if record["reference_id"] != reference_id:
            raise ValueError("Record belongs to another reference")
    status, candidates = response["status"], response["candidate_issues"]
    if status not in ("ok", "insufficient_evidence", "failed"):
        raise ValueError("Unsupported normalized response status")
    if not isinstance(candidates, list) or (status != "ok" and candidates):
        raise ValueError("Only ok predictions may contain candidates")
    # The upstream predictor validator owns candidate value/evidence validation.
    # Check their shape here so link indices always address normalized records.
    for candidate in candidates:
        keys(candidate, ("description", "execution_evidence_s", "reference_evidence_s"),
             ("confidence", "issue_key"))
    if status == "failed":
        keys(response["error"], ("type", "message"))
        if any(not isinstance(value, str) or not value.strip() for value in response["error"].values()):
            raise ValueError("Failure type and message must be nonblank strings")
    elif response["error"] is not None:
        raise ValueError("Only a failed prediction may contain an error")

    available = number(prior_issues["available_at_s"])
    if available > dispatch:
        raise ValueError("Prior issue context was unavailable at dispatch")
    if not isinstance(prior_issues["issues"], list):
        raise ValueError("Prior issues must be a list")
    known = {}
    for issue in prior_issues["issues"]:
        keys(issue, ("issue_id", "state", "last_observation_cutoff_s"))
        identifier = issue["issue_id"]
        uuid_string(identifier)
        if identifier in known:
            raise ValueError("Prior issue IDs must be unique")
        if issue["state"] not in ("candidate", "active", "uncertain", "resolved"):
            raise ValueError("Unsupported prior issue state")
        last_cutoff = number(issue["last_observation_cutoff_s"])
        if last_cutoff > available or last_cutoff > cutoff:
            raise ValueError("Prior context contains unavailable or newer observations")
        known[identifier] = issue

    def open_issue(identifier):
        uuid_string(identifier)
        if identifier not in known or known[identifier]["state"] == "resolved":
            raise ValueError("Issue ID is unknown or its episode is already closed")

    links, transitions = updates["candidate_links"], updates["issue_updates"]
    if not isinstance(links, list) or not isinstance(transitions, list):
        raise ValueError("Candidate links and issue updates must be lists")
    if status == "failed" and (links or transitions):
        raise ValueError("Failed predictions cannot propose issue changes")
    indices, linked_ids = set(), set()
    for link in links:
        keys(link, ("candidate_index", "issue_id"))
        index = link["candidate_index"]
        if type(index) is not int or not 0 <= index < len(candidates) or index in indices:
            raise ValueError("Each candidate must have one unique in-range link index")
        indices.add(index)
        identifier = link["issue_id"]
        if identifier is not None:
            open_issue(identifier)
            if identifier in linked_ids:
                raise ValueError("Multiple candidates cannot claim the same issue identity")
            linked_ids.add(identifier)
    if len(indices) != len(candidates):
        raise ValueError("Every candidate requires an explicit link or null association")
    updated_ids = set()
    for transition in transitions:
        keys(transition, ("issue_id", "state", "execution_evidence_s", "reference_evidence_s"))
        identifier = transition["issue_id"]
        open_issue(identifier)
        if identifier in updated_ids or identifier in linked_ids:
            raise ValueError("Conflicting or duplicate updates to an issue")
        updated_ids.add(identifier)
        state = transition["state"]
        if state not in ("resolved", "uncertain"):
            raise ValueError("Explicit issue transitions must be resolved or uncertain")
        if status == "insufficient_evidence" and state != "uncertain":
            raise ValueError("Insufficient evidence cannot resolve an issue")
        for field, source in (("execution_evidence_s", "execution_frames"),
                              ("reference_evidence_s", "reference_frames")):
            evidence = transition[field]
            if not isinstance(evidence, list) or (field == "execution_evidence_s" and not evidence):
                raise ValueError("Transition evidence must be lists with nonempty execution evidence")
            for timestamp in evidence:
                if number(timestamp) not in frame_times[source]:
                    raise ValueError("Transition evidence must identify an actual supplied frame")

    return {"schema_version": 1, "session_id": session_id, "observation_id": observation_id,
            "reference_id": reference_id, "observation_cutoff_s": cutoff,
            "dispatched_at_s": dispatched_at_s, "prior_available_at_s": available,
            "candidate_links": deepcopy(links), "issue_updates": deepcopy(transitions)}
