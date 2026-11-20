"""A stateless, scripted predictor for exercising the visual pipeline."""

from copy import deepcopy
import math

from .visual_input import validate_visual_input


def mock_predict(model_input, *, response=None):
    """Return insufficient evidence or a validated fictional prediction.

    Accept only build_observation's ready model_input, never its audit wrapper.
    A supplied response is a complete normalized fixture bound to this request.
    Evidence must cite exact supplied frame times; nothing is inferred, rounded
    or repaired. Multiple candidate issues are allowed; notification selection
    belongs to a separate policy. No state, I/O, model call or display is used.

    JPEG signatures are checked without decoding; upstream preparation and
    observation assembly own image integrity and capture/receipt eligibility.
    Invalid fixtures raise ValueError and remain untouched for the caller to
    inspect or record. This helper does not persist raw predictions or traces.
    """
    def keys(value, required, optional=()):
        if (not isinstance(value, dict) or not set(required) <= value.keys()
                or not value.keys() <= set(required) | set(optional)):
            raise ValueError("Record has missing or unsupported fields")

    def finite_number(value):
        if type(value) not in (int, float):
            raise ValueError("Timestamps and scores must be finite numbers, not booleans")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Timestamps and scores must be finite")
        return value

    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected a nonblank string")

    def version(record):
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("Expected schema version 1")

    available_times = validate_visual_input(model_input)

    if response is None:
        return {"schema_version": 1, "session_id": model_input["session_id"],
                "observation_id": model_input["observation_id"],
                "status": "insufficient_evidence", "candidate_issues": [], "error": None}
    keys(response, ("schema_version", "session_id", "observation_id", "status",
                    "candidate_issues", "error"))
    version(response)
    if any(response[name] != model_input[name] for name in ("session_id", "observation_id")):
        raise ValueError("Response belongs to another session or observation")
    status = response["status"]
    if status not in ("ok", "insufficient_evidence", "failed"):
        raise ValueError("Unsupported prediction status")
    candidates = response["candidate_issues"]
    if not isinstance(candidates, list) or (status != "ok" and candidates):
        raise ValueError("Only ok predictions may contain candidate issues")
    if status == "failed":
        keys(response["error"], ("type", "message"))
        text(response["error"]["type"])
        text(response["error"]["message"])
    elif response["error"] is not None:
        raise ValueError("Only failed predictions may contain an error")
    for issue in candidates:
        keys(issue, ("description", "execution_evidence_s", "reference_evidence_s"),
             ("confidence", "issue_key"))
        text(issue["description"])
        for field, source in (("execution_evidence_s", "execution_frames"),
                              ("reference_evidence_s", "reference_frames")):
            evidence = issue[field]
            if not isinstance(evidence, list) or (field == "execution_evidence_s" and not evidence):
                raise ValueError("Execution evidence must be nonempty; all evidence must be lists")
            for timestamp in evidence:
                if finite_number(timestamp) not in available_times[source]:
                    raise ValueError("Evidence timestamp does not identify a supplied frame")
        if issue.get("confidence") is not None:
            if not 0 <= finite_number(issue["confidence"]) <= 1:
                raise ValueError("Confidence must be between zero and one")
        if issue.get("issue_key") is not None:
            text(issue["issue_key"])
    return deepcopy(response)
