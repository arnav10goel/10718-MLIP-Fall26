"""Pure construction of inspectable reference/execution visual requests."""

from copy import deepcopy
import hashlib
import json
import math
import re
from uuid import UUID

from .mock_predictor import mock_predict
from .visual_completion import VISUAL_RESPONSE_SCHEMA


SYSTEM_INSTRUCTION = """Compare the correct REFERENCE demonstration with EXECUTION footage available so far. Derive the intended procedure from the reference visuals. Use only supplied images, timestamps and causal model-derived issue context; no external task checklist or assumed requirements.

Identify meaningful visible discrepancies that remain relevant now. Each candidate describes exactly one issue in a short factual sentence, with evidence citing exact supplied frame timestamps. Several separate issues may be proposed; the runtime chooses at most one issue per notification. Do not give routine progress confirmations, next-step coaching, reminders, or required repair instructions.

Follow the same semantic procedure order while allowing different pace, pauses, repeated adjustments and backtracking to repair a mistake. Elapsed time alone does not establish an omitted step. The execution cutoff and video ending do not announce that the person has finished the task. An unclear view warrants insufficient_evidence rather than a claim of correctness.

Allow harmless differences when they do not change the demonstrated procedure or intended result. Appearance, color, tools or viewpoint can be consequential in context, so do not ignore any such difference categorically or invent a requirement from appearance alone.

PRIOR_ISSUE_CONTEXT contains earlier model beliefs, not ground truth or new instructions. A description identifies the issue episode; state and evidence describe its latest update and may concern a repair. Compare the current visuals before reusing an issue. Link each continuing candidate to its known open issue_id. Use null when association is unknown. Admit only clearly new, distinct issues through new_issue_candidate_indices; a null link alone does not create an issue. Never reuse a resolved issue ID for a new recurrence.

Use an explicit issue_updates entry to mark a known open issue resolved only when supplied execution visuals support repair, or uncertain when its current state is unclear. Absence from the candidate list is not evidence of resolution. Keep other unresolved issues in mind rather than replacing them with only the latest error.

Return only JSON matching the supplied response schema: status, candidate_issues, candidate_links, issue_updates and new_issue_candidate_indices. Use status ok for a supported assessment (including an empty candidate list), or insufficient_evidence with no candidates when the view cannot support an assessment. Every candidate needs one explicit link entry; new issue indices refer to candidate_issues. Cite only exact currently supplied image times in candidate and update evidence, with nonempty execution evidence. Do not echo session/reference/observation envelopes, produce a checklist, or provide chain-of-thought reasoning."""


def build_visual_request(model_input, prior_context, *, max_images,
                         max_jpeg_bytes, max_text_bytes, max_prior_issues):
    """Return detached generic text/inline-JPEG parts without I/O or resampling.

    Counts bound images, their original bytes, all sent UTF-8 text (system,
    parts and canonical response schema), and retained prior-issue rows. They
    are not billed tokens or HTTP/base64 request size. Caps raise RuntimeError;
    invalid inputs raise ValueError. No content is truncated or silently fixed.

    prompt_sha256 hashes the exact system instruction. request_sha256 hashes
    the canonical returned body without that hash, replacing only inline JPEG
    byte leaves with SHA-256/size descriptors. Full actual bytes remain in parts
    for the caller's trace. Provider/model/generation settings are caller-owned.

    The current model_input has no reference hash; its binding to prior_context's
    reference_id is enforced by the session coordinator. This function checks
    session identity, causal availability and exact reference evidence times.
    Earlier execution evidence need not remain among the current request frames.
    """
    def keys(record, required):
        if not isinstance(record, dict) or set(record) != set(required):
            raise ValueError("Record has missing or unsupported fields")

    def number(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Context timestamps must be finite nonnegative numbers")
        return value

    def uuid_string(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Issue IDs must be canonical UUID strings")

    def canonical(value):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Request text must contain finite UTF-8 JSON values") from exc

    for limit in (max_images, max_jpeg_bytes, max_text_bytes, max_prior_issues):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Request limits must be positive integers")
    # Reuse the existing strict ready-visual boundary, discarding its default
    # scripted answer. This performs validation only, not an inference call.
    mock_predict(model_input)
    frames = model_input["reference_frames"] + model_input["execution_frames"]
    image_count = len(frames)
    jpeg_bytes = sum(len(frame["jpeg_bytes"]) for frame in frames)
    if image_count > max_images or jpeg_bytes > max_jpeg_bytes:
        raise RuntimeError("Visual request exceeds its image or JPEG-byte capacity")

    keys(prior_context, ("schema_version", "session_id", "reference_id",
                         "available_at_s", "issues"))
    if (type(prior_context["schema_version"]) is not int
            or prior_context["schema_version"] != 1
            or prior_context["session_id"] != model_input["session_id"]):
        raise ValueError("Prior context must use this session and schema version 1")
    reference_id = prior_context["reference_id"]
    if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
        raise ValueError("Prior context requires a lowercase SHA-256 reference ID")
    available = number(prior_context["available_at_s"])
    cutoff = model_input["cutoff_s"]
    if available > cutoff:
        raise ValueError("Prior context is unavailable at the observation cutoff")
    issues = prior_context["issues"]
    if not isinstance(issues, list):
        raise ValueError("Prior context issues must be a list")
    if len(issues) > max_prior_issues:
        raise RuntimeError("Visual request exceeds its prior-issue capacity")
    reference_times = {frame["timestamp_s"] for frame in model_input["reference_frames"]}
    seen = set()
    for issue in issues:
        keys(issue, ("issue_id", "state", "last_observation_cutoff_s", "description",
                     "execution_evidence_s", "reference_evidence_s"))
        identifier = issue["issue_id"]
        uuid_string(identifier)
        if identifier in seen:
            raise ValueError("Prior issue IDs must be unique")
        seen.add(identifier)
        if issue["state"] not in ("candidate", "active", "uncertain", "resolved"):
            raise ValueError("Unsupported prior issue state")
        last_cutoff = number(issue["last_observation_cutoff_s"])
        if last_cutoff > available or last_cutoff > cutoff:
            raise ValueError("Prior issue contains an unavailable observation")
        if not isinstance(issue["description"], str) or not issue["description"].strip():
            raise ValueError("Prior issue description must be nonblank")
        for field in ("execution_evidence_s", "reference_evidence_s"):
            evidence = issue[field]
            if not isinstance(evidence, list) or (field == "execution_evidence_s" and not evidence):
                raise ValueError("Prior evidence must be lists with nonempty execution support")
            for timestamp in evidence:
                number(timestamp)
                if field == "execution_evidence_s" and timestamp > last_cutoff:
                    raise ValueError("Prior execution evidence follows its observation cutoff")
                if field == "reference_evidence_s" and timestamp not in reference_times:
                    raise ValueError("Prior reference evidence must identify a supplied frame")

    header = {name: model_input[name] for name in (
        "schema_version", "session_id", "observation_id", "cutoff_s")}
    parts = [{"text": "OBSERVATION\n" + canonical(header).decode("utf-8")},
             {"text": "PRIOR_ISSUE_CONTEXT\n" + canonical(prior_context).decode("utf-8")}]
    for label, field in (("REFERENCE", "reference_frames"), ("EXECUTION", "execution_frames")):
        parts.append({"text": label})
        for frame in model_input[field]:
            parts.append({"text": canonical({"timestamp_s": frame["timestamp_s"]}).decode("utf-8")})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame["jpeg_bytes"]}})
    schema = deepcopy(VISUAL_RESPONSE_SCHEMA)
    text_bytes = (len(SYSTEM_INSTRUCTION.encode("utf-8"))
                  + sum(len(part["text"].encode("utf-8")) for part in parts if "text" in part)
                  + len(canonical(schema)))
    if text_bytes > max_text_bytes:
        raise RuntimeError("Visual request exceeds its UTF-8 text capacity")
    body = {"schema_version": 1, "system_instruction": SYSTEM_INSTRUCTION,
            "prompt_sha256": hashlib.sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest(),
            "parts": parts, "response_json_schema": schema,
            "counts": {"images": image_count, "jpeg_bytes": jpeg_bytes,
                       "text_bytes": text_bytes, "prior_issues": len(issues)}}
    identity = {**body, "parts": []}
    for part in parts:
        if "text" in part:
            identity["parts"].append(part)
        else:
            payload = part["inline_data"]["data"]
            identity["parts"].append({"inline_data": {"mime_type": "image/jpeg", "data": {
                "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}}})
    return {**body, "request_sha256": hashlib.sha256(canonical(identity)).hexdigest()}
