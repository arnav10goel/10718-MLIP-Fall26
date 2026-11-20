"""Native-video completions with owner-managed attempts and spending bounds."""

import base64
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, DecimalException
import hashlib
import json
import math
import re
from threading import Lock
from time import perf_counter
from uuid import UUID, uuid4

from .assessment import decode_assessment, build_assessment_request
from .provider_response import extract_response_text, filter_provider_response
from .frame_binding import validate_host_replay_passage
from .visual_input import validate_visual_input


NATIVE_VIDEO_PROFILES = {
    "gemini/gemini-2.5-flash-lite": {
        "max_input_tokens": 1048576, "max_output_tokens": 2048,
        "input_usd_per_million": "0.10", "output_usd_per_million": "0.40",
    },
    "gemini/gemini-3.8-flash": {
        "max_input_tokens": 1048576, "max_output_tokens": 8192,
        "input_usd_per_million": "0.75", "output_usd_per_million": "3.75",
    },
    "gemini/gemini-4-argon": {
        "max_input_tokens": 1048576, "max_output_tokens": 8192,
        "input_usd_per_million": "2", "output_usd_per_million": "10",
    },
}


class ProviderAdmissionRefused(RuntimeError):
    """A bounded admission was refused before starting provider transport."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


_ASSESSMENT_INSTRUCTION = (
    'You are monitoring someone performing a task shown in a correct reference demonstration. '
    'You receive sampled reference frames, execution frames up to the current time, and a list of previously tracked open issues.\n\n'
    'Compare the execution with the reference. People may move at different speeds, pause, adjust their work, '
    'or backtrack to correct a mistake. Ignore harmless differences and actions that are simply unfinished. '
    'Do not describe an object being positioned as already placed. '
    'Report a step as skipped only when a later visible action shows it was bypassed or an observable finish establishes it was omitted. '
    'If a clearly wrong action is still in progress, describe it as in progress. '
    'Report only meaningful mistakes supported by the visible frames; do not guess hidden details or assume '
    'an issue was fixed because it is no longer visible.\n\n'
    'Report each distinct mistake separately. For each current mistake, give a clear description and describe '
    'what visible correction would resolve it. If it is the same as an open issue, use that issue’s exact ID. '
    'If it is a different mistake, use "new". If you cannot tell whether it matches an existing issue, use null.\n\n'
    'For an open issue, report "resolved" only when you can see that it was corrected. Use "uncertain" when '
    'the evidence is unclear. Do not report the same issue as both present and resolved in one response. '
    'Previous issue descriptions are model judgments, not facts.\n\n'
    'Return a JSON object with:\n'
    '- "status": "ok" or "insufficient_evidence".\n'
    '- "issues": current mistakes, each with "description", "link", and "resolution_condition".\n'
    '- "updates": changes to open issues, each with "issue_id" and "state" ("resolved" or "uncertain").\n\n'
    'If nothing relevant can be judged, use "insufficient_evidence" and leave "issues" empty. '
    'If one issue is unclear but another is visible, report the visible one and mark the unclear one uncertain.'
)
_ASSESSMENT_IMAGE_INSTRUCTION = _ASSESSMENT_INSTRUCTION
_TRACKING_INSTRUCTION = (
    'Monitor a person following a correct reference demonstration. Compare the supplied reference frames '
    'with execution frames so far, using the supplied issue context. Frame times identify video positions; '
    'the context contains earlier model judgments, not established facts.\n\n'
    'Report each meaningful, visible mistake separately. Ignore harmless differences, pauses, unfinished '
    'actions and backtracking to correct a mistake. Do not guess hidden details or describe an action as '
    'completed when it is still in progress. Report a skipped step only if later action or an observable '
    'finish establishes it. If the view is unclear, express uncertainty; silence does not imply correction.\n\n'
    'For each current mistake, use same_open with the exact open issue ID if it is the same mistake; '
    'new with a null issue ID if it is distinct; recurrence with the exact corrected historical issue ID '
    'only when current evidence shows a new occurrence; or uncertain with a null issue ID when unsure. '
    'Use a pending ID only for a current sighting of that pending observation, otherwise null. '
    'Use corrected only when correction is visible, uncertain when an open issue cannot be judged, '
    'and retract when the earlier warning was mistaken. Do not both update and continue the same issue. '
    'Discard a pending observation only when current evidence shows it was mistaken. '
    'Optional citations may use only exact supplied reference and execution frame times. '
    'If nothing relevant can be judged, use insufficient_evidence. '
    'Return the requested JSON, with one observation per mistake; use empty lists when there are no claims.\n\n'
    'If two deviations can each be identified and corrected independently, return a separate observation for each, even when they occur in the same step or arrangement.'
)
_TRACKING_CITATIONS = {
    "type": "object", "additionalProperties": False, "required": ["execution_s", "reference_s"],
    "properties": {role: {"type": "array", "items": {"type": "number", "minimum": 0}}
                   for role in ("execution_s", "reference_s")},
}
_TRACKING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["assessment_contract_version", "status", "observations", "updates", "pending_updates"],
    "properties": {
        "assessment_contract_version": {"type": "integer", "enum": [3]},
        "status": {"type": "string", "enum": ["ok", "insufficient_evidence"]},
        "observations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["description", "association"], "properties": {
                "description": {"type": "string", "minLength": 1},
                "resolution_condition": {"type": "string", "minLength": 1},
                "citations": _TRACKING_CITATIONS,
                "association": {"type": "object", "additionalProperties": False,
                    "required": ["kind", "issue_id", "pending_id"], "properties": {
                        "kind": {"type": "string", "enum": ["same_open", "new", "recurrence", "uncertain"]},
                        "issue_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "pending_id": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}}},
        "updates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["issue_id", "action"], "properties": {
                "issue_id": {"type": "string"},
                "citations": _TRACKING_CITATIONS,
                "action": {"type": "string", "enum": ["corrected", "uncertain", "retract"]}}}},
        "pending_updates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["pending_id", "action", "reason"], "properties": {
                "pending_id": {"type": "string"}, "action": {"type": "string", "enum": ["discard"]},
                "reason": {"type": "string", "minLength": 1}}}},
    },
}
_OBSERVATION_INSTRUCTION = ("Compare the execution so far with the reference. List meaningful visible mistakes, "
    "one per observation with a unique integer ID, and describe what visible correction would resolve each. "
    "Ignore harmless differences and harmless unfinished actions. "
    "Only describe details supported by the supplied frames. Use insufficient evidence only when nothing relevant can be judged. "
    "Still report other clearly visible mistakes. Describe an impending mistake as in progress, not completed.")
_ASSOCIATION_INSTRUCTION = ("Compare the current observations with the existing and pending issues using the videos. "
    "Cover every current and pending observation exactly once. Use one decision with both IDs when a current observation "
    "is a fresh sighting of a pending one; otherwise use null for the missing ID. "
    "Decide same, distinct, or uncertain; use the open issue ID only for same and null otherwise. "
    "A pending item needs a current sighting for same or distinct; otherwise mark it uncertain. "
    "If the observation result has insufficient evidence, use only uncertain decisions and updates. "
    "When another issue is unclear, keep visible issues and mark the unclear issue uncertain. "
    "Mark an existing issue resolved only when correction is visible; if unclear, say uncertain. "
    "Previous descriptions are model judgments, not facts.")
_OBSERVATION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["schema_version", "status", "observations"], "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "status": {"type": "string", "enum": ["ok", "insufficient_evidence"]},
        "observations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["id", "description"], "properties": {
                "id": {"type": "integer", "minimum": 0}, "description": {"type": "string", "minLength": 1},
                "resolution_condition": {"type": "string", "minLength": 1}}}}}}
_ASSOCIATION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["schema_version", "status", "decisions", "updates"], "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "status": {"type": "string", "enum": ["ok", "insufficient_evidence"]},
        "decisions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["current_id", "pending_id", "match", "issue_id"], "properties": {
                "current_id": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
                "pending_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "match": {"type": "string", "enum": ["same", "distinct", "uncertain"]},
                "issue_id": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}},
        "updates": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["issue_id", "state"], "properties": {
                "issue_id": {"type": "string"}, "state": {"type": "string", "enum": ["resolved", "uncertain"]}}}}}}
_ASSESSMENT_CITATIONS = {
    "type": "object", "additionalProperties": False,
    "required": ["execution_s", "reference_s"],
    "properties": {role: {"type": "array", "items": {"type": "number", "minimum": 0}}
                   for role in ("execution_s", "reference_s")},
}
_ASSESSMENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "issues", "updates"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "insufficient_evidence"]},
        "issues": {"type": "array", "description": "One current discrepancy per item, without progress reports or coaching.",
            "items": {"type": "object", "additionalProperties": False,
                "required": ["description", "link"], "properties": {
                    "description": {"type": "string", "minLength": 1},
                    "resolution_condition": {"type": "string", "minLength": 1},
                    "link": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
                        "description": "Use new for a clearly distinct episode, an open prior issue ID for continuation, or null if association is unclear."},
                    "citations": _ASSESSMENT_CITATIONS}}},
        "updates": {"type": "array",
            "description": "Explicit changes to known open issues. Resolve only after visible correction; silence or video ending never resolves an issue.",
            "items": {"type": "object", "additionalProperties": False,
                "required": ["issue_id", "state"], "properties": {
                    "issue_id": {"type": "string", "minLength": 1},
                    "state": {"type": "string", "enum": ["resolved", "uncertain"]},
                    "citations": _ASSESSMENT_CITATIONS}}},
    },
}


@dataclass(frozen=True, eq=False)
class _PreparedRequest:
    """Opaque detached input; wire bytes cannot change after validation."""

    owner: object = field(repr=False)
    model_input: dict = field(repr=False)
    prior: dict = field(repr=False)
    wire: bytes = field(repr=False)
    provenance: dict = field(repr=False)
    prompt_hash: str
    short: bool
    input_representation: str = "native_video"
    stage: str | None = None
    integrity_sha256: str | None = None


@dataclass(frozen=True, eq=False)
class _AttemptTicket:
    """The lock protects only the one-shot transport/cancellation gate."""

    owner: object = field(repr=False)
    prepared: object = field(repr=False)
    lock: object = field(default_factory=Lock, repr=False)
    state: str = "ready"
    started: bool = False
    outcome: object = field(default=None, repr=False)
    completion_claimed: bool = False


@dataclass(frozen=True, eq=False)
class _TransportOutcome:
    ticket: _AttemptTicket = field(repr=False)
    status: object
    body: bytes = field(repr=False)
    error: object = field(repr=False)
    started_at_s: float
    received_at_s: float


class NativeVideoCompletionPlan:
    """Send verified native videos or opt-in selected JPEGs through a gateway.

    JPEG observations are validated timestamp registries, never wire images.
    The resolver owns physical video preparation; this boundary verifies its
    bytes, source identities and timestamp witness. Evidence must cite exact
    registry anchors, not approximate provider-selected video sample times.
    Optional media_detail requests low or high resolution for both videos;
    omission leaves provider resolution unspecified.

    The contract-2 jpeg_frames option instead sends the ready observation's
    explicitly selected JPEGs, retaining their actual timestamp/hash witness.
    It requires a streaming client and never resolves or invents video assets.
    Native defaults, wire format and buffered transport remain unchanged.

    Explicitly capped runs retain each full configured-context reservation.
    With no call-count cap, complete reported usage settles the active budget
    commitment; unknown charges retain their full reservation. Cumulative
    reservations remain available separately for audit.
    This is a conservative rate equivalent, not an invoice guarantee: proxy
    retry behavior and actual billing remain outside this class. No counting,
    upload, retry, cache or fallback request is made. The caller configures
    authentication, HTTP timeout and transport retries, and owns client closure.
    Use callback_errors_fatal=True so persistence/budget failures abort replay.

    Prepare may run in a separate serialized preparation lane. Admit, complete,
    cancel_unsent and snapshot have one coordinator owner; they are not APIs for
    concurrent ledger mutation. Only perform runs on HTTP workers. Its per-ticket
    lock allows one transport claim or one provably-unsent cancellation to win.
    """

    _COUNT_BASIS = "configured_model_context_upper_bound"

    def __init__(self, *, client, model, reference_id, reference_source_sha256,
                 execution_source_sha256, resolve_assets=None, trace, limits,
                 timeout_s, max_request_bytes, max_response_bytes, fps=1,
                 media_detail=None, clock=perf_counter, event_clock=None,
                 assessment_contract_version=1, input_representation="native_video",
                 image_limits=None):
        if type(assessment_contract_version) is not int or assessment_contract_version not in (1, 2, 3):
            raise ValueError("Assessment contract must be 1, 2 or 3")
        if type(input_representation) is not str or input_representation not in ("native_video", "jpeg_frames"):
            raise ValueError("Unsupported provider input representation")
        images = input_representation == "jpeg_frames"
        if images:
            if (assessment_contract_version not in (2, 3) or resolve_assets is not None
                    or type(fps) not in (int, float) or fps != 1 or media_detail is not None):
                raise ValueError("JPEG input requires contract 2 or 3 without native asset or media settings")
            if (not isinstance(image_limits, dict) or set(image_limits) != {"max_images", "max_jpeg_bytes"}
                    or any(type(value) is not int or value <= 0 for value in image_limits.values())):
                raise ValueError("JPEG input requires explicit positive image and JPEG-byte limits")
        elif assessment_contract_version == 3:
            raise ValueError("Contract 3 requires JPEG input")
        elif image_limits is not None:
            raise ValueError("Image limits require JPEG input")
        if not isinstance(model, str) or model not in NATIVE_VIDEO_PROFILES:
            raise ValueError("Native model must match the frozen context and rate profile")
        profile = NATIVE_VIDEO_PROFILES[model]
        for value in (reference_id, reference_source_sha256, execution_source_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Reference and source identities require lowercase SHA-256 values")
        fields = {"max_generation_calls", "max_input_tokens", "max_output_tokens",
                  "budget_usd", "input_usd_per_million", "output_usd_per_million"}
        if not isinstance(limits, dict) or set(limits) != fields:
            raise ValueError("Limits require explicit call, context and monetary settings")
        cap = limits["max_generation_calls"]
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("Call limit must be a positive integer or null")
        for key in ("max_input_tokens", "max_output_tokens"):
            if type(limits[key]) is not int or limits[key] <= 0:
                raise ValueError("Call and context limits must be positive integers")
        if (limits["max_input_tokens"] != profile["max_input_tokens"]
                or limits["max_output_tokens"] != profile["max_output_tokens"]):
            raise ValueError("Context limits must match the frozen model profile")
        amounts = []
        try:
            for key in ("budget_usd", "input_usd_per_million", "output_usd_per_million"):
                value = limits[key]
                if type(value) is not str or not value or value.strip() != value:
                    raise ValueError("Monetary settings must be explicit decimal strings")
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0:
                    raise ValueError("Monetary settings must be finite and nonnegative")
                amounts.append(amount)
        except DecimalException as exc:
            raise ValueError("Monetary settings must be valid decimals") from exc
        self._budget, self._input_rate, self._output_rate = amounts
        if (self._input_rate != Decimal(profile["input_usd_per_million"])
                or self._output_rate != Decimal(profile["output_usd_per_million"])):
            raise ValueError("Rates must match the frozen model profile")
        for value in (max_request_bytes, max_response_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Request and response limits must be positive integers")
        if self._number(timeout_s) <= 0:
            raise ValueError("A positive timeout is required")
        if type(fps) is dict:
            if set(fps) != {"reference", "execution"}:
                raise ValueError("Native FPS requires exactly reference and execution rates")
            if any(not 0 < self._number(rate) <= 24 for rate in fps.values()):
                raise ValueError("Native FPS rates must be positive and at most 24")
        elif type(fps) not in (int, float) or fps != 1:
            raise ValueError("Scalar native FPS must use the legacy one-FPS profile")
        if media_detail is not None and (
                type(media_detail) is not str or media_detail not in ("low", "high")):
            raise ValueError("Native media detail must be low, high or null")
        transport = getattr(client, "stream" if images else "post", None)
        if (not all(callable(value) for value in (transport, getattr(trace, "append", None), clock))
                or not images and not callable(resolve_assets)):
            raise ValueError("Client, asset resolver, trace and clock must be callable")
        if event_clock is not None and not callable(event_clock):
            raise ValueError("Event clock must be callable when supplied")
        self._client, self._model = client, model
        self._assessment_contract_version = assessment_contract_version
        self._input_representation = input_representation
        self._image_limits = deepcopy(image_limits)
        self._reference_id = reference_id
        self._sources = (reference_source_sha256, execution_source_sha256)
        self._resolve, self._trace, self._clock = resolve_assets, trace, clock
        self._limits = deepcopy(limits)
        self._timeout_s, self._fps = timeout_s, deepcopy(fps)
        self._media_detail = media_detail
        self._max_request_bytes, self._max_response_bytes = max_request_bytes, max_response_bytes
        self._reservation = (limits["max_input_tokens"] * self._input_rate
            + limits["max_output_tokens"] * self._output_rate) / 1000000
        self._reserved = self._known = Decimal(0)
        self._committed = Decimal(0)
        self._settled_usage = cap is None
        self._skipped = 0
        self._attempts = []
        self._tickets = {}
        self._owner = object()
        self._admissions = 0
        self._event_clock, self._last_event_s = event_clock, None
        self._stopped = False
        self._session_id = self._last_cutoff = None

    @staticmethod
    def _number(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Timestamps and clocks must be finite nonnegative numbers")
        return value

    @staticmethod
    def _json(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    def _emit(self, kind, payload, cutoff):
        # Persistence exceptions stay outside provider/decoder catches.
        try:
            at_s = (max(cutoff, self._last_cutoff or 0, self._last_event_s or 0)
                    if self._event_clock is None else self._number(self._event_clock()))
            if (self._assessment_contract_version == 1 and at_s < cutoff
                    or self._last_event_s is not None and at_s < self._last_event_s):
                raise ValueError("Coordinator event time cannot precede its input or reverse")
            value = deepcopy(payload)
            value["observation_cutoff_s"] = cutoff
            if self._assessment_contract_version in (2, 3):
                value["assessment_contract_version"] = self._assessment_contract_version
            self._trace.append(kind, value, at_s=at_s)
            self._last_event_s = at_s
        except BaseException:
            self._stopped = True
            raise

    def _asset(self, value, *, source, cutoff, anchors):
        required = {"schema_version", "status", "source_sha256", "source_size_bytes",
                    "source_first_pts_s", "cutoff_s", "source_eof", "mime_type",
                    "video_bytes", "video_sha256", "size_bytes", "frame_map", "config"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("Native asset has missing or unsupported fields")
        if (type(value["schema_version"]) is not int or value["schema_version"] != 1
                or value["status"] != "ready" or value["source_sha256"] != source
                or self._number(value["source_first_pts_s"]) != 0
                or value["cutoff_s"] != cutoff or type(value["source_eof"]) is not bool
                or value["mime_type"] != "video/mp4" or not isinstance(value["config"], dict)):
            raise ValueError("Native asset identity, origin or cutoff disagrees with the observation")
        if cutoff is None and not value["source_eof"]:
            raise ValueError("Reference asset must contain the complete reference")
        for field in ("source_size_bytes", "size_bytes"):
            if type(value[field]) is not int or value[field] <= 0:
                raise ValueError("Ready native assets require positive byte sizes")
        video = value["video_bytes"]
        if (type(video) is not bytes or len(video) != value["size_bytes"]
                or hashlib.sha256(video).hexdigest() != value["video_sha256"]):
            raise ValueError("Native video bytes disagree with their digest or size")
        rows = value["frame_map"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Ready native assets require a source-frame witness")
        previous = None
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                    "frame_index", "source_pts_s", "elapsed_s", "video_pts_s"}:
                raise ValueError("Native frame witnesses have invalid fields")
            index = row["frame_index"]
            times = tuple(self._number(row[key]) for key in (
                "source_pts_s", "elapsed_s", "video_pts_s"))
            if (type(index) is not int or index < 0 or times[0] != times[1]
                    or abs(times[2] - times[1]) > 0.000001
                    or (cutoff is not None and times[1] > cutoff)
                    or (previous is not None and (index <= previous[0]
                        or any(t <= p for t, p in zip(times, previous[1:]))))):
                raise ValueError("Native frame witness must preserve ordered causal source times")
            previous = (index, *times)
        if rows[0]["frame_index"] != 0 or rows[0]["elapsed_s"] != 0:
            raise ValueError("Native video must begin at source frame zero")
        source_times = {row["elapsed_s"] for row in rows}
        if any(frame["timestamp_s"] not in source_times for frame in anchors):
            raise ValueError("Every evidence anchor must identify an actual native source frame")
        return {key: deepcopy(value[key]) for key in value if key != "video_bytes"}

    _filtered = staticmethod(filter_provider_response)

    def _usage(self, raw, attempt):
        usage = raw.get("usage", {})
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        prompt = prompt if type(prompt) is int and prompt >= 0 else None
        output = output if type(output) is int and output >= 0 else None
        attempt.update(prompt_tokens=prompt, completion_tokens=output,
                       cost_complete=prompt is not None and output is not None)
        known = ((prompt or 0) * self._input_rate + (output or 0) * self._output_rate) / 1000000
        self._known += known
        attempt["known_rate_equivalent_usd"] = format(known, "f")

    def _settle_budget(self, attempt, cutoff):
        """Settle only after the response was durably recorded; never free unknown cost."""
        prompt, output = attempt["prompt_tokens"], attempt["completion_tokens"]
        known = Decimal(attempt["known_rate_equivalent_usd"])
        invalid = (prompt is not None and prompt > self._limits["max_input_tokens"]
                   or output is not None and output > self._limits["max_output_tokens"]
                   or known > self._reservation)
        committed = self._reservation
        if invalid:
            committed = max(committed, known)
            attempt["cost_complete"] = False
        elif attempt["cost_complete"]:
            committed = known
        total = self._committed + committed - self._reservation
        self._emit("budget_settlement", {"observation_id": attempt["observation_id"],
            "attempt_id": attempt["attempt_id"],
            "request_sha256": attempt["request_sha256"], "cutoff_s": cutoff,
            "accounting_mode": "settled_usage", "reserved_usd": attempt["reserved_usd"],
            "budget_committed_usd": format(committed, "f"),
            "total_budget_committed_usd": format(total, "f"),
            "cost_complete": attempt["cost_complete"], "usage_valid": not invalid}, cutoff)
        # Only a durable settlement can release the full outstanding hold.
        self._committed = total
        attempt["budget_committed_usd"] = format(committed, "f")
        return not invalid

    _text = staticmethod(extract_response_text)

    def _assessment_prior(self, model_input, prior):
        """Validate only the model-visible whitelist; causal revision/time stays with the runner."""
        if (not isinstance(prior, dict) or set(prior) != {
                "assessment_contract_version", "session_id", "reference_id", "issues"}
                or type(prior["assessment_contract_version"]) is not int
                or prior["assessment_contract_version"] != 2
                or prior["session_id"] != model_input["session_id"]
                or prior["reference_id"] != self._reference_id):
            raise ValueError("Prior context must use this session, reference and assessment contract 2")
        issues = prior["issues"]
        if not isinstance(issues, list):
            raise ValueError("Prior issues must be an explicit list")
        if len(issues) > self._max_request_bytes:
            raise RuntimeError("Prior context exceeds its request capacity")
        seen = set()
        for issue in issues:
            if not isinstance(issue, dict):
                raise ValueError("Prior issue has missing or unsupported fields")
            rich_fields = {"issue_id", "state", "initial_description", "latest_description", "repair_pending"}
            rich = set(issue) in (rich_fields, rich_fields | {"resolution_condition"})
            if not rich and set(issue) != {"issue_id", "state", "description"}:
                raise ValueError("Prior issue has missing or unsupported fields")
            descriptions = ("initial_description", "latest_description") if rich else ("description",)
            identifier = issue["issue_id"]
            if not isinstance(identifier, str) or str(UUID(identifier)) != identifier or identifier in seen:
                raise ValueError("Prior issue IDs must be unique canonical UUIDs")
            if issue["state"] not in ("candidate", "active", "uncertain", "resolved"):
                raise ValueError("Unsupported prior issue state")
            if (any(not isinstance(issue[name], str) or not issue[name].strip() for name in descriptions)
                    or "resolution_condition" in issue and
                       (not isinstance(issue["resolution_condition"], str) or not issue["resolution_condition"].strip())
                    or rich and type(issue["repair_pending"]) is not bool):
                raise ValueError("Prior descriptions must be nonblank UTF-8 text")
            seen.add(identifier)
        if len(self._json(prior)) > self._max_request_bytes:
            raise RuntimeError("Prior context exceeds its request capacity")

    def _assessment_result(self, prepared, *, outcome, assessment=None, error=None, attempt=None):
        """Expose frozen request provenance and raw transport-clock measurements, not a frame-clock mapping."""
        provenance_key = "jpeg_provenance" if prepared.input_representation == "jpeg_frames" else "native_provenance"
        return {"assessment_contract_version": self._assessment_contract_version, "outcome": outcome,
            "assessment": assessment, "error": error, "observation_id": prepared.model_input["observation_id"],
            "request_sha256": None if attempt is None else attempt["request_sha256"],
            provenance_key: None if attempt is None else deepcopy(prepared.provenance),
            "transport_started_at_s": None if attempt is None else attempt.get("transport_started_at_s"),
            "transport_received_at_s": None if attempt is None else attempt.get("transport_received_at_s")}

    def _failed(self, code, model_input, prior, attempt, *, phase, status=None, prepared=None,
                recoverable=False):
        if not recoverable:
            self._stopped = True
        attempt.update(status="failed", code=code)
        self._emit("provider_assessment_skipped" if recoverable else "provider_failure", {"observation_id": model_input["observation_id"],
            "attempt_id": attempt["attempt_id"],
            "code": code, "phase": phase, "http_status": status,
            "elapsed_s": attempt.get("elapsed_s"),
            "transport_started_at_s": attempt.get("transport_started_at_s"),
            "transport_received_at_s": attempt.get("transport_received_at_s")}, model_input["cutoff_s"])
        if self._assessment_contract_version in (2, 3):
            return self._assessment_result(prepared, outcome="failed", attempt=attempt,
                error={"type": "recoverable_invalid_completion" if recoverable else code,
                       "message": "Visual provider completion failed"})
        return {"response": {"schema_version": 1, "session_id": model_input["session_id"],
                    "observation_id": model_input["observation_id"], "status": "failed",
                    "candidate_issues": [], "error": {"type": code,
                        "message": "Visual provider completion failed"}},
                "updates": {"schema_version": 1, "session_id": model_input["session_id"],
                    "observation_id": model_input["observation_id"],
                    "reference_id": prior["reference_id"], "candidate_links": [], "issue_updates": []},
                "new_issue_candidate_indices": []}

    def prepare(self, model_input, prior_context, *, assets=None, observation=None,
                selected_indices=None, stage=None):
        """Prepare detached wire/context without ledger, clock or trace mutation.

        Callers may supply physically prepared assets from a separate serialized
        preparation lane; otherwise the existing resolver is used. Only owner
        admission/completion may mutate the plan. Prepared inputs and ticket
        fields are opaque implementation details, not mutable caller settings.
        """
        model_input, prior = deepcopy(model_input), deepcopy(prior_context)
        if stage not in (None, "observation", "association") or stage is not None and (
                self._assessment_contract_version != 2 or self._input_representation != "jpeg_frames"):
            raise ValueError("Two-stage requests require contract 2 with JPEG frames")
        if self._input_representation == "jpeg_frames":
            if assets is not None or not isinstance(observation, dict) or selected_indices is None:
                raise ValueError("JPEG preparation requires a ready observation and selected indices, not native assets")
            observation, selected_indices = deepcopy(observation), deepcopy(selected_indices)
            if observation.get("model_input") != model_input or observation.get("reference_id") != self._reference_id:
                raise ValueError("JPEG observation differs from the supplied model input or pinned reference")
            instruction, schema, schema_name = {
                None: (_ASSESSMENT_IMAGE_INSTRUCTION, _ASSESSMENT_SCHEMA, "assessment_completion"),
                "observation": (_OBSERVATION_INSTRUCTION, _OBSERVATION_SCHEMA, "assessment_observation"),
                "association": (_ASSOCIATION_INSTRUCTION, _ASSOCIATION_SCHEMA, "assessment_association"),
            }[stage]
            if self._assessment_contract_version == 3:
                instruction, schema, schema_name = (_TRACKING_INSTRUCTION,
                    _TRACKING_SCHEMA, "issue_assessment")
            template = {"model": self._model, "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": []}], "max_tokens": self._limits["max_output_tokens"], "n": 1,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema_name, "schema": deepcopy(schema)}}}
            options = dict(request_template=template, source_sha256=dict(zip(("reference", "execution"), self._sources)),
                selected_indices=selected_indices, max_images=self._image_limits["max_images"],
                max_jpeg_bytes=self._image_limits["max_jpeg_bytes"], max_request_bytes=self._max_request_bytes)
            if self._assessment_contract_version == 3:
                built = build_assessment_request(observation, prior, **options)
            else:
                from .assessment_image_request import build_assessment_image_request
                built = build_assessment_image_request(observation, prior, stage=stage, **options)
            integrity = None
            if self._assessment_contract_version == 3:
                identity = {key: model_input[key] for key in ("session_id", "observation_id", "cutoff_s")}
                integrity = hashlib.sha256(built["wire"] + self._json({"identity": identity, "prior": prior,
                    "provenance": built["jpeg_provenance"]})).hexdigest()
            return _PreparedRequest(self._owner, model_input, prior, built["wire"], built["jpeg_provenance"],
                hashlib.sha256(instruction.encode("utf-8")).hexdigest(), False, "jpeg_frames", stage, integrity)
        if observation is not None or selected_indices is not None:
            raise ValueError("Observation selection requires JPEG input")
        if self._assessment_contract_version == 2:
            # This existing pure boundary validates visuals only. Its scripted
            # result is discarded; no model assessment or legacy prior is made.
            validate_visual_input(model_input)
            self._assessment_prior(model_input, prior)
            if sum(len(frame["jpeg_bytes"]) for role in ("reference", "execution")
                   for frame in model_input[role + "_frames"]) > self._max_request_bytes:
                raise RuntimeError("Visual registry exceeds its JPEG-byte capacity")
        else:
            from .visual_request import build_visual_request
            # The existing pure builder validates JPEG registries, causal context and
            # identity envelopes. Discard all of its JPEG wire parts immediately.
            built = build_visual_request(model_input, prior,
                max_images=max(1, sum(len(model_input.get(key, [])) for key in (
                    "reference_frames", "execution_frames"))), max_jpeg_bytes=self._max_request_bytes,
                max_text_bytes=self._max_request_bytes, max_prior_issues=self._max_request_bytes)
            del built
        cutoff = model_input["cutoff_s"]
        if prior["reference_id"] != self._reference_id:
            raise ValueError("Native completion cannot change reference")
        if cutoff == 0:
            return _PreparedRequest(self._owner, model_input, prior, b"", {}, "", False)
        assets = deepcopy(self._resolve(cutoff) if assets is None else assets)
        if not isinstance(assets, dict) or set(assets) != {"reference", "execution"}:
            raise ValueError("Resolver must supply exactly reference and execution assets")
        provenance = {}
        parts = []
        for role, source, end in zip(("reference", "execution"), self._sources, (None, cutoff)):
            value = assets[role]
            provenance[role] = self._asset(value, source=source, cutoff=end,
                                           anchors=model_input[role + "_frames"])
            file_part = {"file_data": "data:video/mp4;base64," +
                base64.b64encode(value["video_bytes"]).decode("ascii"),
                "video_metadata": {"fps": self._fps[role] if type(self._fps) is dict else self._fps}}
            if self._media_detail is not None:
                file_part["detail"] = self._media_detail
            parts.extend([{"type": "text", "text": role.upper()}, {"type": "file", "file": file_part}])
        header = {key: model_input[key] for key in (
            "schema_version", "session_id", "observation_id", "cutoff_s")}
        anchors = {role: [frame["timestamp_s"] for frame in model_input[role + "_frames"]]
                   for role in ("reference", "execution")}
        for label, value in (("OBSERVATION", header), ("EVIDENCE_ANCHORS", anchors),
                             ("PRIOR_ISSUE_CONTEXT", prior)):
            parts.append({"type": "text", "text": label + "\n" + self._json(value).decode("utf-8")})
        short = self._model == "gemini/gemini-3.8-flash"
        if self._assessment_contract_version == 2:
            instruction = _ASSESSMENT_INSTRUCTION
            schema, schema_name = _ASSESSMENT_SCHEMA, "assessment_completion"
        elif short:
            from .short_video_completion import SHORT_VIDEO_RESPONSE_SCHEMA, SHORT_VIDEO_SYSTEM_INSTRUCTION
            instruction = SHORT_VIDEO_SYSTEM_INSTRUCTION
            schema, schema_name = SHORT_VIDEO_RESPONSE_SCHEMA, "short_video_completion"
        else:
            from .visual_completion import VISUAL_RESPONSE_SCHEMA
            from .visual_request import SYSTEM_INSTRUCTION
            instruction = SYSTEM_INSTRUCTION.replace("supplied images", "supplied silent videos").replace(
                "exact supplied frame timestamps", "exact supplied EVIDENCE_ANCHORS timestamps").replace(
                "exact currently supplied image times", "exact currently supplied EVIDENCE_ANCHORS times")
            instruction += ("\nThe two MP4 files contain the complete REFERENCE and causal EXECUTION "
                "prefix. EVIDENCE_ANCHORS lists source elapsed times of actual frames in these videos. "
                "Use only those exact times for evidence; they are citation anchors, not a claim about "
                "the provider's internal video sampling. Do not invent or round a timestamp.")
            schema, schema_name = VISUAL_RESPONSE_SCHEMA, "visual_completion"
        request = {"model": self._model, "messages": [
            {"role": "system", "content": instruction}, {"role": "user", "content": parts}],
            "max_tokens": self._limits["max_output_tokens"], "n": 1,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema_name, "schema": deepcopy(schema)}}}
        if not short and self._model != "gemini/gemini-4-argon":
            request["reasoning_effort"] = "none"
        encoded = self._json(request)
        if len(encoded) > self._max_request_bytes:
            raise RuntimeError("Native request exceeds its encoded-byte limit")
        return _PreparedRequest(self._owner, model_input, prior, encoded, provenance,
            hashlib.sha256(instruction.encode("utf-8")).hexdigest(), short)

    def describe_prepared(self, prepared):
        """Describe one owner-created positive-duration request without side effects.

        This detached witness lets the coordinator freeze native-tail timing
        before admission. It exposes neither the opaque wire nor media bytes.
        """
        if (not isinstance(prepared, _PreparedRequest) or prepared.owner is not self._owner
                or prepared.model_input["cutoff_s"] <= 0 or not prepared.wire):
            raise ValueError("A positive-duration request prepared by this provider is required")
        provenance_key = "jpeg_provenance" if prepared.input_representation == "jpeg_frames" else "native_provenance"
        return {"assessment_contract_version": self._assessment_contract_version,
                "observation_id": prepared.model_input["observation_id"],
                "request_sha256": hashlib.sha256(prepared.wire).hexdigest(),
                provenance_key: deepcopy(prepared.provenance)}

    def _bind(self, prepared):
        if not isinstance(prepared, _PreparedRequest) or prepared.owner is not self._owner:
            raise ValueError("Prepared input belongs to another provider plan")
        model_input = prepared.model_input
        cutoff = model_input["cutoff_s"]
        if (self._session_id not in (None, model_input["session_id"])
                or self._last_cutoff is not None and cutoff < self._last_cutoff):
            raise ValueError("Native completion cannot change session or reverse admitted cutoffs")
        return cutoff

    def _capacity(self):
        if self._stopped:
            raise RuntimeError("Native completion batch stopped after a provider or schema failure")
        cap = self._limits["max_generation_calls"]
        if cap is not None and self._admissions >= cap:
            raise ProviderAdmissionRefused("generation_call_limit_reached")
        if self._committed + self._reservation > self._budget:
            raise ProviderAdmissionRefused("budget_reservation_unavailable")

    def admit(self, prepared, *, replay_clock_assertion=None, browser_playback_witness=None):
        """Owner-only admission reserves every outstanding attempt before HTTP.

        ``provider_dispatch`` records transport admission, not exact HTTP start.
        Transport start/receipt clocks are recorded on the later response event.
        Without event_clock, append times are logical admitted-cutoff times,
        never measured wall/display latency. Browser adapters may stamp their
        actual coordinator clock instead.
        """
        cutoff = self._bind(prepared)
        if self._assessment_contract_version == 3:
            # File replay itself supplies this convention; it is a caller
            # assertion, not evidence of camera capture or host-clock mapping.
            identity = {key: prepared.model_input[key] for key in ("session_id", "observation_id", "cutoff_s")}
            actual_integrity = hashlib.sha256(prepared.wire + self._json({"identity": identity,
                "prior": prepared.prior, "provenance": prepared.provenance})).hexdigest()
            if actual_integrity != prepared.integrity_sha256:
                raise ProviderAdmissionRefused("contract_3_prepared_input_changed")
            fields = {"kind", "session_id", "observation_id", "execution_source_sha256"}
            assertion = replay_clock_assertion
            rows = prepared.provenance.get("execution", {}).get("frames", [])
            if (prepared.prior["source_cutoff_s"] != cutoff or not rows
                    or any(row["elapsed_s"] > cutoff for row in rows)
                    or (assertion is None) == (browser_playback_witness is None)):
                raise ProviderAdmissionRefused("contract_3_clock_mapping_unavailable")
            if assertion is not None:
                if (not isinstance(assertion, dict) or set(assertion) != fields
                        or assertion["kind"] != "offline_replay_source_elapsed"
                        or assertion["session_id"] != prepared.model_input["session_id"]
                        or assertion["observation_id"] != prepared.model_input["observation_id"]
                        or assertion["execution_source_sha256"] != self._sources[1]
                        or any(row["elapsed_s"] != row["timestamp_s"] for row in rows)):
                    raise ProviderAdmissionRefused("contract_3_clock_mapping_unavailable")
            else:
                # This witness is supplied by the coordinator; it validates
                # consistency and chronology, not physical playback truth.
                try:
                    witness = browser_playback_witness
                    expected = {"session_id", "observation_id", "reference_id", "execution_source_sha256",
                        "clock_id", "timing_basis", "request_sha256", "tail_frame_index",
                        "tail_source_elapsed_s", "mapping_frozen_at_s", "admitted_at_s", "playback_mapping"}
                    if not isinstance(witness, dict) or set(witness) != expected or self._event_clock is None:
                        raise ValueError("Invalid browser witness")
                    clock_id = witness["clock_id"]
                    tail = rows[-1]
                    if (not isinstance(clock_id, str) or not clock_id.strip()
                            or witness["timing_basis"] != "session_monotonic"
                            or witness["session_id"] != prepared.model_input["session_id"]
                            or witness["observation_id"] != prepared.model_input["observation_id"]
                            or witness["reference_id"] != self._reference_id
                            or witness["execution_source_sha256"] != self._sources[1]
                            or witness["request_sha256"] != hashlib.sha256(prepared.wire).hexdigest()
                            or type(witness["tail_frame_index"]) is not int
                            or witness["tail_frame_index"] != tail["frame_index"]
                            or self._number(witness["tail_source_elapsed_s"]) != tail["elapsed_s"]):
                        raise ValueError("Browser witness identity disagrees")
                    clock_id.encode("utf-8")
                    mapping = witness["playback_mapping"]
                    mapping_fields = {"schema_version", "session_id", "execution_source_sha256", "clock_id",
                        "client_epoch_id", "segment_id", "kind", "status", "reason", "host_interval_s",
                        "support", "received_at_s"}
                    if not isinstance(mapping, dict) or set(mapping) != mapping_fields:
                        raise ValueError("Invalid playback mapping")
                    if (type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
                            or mapping["session_id"] != witness["session_id"]
                            or mapping["execution_source_sha256"] != self._sources[1]
                            or mapping["clock_id"] != clock_id or mapping["kind"] not in ("replay_passage", "host_replay_passage")
                            or mapping["status"] != "mapped" or mapping["reason"] is not None):
                        raise ValueError("Playback mapping identity disagrees")
                    for name in ("client_epoch_id", "segment_id"):
                        value = mapping[name]
                        if not isinstance(value, str) or str(UUID(value)) != value:
                            raise ValueError("Invalid playback identity")
                    support, interval = mapping["support"], mapping["host_interval_s"]
                    if mapping["kind"] == "host_replay_passage":
                        if not validate_host_replay_passage(mapping, tail["elapsed_s"], witness["mapping_frozen_at_s"]):
                            raise ValueError("Host replay passage unavailable")
                    else:
                        if (not isinstance(support, dict) or set(support) != {"source_elapsed_s", "sample_ids"}
                                or self._number(support["source_elapsed_s"]) != tail["elapsed_s"]
                                or not isinstance(support["sample_ids"], list) or len(support["sample_ids"]) != 2
                                or support["sample_ids"][0] == support["sample_ids"][1]
                                or not isinstance(interval, list) or len(interval) != 2):
                            raise ValueError("Playback mapping identifies another frame")
                        for value in support["sample_ids"]:
                            if not isinstance(value, str) or str(UUID(value)) != value:
                                raise ValueError("Invalid playback sample")
                    lower, upper = (self._number(value) for value in interval)
                    received = self._number(mapping["received_at_s"])
                    frozen, admitted = self._number(witness["mapping_frozen_at_s"]), self._number(witness["admitted_at_s"])
                    current = self._number(self._event_clock())
                    if (not lower <= upper == received <= frozen <= admitted <= current
                            or self._last_event_s is not None and current < self._last_event_s):
                        raise ValueError("Playback mapping chronology disagrees")
                except (KeyError, TypeError, ValueError, OverflowError, UnicodeError) as exc:
                    raise ProviderAdmissionRefused("contract_3_clock_mapping_unavailable") from exc
        elif replay_clock_assertion is not None or browser_playback_witness is not None:
            raise ValueError("Clock witnesses are only supported for contract 3")
        if cutoff == 0:
            raise ValueError("Zero-cutoff input cannot be admitted for HTTP")
        self._capacity()
        # Detach original decode contexts from the preparation handle as well.
        prepared = _PreparedRequest(self._owner, deepcopy(prepared.model_input),
            deepcopy(prepared.prior), prepared.wire, deepcopy(prepared.provenance),
            prepared.prompt_hash, prepared.short, prepared.input_representation, prepared.stage,
            prepared.integrity_sha256)
        model_input = prepared.model_input
        self._session_id, self._last_cutoff = model_input["session_id"], cutoff
        encoded = prepared.wire
        request = json.loads(encoded)
        request_hash = hashlib.sha256(encoded).hexdigest()
        provenance_key = "jpeg_provenance" if prepared.input_representation == "jpeg_frames" else "assets"
        self._emit("provider_request", {"observation_id": model_input["observation_id"],
            "request_sha256": request_hash,
            "prompt_sha256": prepared.prompt_hash,
            "request": request, provenance_key: prepared.provenance, "request_bytes": len(encoded),
            "timeout_s": self._timeout_s}, cutoff)
        self._reserved += self._reservation
        self._committed += self._reservation
        self._admissions += 1
        attempt = {"attempt_id": str(uuid4()),
            "observation_id": model_input["observation_id"], "cutoff_s": cutoff,
            "request_sha256": request_hash, "reserved_usd": format(self._reservation, "f"),
            "status": "reserved", "prompt_tokens": None, "completion_tokens": None,
            "known_rate_equivalent_usd": "0", "cost_complete": False}
        if self._settled_usage:
            attempt["budget_committed_usd"] = format(self._reservation, "f")
        self._attempts.append(attempt)
        ticket = _AttemptTicket(self._owner, prepared)
        self._tickets[ticket] = attempt
        reservation = {**attempt, "total_reserved_usd": format(self._reserved, "f"),
                       "count_basis": self._COUNT_BASIS}
        if self._settled_usage:
            reservation.update(accounting_mode="settled_usage",
                total_budget_committed_usd=format(self._committed, "f"))
        try:
            self._emit("budget_reservation", reservation, cutoff)
            self._emit("provider_dispatch", {"observation_id": model_input["observation_id"],
                "attempt_id": attempt["attempt_id"], "dispatch_basis": "transport_admission",
                "request_sha256": request_hash}, cutoff)
        except BaseException:
            object.__setattr__(ticket, "state", "cancelled")
            object.__setattr__(ticket, "prepared", None)
            raise
        return ticket

    def perform(self, ticket):
        """Worker-only, one-shot HTTP transport; no ledger/trace/clock mutation."""
        if not isinstance(ticket, _AttemptTicket) or ticket.owner is not self._owner:
            raise ValueError("Transport ticket belongs to another provider plan")
        with ticket.lock:
            if ticket.state != "ready":
                raise RuntimeError("Transport ticket has already started or was cancelled")
            object.__setattr__(ticket, "state", "running")
            object.__setattr__(ticket, "started", True)
        started = self._number(self._clock())
        import httpx
        status, body, error = None, b"", None
        response_buffer = None
        try:
            if ticket.prepared.input_representation == "jpeg_frames":
                response_buffer = bytearray()
                with self._client.stream("POST", "chat/completions", content=ticket.prepared.wire,
                        headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
                        follow_redirects=False) as response:
                    if not isinstance(response, httpx.Response):
                        raise TypeError("JPEG client must stream an httpx.Response")
                    status = response.status_code
                    for chunk in response.iter_raw():
                        response_buffer.extend(chunk[:self._max_response_bytes + 1 - len(response_buffer)])
                        if len(response_buffer) > self._max_response_bytes:
                            error = "provider_response_too_large"
                            break
            else:
                response = self._client.post("chat/completions", json=json.loads(ticket.prepared.wire))
                if not isinstance(response, httpx.Response):
                    raise TypeError("Native client must return an httpx.Response")
                status = response.status_code
                if len(response.content) > self._max_response_bytes:
                    error = "provider_response_too_large"
                else:
                    body = bytes(response.content)
        except (TimeoutError, httpx.TimeoutException):
            error = "provider_timeout"
        except httpx.RequestError:
            error = "provider_error"
        except Exception as exc:
            error = exc
        if response_buffer is not None:
            body = bytes(response_buffer[:self._max_response_bytes])
        received = self._number(self._clock())
        if received < started:
            raise ValueError("Transport clock cannot reverse during HTTP")
        outcome = _TransportOutcome(ticket, status, body, error, started, received)
        with ticket.lock:
            object.__setattr__(ticket, "state", "finished")
            object.__setattr__(ticket, "outcome", outcome)
        return outcome

    def _ticket(self, ticket):
        if (not isinstance(ticket, _AttemptTicket) or ticket.owner is not self._owner
                or ticket not in self._tickets):
            raise ValueError("Unknown provider ticket")
        return self._tickets[ticket]

    def cancel_unsent(self, ticket):
        """Owner-only release, only when cancellation wins before HTTP claim."""
        attempt = self._ticket(ticket)
        with ticket.lock:
            if ticket.state != "ready":
                return False
            object.__setattr__(ticket, "state", "cancelled")
            object.__setattr__(ticket, "prepared", None)
        total = self._committed - self._reservation
        self._emit("provider_cancelled", {"attempt_id": attempt["attempt_id"],
            "observation_id": attempt["observation_id"], "request_sha256": attempt["request_sha256"],
            "reason": "provably_unsent", "released_usd": format(self._reservation, "f"),
            "total_budget_committed_usd": format(total, "f")}, attempt["cutoff_s"])
        self._committed, self._admissions = total, self._admissions - 1
        attempt.update(status="cancelled", prompt_tokens=0, completion_tokens=0, cost_complete=True)
        if self._settled_usage:
            attempt["budget_committed_usd"] = "0"
        return True

    def complete(self, ticket, outcome):
        """Owner-only persistence/accounting/decode, even after another failure."""
        attempt = self._ticket(ticket)
        with ticket.lock:
            if (ticket.state != "finished" or ticket.completion_claimed
                    or outcome is not ticket.outcome):
                raise ValueError("Completion must match one unfinished transport result")
            object.__setattr__(ticket, "completion_claimed", True)
        try:
            return self._complete_outcome(ticket, outcome, attempt)
        finally:
            # Preserve only lightweight gate/accounting identity after a result.
            # In particular, synchronous replay must not retain every MP4 wire.
            object.__setattr__(ticket, "prepared", None)
            object.__setattr__(ticket, "outcome", None)

    def _complete_outcome(self, ticket, outcome, attempt):
        prepared = ticket.prepared
        model_input, prior = prepared.model_input, prepared.prior
        cutoff = model_input["cutoff_s"]
        attempt.update(status="dispatched", elapsed_s=outcome.received_at_s - outcome.started_at_s,
            transport_started_at_s=outcome.started_at_s,
            transport_received_at_s=outcome.received_at_s)
        if isinstance(outcome.error, Exception):
            self._stopped = True
            raise outcome.error
        status = outcome.status
        if outcome.error is not None:
            phase = "response" if outcome.error == "provider_response_too_large" else "generation"
            return self._failed(outcome.error, model_input, prior, attempt, phase=phase, status=status, prepared=prepared)
        try:
            raw = json.loads(outcome.body)
            if prepared.input_representation == "jpeg_frames":
                self._json(raw)  # Reject nonfinite JSON rather than passing it into traces/accounting.
        except (ValueError, UnicodeDecodeError):
            raw = None
        except (RecursionError, OverflowError):
            if prepared.input_representation != "jpeg_frames":
                raise
            raw = None
        filtered = self._filtered(raw) if isinstance(raw, dict) else {}
        self._usage(filtered, attempt)
        self._emit("provider_response", {"observation_id": model_input["observation_id"],
            "attempt_id": attempt["attempt_id"],
            "request_sha256": attempt["request_sha256"], "http_status": status, "response": filtered,
            "transport_started_at_s": outcome.started_at_s, "transport_received_at_s": outcome.received_at_s,
            "elapsed_s": attempt["elapsed_s"]}, cutoff)
        if self._settled_usage and not self._settle_budget(attempt, cutoff):
            return self._failed("invalid_provider_usage", model_input, prior, attempt,
                                phase="response", status=status, prepared=prepared)
        if status != 200:
            return self._failed("provider_http_error", model_input, prior, attempt,
                                phase="response", status=status, prepared=prepared)
        try:
            text = self._text(raw)
        except ValueError:
            return self._failed("invalid_provider_response", model_input, prior, attempt,
                                phase="response", status=status, prepared=prepared)
        host_context_validated = False
        try:
            if prepared.stage == "observation":
                from .assessment_observation import decode_observations
                assessment = decode_observations(text, max_response_bytes=self._max_response_bytes)
                result = self._assessment_result(prepared, outcome="completed", assessment=assessment, attempt=attempt)
            elif prepared.stage == "association":
                from .assessment_association import adapt_association
                assessment = adapt_association(prior["observations"], text,
                    open_issue_ids=[row["issue_id"] for row in prior["issues"] if row["state"] != "resolved"],
                    pending_observations=prior["pending_observations"], max_response_bytes=self._max_response_bytes)
                result = self._assessment_result(prepared, outcome="completed", assessment=assessment, attempt=attempt)
            elif self._assessment_contract_version == 3:
                frame_registry = {role + "_s": [row["elapsed_s"] for row in prepared.provenance[role]["frames"]]
                                  for role in ("reference", "execution")}
                decode_assessment('{"assessment_contract_version":3,"status":"ok",'
                    '"observations":[],"updates":[],"pending_updates":[]}', context=prior,
                    frame_registry=frame_registry, max_response_bytes=self._max_response_bytes)
                host_context_validated = True
                assessment = decode_assessment(text, context=prior,
                    frame_registry=frame_registry, max_response_bytes=self._max_response_bytes)
                result = self._assessment_result(prepared, outcome="completed", assessment=assessment, attempt=attempt)
            elif self._assessment_contract_version == 2:
                from .assessment_completion import decode_assessment_completion
                images = prepared.input_representation == "jpeg_frames"
                known_ids = [row["issue_id"] for row in prior["issues"] if row["state"] != "resolved"]
                frame_registry = {role + "_s": [row["timestamp_s" if images else "elapsed_s"]
                        for row in prepared.provenance[role]["frames" if images else "frame_map"]]
                                  for role in ("reference", "execution")}
                # Validate host-owned identifiers and frame registry separately so
                # malformed provenance is never mistaken for bad model content.
                decode_assessment_completion('{"status":"ok","issues":[],"updates":[]}',
                    known_open_issue_ids=known_ids, frame_registry=frame_registry,
                    max_response_bytes=self._max_response_bytes)
                host_context_validated = True
                assessment = decode_assessment_completion(text,
                    known_open_issue_ids=known_ids, frame_registry=frame_registry,
                    max_response_bytes=self._max_response_bytes)
                result = self._assessment_result(prepared, outcome="completed", assessment=assessment, attempt=attempt)
            else:
                from .short_video_completion import decode_short_video_completion
                from .visual_completion import decode_visual_completion
                decode = decode_short_video_completion if prepared.short else decode_visual_completion
                result = decode(text, model_input=model_input, prior_issues=prior,
                    dispatched_at_s=cutoff, max_response_bytes=self._max_response_bytes)
        except ValueError:
            recoverable = (self._assessment_contract_version in (2, 3) and prepared.stage is None
                           and status == 200 and self._settled_usage and attempt["cost_complete"]
                           and host_context_validated and not self._stopped)
            return self._failed("invalid_completion", model_input, prior, attempt,
                                phase="decode", status=status, prepared=prepared, recoverable=recoverable)
        except RuntimeError:
            return self._failed("invalid_completion", model_input, prior, attempt,
                                phase="decode", status=status, prepared=prepared)
        attempt["status"] = "completed"
        return result

    def __call__(self, model_input, prior_context, *, observation=None, selected_indices=None):
        """Compatibility wrapper; existing replay stays synchronous."""
        if self._stopped:
            raise RuntimeError("Native completion batch stopped after a provider or schema failure")
        if self._input_representation == "jpeg_frames":
            prepared = self.prepare(model_input, prior_context, observation=observation, selected_indices=selected_indices)
        else:
            if observation is not None or selected_indices is not None:
                raise ValueError("Observation selection requires JPEG input")
            prepared = self.prepare(model_input, prior_context)
        cutoff = self._bind(prepared)
        if cutoff == 0:
            self._session_id, self._last_cutoff = prepared.model_input["session_id"], cutoff
            self._emit("provider_skipped", {"observation_id": prepared.model_input["observation_id"],
                "reason": "host_waiting_for_positive_duration_execution", "model_assessed": False}, cutoff)
            self._skipped += 1
            if self._assessment_contract_version in (2, 3):
                return self._assessment_result(prepared, outcome="not_assessed")
            from .visual_completion import decode_visual_completion
            return decode_visual_completion(self._json({"status": "insufficient_evidence",
                "candidate_issues": [], "candidate_links": [], "issue_updates": [],
                "new_issue_candidate_indices": []}).decode("utf-8"), model_input=prepared.model_input,
                prior_issues=prepared.prior, dispatched_at_s=cutoff, max_response_bytes=self._max_response_bytes)
        ticket = self.admit(prepared)
        return self.complete(ticket, self.perform(ticket))

    def snapshot(self):
        """Return detached accounting; missing usage never implies no charge."""
        result = {"generation_calls": sum(ticket.started for ticket in self._tickets),
            "skipped_calls": self._skipped,
            "reserved_usd": format(self._reserved, "f"),
            "known_rate_equivalent_usd": format(self._known, "f"),
            "cost_complete": all(item["cost_complete"] for item in self._attempts),
            "cost_is_invoice": False, "count_basis": self._COUNT_BASIS,
            "attempts": deepcopy(self._attempts), "stopped": self._stopped}
        if self._settled_usage:
            result.update(accounting_mode="settled_usage",
                          budget_committed_usd=format(self._committed, "f"))
        return result
