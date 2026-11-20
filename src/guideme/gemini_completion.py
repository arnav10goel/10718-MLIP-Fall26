"""A bounded Gemini completion callback with exact semantic request tracing."""

from copy import deepcopy
import math
from time import perf_counter

from google.genai import errors
import httpx


class GeminiCompletionPlan:
    """Call an injected Gemini client through pure request/response adapters.

    The caller supplies authentication, a fixed model, bounded build_request and
    decode_response callables, and a bound trace. No environment, media paths,
    uploads, tools, cache creation or retries are used. Input bytes count the
    builder's JPEG and UTF-8 text/schema bytes, not HTTP/base64 overhead, tokens,
    dollar cost or total Python memory. Call/input reservations are not refunded
    after provider errors. Use callback_errors_fatal=True in content replay so
    persistence, budget and unexpected errors abort with the original exception.

    Trace times are observation cutoffs. elapsed_s measures only the synchronous
    provider call on the injected wall clock. HTTP timeouts round up to whole
    milliseconds and are not a guaranteed end-to-end wall deadline. Minimal
    thinking is requested, so the chosen model must support that setting. The
    client remains caller-owned. media_resolution=None preserves the provider
    default; LOW/MEDIUM/HIGH set a request-wide token allocation while leaving
    the supplied JPEGs unchanged.
    """

    _PROVIDER_STATUSES = frozenset({
        "OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
        "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "UNAUTHENTICATED",
        "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE",
        "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE", "DATA_LOSS",
    })

    def __init__(self, *, client, model, build_request, decode_response, trace,
                 max_calls, max_total_input_bytes, timeout_s, max_output_tokens,
                 media_resolution=None, clock=perf_counter):
        if media_resolution is not None and (type(media_resolution) is not str
                or media_resolution not in ("LOW", "MEDIUM", "HIGH")):
            raise ValueError("Media resolution must be None, LOW, MEDIUM or HIGH")
        generate = getattr(getattr(client, "models", None), "generate_content", None)
        if not all(callable(value) for value in (
                generate, build_request, decode_response, getattr(trace, "append", None), clock)):
            raise ValueError("Client, request/response adapters, trace and clock must be callable")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be an explicit nonblank string")
        for limit in (max_calls, max_total_input_bytes, max_output_tokens):
            if type(limit) is not int or limit <= 0:
                raise ValueError("Call, input-byte and output-token limits must be positive integers")
        self._number(timeout_s)
        self._number(timeout_s * 1000)
        if timeout_s <= 0:
            raise ValueError("HTTP timeout must be positive and representable in milliseconds")
        self._generate = generate
        self._model = model
        self._build_request, self._decode_response = build_request, decode_response
        self._trace, self._clock = trace, clock
        self._max_calls, self._max_input_bytes = max_calls, max_total_input_bytes
        self._timeout_ms, self._max_output_tokens = math.ceil(timeout_s * 1000), max_output_tokens
        self._media_resolution = (None if media_resolution is None else
                                  "MEDIA_RESOLUTION_" + media_resolution)
        self._calls = self._input_bytes = 0
        self._last_wall_s = None

    @property
    def calls(self):
        return self._calls

    @property
    def input_bytes(self):
        return self._input_bytes

    @staticmethod
    def _number(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Clock and timeout values must be finite nonnegative numbers")
        return value

    def _now(self):
        value = self._number(self._clock())
        if self._last_wall_s is not None and value < self._last_wall_s:
            raise ValueError("Provider wall clock cannot move backward")
        self._last_wall_s = value
        return value

    def _emit(self, kind, payload, cutoff):
        # Never place persistence inside a provider/decoder exception handler.
        self._trace.append(kind, payload, at_s=cutoff)

    def _failed(self, code, model_input, prior_issues, *, elapsed_s=None, exception=None):
        # Retain bounded diagnostic scalars, never exception text, response
        # objects or request/header data. SDK status can be arbitrary JSON.
        http_status = getattr(exception, "code", None)
        if type(http_status) is not int or not 100 <= http_status <= 599:
            http_status = None
        provider_status = getattr(exception, "status", None)
        if type(provider_status) is not str or provider_status not in self._PROVIDER_STATUSES:
            provider_status = None
        self._emit("provider_failure", {"observation_id": model_input["observation_id"],
            "code": code, "elapsed_s": elapsed_s,
            "exception_type": type(exception).__name__ if exception is not None else None,
            "http_status": http_status, "provider_status": provider_status}, model_input["cutoff_s"])
        return {"response": {"schema_version": 1,
                    "session_id": model_input["session_id"],
                    "observation_id": model_input["observation_id"],
                    "status": "failed", "candidate_issues": [],
                    "error": {"type": code, "message": "Visual provider completion failed"}},
                "updates": {"schema_version": 1,
                    "session_id": model_input["session_id"],
                    "observation_id": model_input["observation_id"],
                    "reference_id": prior_issues["reference_id"],
                    "candidate_links": [], "issue_updates": []},
                "new_issue_candidate_indices": []}

    @staticmethod
    def _text(raw):
        feedback = raw.get("prompt_feedback")
        if feedback is not None and (not isinstance(feedback, dict)
                or feedback.get("block_reason") not in (None, "BLOCKED_REASON_UNSPECIFIED")):
            raise ValueError("Provider blocked the prompt")
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Provider must return exactly one candidate")
        candidate = candidates[0]
        if not isinstance(candidate, dict) or candidate.get("finish_reason") != "STOP":
            raise ValueError("Provider candidate did not finish")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts:
            raise ValueError("Provider returned no text parts")
        texts = []
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise ValueError("Provider returned nontext output")
            if any(value is not None and key not in ("text", "thought", "thought_signature")
                   for key, value in part.items()):
                raise ValueError("Provider returned mixed text and nontext output")
            if part.get("thought") is not None and type(part["thought"]) is not bool:
                raise ValueError("Provider returned an invalid thought flag")
            if not part.get("thought", False):
                texts.append(part["text"])
        text = "".join(texts)
        if not text.strip():
            raise ValueError("Provider returned no ordinary completion text")
        return text

    def __call__(self, model_input, prior_issues):
        built = deepcopy(self._build_request(deepcopy(model_input), deepcopy(prior_issues)))
        counts = built["counts"]
        if (not isinstance(counts, dict)
                or set(counts) != {"images", "jpeg_bytes", "text_bytes", "prior_issues"}
                or any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("Request counts must be nonnegative ordinary integers")
        size = counts["jpeg_bytes"] + counts["text_bytes"]
        if self._calls >= self._max_calls:
            raise RuntimeError("Provider call limit exceeded")
        if size > self._max_input_bytes - self._input_bytes:
            raise RuntimeError("Provider total input-byte limit exceeded")
        request = {"model": self._model,
            "contents": [{"role": "user", "parts": built["parts"]}],
            "config": {"system_instruction": built["system_instruction"],
                "response_mime_type": "application/json",
                "response_json_schema": built["response_json_schema"],
                "candidate_count": 1, "max_output_tokens": self._max_output_tokens,
                "automatic_function_calling": {"disable": True},
                "thinking_config": {"thinking_level": "MINIMAL", "include_thoughts": False},
                "http_options": {"timeout": self._timeout_ms,
                                 "retry_options": {"attempts": 1}}}}
        if self._media_resolution is not None:
            request["config"]["media_resolution"] = self._media_resolution
        cutoff = model_input["cutoff_s"]
        self._emit("provider_request", {"observation_id": model_input["observation_id"],
            "request_sha256": built["request_sha256"], "prompt_sha256": built["prompt_sha256"],
            "counts": counts, "request": deepcopy(request)}, cutoff)
        started = self._now()
        self._calls += 1
        self._input_bytes += size
        try:
            response = self._generate(**request)
        except (TimeoutError, httpx.TimeoutException) as exc:
            return self._failed("provider_timeout", model_input, prior_issues,
                                elapsed_s=self._now() - started, exception=exc)
        except (errors.APIError, httpx.RequestError) as exc:
            return self._failed("provider_error", model_input, prior_issues,
                                elapsed_s=self._now() - started, exception=exc)
        except errors.UnknownApiResponseError as exc:
            return self._failed("invalid_provider_response", model_input, prior_issues,
                                elapsed_s=self._now() - started, exception=exc)
        elapsed = self._now() - started
        raw = response.model_dump(mode="json")
        if not isinstance(raw, dict):
            return self._failed("invalid_provider_response", model_input, prior_issues,
                                elapsed_s=elapsed)
        # The SDK's full HTTP response can include authentication-bearing
        # headers. Save only provider output fields, including raw text parts.
        raw = {key: deepcopy(raw[key]) for key in (
            "candidates", "prompt_feedback", "usage_metadata", "model_version",
            "response_id", "create_time", "model_status") if key in raw}
        self._emit("provider_response", {"observation_id": model_input["observation_id"],
            "request_sha256": built["request_sha256"], "response": raw,
            "elapsed_s": elapsed}, cutoff)
        try:
            text = self._text(raw)
        except ValueError:
            return self._failed("invalid_provider_response", model_input, prior_issues,
                                elapsed_s=elapsed)
        try:
            return self._decode_response(text, model_input=deepcopy(model_input),
                prior_issues=deepcopy(prior_issues), dispatched_at_s=cutoff)
        except (ValueError, RuntimeError):
            return self._failed("invalid_completion", model_input, prior_issues,
                                elapsed_s=elapsed)
