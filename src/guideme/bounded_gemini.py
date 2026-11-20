"""Serial, injected token and spending guards for the existing Gemini adapter."""

from copy import deepcopy
from decimal import Decimal, DecimalException, Inexact, localcontext
import math
from time import perf_counter

from .gemini_token_request import build_count_tokens_request


class BoundedGeminiClient:
    """Expose one count-before-generation path, without authentication or retries.

    The caller owns the SDK client and its one-attempt configuration. The strict
    count mapper validates request settings, including disabled SDK retries.
    Reserve the maximum declared input/output cost before each generation; an
    exception never refunds that reservation. Rates are caller-supplied listed
    rates, not a billing quote. Known cost is a partial rate-equivalent subtotal
    whenever a generation has missing usage or fails. Missing thinking counts
    are treated as zero, matching the existing adapter's usage convention.

    ``emit(kind, payload)`` supplies persistence and the source clock outside
    this transport. Phase durations use the injected monotonic wall clock.
    Provider exceptions propagate unchanged; persistence, clock, budget and
    unexpected errors must abort replay through callback_errors_fatal=True.
    This class supports serial calls only and never creates or closes a client.
    """

    _PROVIDER_STATUSES = frozenset({
        "OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
        "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "UNAUTHENTICATED",
        "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE",
        "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE", "DATA_LOSS",
    })

    def __init__(self, client, *, emit, limits, clock=perf_counter):
        integer_fields = ("max_token_checks", "max_generation_calls", "max_input_tokens",
                          "max_output_tokens")
        money_fields = ("budget_usd", "input_usd_per_million", "output_usd_per_million")
        if not isinstance(limits, dict) or set(limits) != set(integer_fields + money_fields):
            raise ValueError("Limits must contain the explicit call, token and monetary settings")
        for key in integer_fields:
            if type(limits[key]) is not int or limits[key] <= 0:
                raise ValueError("Call and token limits must be positive ordinary integers")
        amounts = []
        try:
            for key in money_fields:
                value = limits[key]
                if type(value) is not str or not value or value.strip() != value:
                    raise ValueError("Monetary settings must be decimal strings")
                amount = Decimal(value)
                if not amount.is_finite() or amount < 0:
                    raise ValueError("Monetary settings must be finite and nonnegative")
                amounts.append(amount)
            self._budget, self._input_rate, self._output_rate = amounts
            # Do not silently round a reservation down for unrepresentable rates.
            with localcontext() as context:
                context.traps[Inexact] = True
                self._reservation = (limits["max_input_tokens"] * self._input_rate
                    + limits["max_output_tokens"] * self._output_rate) / Decimal(1_000_000)
        except DecimalException as exc:
            raise ValueError("Monetary settings cannot be represented exactly") from exc
        models = getattr(client, "models", None)
        count = getattr(models, "count_tokens", None)
        generate = getattr(models, "generate_content", None)
        if not all(callable(item) for item in (count, generate, emit, clock)):
            raise ValueError("Client methods, emit and clock must be callable")
        self._count, self._generate = count, generate
        self._emit_callback, self._clock = emit, clock
        self._limits = deepcopy(limits)
        self._token_checks = self._generation_calls = 0
        self._reserved = self._known = Decimal(0)
        self._attempts = []
        self._last_clock = None
        self._halted = False

    @property
    def models(self):
        return self

    def _emit(self, kind, payload):
        try:
            self._emit_callback(kind, deepcopy(payload))
        except BaseException:
            self._halted = True
            raise

    def _now(self):
        try:
            value = self._clock()
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (self._last_clock is not None and value < self._last_clock)):
                raise ValueError("Wall clock must be finite, nonnegative and monotonic")
        except BaseException:
            self._halted = True
            raise
        self._last_clock = value
        return value

    @staticmethod
    def _field(record, name):
        return record.get(name) if isinstance(record, dict) else getattr(record, name, None)

    def _call(self, method, request, attempt, phase):
        started = self._now()
        try:
            result = method(**deepcopy(request))
        except BaseException as exc:
            attempt[phase + "_s"] = self._now() - started
            status = getattr(exc, "status", None)
            http_status = getattr(exc, "code", None)
            failure = {"phase": phase, "exception_type": type(exc).__name__,
                "http_status": http_status if type(http_status) is int and 100 <= http_status <= 599 else None,
                "provider_status": status if type(status) is str and status in self._PROVIDER_STATUSES else None}
            attempt.update(status=phase + "_failed", failure=failure)
            self._emit(phase + "_failure", failure)
            raise
        attempt[phase + "_s"] = self._now() - started
        return result

    def generate_content(self, **request):
        """Validate, count, reserve and attempt exactly one generation."""
        request = deepcopy(request)
        counted_request = build_count_tokens_request(request)
        if request["config"]["max_output_tokens"] != self._limits["max_output_tokens"]:
            raise ValueError("Request output cap must equal the approved token limit")
        if (self._halted or self._token_checks >= self._limits["max_token_checks"]
                or self._generation_calls >= self._limits["max_generation_calls"]):
            raise RuntimeError("Transport is stopped or its call allowance is exhausted")
        with localcontext() as context:
            context.traps[Inexact] = True
            reserved = self._reserved + self._reservation
        if reserved > self._budget:
            raise RuntimeError("Remaining budget cannot cover the maximum generation reservation")
        attempt = {"status": "token_check_pending", "token_check_s": None,
            "generation_s": None, "counted_input_tokens": None, "reserved_usd": "0",
            "usage": None, "known_rate_equivalent_usd": None, "cost_unknown": False,
            "failure": None}
        self._attempts.append(attempt)
        self._emit("token_preflight_request", {"request": counted_request})
        self._token_checks += 1
        counted = self._call(self._count, counted_request, attempt, "token_check")
        tokens = self._field(counted, "total_tokens")
        valid = type(tokens) is int and tokens >= 0
        self._emit("token_preflight_response", {"total_tokens": tokens if valid else None,
            "elapsed_s": attempt["token_check_s"]})
        if not valid or tokens > self._limits["max_input_tokens"]:
            attempt["status"] = "input_token_limit" if valid else "invalid_token_count"
            raise RuntimeError("Token preflight did not establish an input within its limit")
        attempt["counted_input_tokens"] = tokens
        self._emit("budget_reservation", {"reserved_call_usd": str(self._reservation),
            "reserved_total_usd": str(reserved), "budget_usd": str(self._budget),
            "cost_is_invoice": False})
        self._reserved = reserved
        self._generation_calls += 1
        attempt.update(status="generation_pending", reserved_usd=str(self._reservation), cost_unknown=True)
        response = self._call(self._generate, request, attempt, "generation")
        raw_usage = self._field(response, "usage_metadata")
        if raw_usage is not None:
            usage = {}
            for name in ("prompt_token_count", "candidates_token_count", "thoughts_token_count",
                         "cached_content_token_count"):
                value = self._field(raw_usage, name)
                usage[name] = value if type(value) is int and value >= 0 else None
            attempt["usage"] = usage
            inputs, outputs = usage["prompt_token_count"], usage["candidates_token_count"]
            thoughts = self._field(raw_usage, "thoughts_token_count")
            thoughts = 0 if thoughts is None else thoughts
            if all(type(value) is int and value >= 0 for value in (inputs, outputs, thoughts)):
                with localcontext() as context:
                    context.traps[Inexact] = True
                    cost = (inputs * self._input_rate + (outputs + thoughts) * self._output_rate) / Decimal(1_000_000)
                    self._known += cost
                attempt.update(known_rate_equivalent_usd=str(cost), cost_unknown=False)
                if (inputs > self._limits["max_input_tokens"]
                        or outputs + thoughts > self._limits["max_output_tokens"]
                        or cost > self._reservation):
                    self._halted = True
                    attempt["status"] = "usage_limit"
                    self._emit("generation_usage_limit", attempt)
                    raise RuntimeError("Returned usage exceeded the reserved generation limits")
        attempt["status"] = "completed"
        self._emit("generation_finished", attempt)
        return response

    def snapshot(self):
        """Copy counters, retained reservations and explicitly partial cost evidence."""
        return {"token_checks": self._token_checks, "generation_calls": self._generation_calls,
            "reserved_usd": str(self._reserved), "known_rate_equivalent_usd": str(self._known),
            "cost_complete": not any(attempt["cost_unknown"] for attempt in self._attempts),
            "cost_is_invoice": False, "known_cost_is_partial": any(
                attempt["cost_unknown"] for attempt in self._attempts),
            "stopped": self._halted, "attempts": deepcopy(self._attempts)}
