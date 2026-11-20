"""Bounded preparation and transport work for one serialized coordinator."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import math
from queue import Empty, SimpleQueue
from time import monotonic


@dataclass(frozen=True, eq=False)
class _Prepared:
    owner: object = field(repr=False)
    job_id: int
    cutoff_s: float
    context: object = field(repr=False)
    value: object = field(repr=False)
    error: object = field(repr=False)


@dataclass(frozen=True)
class _Completed:
    cutoff_s: float
    context: object = field(repr=False)
    ticket: object = field(repr=False)
    value: object = field(repr=False)
    error: object = field(repr=False)
    allow_state: bool


class CompletionScheduler:
    """Run bounded work without owning provider admission, state or accounting.

    One coordinator calls every public method. Only ``prepare(cutoff, context)``
    and ``perform(ticket)`` run on workers. Preparation has one lane; transport
    has configurable capacity. Ready, unconsumed work retains its capacity.

    The coordinator takes a prepared handle, checks its frozen context, obtains
    a provider admission ticket, then dispatches it. The scheduler never admits
    provider work or writes traces. A dispatch submission error returns that
    ticket in a completed record so its owner can account for/cancel it safely.

    Stop disables new work and future state effects, not physical HTTP. Poll and
    consume outstanding results after Stop to finish accounting. A drain deadline
    only reports unknown work. It neither cancels workers nor closes their client.
    Supplied executors are caller-owned and must return concurrent Futures.
    """

    def __init__(self, *, max_in_flight, prepare, perform, clock=monotonic,
                 prepare_executor=None, transport_executor=None):
        if type(max_in_flight) is not int or max_in_flight <= 0:
            raise ValueError("Outstanding work capacity must be a positive integer")
        if not all(callable(value) for value in (prepare, perform, clock)):
            raise ValueError("Preparation, transport and clock must be callable")
        for executor in (prepare_executor, transport_executor):
            if executor is not None and not callable(getattr(executor, "submit", None)):
                raise ValueError("Injected executors must provide submit")
        self._capacity = max_in_flight
        self._prepare, self._perform, self._clock = prepare, perform, clock
        self._owns_prepare = prepare_executor is None
        self._owns_transport = transport_executor is None
        self._prepare_executor = (ThreadPoolExecutor(max_workers=1)
                                  if self._owns_prepare else prepare_executor)
        self._transport_executor = (ThreadPoolExecutor(max_workers=max_in_flight)
                                    if self._owns_transport else transport_executor)
        self._owner = object()
        self._jobs = {}
        self._prepared = deque()
        self._completed = deque()
        self._finished = SimpleQueue()
        self._pending = None
        self._last_offered = self._last_clock = None
        self._last_context_revision = None
        self._next_id = self._coalesced = 0
        self._playing = self._stopped = self._closed = False
        self._drain_deadline = None

    @staticmethod
    def _number(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative number")
        return value

    def _now(self):
        value = self._number(self._clock())
        if self._last_clock is not None and value < self._last_clock:
            raise ValueError("Scheduler clock cannot reverse")
        self._last_clock = value
        return value

    def _open(self):
        if self._closed:
            raise RuntimeError("Scheduler is closed")

    def set_playing(self, playing):
        self._open()
        if type(playing) is not bool:
            raise ValueError("Playback state must be a boolean")
        if self._stopped and playing:
            raise RuntimeError("Stopped scheduler cannot resume")
        self._playing = playing

    def offer(self, cutoff_s, context, *, context_revision=None):
        """Keep the latest opportunity, optionally refreshing equal-C context.

        Ordinary callers must advance source time. Explicit context revisions
        allow the same source prefix once per newer revision, without changing
        its timestamp, capacity or any existing drain deadline.
        """
        self._open()
        cutoff_s = self._number(cutoff_s)
        if context_revision is not None and (type(context_revision) is not int or context_revision < 0):
            raise ValueError("Context revision must be a nonnegative integer or None")
        if self._stopped:
            return False
        if self._last_offered is not None and (cutoff_s < self._last_offered or cutoff_s == self._last_offered and (
                context_revision is None or self._last_context_revision is None
                or context_revision <= self._last_context_revision)):
            raise ValueError("Offer must advance its cutoff or equal-cutoff context revision")
        frozen = deepcopy(context)
        if self._pending is not None:
            self._coalesced += 1
        self._pending = (cutoff_s, frozen)
        self._last_offered = cutoff_s
        self._last_context_revision = context_revision
        return True

    def _submit(self, job_id, stage, executor, function, *args):
        job = self._jobs[job_id]
        job["stage"] = stage
        try:
            future = executor.submit(function, *args)
        except BaseException as exc:
            self._record_result(job_id, stage, None, exc)
            return
        job["future"] = future
        # The callback only posts a bounded ready notification. It never mutates
        # coordinator state, invokes provider admission or appends trace events.
        future.add_done_callback(lambda done: self._finished.put((job_id, stage, done)))

    def _record_result(self, job_id, stage, value, error):
        job = self._jobs[job_id]
        job["future"] = None
        if stage == "preparing":
            handle = _Prepared(self._owner, job_id, job["cutoff_s"],
                               deepcopy(job["context"]), value, error)
            job.update(stage="prepared", handle=handle)
            self._prepared.append(handle)
        else:
            job.update(stage="completed", value=value, error=error, handle=None)
            self._completed.append(job_id)

    def poll(self):
        """Harvest ready work and start at most one eligible preparation; never wait."""
        self._open()
        self._now()
        while True:
            try:
                job_id, stage, future = self._finished.get_nowait()
            except Empty:
                break
            job = self._jobs.get(job_id)
            if job is None or job["future"] is not future or job["stage"] != stage:
                continue
            try:
                value, error = future.result(), None
            except BaseException as exc:
                value, error = None, exc
            self._record_result(job_id, stage, value, error)
        if (self._stopped or not self._playing or self._pending is None
                or len(self._jobs) >= self._capacity
                or any(job["stage"] == "preparing" for job in self._jobs.values())):
            return
        cutoff, context = self._pending
        worker_context = deepcopy(context)
        self._pending = None
        self._next_id += 1
        job_id = self._next_id
        self._jobs[job_id] = {"cutoff_s": cutoff, "context": context, "ticket": None,
                              "future": None, "stage": "preparing", "handle": None}
        self._submit(job_id, "preparing", self._prepare_executor,
                     self._prepare, cutoff, worker_context)

    def take_prepared(self):
        """Hand out a preparation once; its slot remains reserved until consumed."""
        self._open()
        while self._prepared:
            handle = self._prepared.popleft()
            job = self._jobs.get(handle.job_id)
            if job is not None and job["handle"] is handle and job["stage"] == "prepared":
                job["stage"] = "prepared_taken"
                return handle
        return None

    def _prepared_job(self, handle):
        if not isinstance(handle, _Prepared) or handle.owner is not self._owner:
            raise ValueError("Prepared handle belongs to another scheduler")
        job = self._jobs.get(handle.job_id)
        if (job is None or job["handle"] is not handle
                or job["stage"] not in ("prepared", "prepared_taken")):
            raise ValueError("Prepared handle was already dispatched or discarded")
        return job

    def dispatch(self, handle, ticket):
        """Submit an already admitted ticket; the coordinator owns admission."""
        self._open()
        job = self._prepared_job(handle)
        if self._stopped or not self._playing:
            raise RuntimeError("Playback does not permit new transport dispatch")
        if handle.error is not None:
            raise ValueError("Failed preparation cannot be dispatched")
        job["ticket"] = ticket
        job["handle"] = None
        self._submit(handle.job_id, "transport", self._transport_executor,
                     self._perform, ticket)

    def discard_prepared(self, handle):
        """Release prepared work that never received a transport dispatch."""
        self._open()
        self._prepared_job(handle)
        del self._jobs[handle.job_id]

    def take_completed(self):
        """Consume one ready result and free its slot, preserving its ticket.

        ``allow_state`` reflects Stop at consumption. The coordinator must still
        check its live session before applying state or displaying a notification.
        """
        self._open()
        if not self._completed:
            return None
        job = self._jobs.pop(self._completed.popleft())
        return _Completed(job["cutoff_s"], deepcopy(job["context"]), job["ticket"],
                          job["value"], job["error"], not self._stopped)

    def stop(self, *, drain_timeout_s):
        """Disable new work immediately; deadlines report uncertainty, not cancellation."""
        self._open()
        timeout = self._number(drain_timeout_s)
        if not self._stopped:
            now = self._now()
            deadline = now + timeout
            self._number(deadline)
            self._playing = False
            self._stopped = True
            self._pending = None
            self._drain_deadline = deadline

    def snapshot(self):
        now = self._now()
        expired = self._stopped and now >= self._drain_deadline
        unknown = sum(job["future"] is not None and not job["future"].done()
                      for job in self._jobs.values()) if expired else 0
        return {"outstanding": len(self._jobs), "stopped": self._stopped,
                "unknown_outstanding": unknown,
                "queued_cutoff_s": None if self._pending is None else self._pending[0],
                "coalesced_opportunities": self._coalesced, "closed": self._closed}

    def close(self):
        """Close owned executors only after all results/preparations are consumed."""
        if self._closed:
            return
        if self._jobs:
            raise RuntimeError("Outstanding work must be drained and consumed before closing")
        self._pending = None
        self._playing = False
        self._closed = True
        if self._owns_prepare:
            self._prepare_executor.shutdown(wait=True)
        if self._owns_transport:
            self._transport_executor.shutdown(wait=True)
