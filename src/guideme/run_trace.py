"""Bounded, incremental JSONL traces with explicit durable-write receipts."""

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from uuid import UUID


class RunTraceWriter:
    """Create one exclusive trace and acknowledge each fully synchronized event.

    Callers supply credential-free events, including configuration/start/completion
    events explicitly. JSON values retain their values; bytes are represented by
    null placeholders and a separate, ordered binary table. No model, environment
    configuration or HTTP headers are automatically logged.

    Each successful append writes a complete line, flushes and fsyncs the file.
    This preserves acknowledged prefixes across process interruption; it does not
    guarantee parent-directory or power-loss durability. After uncertain I/O the
    writer closes permanently. Its last suffix may be torn or complete without an
    acknowledgment, so parseability alone cannot establish writer acknowledgment.
    There is no resume, reader, rollback or recovery mechanism. Use one serialized
    caller and explicit close or a context manager.

    Optional payload deduplication selects schema 2. Immutable bytes and UTF-8
    strings of at least 4096 bytes are defined once, then referenced by their
    type, digest and size. The cache retains only those identities and advances
    after a durable append; references may also follow definitions in one record.
    """

    def __init__(self, path, *, run_id, session_id, reference_id,
                 max_events, max_bytes, data_root=None, deduplicate_payloads=False):
        if type(deduplicate_payloads) is not bool:
            raise ValueError("deduplicate_payloads must be boolean")
        for identity in (run_id, session_id):
            try:
                valid = isinstance(identity, str) and str(UUID(identity)) == identity
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("Run and session IDs must be canonical UUID strings")
        if not isinstance(reference_id, str) or re.fullmatch(r"[0-9a-f]{64}", reference_id) is None:
            raise ValueError("Reference ID must be a lowercase SHA-256 identity")
        for limit in (max_events, max_bytes):
            if type(limit) is not int or limit <= 0:
                raise ValueError("Event and byte limits must be positive integers")
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("Path must be a nonblank local path")
        destination = Path(path).expanduser()
        root = None
        if not destination.is_absolute():
            configured = (data_root if data_root is not None else
                          os.environ.get("GUIDEME_DATA_ROOT") or
                          Path(__file__).resolve().parents[2] / "data")
            if not isinstance(configured, (str, Path)) or not str(configured).strip():
                raise ValueError("Data root must be a nonblank absolute path")
            root = Path(configured).expanduser()
            if not root.is_absolute():
                raise ValueError("Data root must be absolute")
            root = root.resolve()
            destination = root / destination
        # Preserve the final component: resolving it would follow a dangling
        # symlink before exclusive creation and accidentally create its target.
        parent = destination.parent.resolve()
        if root is not None and not parent.is_relative_to(root):
            raise ValueError("Relative trace path must remain within the data root")
        self._path = parent / destination.name
        self._run_id, self._session_id = run_id, session_id
        self._reference_id = reference_id
        self._max_events, self._max_bytes = max_events, max_bytes
        self._events_written = self._bytes_written = 0
        self._last_at_s = None
        self._closed = self._failed = False
        self._deduplicate_payloads = deduplicate_payloads
        self._payload_keys = set()
        self._file = open(self._path, "xb", buffering=0)
        try:
            if not stat.S_ISREG(os.fstat(self._file.fileno()).st_mode):
                raise ValueError("Trace destination must be a regular file")
        except BaseException:
            self._fail_close()
            raise

    @property
    def path(self):
        return self._path

    @property
    def closed(self):
        return self._closed

    @property
    def failed(self):
        return self._failed

    @property
    def events_written(self):
        """Number of acknowledged events, excluding uncertain failed writes."""
        return self._events_written

    @property
    def bytes_written(self):
        """Acknowledged encoded bytes; a failed suffix may occupy more on disk."""
        return self._bytes_written

    @staticmethod
    def _payload(value, path, binaries, ancestors):
        kind = type(value)
        if value is None or kind in (bool, int, str):
            return value
        if kind is float:
            if not math.isfinite(value):
                raise ValueError("Payload numbers must be finite")
            return value
        if kind is bytes:
            binaries.append({"path": path, "encoding": "base64",
                             "data": base64.b64encode(value).decode("ascii"),
                             "sha256": hashlib.sha256(value).hexdigest(),
                             "size_bytes": len(value)})
            return None
        if kind not in (dict, list):
            raise ValueError("Payload must contain only JSON values and immutable bytes")
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Payload cannot contain cycles")
        ancestors.add(identity)
        try:
            if kind is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError("Payload object keys must be strings")
                return {key: RunTraceWriter._payload(value[key], path + [key], binaries, ancestors)
                        for key in sorted(value)}
            return [RunTraceWriter._payload(item, path + [index], binaries, ancestors)
                    for index, item in enumerate(value)]
        finally:
            ancestors.remove(identity)

    def _deduplicated_payload(self, value, path, binaries, ancestors, pending):
        kind = type(value)
        if kind in (bytes, str):
            raw = value if kind is bytes else value.encode("utf-8")
            if kind is str and len(raw) < 4096:
                return value
            digest = hashlib.sha256(raw).hexdigest()
            key = (kind, digest, len(raw))
            reference = key in self._payload_keys or key in pending
            encoding = ("sha256" if reference else "base64")
            if kind is str:
                encoding += "-utf8"
            entry = {"path": path, "encoding": encoding,
                     "sha256": digest, "size_bytes": len(raw)}
            if not reference:
                entry["data"] = base64.b64encode(raw).decode("ascii")
                pending.add(key)
            binaries.append(entry)
            return None
        if value is None or kind in (bool, int):
            return value
        if kind is float:
            if not math.isfinite(value):
                raise ValueError("Payload numbers must be finite")
            return value
        if kind not in (dict, list):
            raise ValueError("Payload must contain only JSON values and immutable bytes")
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Payload cannot contain cycles")
        ancestors.add(identity)
        try:
            if kind is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError("Payload object keys must be strings")
                return {key: self._deduplicated_payload(value[key], path + [key], binaries,
                                                       ancestors, pending)
                        for key in sorted(value)}
            return [self._deduplicated_payload(item, path + [index], binaries, ancestors, pending)
                    for index, item in enumerate(value)]
        finally:
            ancestors.remove(identity)

    def append(self, event_type, payload, *, at_s):
        """Return sequence, exact-line SHA-256, line bytes and total ACK bytes.

        Validation raises ValueError and capacity raises RuntimeError before any
        I/O or retained-state change. Write/flush/fsync failures, including process
        interruptions, propagate their original exception and poison the writer.
        Event time is nonnegative and cannot precede the last acknowledged event;
        numerical payload values, including negative source PTS, are unchanged.
        """
        if self._closed or self._failed:
            raise RuntimeError("Trace writer is closed or failed")
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("Event type must be a nonblank string")
        try:
            finite_time = type(at_s) in (int, float) and math.isfinite(at_s)
        except OverflowError:
            finite_time = False
        if not finite_time or at_s < 0:
            raise ValueError("Event time must be an ordinary finite nonnegative number")
        if self._last_at_s is not None and at_s < self._last_at_s:
            raise ValueError("Event time cannot move backward")
        binaries, pending = [], set()
        try:
            converted = (self._deduplicated_payload(payload, [], binaries, set(), pending)
                         if self._deduplicate_payloads else self._payload(payload, [], binaries, set()))
            record = {"schema_version": 2 if self._deduplicate_payloads else 1, "run_id": self._run_id,
                      "session_id": self._session_id, "reference_id": self._reference_id,
                      "sequence": self._events_written, "event_type": event_type,
                      "at_s": at_s, "payload": converted, "binary_payloads": binaries}
            line = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Event must contain finite UTF-8 JSON values and bytes") from exc
        line_bytes = len(line)
        total_bytes = self._bytes_written + line_bytes
        if self._events_written >= self._max_events or total_bytes > self._max_bytes:
            raise RuntimeError("Trace event or encoded-byte limit exceeded")
        receipt = {"sequence": self._events_written,
                   "sha256": hashlib.sha256(line).hexdigest(),
                   "line_bytes": line_bytes, "total_bytes": total_bytes}
        try:
            remaining = memoryview(line)
            while remaining:
                written = self._file.write(remaining)
                if type(written) is not int or written <= 0 or written > len(remaining):
                    raise OSError("Trace write made no valid forward progress")
                remaining = remaining[written:]
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self._fail_close()
            raise
        self._payload_keys.update(pending)
        self._events_written += 1
        self._bytes_written = total_bytes
        self._last_at_s = at_s
        return receipt

    def _fail_close(self):
        self._failed = self._closed = True
        try:
            self._file.close()
        except BaseException:
            # Keep the original I/O error or interruption, never a cleanup error.
            pass

    def close(self):
        """Close once; a close error marks the writer failed and propagates."""
        if self._closed:
            return
        self._closed = True
        try:
            self._file.close()
        except BaseException:
            self._failed = True
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
