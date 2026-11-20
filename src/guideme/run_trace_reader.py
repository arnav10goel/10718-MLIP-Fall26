"""Bounded, read-only validation of exact-payload run-trace prefixes."""

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from uuid import UUID


def read_run_trace(path, *, max_events, max_bytes, data_root=None):
    """Return the valid decoded prefix and the first damaged suffix, if any.

    ``valid`` describes canonical syntax and the lossless payload codec only;
    neither a full final newline nor a completion event proves writer
    acknowledgment, run completion, or physical-task correctness. The writer can
    leave a complete unacknowledged line after uncertain I/O. This reader does
    not repair, truncate, resume, authenticate, or establish a concurrent-file
    snapshot. It reads at most ``max_bytes + 1`` bytes and raises RuntimeError
    on either capacity limit. Complete newline frames count toward max_events;
    an unterminated final fragment does not. Unknown I/O errors propagate.

    Schema 2 definitions and backward references restore immutable bytes or
    UTF-8 strings without copying repeated values. Definitions become available
    across records only after their complete containing record validates. The
    returned envelope stays schema 1 for existing downstream consumers.
    """
    for limit in (max_events, max_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Event and byte limits must be positive integers")
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Path must be a nonblank local path")
    source_path = Path(path).expanduser()
    root = None
    if not source_path.is_absolute():
        configured = (data_root if data_root is not None else
                      os.environ.get("GUIDEME_DATA_ROOT") or
                      Path(__file__).resolve().parents[2] / "data")
        if not isinstance(configured, (str, Path)) or not str(configured).strip():
            raise ValueError("Data root must be a nonblank absolute path")
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError("Data root must be absolute")
        root = root.resolve()
        source_path = root / source_path
    parent = source_path.parent.resolve()
    if root is not None and not parent.is_relative_to(root):
        raise ValueError("Relative trace path must remain within the data root")
    source_path = parent / source_path.name
    # Nonblocking open prevents a FIFO from hanging before the regular-file
    # check. Do not follow a final symlink out of a relative data-root path.
    descriptor = os.open(source_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Trace source must be a regular file")
        content = bytearray()
        while len(content) <= max_bytes:
            chunk = os.read(descriptor, min(65_536, max_bytes + 1 - len(content)))
            if chunk == b"":
                break
            content.extend(chunk)
        if len(content) > max_bytes:
            raise RuntimeError("Trace encoded-byte limit exceeded")
    finally:
        os.close(descriptor)

    def reject(message):
        raise ValueError(message)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                reject("Duplicate JSON object key")
            result[key] = value
        return result

    def canonical_uuid(value):
        return type(value) is str and str(UUID(value)) == value

    def sha256_identity(value):
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    record_fields = {"schema_version", "run_id", "session_id", "reference_id",
                     "sequence", "event_type", "at_s", "payload", "binary_payloads"}
    binary_fields = {"path", "encoding", "data", "sha256", "size_bytes"}
    reference_fields = binary_fields - {"data"}
    definitions = {}

    def decode_record(line, sequence, bindings, last_at_s):
        record = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=reject)
        canonical = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        if canonical != line:
            reject("Trace record is not canonical JSONL")
        if type(record) is not dict or set(record) != record_fields:
            reject("Invalid trace record fields")
        version = record["schema_version"]
        if type(version) is not int or version not in (1, 2):
            reject("Invalid trace schema")
        if not canonical_uuid(record["run_id"]) or not canonical_uuid(record["session_id"]):
            reject("Invalid trace UUID")
        if not sha256_identity(record["reference_id"]):
            reject("Invalid trace reference identity")
        current_bindings = (record["run_id"], record["session_id"], record["reference_id"], version)
        if bindings is not None and current_bindings != bindings:
            reject("Inconsistent trace bindings")
        if type(record["sequence"]) is not int or record["sequence"] != sequence:
            reject("Invalid trace sequence")
        at_s = record["at_s"]
        if type(at_s) not in (int, float) or not math.isfinite(at_s) or at_s < 0:
            reject("Invalid trace event time")
        if last_at_s is not None and at_s < last_at_s:
            reject("Trace event time moved backward")
        if type(record["event_type"]) is not str or not record["event_type"].strip():
            reject("Invalid trace event type")
        if type(record["binary_payloads"]) is not list:
            reject("Invalid binary payload table")
        # Validate all paths and null placeholders before replacing any of them.
        replacements = []
        paths = []
        staged_definitions = {}
        for binary in record["binary_payloads"]:
            if type(binary) is not dict:
                reject("Invalid binary payload fields")
            encoding = binary.get("encoding")
            is_reference = version == 2 and encoding in ("sha256", "sha256-utf8")
            if (set(binary) != (reference_fields if is_reference else binary_fields)
                    or (version == 1 and encoding != "base64")
                    or (version == 2 and encoding not in (
                        "base64", "base64-utf8", "sha256", "sha256-utf8"))):
                reject("Invalid binary payload encoding or fields")
            binary_path = binary["path"]
            if type(binary_path) is not list or any(
                    type(part) is not str and (type(part) is not int or part < 0)
                    for part in binary_path):
                reject("Invalid binary payload path")
            typed_path = tuple(binary_path)
            if any(typed_path[:len(other)] == other or other[:len(typed_path)] == typed_path
                   for other in paths):
                reject("Duplicate or overlapping binary payload paths")
            paths.append(typed_path)
            target = record["payload"]
            for part in binary_path:
                if type(part) is str and type(target) is dict and part in target:
                    target = target[part]
                elif type(part) is int and type(target) is list and part < len(target):
                    target = target[part]
                else:
                    reject("Binary payload path does not exist")
            if target is not None:
                reject("Binary payload must replace a null placeholder")
            if type(binary["size_bytes"]) is not int or binary["size_bytes"] < 0:
                reject("Invalid binary payload size")
            if not sha256_identity(binary["sha256"]):
                reject("Binary payload digest mismatch")
            kind = "utf8" if encoding.endswith("-utf8") else "bytes"
            identity = (kind, binary["sha256"], binary["size_bytes"])
            if is_reference:
                if identity in staged_definitions:
                    decoded = staged_definitions[identity]
                elif identity in definitions:
                    decoded = definitions[identity]
                else:
                    reject("Unknown or forward binary payload reference")
            else:
                if type(binary["data"]) is not str:
                    reject("Invalid binary payload encoding")
                encoded = binary["data"].encode("ascii")
                decoded = base64.b64decode(encoded, validate=True)
                if base64.b64encode(decoded) != encoded:
                    reject("Noncanonical binary payload encoding")
                if len(decoded) != binary["size_bytes"]:
                    reject("Binary payload size mismatch")
                if hashlib.sha256(decoded).hexdigest() != binary["sha256"]:
                    reject("Binary payload digest mismatch")
                if kind == "utf8":
                    decoded = decoded.decode("utf-8")
                if version == 2:
                    decoded = staged_definitions.setdefault(identity, definitions.get(identity, decoded))
            replacements.append((binary_path, decoded))
        for binary_path, decoded in replacements:
            if not binary_path:
                record["payload"] = decoded
                continue
            parent_value = record["payload"]
            for part in binary_path[:-1]:
                parent_value = parent_value[part]
            parent_value[binary_path[-1]] = decoded
        return record, current_bindings, at_s, staged_definitions

    records = []
    offset = 0
    bindings = None
    last_at_s = None
    problem = None
    while offset < len(content):
        end = content.find(b"\n", offset)
        if end < 0:
            problem = {"offset": offset, "reason": "unterminated_line"}
            break
        if len(records) >= max_events:
            raise RuntimeError("Trace event limit exceeded")
        line = bytes(content[offset:end + 1])
        try:
            record, current_bindings, at_s, staged = decode_record(line, len(records), bindings, last_at_s)
        except (ValueError, TypeError, OverflowError, RecursionError):
            problem = {"offset": offset, "reason": "invalid_record"}
            break
        records.append(record)
        definitions.update(staged)
        bindings, last_at_s = current_bindings, at_s
        offset = end + 1
    return {"schema_version": 1,
            "run_id": bindings[0] if bindings is not None else None,
            "session_id": bindings[1] if bindings is not None else None,
            "reference_id": bindings[2] if bindings is not None else None,
            "status": "damaged" if problem is not None else "valid",
            "records": records, "valid_bytes": offset, "file_bytes": len(content),
            "problem": problem}
