"""Deterministic identities for explicit run settings and bounded local files."""

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import stat


def _copy_json(value, ancestors):
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind not in (dict, list):
        raise ValueError("Metadata must contain only finite JSON values")
    identity = id(value)
    if identity in ancestors:
        raise ValueError("Metadata cannot contain cycles")
    ancestors.add(identity)
    try:
        if kind is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("Metadata object keys must be strings")
            return {key: _copy_json(value[key], ancestors) for key in sorted(value)}
        return [_copy_json(item, ancestors) for item in value]
    finally:
        ancestors.remove(identity)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _check_source(source, resolved, expected):
    try:
        changed = (source.resolve(strict=True) != resolved or
                   _identity(resolved.stat()) != expected)
    except OSError as exc:
        raise RuntimeError("Source changed while freezing the manifest") from exc
    if changed:
        raise RuntimeError("Source changed while freezing the manifest")


def freeze_run_manifest(configuration, files, *, git_state, provenance,
                        max_files, max_total_bytes):
    """Return detached settings and hashes of every explicitly supplied file.

    ``files`` maps logical roles to absolute local paths, including any relevant
    untracked source. Aliases each consume the file-count and byte budgets. Paths
    resolve symlinks and participate in the manifest identity. No artifact is
    written, and no Git, environment, credentials or network data is collected.

    Git state and provenance are caller-supplied finite JSON objects, not verified
    claims. The caller owns their truth, credential safety and file-list coverage.
    Hashes identify provided bytes; they do not establish a complete code archive
    or reproducible environment. Before/after filesystem checks detect ordinary
    concurrent changes but cannot provide an atomic snapshot of several files.

    Invalid values/nonregular sources raise ValueError, capacity or observed
    source changes raise RuntimeError, and initially unavailable sources propagate
    OSError. File reads use bounded chunks and never modify the sources.
    """
    for limit in (max_files, max_total_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("File and byte limits must be positive integers")
    copied = []
    try:
        for metadata in (configuration, git_state, provenance):
            if type(metadata) is not dict:
                raise ValueError("Configuration, Git state and provenance must be JSON objects")
            detached = _copy_json(metadata, set())
            _canonical(detached)
            copied.append(detached)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Metadata must contain finite UTF-8 JSON values") from exc
    if not isinstance(files, Mapping):
        raise ValueError("Files must map logical roles to absolute paths")
    entries = list(files.items())
    if len(entries) > max_files:
        raise RuntimeError("Manifest file-count limit exceeded")
    sources = {}
    for role, path in entries:
        if type(role) is not str or not role.strip():
            raise ValueError("File roles must be nonblank strings")
        if not isinstance(path, (str, Path)):
            raise ValueError("File paths must be absolute strings or Paths")
        try:
            role.encode("utf-8")
            str(path).encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("File roles and paths must be valid UTF-8") from exc
        source = Path(path)
        if not source.is_absolute():
            raise ValueError("File paths must be absolute")
        sources[role] = source

    records, snapshots = {}, []
    total = 0
    for role in sorted(sources):
        source = sources[role]
        resolved = source.resolve(strict=True)
        initial = resolved.stat()
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("Manifest sources must be regular files")
        expected = _identity(initial)
        if initial.st_size > max_total_bytes - total:
            raise RuntimeError("Manifest total-byte limit exceeded")
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("Manifest sources must be regular files")
            if _identity(opened) != expected:
                raise RuntimeError("Source changed while freezing the manifest")
            digest = hashlib.sha256()
            size = 0
            while True:
                # One extra byte is sufficient to establish an exceeded budget.
                chunk = os.read(descriptor, min(1024 * 1024,
                                max_total_bytes - total - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_total_bytes - total:
                    raise RuntimeError("Manifest total-byte limit exceeded")
                digest.update(chunk)
            if size != initial.st_size or _identity(os.fstat(descriptor)) != expected:
                raise RuntimeError("Source changed while freezing the manifest")
            _check_source(source, resolved, expected)
        finally:
            os.close(descriptor)
        records[role] = {"path": str(resolved), "size_bytes": size,
                         "sha256": digest.hexdigest()}
        total += size
        snapshots.append((source, resolved, expected))
    # Catch changes to an earlier file while later entries were being hashed.
    for source, resolved, expected in snapshots:
        _check_source(source, resolved, expected)
    body = {"schema_version": 1,
            "configuration_sha256": hashlib.sha256(_canonical(copied[0])).hexdigest(),
            "configuration": copied[0], "git_state": copied[1],
            "provenance": copied[2], "files": records, "total_file_bytes": total}
    return {**body, "manifest_id": hashlib.sha256(_canonical(body)).hexdigest()}
