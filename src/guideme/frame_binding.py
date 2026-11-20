"""Bind an assessment to its supplied media and declared clocks."""

import hashlib
import json
import math
import re
from copy import deepcopy
from uuid import UUID

from .frame_validation import validate_assessment_frame
from .validation import require_keys, require_nonnegative_number, require_sha256, require_uuid


def validate_host_replay_passage(mapping, elapsed, dispatch):
    """Validate a host-controlled replay clock, never browser display evidence."""
    def number(value):
        if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
            raise ValueError("Expected finite nonnegative host replay time")
        return value

    support = mapping["support"]
    if not isinstance(support, dict) or set(support) != {"source_elapsed_s", "anchor_source_s", "anchor_host_s"}:
        raise ValueError("Invalid host replay support")
    if number(support["source_elapsed_s"]) != elapsed:
        raise ValueError("Host replay support identifies another frame")
    if mapping["status"] == "mapped":
        anchor_source, anchor_host = number(support["anchor_source_s"]), number(support["anchor_host_s"])
        interval = mapping["host_interval_s"]
        passage = anchor_host + elapsed - anchor_source
        if (elapsed < anchor_source or not isinstance(interval, list) or len(interval) != 2
                or mapping["reason"] is not None or number(interval[0]) != number(interval[1])
                or abs(interval[0] - passage) > 0.000001
                or number(mapping["received_at_s"]) != interval[1] or interval[1] > dispatch):
            raise ValueError("Host replay passage or availability disagrees")
        return True
    if (mapping["status"] != "unavailable" or mapping["reason"] != "missing_host_passage"
            or mapping["host_interval_s"] is not None or mapping["received_at_s"] is not None
            or support["anchor_source_s"] is not None or support["anchor_host_s"] is not None):
        raise ValueError("Unavailable host replay mapping disagrees")
    return False


def _require_binding_version(binding):
    if type(binding["assessment_contract_version"]) is not int or binding["assessment_contract_version"] != 2:
        raise ValueError("Expected assessment contract 2")


def validate_tracker_binding(binding, applied, *, session_id, reference_id,
                             execution_source_sha256, clock_id, timing_basis, playback_identity):
    """Validate a state update against its recorded media, session, and clocks."""
    playback = isinstance(binding, dict) and "playback_mapping" in binding
    fields = ("assessment_contract_version", "session_id", "observation_id", "reference_id",
        "observation_manifest_sha256", "input_manifest_sha256", "request_sha256", "observation_cutoff_s",
        "latest_execution_frame", "clock_id", "timing_basis", "dispatched_at_s", "completed_at_s",
        "timing_status", "timing_reason", "latest_execution_captured_at_s", "latest_execution_received_at_s")
    require_keys(binding, fields + (("playback_mapping",) if playback else ()))
    _require_binding_version(binding)
    require_uuid(binding["observation_id"])
    if (binding["session_id"] != session_id or binding["reference_id"] != reference_id
            or binding["clock_id"] != clock_id or binding["timing_basis"] != timing_basis):
        raise ValueError("Binding belongs to another session, reference or clock")
    for field in ("observation_manifest_sha256", "input_manifest_sha256", "request_sha256"):
        require_sha256(binding[field])
    cutoff = require_nonnegative_number(binding["observation_cutoff_s"])
    dispatch, completion = (require_nonnegative_number(binding[name]) for name in ("dispatched_at_s", "completed_at_s"))
    if not dispatch <= completion <= applied:
        raise ValueError("Dispatch, completion and application chronology disagrees")
    elapsed = validate_assessment_frame(binding["latest_execution_frame"], execution_source_sha256, cutoff)
    captured, received = binding["latest_execution_captured_at_s"], binding["latest_execution_received_at_s"]
    if playback:
        mapping = binding["playback_mapping"]
        require_keys(mapping, ("schema_version", "session_id", "execution_source_sha256", "clock_id",
            "client_epoch_id", "segment_id", "kind", "status", "reason", "host_interval_s",
            "support", "received_at_s"))
        if (type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
                or timing_basis != "session_monotonic" or mapping["kind"] not in ("replay_passage", "host_replay_passage")
                or mapping["session_id"] != session_id or mapping["clock_id"] != clock_id
                or mapping["execution_source_sha256"] != execution_source_sha256
                or captured is not None or received is not None):
            raise ValueError("Playback identity or scalar timing disagrees")
        identity = (mapping["client_epoch_id"], mapping["segment_id"])
        for value in identity:
            require_uuid(value)
        if playback_identity is not None and identity != playback_identity:
            raise ValueError("Playback epoch or segment changed")
        if mapping["kind"] == "host_replay_passage":
            mapped = validate_host_replay_passage(mapping, elapsed, dispatch)
            if ((mapped and (binding["timing_status"] != "bounded" or binding["timing_reason"] is not None))
                    or (not mapped and (binding["timing_status"] != "unavailable" or binding["timing_reason"] != "missing_clock_mapping"))):
                raise ValueError("Host replay timing status disagrees")
            return identity
        support = mapping["support"]
        require_keys(support, ("source_elapsed_s", "sample_ids"))
        ids = support["sample_ids"]
        if require_nonnegative_number(support["source_elapsed_s"]) != elapsed or not isinstance(ids, list):
            raise ValueError("Playback support identifies another native tail")
        for value in ids:
            require_uuid(value)
        if mapping["status"] == "mapped":
            bounds = mapping["host_interval_s"]
            if (not isinstance(bounds, list) or len(bounds) != 2 or len(ids) != 2 or ids[0] == ids[1]
                    or mapping["reason"] is not None or binding["timing_status"] != "bounded"
                    or binding["timing_reason"] is not None
                    or not require_nonnegative_number(bounds[0]) <= require_nonnegative_number(bounds[1]) == require_nonnegative_number(mapping["received_at_s"]) <= dispatch):
                raise ValueError("Playback bounds or availability disagree")
        elif (mapping["status"] != "unavailable" or mapping["reason"] not in
              ("missing_lower_sample", "missing_upper_sample") or ids
              or mapping["host_interval_s"] is not None or mapping["received_at_s"] is not None
              or binding["timing_status"] != "unavailable" or binding["timing_reason"] != "missing_clock_mapping"):
            raise ValueError("Unavailable playback mapping disagrees")
        return identity
    elif binding["timing_status"] == "mapped":
        if binding["timing_reason"] is not None or not require_nonnegative_number(captured) <= require_nonnegative_number(received) <= dispatch:
            raise ValueError("Mapped capture, receipt and dispatch chronology disagrees")
    elif (binding["timing_status"] != "unavailable" or binding["timing_reason"] != "missing_clock_mapping"
            or captured is not None or received is not None):
        raise ValueError("Unavailable timing must not invent capture or receipt")

def build_assessment_binding(observation, *, native_provenance=None, jpeg_provenance=None, request_sha256,
                             dispatched_at_s, completed_at_s, clock_context):
    """Return contract-2 provenance without model claims or state changes.

    Inputs come from the existing ready observation and either validated native
    preparation or an explicit sent-JPEG witness. JPEG hashes are checked against
    supplied bytes; this does not authenticate their source or a caller's clock
    mapping. The request hash is caller-owned.
    Native media contains more frames than the JPEG citation anchors; its actual
    final execution frame determines the recorded input boundary. This does not
    establish which frames a provider sampled internally. JPEG input instead uses
    its actually sent final image. Source-video times remain separate from the
    observation's session capture/receipt times.

    Browser passage bounds remain bounds with null scalar capture/receipt times;
    they describe replay passage, not a camera's physical capture time.
    Unknown capture mapping stays unavailable. No clocks are read, timestamps
    inferred, freshness calculated, identities minted or delivery authorized.
    """
    def keys(record, fields):
        if not isinstance(record, dict) or set(record) != set(fields):
            raise ValueError("Record has missing or unsupported fields")

    def number(value):
        try:
            valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative timestamp")
        return value

    def digest(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Expected a lowercase SHA-256 identity")
        return value

    def source_pts(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite source timestamp")
        return value

    def uuid_string(value):
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError("Expected a canonical UUID identity")
        return value

    def hash_json(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()

    try:
        if (native_provenance is None) == (jpeg_provenance is None):
            raise ValueError("Exactly one supplied-media representation is required")
        if (not isinstance(observation, dict) or observation["status"] != "ready"
                or observation.get("reason") is not None):
            raise ValueError("A ready observation is required")
        model, manifest = observation["model_input"], observation["manifest"]
        for record in (observation, model, manifest):
            if not isinstance(record, dict) or type(record["schema_version"]) is not int or record["schema_version"] != 1:
                raise ValueError("Expected existing observation schema version 1")
        session_id = uuid_string(observation["session_id"])
        observation_id = uuid_string(observation["observation_id"])
        reference_id = digest(observation["reference_id"])
        cutoff = number(observation["cutoff_s"])
        for record in (model, manifest):
            if (record["session_id"] != session_id or record["observation_id"] != observation_id
                    or number(record["cutoff_s"]) != cutoff):
                raise ValueError("Observation headers disagree")
        if (manifest["reference_id"] != reference_id or manifest["representation"] != "full_prefix"
                or manifest["timeline"] != "session_elapsed"):
            raise ValueError("Observation reference or representation disagrees")
        observation_hash = digest(observation["manifest_sha256"])
        if hash_json(manifest) != observation_hash:
            raise ValueError("Observation manifest hash disagrees")
        digest(request_sha256)
        if native_provenance is not None:
            keys(native_provenance, ("reference", "execution"))
            for role in ("reference", "execution"):
                asset = native_provenance[role]
                if (not isinstance(asset, dict) or "video_bytes" in asset
                        or type(asset["schema_version"]) is not int or asset["schema_version"] != 1
                        or asset["status"] != "ready" or type(asset["source_eof"]) is not bool):
                    raise ValueError("Expected validated native metadata without video bytes")
                digest(asset["source_sha256"])
                digest(asset["video_sha256"])
                if role == "reference":
                    if (asset["source_sha256"] != digest(manifest["reference_source_sha256"])
                            or asset["cutoff_s"] is not None or not asset["source_eof"]):
                        raise ValueError("Native reference must match the complete reference")
                elif number(asset["cutoff_s"]) != cutoff:
                    raise ValueError("Native execution cutoff disagrees")
                rows = asset["frame_map"]
                if not isinstance(rows, list) or not rows:
                    raise ValueError("Native frame maps must be nonempty")
                previous = None
                for row in rows:
                    keys(row, ("frame_index", "source_pts_s", "elapsed_s", "video_pts_s"))
                    index = row["frame_index"]
                    times = tuple(number(row[name]) for name in ("source_pts_s", "elapsed_s", "video_pts_s"))
                    if (type(index) is not int or index < 0 or times[0] != times[1]
                            or abs(times[2] - times[1]) > 0.000001
                            or role == "execution" and times[1] > cutoff
                            or previous is not None and (index <= previous[0]
                                or any(time <= earlier for time, earlier in zip(times, previous[1:])))):
                        raise ValueError("Native frame maps must preserve ordered causal source times")
                    previous = (index, *times)
                if rows[0]["frame_index"] != 0 or rows[0]["elapsed_s"] != 0:
                    raise ValueError("Native media must begin at source frame zero")
                frames, metadata = model[role + "_frames"], manifest[role + "_frames"]
                if (not isinstance(frames, list) or not frames or not isinstance(metadata, list)
                        or len(frames) != len(metadata)):
                    raise ValueError("Observation frames and manifest disagree")
                source_times, last_anchor = {row["elapsed_s"] for row in rows}, None
                for frame, item in zip(frames, metadata):
                    anchor = number(frame["timestamp_s"])
                    captured = number(item["captured_at_s"] if role == "execution" else item["elapsed_s"])
                    if (anchor != captured or anchor not in source_times
                            or last_anchor is not None and anchor <= last_anchor):
                        raise ValueError("Observation anchors must match supplied native frames")
                    if role == "execution" and not captured <= number(item["received_at_s"]) <= cutoff:
                        raise ValueError("Execution anchor was unavailable at the observation cutoff")
                    last_anchor = anchor
            input_hash = hash_json({"observation_manifest_sha256": observation_hash,
                                    "native_provenance": native_provenance,
                                    "request_sha256": request_sha256})
        else:
            keys(jpeg_provenance, ("reference", "execution"))
            for role in ("reference", "execution"):
                asset = jpeg_provenance[role]
                keys(asset, ("source_sha256", "frames"))
                digest(asset["source_sha256"])
                if role == "reference" and asset["source_sha256"] != digest(manifest["reference_source_sha256"]):
                    raise ValueError("JPEG reference source disagrees")
                frames, metadata = model[role + "_frames"], manifest[role + "_frames"]
                if (not isinstance(frames, list) or not frames or not isinstance(metadata, list)
                        or len(frames) != len(metadata)):
                    raise ValueError("Observation frames and manifest disagree")
                pool, previous, origin = {}, None, None
                for frame, item in zip(frames, metadata):
                    index = item["frame_index"]
                    pts, elapsed = source_pts(item["source_pts_s"]), number(item["elapsed_s"])
                    caption = number(frame["timestamp_s"])
                    expected_caption = number(item["captured_at_s"] if role == "execution" else item["elapsed_s"])
                    payload = frame["jpeg_bytes"]
                    if (type(index) is not int or index < 0 or caption != expected_caption
                            or frame["mime_type"] != "image/jpeg" or item["mime_type"] != "image/jpeg"
                            or type(payload) is not bytes or not payload.startswith(b"\xff\xd8")
                            or not payload.endswith(b"\xff\xd9")
                            or hashlib.sha256(payload).hexdigest() != digest(item["sha256"])):
                        raise ValueError("JPEG observation bytes or metadata disagree")
                    current = (index, pts, elapsed, caption)
                    current_origin = source_pts(pts - elapsed)
                    if (previous is not None and any(a <= b for a, b in zip(current, previous))
                            or origin is not None and abs(current_origin - origin) > 0.000001):
                        raise ValueError("JPEG source frames must preserve order and source origin")
                    if role == "execution" and not expected_caption <= number(item["received_at_s"]) <= cutoff:
                        raise ValueError("JPEG execution frame was unavailable at the observation cutoff")
                    previous = current
                    if origin is None:
                        origin = current_origin
                    pool[index] = {"frame_index": index, "source_pts_s": pts, "elapsed_s": elapsed,
                        "timestamp_s": caption, "jpeg_sha256": item["sha256"], "jpeg_size_bytes": len(payload)}
                rows = asset["frames"]
                if not isinstance(rows, list) or not rows:
                    raise ValueError("Sent JPEG witnesses must be nonempty")
                previous_index = -1
                for row in rows:
                    keys(row, ("frame_index", "source_pts_s", "elapsed_s", "timestamp_s",
                               "jpeg_sha256", "jpeg_size_bytes"))
                    index, size = row["frame_index"], row["jpeg_size_bytes"]
                    source_pts(row["source_pts_s"]); number(row["elapsed_s"]); number(row["timestamp_s"])
                    digest(row["jpeg_sha256"])
                    if (type(index) is not int or index <= previous_index
                            or type(size) is not int or size <= 0 or row != pool.get(index)):
                        raise ValueError("Sent JPEG witness must match an ordered observation subset")
                    previous_index = index
            input_hash = hash_json({"observation_manifest_sha256": observation_hash,
                                    "jpeg_provenance": jpeg_provenance,
                                    "request_sha256": request_sha256})
        dispatch, completion = number(dispatched_at_s), number(completed_at_s)
        if completion < dispatch:
            raise ValueError("Completion cannot precede dispatch")
        playback = isinstance(clock_context, dict) and "playback_mapping" in clock_context
        keys(clock_context, ("session_id", "observation_id", "clock_id", "timing_basis",
                             "playback_mapping" if playback else "latest_frame"))
        if clock_context["session_id"] != session_id or clock_context["observation_id"] != observation_id:
            raise ValueError("Clock context belongs to another observation")
        clock_id, basis = clock_context["clock_id"], clock_context["timing_basis"]
        if not isinstance(clock_id, str) or not clock_id.strip():
            raise ValueError("A declared clock identity is required")
        clock_id.encode("utf-8")
        if basis not in ("ideal_content_cutoff", "session_monotonic"):
            raise ValueError("Unsupported timing basis")
        if native_provenance is not None:
            execution = native_provenance["execution"]
            latest = {"source_sha256": execution["source_sha256"], **execution["frame_map"][-1]}
        else:
            execution = jpeg_provenance["execution"]
            latest = {"media_type": "image/jpeg", "source_sha256": execution["source_sha256"],
                **{name: execution["frames"][-1][name] for name in
                   ("frame_index", "source_pts_s", "elapsed_s", "jpeg_sha256", "jpeg_size_bytes")}}
        mapping = clock_context["playback_mapping" if playback else "latest_frame"]
        captured = received = None
        timing_status, timing_reason = "unavailable", "missing_clock_mapping"
        if playback:
            keys(mapping, ("schema_version", "session_id", "execution_source_sha256", "clock_id",
                "client_epoch_id", "segment_id", "kind", "status", "reason", "host_interval_s",
                "support", "received_at_s"))
            if (type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
                    or basis != "session_monotonic" or mapping["kind"] not in ("replay_passage", "host_replay_passage")
                    or mapping["session_id"] != session_id or mapping["clock_id"] != clock_id
                    or mapping["execution_source_sha256"] != latest["source_sha256"]):
                raise ValueError("Playback mapping identity or kind disagrees")
            uuid_string(mapping["client_epoch_id"]); uuid_string(mapping["segment_id"])
            if mapping["kind"] == "host_replay_passage":
                if validate_host_replay_passage(mapping, latest["elapsed_s"], dispatch):
                    timing_status, timing_reason = "bounded", None
            else:
                support = mapping["support"]
                keys(support, ("source_elapsed_s", "sample_ids"))
                if number(support["source_elapsed_s"]) != latest["elapsed_s"]:
                    raise ValueError("Playback mapping must identify the actual native tail")
                ids = support["sample_ids"]
                if not isinstance(ids, list):
                    raise ValueError("Playback support must be an explicit list")
                for identifier in ids:
                    uuid_string(identifier)
                if mapping["status"] == "mapped":
                    interval = mapping["host_interval_s"]
                    if (not isinstance(interval, list) or len(interval) != 2 or len(ids) != 2
                            or ids[0] == ids[1] or mapping["reason"] is not None
                            or not number(interval[0]) <= number(interval[1]) == number(mapping["received_at_s"]) <= dispatch):
                        raise ValueError("Playback bounds or their availability disagree")
                    timing_status, timing_reason = "bounded", None
                elif (mapping["status"] != "unavailable" or mapping["reason"] not in
                      ("missing_lower_sample", "missing_upper_sample") or ids
                      or mapping["host_interval_s"] is not None or mapping["received_at_s"] is not None):
                    raise ValueError("Unavailable playback mapping must preserve its diagnostic")
        elif mapping is not None:
            keys(mapping, ("source_sha256", "frame_index", "source_elapsed_s", "captured_at_s", "received_at_s"))
            if (mapping["source_sha256"] != latest["source_sha256"]
                    or type(mapping["frame_index"]) is not int or mapping["frame_index"] != latest["frame_index"]
                    or number(mapping["source_elapsed_s"]) != latest["elapsed_s"]):
                raise ValueError("Clock mapping must identify the actual latest supplied frame")
            captured, received = number(mapping["captured_at_s"]), number(mapping["received_at_s"])
            if not captured <= received <= dispatch:
                raise ValueError("Capture, receipt and dispatch chronology disagrees")
            timing_status, timing_reason = "mapped", None
        result = {"assessment_contract_version": 2, "session_id": session_id,
                "observation_id": observation_id, "reference_id": reference_id,
                "observation_manifest_sha256": observation_hash, "input_manifest_sha256": input_hash,
                "request_sha256": request_sha256, "observation_cutoff_s": cutoff,
                "latest_execution_frame": latest, "clock_id": clock_id, "timing_basis": basis,
                "dispatched_at_s": dispatch, "completed_at_s": completion,
                "timing_status": timing_status, "timing_reason": timing_reason,
                "latest_execution_captured_at_s": captured, "latest_execution_received_at_s": received}
        if playback:
            result["playback_mapping"] = deepcopy(mapping)
        return result
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid assessment binding metadata") from exc
