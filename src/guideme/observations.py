"""Stateless visual observations, with a separate provenance manifest."""

import hashlib
import json
import math
import re
from uuid import UUID


def build_observation(reference, execution_samples, *, session_id, observation_id,
                      cutoff_s, max_images, max_jpeg_bytes):
    """Combine the complete reference with the received causal execution prefix.

    Capture and receipt times must already share the session elapsed clock.
    Source-video timestamps are retained for audit, not used as session times.
    This function performs no I/O and trusts the caller's clock/receipt records;
    hashes detect inconsistent records, rather than authenticate their origin.
    Limits cover returned visual payloads, not caller-owned prefix storage.
    Prior model memory and delivery scheduling are intentionally outside this unit.
    """
    def integer(value, name, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value

    def number(value, name, minimum=None):
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (minimum is not None and value < minimum):
            raise ValueError(f"{name} must be finite and >= {minimum}")
        return value

    def digest(value, name):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        return value

    def json_hash(value):
        try:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Metadata must be finite JSON values") from exc
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def uuid_string(value, name):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a canonical UUID string")

    def image_metadata(image):
        if not isinstance(image, dict):
            raise ValueError("Each image must be a JPEG record")
        metadata = {"frame_index": integer(image.get("frame_index"), "frame_index"),
                    "source_pts_s": number(image.get("source_pts_s"), "source_pts_s"),
                    "elapsed_s": number(image.get("elapsed_s"), "elapsed_s", 0)}
        for name in ("source_width", "source_height", "width", "height"):
            metadata[name] = integer(image.get(name), name, 1)
        payload = image.get("jpeg_bytes")
        if (image.get("mime_type") != "image/jpeg" or type(payload) is not bytes
                or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9")):
            raise ValueError("Image must contain immutable JPEG bytes and image/jpeg MIME type")
        sha = digest(image.get("sha256"), "image sha256")
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError("JPEG bytes do not match their digest")
        encoding = image.get("encoding")
        if not isinstance(encoding, dict):
            raise ValueError("Image encoding settings are required")
        side = integer(encoding.get("max_side"), "max_side", 1)
        quality = integer(encoding.get("jpeg_quality"), "jpeg_quality", 1)
        interpolation = encoding.get("resize_interpolation")
        version = encoding.get("encoder_version")
        if (quality > 100 or interpolation not in ("area", "none")
                or encoding.get("encoder") != "opencv" or not isinstance(version, str)
                or not version.strip()):
            raise ValueError("Invalid JPEG encoding settings")
        sw, sh = metadata["source_width"], metadata["source_height"]
        scale = min(1.0, side / max(sw, sh))
        expected = (max(1, round(sw * scale)), max(1, round(sh * scale)))
        if ((metadata["width"], metadata["height"]) != expected
                or interpolation != ("area" if scale < 1 else "none")):
            raise ValueError("Image dimensions do not match encoding settings")
        metadata.update(mime_type="image/jpeg", sha256=sha, encoding={
            "max_side": side, "jpeg_quality": quality,
            "resize_interpolation": interpolation, "encoder": "opencv", "encoder_version": version})
        return metadata, payload

    def selection(record, elapsed):
        target = number(record.get("target_s"), "target_s", 0)
        skipped = integer(record.get("skipped_targets"), "skipped_targets")
        if target > elapsed:
            raise ValueError("Sampling target cannot follow the selected frame")
        return target, skipped

    def check_order(previous, current):
        if previous is not None and any(a >= b for a, b in zip(previous, current)):
            raise ValueError("Frame indices, source times and capture times must strictly increase")

    uuid_string(session_id, "session_id")
    uuid_string(observation_id, "observation_id")
    number(cutoff_s, "cutoff_s", 0)
    integer(max_images, "max_images", 1)
    integer(max_jpeg_bytes, "max_jpeg_bytes", 1)
    if not isinstance(reference, dict) or not isinstance(execution_samples, list):
        raise ValueError("Expected a prepared reference and a list of execution samples")
    if type(reference.get("schema_version")) is not int or reference["schema_version"] != 1 or reference.get("status") != "ready":
        raise ValueError("Reference must be a ready schema-version-1 preparation")
    reference_id = digest(reference.get("reference_id"), "reference_id")
    source_sha = digest(reference.get("source_sha256"), "source_sha256")
    config_sha = digest(reference.get("config_sha256"), "config_sha256")
    config = reference.get("config")
    if not isinstance(config, dict) or json_hash(config) != config_sha:
        raise ValueError("Reference configuration digest mismatch")
    number(config.get("fps"), "reference fps", 0)
    if config["fps"] == 0:
        raise ValueError("Reference fps must be positive")
    integer(config.get("max_frames"), "reference max_frames", 1)
    integer(config.get("max_side"), "reference max_side", 1)
    integer(config.get("jpeg_quality"), "reference jpeg_quality", 1)
    if (config["jpeg_quality"] > 100 or config.get("endpoint_policy") != "include_first_and_last"
            or config.get("budget_policy") != "error"):
        raise ValueError("Unsupported reference preparation settings")
    frames = reference.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Prepared reference must contain frames")
    decoded_count = integer(reference.get("decoded_frame_count"), "decoded_frame_count", 1)
    if (integer(reference.get("sample_count"), "sample_count", 1) != len(frames)
            or len(frames) > min(decoded_count, config["max_frames"])):
        raise ValueError("Inconsistent reference frame counts")
    reference_metadata, reference_visuals = [], []
    previous = None
    reference_bytes = 0
    for i, frame in enumerate(frames):
        metadata, payload = image_metadata(frame)
        current = tuple(metadata[name] for name in ("frame_index", "source_pts_s", "elapsed_s"))
        check_order(previous, current)
        previous = current
        reason = frame.get("selection_reason")
        if reason == "cadence":
            target, skipped = selection(frame, metadata["elapsed_s"])
        elif (reason == "final_frame" and i == len(frames) - 1 and i > 0
              and frame.get("target_s") is None and frame.get("skipped_targets") is None):
            target, skipped = None, None
        else:
            raise ValueError("Invalid reference selection metadata")
        if any(metadata["encoding"][name] != config[name] for name in ("max_side", "jpeg_quality")):
            raise ValueError("Reference encoding settings disagree with configuration")
        metadata.update(selection_reason=reason, target_s=target, skipped_targets=skipped)
        reference_metadata.append(metadata)
        reference_visuals.append({"timestamp_s": metadata["elapsed_s"], "mime_type": "image/jpeg", "jpeg_bytes": payload})
        reference_bytes += len(payload)
    first, last = reference_metadata[0], reference_metadata[-1]
    first_pts = number(reference.get("first_source_pts_s"), "first_source_pts_s")
    last_pts = number(reference.get("last_source_pts_s"), "last_source_pts_s")
    last_elapsed = number(reference.get("last_elapsed_s"), "last_elapsed_s", 0)
    if (first["frame_index"] != 0 or first["elapsed_s"] != 0 or first["target_s"] != 0
            or first["source_pts_s"] != first_pts or last["source_pts_s"] != last_pts
            or last["frame_index"] != decoded_count - 1 or last["elapsed_s"] != last_elapsed
            or integer(reference.get("total_jpeg_bytes"), "total_jpeg_bytes", 1) != reference_bytes):
        raise ValueError("Inconsistent reference endpoints or byte count")
    identity = {name: reference[name] for name in (
        "schema_version", "source_sha256", "config_sha256", "decoded_frame_count",
        "first_source_pts_s", "last_source_pts_s", "last_elapsed_s")}
    # Match prepare_reference's original identity, including unrecognized frame
    # metadata. Only the explicit whitelist above may enter returned manifests.
    identity["frames"] = [{key: value for key, value in frame.items() if key != "jpeg_bytes"}
                          for frame in frames]
    if json_hash(identity) != reference_id:
        raise ValueError("Prepared reference identity mismatch")

    included, excluded, execution_visuals = [], [], []
    previous = None
    execution_bytes = 0
    for record in execution_samples:
        if (not isinstance(record, dict) or record.get("session_id") != session_id
                or record.get("reference_id") != reference_id):
            raise ValueError("Execution sample belongs to a different session or reference")
        captured = number(record.get("captured_at_s"), "captured_at_s", 0)
        received = number(record.get("received_at_s"), "received_at_s", 0)
        if received < captured:
            raise ValueError("Receipt cannot precede capture on the session clock")
        metadata, payload = image_metadata(record.get("image"))
        target, skipped = selection(record, metadata["elapsed_s"])
        current = tuple(metadata[name] for name in ("frame_index", "source_pts_s", "elapsed_s")) + (captured,)
        check_order(previous, current)
        previous = current
        metadata.update(captured_at_s=captured, received_at_s=received,
                        target_s=target, skipped_targets=skipped)
        reasons = []
        if captured > cutoff_s:
            reasons.append("future_capture")
        if received > cutoff_s:
            reasons.append("not_received_by_cutoff")
        if reasons:
            metadata["reasons"] = reasons
            excluded.append(metadata)
        else:
            included.append(metadata)
            execution_visuals.append({"timestamp_s": captured, "mime_type": "image/jpeg", "jpeg_bytes": payload})
            execution_bytes += len(payload)

    image_count = len(reference_visuals) + len(execution_visuals)
    total_bytes = reference_bytes + execution_bytes
    if image_count > max_images or total_bytes > max_jpeg_bytes:
        raise RuntimeError("Observation exceeds its image or JPEG-byte budget")
    manifest = {
        "schema_version": 1, "representation": "full_prefix", "timeline": "session_elapsed",
        "session_id": session_id, "observation_id": observation_id, "reference_id": reference_id,
        "reference_source_sha256": source_sha, "reference_config_sha256": config_sha,
        "cutoff_s": cutoff_s, "limits": {"max_images": max_images, "max_jpeg_bytes": max_jpeg_bytes},
        "reference_frames": reference_metadata, "execution_frames": included,
        "excluded_execution_frames": excluded, "image_count": image_count, "total_jpeg_bytes": total_bytes,
    }
    model_input = None
    if execution_visuals:
        model_input = {"schema_version": 1, "session_id": session_id,
                       "observation_id": observation_id, "cutoff_s": cutoff_s,
                       "reference_frames": reference_visuals, "execution_frames": execution_visuals}
    return {
        "schema_version": 1, "status": "ready" if model_input is not None else "not_ready",
        "reason": None if model_input is not None else "no_available_execution_frames",
        "session_id": session_id, "observation_id": observation_id, "reference_id": reference_id,
        "cutoff_s": cutoff_s, "model_input": model_input, "manifest": manifest,
        "manifest_sha256": json_hash(manifest),
    }
