"""Validate and select timestamped JPEG frames for assessment requests."""

import base64
import hashlib
import json
import math
import re
from uuid import UUID

from .visual_input import validate_visual_input


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Record has missing or unsupported fields")

def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Expected an ordinary integer within its bound")
    return value

def number(value, *, signed=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (not signed and value < 0):
        raise ValueError("Expected a finite timestamp within its bound")
    return value

def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 identity")
    return value

def uuid_string(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Expected a canonical UUID identity")
    return value

def canonical(value, *, ascii=False):
    def json_value(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                json_value(child)
        elif isinstance(item, list):
            for child in item:
                json_value(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("Expected JSON values without implicit conversion")
    json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ascii, allow_nan=False).encode("utf-8")



def select_assessment_jpegs(observation, *, source_sha256, selected_indices,
                            max_images, max_jpeg_bytes):
    """Validate and select JPEG evidence without constructing a model request.

    Execution timestamps are capture times; provenance separately retains source
    elapsed time. All observation frames are validated, including unselected ones.
    """
    roles = ("reference", "execution")
    try:
        for limit in (max_images, max_jpeg_bytes):
            integer(limit, 1)
        keys(observation, ("schema_version", "status", "reason", "session_id",
            "observation_id", "reference_id", "cutoff_s", "model_input", "manifest",
            "manifest_sha256"))
        if observation["status"] != "ready" or observation["reason"] is not None:
            raise ValueError("A ready observation is required")
        model, manifest = observation["model_input"], observation["manifest"]
        validate_visual_input(model)
        keys(manifest, ("schema_version", "representation", "timeline", "session_id",
            "observation_id", "reference_id", "reference_source_sha256",
            "reference_config_sha256", "cutoff_s", "limits", "reference_frames",
            "execution_frames", "excluded_execution_frames", "image_count", "total_jpeg_bytes"))
        session = uuid_string(observation["session_id"])
        observation_id = uuid_string(observation["observation_id"])
        reference_id = digest(observation["reference_id"])
        cutoff = number(observation["cutoff_s"])
        for record in (observation, model, manifest):
            if (type(record["schema_version"]) is not int or record["schema_version"] != 1
                    or record["session_id"] != session or record["observation_id"] != observation_id
                    or number(record["cutoff_s"]) != cutoff):
                raise ValueError("Observation identity or cutoff disagrees")
        if (manifest["reference_id"] != reference_id or manifest["representation"] != "full_prefix"
                or manifest["timeline"] != "session_elapsed"
                or hashlib.sha256(canonical(manifest, ascii=True)).hexdigest()
                    != digest(observation["manifest_sha256"])):
            raise ValueError("Observation manifest identity disagrees")
        digest(manifest["reference_config_sha256"])
        keys(source_sha256, roles)
        keys(selected_indices, roles)
        for role in roles:
            digest(source_sha256[role])
        if source_sha256["reference"] != digest(manifest["reference_source_sha256"]):
            raise ValueError("Reference source identity disagrees")
        keys(manifest["limits"], ("max_images", "max_jpeg_bytes"))
        for value in manifest["limits"].values():
            integer(value, 1)
        if not isinstance(manifest["excluded_execution_frames"], list):
            raise ValueError("Excluded observations must be an explicit list")

        parts, provenance, registry = [], {}, {}
        image_count = jpeg_count = total_images = total_bytes = 0
        base_metadata = {"frame_index", "source_pts_s", "elapsed_s", "source_width",
            "source_height", "width", "height", "mime_type", "sha256", "encoding"}
        for role in roles:
            frames, metadata = model[role + "_frames"], manifest[role + "_frames"]
            if not isinstance(metadata, list) or len(frames) != len(metadata):
                raise ValueError("Observation JPEGs and metadata disagree")
            indices = selected_indices[role]
            if (not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(frames) for i in indices)
                    or any(a >= b for a, b in zip(indices, indices[1:]))):
                raise ValueError("Selected indices must be nonempty, ordered and unique")
            selected = set(indices)
            rows, times, previous, origin = [], [], None, None
            extra = ({"captured_at_s", "received_at_s", "target_s", "skipped_targets"}
                     if role == "execution" else {"selection_reason", "target_s", "skipped_targets"})
            for ordinal, (frame, item) in enumerate(zip(frames, metadata)):
                keys(item, base_metadata | extra)
                index = integer(item["frame_index"])
                pts, elapsed = number(item["source_pts_s"], signed=True), number(item["elapsed_s"])
                offset = number(pts - elapsed, signed=True)
                if origin is None:
                    origin = offset
                if (abs(offset - origin) > 0.000001 or previous is not None
                        and any(a <= b for a, b in zip((index, pts, elapsed), previous))):
                    raise ValueError("Source rows must be ordered with a consistent role origin")
                previous = index, pts, elapsed
                data = frame["jpeg_bytes"]
                if (item["mime_type"] != "image/jpeg"
                        or hashlib.sha256(data).hexdigest() != digest(item["sha256"])):
                    raise ValueError("JPEG bytes and metadata hash disagree")
                for name in ("source_width", "source_height", "width", "height"):
                    integer(item[name], 1)
                encoding = item["encoding"]
                keys(encoding, ("max_side", "jpeg_quality", "resize_interpolation", "encoder", "encoder_version"))
                side, quality = integer(encoding["max_side"], 1), integer(encoding["jpeg_quality"], 1)
                scale = min(1, side / max(item["source_width"], item["source_height"]))
                if (quality > 100 or encoding["encoder"] != "opencv"
                        or not isinstance(encoding["encoder_version"], str) or not encoding["encoder_version"].strip()
                        or encoding["resize_interpolation"] != ("area" if scale < 1 else "none")
                        or (item["width"], item["height"]) != tuple(max(1, round(item[k] * scale))
                            for k in ("source_width", "source_height"))):
                    raise ValueError("JPEG encoding metadata disagrees")
                timestamp = frame["timestamp_s"]
                if role == "execution":
                    captured, received = number(item["captured_at_s"]), number(item["received_at_s"])
                    if timestamp != captured or not captured <= received <= cutoff:
                        raise ValueError("Execution JPEG was unavailable at the observation cutoff")
                elif timestamp != elapsed:
                    raise ValueError("Reference JPEG timestamp disagrees")
                if role == "reference" and item["selection_reason"] == "final_frame":
                    if ordinal != len(frames) - 1 or ordinal == 0 or item["target_s"] is not None or item["skipped_targets"] is not None:
                        raise ValueError("Invalid reference endpoint metadata")
                else:
                    if (role == "reference" and item["selection_reason"] != "cadence"
                            or number(item["target_s"]) > elapsed):
                        raise ValueError("Invalid cadence target")
                    integer(item["skipped_targets"])
                total_images += 1
                total_bytes += len(data)
                if ordinal not in selected:
                    continue
                image_count += 1
                jpeg_count += len(data)
                if image_count > max_images or jpeg_count > max_jpeg_bytes:
                    raise RuntimeError("Selected JPEG request exceeds its image or byte capacity")
                rows.append({"frame_index": index, "source_pts_s": pts, "elapsed_s": elapsed,
                    "timestamp_s": timestamp, "jpeg_sha256": item["sha256"], "jpeg_size_bytes": len(data)})
                times.append(timestamp)
                parts.extend([{"type": "text", "text": role.upper() + " " + repr(timestamp) + " s"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}}])
            provenance[role] = {"source_sha256": source_sha256[role], "frames": rows}
            registry[role + "_s"] = times
        if (integer(manifest["image_count"]) != total_images or integer(manifest["total_jpeg_bytes"]) != total_bytes
                or total_images > manifest["limits"]["max_images"] or total_bytes > manifest["limits"]["max_jpeg_bytes"]):
            raise ValueError("Original observation image accounting disagrees")

        header = {key: model[key] for key in ("schema_version", "session_id", "observation_id", "cutoff_s")}
        return {"parts": parts, "jpeg_provenance": provenance,
                "frame_registry": registry, "header": header}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid assessment JPEG request or observation") from exc
