"""Build contract-2 requests from selected JPEG frames."""

from copy import deepcopy
import hashlib

from .frame_selection import (canonical, digest, integer, keys, number,
                                      select_assessment_jpegs, uuid_string)


def build_assessment_image_request(observation, prior_context, *, request_template,
                                   source_sha256, selected_indices, max_images,
                                   max_jpeg_bytes, max_request_bytes, stage=None):
    """Return exact wire bytes, a sent-JPEG witness and a sent-only citation index.

    Selection is caller-owned: indices refer to the original ready observation,
    which is neither resampled nor rewritten. JPEG signatures and hashes are
    checked without decoding. Source metadata and clock records remain caller
    assertions; particularly, the observation has no execution-source hash to
    authenticate the supplied execution source identity.

    Sent execution labels are capture times on the observation's clock. Source
    elapsed time and signed source PTS are separate audit coordinates; their
    offset is checked within each role, not assumed to be zero. No capture or
    availability clock is invented. Limits cover selected JPEGs and the complete
    encoded request, including base64/text/schema overhead, not Python memory.
    """
    roles = ("reference", "execution")
    try:
        if stage not in (None, "observation", "association"):
            raise ValueError("Unsupported request stage")
        for limit in (max_images, max_jpeg_bytes, max_request_bytes):
            integer(limit, 1)
        selected = select_assessment_jpegs(observation, source_sha256=source_sha256,
            selected_indices=selected_indices, max_images=max_images, max_jpeg_bytes=max_jpeg_bytes)
        parts, provenance, registry = selected["parts"], selected["jpeg_provenance"], selected["frame_registry"]
        session, reference_id, cutoff = observation["session_id"], observation["reference_id"], observation["cutoff_s"]

        if stage == "observation":
            if prior_context is not None:
                raise ValueError("Observation stage cannot receive prior context")
        else:
            fields = ("assessment_contract_version", "session_id", "reference_id", "issues")
            keys(prior_context, fields + (("pending_observations", "observations") if stage == "association" else ()))
            if (type(prior_context["assessment_contract_version"]) is not int or prior_context["assessment_contract_version"] != 2
                    or prior_context["session_id"] != session or prior_context["reference_id"] != reference_id
                    or not isinstance(prior_context["issues"], list)):
                raise ValueError("Prior context identity or contract disagrees")
        seen = set()
        for issue in (() if stage == "observation" else prior_context["issues"]):
            if not isinstance(issue, dict):
                raise ValueError("Invalid prior issue")
            rich_fields = {"issue_id", "state", "initial_description", "latest_description", "repair_pending"}
            rich = set(issue) in (rich_fields, rich_fields | {"resolution_condition"})
            if not rich and set(issue) != {"issue_id", "state", "description"}:
                raise ValueError("Invalid prior issue")
            descriptions = ("initial_description", "latest_description") if rich else ("description",)
            identifier = uuid_string(issue["issue_id"])
            if (identifier in seen or issue["state"] not in ("candidate", "active", "uncertain", "resolved")
                    or any(not isinstance(issue[name], str) or not issue[name].strip() for name in descriptions)
                    or rich and type(issue["repair_pending"]) is not bool
                    or "resolution_condition" in issue and
                    (not isinstance(issue["resolution_condition"], str) or not issue["resolution_condition"].strip())):
                raise ValueError("Invalid prior issue")
            seen.add(identifier)
        if stage == "association":
            from .assessment_observation import decode_observations
            stage_observations = prior_context["observations"]
            encoded_observations = canonical(stage_observations)
            decode_observations(encoded_observations.decode("utf-8"), max_response_bytes=max_request_bytes)
            pending = prior_context["pending_observations"]
            if not isinstance(pending, dict):
                raise ValueError("Pending observations must be a mapping")
            for pending_id, item in pending.items():
                uuid_string(pending_id)
                if pending_id in seen or not isinstance(item, dict):
                    raise ValueError("Pending identity collides or record is invalid")
                fields = {"description", "first_observation_id", "first_cutoff_s", "latest_observation_id", "latest_cutoff_s"}
                if set(item) not in (fields, fields | {"resolution_condition"}):
                    raise ValueError("Invalid pending observation record")
                for name in ("description", "resolution_condition"):
                    if name in item and (not isinstance(item[name], str) or not item[name].strip()):
                        raise ValueError("Pending text must be nonblank")
                uuid_string(item["first_observation_id"])
                uuid_string(item["latest_observation_id"])
                if not number(item["first_cutoff_s"]) <= number(item["latest_cutoff_s"]) <= cutoff:
                    raise ValueError("Pending provenance must precede the current cutoff")
        keys(request_template, ("model", "messages", "max_tokens", "n", "response_format"))
        if (not isinstance(request_template["model"], str) or not request_template["model"].strip()
                or type(request_template["n"]) is not int or request_template["n"] != 1
                or not isinstance(request_template["response_format"], dict)):
            raise ValueError("Invalid request template settings")
        integer(request_template["max_tokens"], 1)
        messages = request_template["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("Request template requires exactly two messages")
        for message in messages:
            keys(message, ("role", "content"))
        if (messages[0]["role"] != "system" or not isinstance(messages[0]["content"], str)
                or messages[1]["role"] != "user" or messages[1]["content"] != []):
            raise ValueError("Request template requires system text and empty user content")
        canonical(request_template)
        header = selected["header"]
        anchors = {role: registry[role + "_s"] for role in roles}
        context_parts = [("OBSERVATION", header), ("EVIDENCE_ANCHORS", anchors)]
        if stage is None:
            context_parts.append(("PRIOR_ISSUE_CONTEXT", prior_context))
        elif stage == "association":
            context_parts.append(("STAGE_2_CONTEXT", prior_context))
        for label, value in context_parts:
            parts.append({"type": "text", "text": label + "\n" + canonical(value).decode("utf-8")})
        request = deepcopy(request_template)
        request["messages"][1]["content"] = parts
        wire = canonical(request)
        if len(wire) > max_request_bytes:
            raise RuntimeError("JPEG request exceeds its complete encoded-byte capacity")
        return {"wire": wire, "request_sha256": hashlib.sha256(wire).hexdigest(),
                "jpeg_provenance": provenance, "frame_registry": registry}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid assessment JPEG request or observation") from exc
