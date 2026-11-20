"""Build tracking context and requests, and decode model assessments."""

from copy import deepcopy
import hashlib
import json
import math
import re
from uuid import UUID

from .frame_selection import (canonical, digest, integer, keys, number,
                                       select_assessment_jpegs, uuid_string)


def build_issue_context(*, session_id, reference_id, observation_id,
        manifest_hash, source_cutoff_s, context_revision, open_issues,
        resolved_history, pending_observations, max_open_issues,
        max_pending_observations, max_resolved_history, max_context_bytes):
    """Validate and copy a causal context; never truncate active or pending state."""

    def identity(value):
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("Expected a canonical UUID")

    def digest(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Expected a lowercase SHA-256 digest")

    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected nonblank text")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Expected UTF-8 text") from exc

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Expected a finite nonnegative cutoff")
        try:
            valid = math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite nonnegative cutoff")

    def count(value, *, positive=False):
        if type(value) is not int or value < int(positive):
            raise ValueError("Invalid revision or capacity")

    identity(session_id)
    identity(observation_id)
    digest(reference_id)
    digest(manifest_hash)
    number(source_cutoff_s)
    count(context_revision)
    for capacity in (max_open_issues, max_pending_observations, max_resolved_history):
        count(capacity)
    count(max_context_bytes, positive=True)
    if any(not isinstance(rows, list) for rows in (open_issues, resolved_history, pending_observations)):
        raise ValueError("Context collections must be lists")
    if len(open_issues) > max_open_issues or len(pending_observations) > max_pending_observations:
        raise ValueError("Open or pending context capacity exceeded")

    seen = set()

    def record(row, fields, id_field, last_field):
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError("Unexpected context record fields")
        identity(row[id_field])
        if row[id_field] in seen:
            raise ValueError("Duplicate issue or pending ID")
        seen.add(row[id_field])
        first, last = row["first_assessment_cutoff_s"], row[last_field]
        number(first)
        number(last)
        if first > last or last > source_cutoff_s:
            raise ValueError("Assessment cutoff is inconsistent with observation")
        return dict(row)

    open_rows = []
    for row in open_issues:
        item = record(row, ("issue_id", "state", "initial_description", "latest_description",
            "first_assessment_cutoff_s", "latest_assessment_cutoff_s", "pending_repair",
            "resolution_condition"), "issue_id", "latest_assessment_cutoff_s")
        if item["state"] not in ("candidate", "active", "uncertain") or type(item["pending_repair"]) is not bool:
            raise ValueError("Invalid open issue state")
        text(item["initial_description"])
        text(item["latest_description"])
        if item["resolution_condition"] is not None:
            text(item["resolution_condition"])
        open_rows.append(item)

    history_rows = []
    for row in resolved_history:
        item = record(row, ("issue_id", "description", "outcome", "first_assessment_cutoff_s",
            "terminal_assessment_cutoff_s"), "issue_id", "terminal_assessment_cutoff_s")
        text(item["description"])
        if item["outcome"] not in ("corrected", "retracted"):
            raise ValueError("Invalid terminal outcome")
        history_rows.append(item)

    history_by_id = {item["issue_id"]: item for item in history_rows}
    pending_rows = []
    for row in pending_observations:
        item = record(row, ("pending_id", "description", "first_assessment_cutoff_s",
            "latest_assessment_cutoff_s", "recurrence_of"), "pending_id", "latest_assessment_cutoff_s")
        text(item["description"])
        if item["recurrence_of"] is not None:
            identity(item["recurrence_of"])
            prior = history_by_id.get(item["recurrence_of"])
            if (prior is None or prior["outcome"] != "corrected"
                    or prior["terminal_assessment_cutoff_s"] > item["latest_assessment_cutoff_s"]):
                raise ValueError("Pending recurrence requires prior corrected history")
        pending_rows.append(item)

    history_rows.sort(key=lambda row: (-row["terminal_assessment_cutoff_s"], row["issue_id"]))
    context = {"assessment_contract_version": 3, "session_id": session_id,
        "reference_id": reference_id, "observation_id": observation_id,
        "manifest_hash": manifest_hash, "source_cutoff_s": source_cutoff_s,
        "context_revision": context_revision, "open_issues": open_rows,
        "resolved_history": history_rows[:max_resolved_history],
        "pending_observations": pending_rows,
        "omitted_resolved_history_count": max(0, len(history_rows) - max_resolved_history)}
    try:
        encoded = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Context cannot be serialized as UTF-8 JSON") from exc
    if len(encoded) > max_context_bytes:
        raise ValueError("Context byte capacity exceeded")
    return context


def decode_assessment(raw_text, *, context, frame_registry,
                                    max_response_bytes):
    """Validate claims against the supplied request context and frame registry.

    Citations report whether timestamps were supplied, not whether they prove a
    claim. Admission and chronology-dependent decisions remain host duties.
    """
    def keys(value, required, optional=()):
        if (not isinstance(value, dict) or not set(required) <= value.keys()
                or not value.keys() <= set(required) | set(optional)):
            raise ValueError('Missing or unsupported fields')

    def time(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Expected a nonblank string')
        value.encode('utf-8')
        return value

    def identifier(value):
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError('Expected a canonical UUID')
        return value

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError('Nonstandard JSON constant')

    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError('Response-byte limit must be a positive integer')
    if not isinstance(raw_text, str):
        raise ValueError('Response must be a string')
    try:
        size = len(raw_text.encode('utf-8'))
    except UnicodeError as exc:
        raise ValueError('Response must be valid UTF-8') from exc
    if size > max_response_bytes:
        raise RuntimeError('Assessment response-byte limit exceeded')

    try:
        if (not isinstance(context, dict) or
                type(context.get('assessment_contract_version')) is not int or
                context['assessment_contract_version'] != 3 or
                not time(context.get('source_cutoff_s'))):
            raise ValueError('Invalid frozen context')
        collections = {}
        all_ids = set()
        for name, id_key in (('open_issues', 'issue_id'),
                             ('resolved_history', 'issue_id'),
                             ('pending_observations', 'pending_id')):
            records = context[name]
            if not isinstance(records, list):
                raise ValueError('Context records must be lists')
            current = {}
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError('Invalid context record')
                first = record['first_assessment_cutoff_s']
                last = record['terminal_assessment_cutoff_s' if name == 'resolved_history'
                              else 'latest_assessment_cutoff_s']
                if (not time(first) or not time(last) or first > last or
                        last > context['source_cutoff_s']):
                    raise ValueError('Invalid context assessment times')
                key = identifier(record[id_key])
                if key in all_ids:
                    raise ValueError('Duplicate context ID')
                all_ids.add(key)
                current[key] = record
            collections[name] = current
        opened = collections['open_issues']
        resolved = collections['resolved_history']
        pending = collections['pending_observations']
        if any(record.get('outcome') not in ('corrected', 'retracted')
               for record in resolved.values()):
            raise ValueError('Invalid historical outcome')
        roles = ('execution_s', 'reference_s')
        keys(frame_registry, roles)
        registry = {}
        for role in roles:
            values = frame_registry[role]
            if (not isinstance(values, list) or not values or
                    any(not time(item) for item in values) or
                    any(a >= b for a, b in zip(values, values[1:])) or
                    (role == 'execution_s' and values[-1] > context['source_cutoff_s'])):
                raise ValueError('Invalid supplied frame times')
            registry[role] = set(values)

        def citation(claim, target, index):
            result = {'target': target, 'index': index, 'status': 'absent',
                      'reason': None, 'citations': None}
            if 'citations' not in claim:
                return result
            value = claim['citations']
            if (not isinstance(value, dict) or set(value) != set(roles) or
                    any(not isinstance(value[role], list) for role in roles)):
                reason = 'invalid_shape'
            elif any(not time(t) for role in roles for t in value[role]):
                reason = 'invalid_timestamp'
            elif any(t not in registry[role] for role in roles for t in value[role]):
                reason = 'unregistered_timestamp'
            elif any(value[role] for role in roles):
                result.update(status='valid', citations={role: list(value[role]) for role in roles})
                return result
            else:
                return result
            result.update(status='invalid', reason=reason)
            return result

        raw = json.loads(raw_text, object_pairs_hook=unique_object,
                         parse_constant=reject_constant)
        keys(raw, ('assessment_contract_version', 'status', 'observations',
                   'updates', 'pending_updates'))
        if type(raw['assessment_contract_version']) is not int or raw['assessment_contract_version'] != 3:
            raise ValueError('Wrong version')
        status = raw['status']
        if status not in ('ok', 'insufficient_evidence'):
            raise ValueError('Invalid status')
        if any(not isinstance(raw[name], list) for name in ('observations', 'updates', 'pending_updates')):
            raise ValueError('Claims must be lists')
        if status == 'insufficient_evidence' and (raw['observations'] or raw['pending_updates']):
            raise ValueError('Insufficient evidence cannot support positive claims')
        observations, updates, pending_updates, diagnostics = [], [], [], []
        claimed_open, claimed_pending = set(), set()
        for index, claim in enumerate(raw['observations']):
            keys(claim, ('description', 'association'), ('resolution_condition', 'citations'))
            description = text(claim['description'])
            association = claim['association']
            keys(association, ('kind', 'issue_id', 'pending_id'))
            kind, issue_id, pending_id = (association[k] for k in ('kind', 'issue_id', 'pending_id'))
            if kind == 'same_open':
                if not isinstance(issue_id, str) or issue_id not in opened or issue_id in claimed_open:
                    raise ValueError('Invalid open association')
                claimed_open.add(issue_id)
            elif kind == 'recurrence':
                if not isinstance(issue_id, str) or issue_id not in resolved or resolved[issue_id]['outcome'] != 'corrected':
                    raise ValueError('Invalid historical association')
            elif kind in ('new', 'uncertain'):
                if issue_id is not None:
                    raise ValueError('Unexpected issue ID')
            else:
                raise ValueError('Invalid association kind')
            if pending_id is not None:
                if not isinstance(pending_id, str) or pending_id not in pending or pending_id in claimed_pending:
                    raise ValueError('Invalid pending association')
                claimed_pending.add(pending_id)
            item = {'description': description, 'association': dict(association)}
            if 'resolution_condition' in claim:
                item['resolution_condition'] = text(claim['resolution_condition'])
            observations.append(item)
            diagnostics.append(citation(claim, 'observations', index))
        for index, claim in enumerate(raw['updates']):
            keys(claim, ('issue_id', 'action'), ('citations',))
            issue_id, action = claim['issue_id'], claim['action']
            if not isinstance(issue_id, str) or issue_id not in opened or issue_id in claimed_open:
                raise ValueError('Invalid open update')
            if action not in ('corrected', 'uncertain', 'retract') or (status == 'insufficient_evidence' and action != 'uncertain'):
                raise ValueError('Invalid update action')
            claimed_open.add(issue_id)
            updates.append({'issue_id': issue_id, 'action': action})
            diagnostics.append(citation(claim, 'updates', index))
        for claim in raw['pending_updates']:
            keys(claim, ('pending_id', 'action', 'reason'))
            pending_id = claim['pending_id']
            if not isinstance(pending_id, str) or pending_id not in pending or pending_id in claimed_pending or claim['action'] != 'discard':
                raise ValueError('Invalid pending update')
            claimed_pending.add(pending_id)
            pending_updates.append({'pending_id': pending_id, 'action': 'discard', 'reason': text(claim['reason'])})
        return {'assessment_contract_version': 3, 'status': status,
                'observations': observations, 'updates': updates,
                'pending_updates': pending_updates, 'citation_diagnostics': diagnostics}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError('Invalid assessment completion') from exc


def build_assessment_request(observation, context, *, request_template,
        source_sha256, selected_indices, max_images, max_jpeg_bytes, max_request_bytes):
    """Freeze a request and its supplied frames without inferring clock mappings.

    Context is a host-projected snapshot, not evaluator labels. Its visible
    structure and identities are checked here; the coordinator must bind this
    exact snapshot and wire to the tracker. A pending recurrence may refer to
    history omitted by the projector, so its full lineage remains host-owned.
    Execution source identity is caller-supplied, not authenticated by the
    observation. No capture/source clock mapping is asserted. The returned
    request is preparation only and is not dispatchable in this state.
    """
    try:
        integer(max_request_bytes, 1)
        selected = select_assessment_jpegs(observation, source_sha256=source_sha256,
            selected_indices=selected_indices, max_images=max_images,
            max_jpeg_bytes=max_jpeg_bytes)
        frozen = deepcopy(context)
        keys(frozen, ("assessment_contract_version", "session_id", "reference_id",
            "observation_id", "manifest_hash", "source_cutoff_s", "context_revision",
            "open_issues", "resolved_history", "pending_observations",
            "omitted_resolved_history_count"))
        if (type(frozen["assessment_contract_version"]) is not int
                or frozen["assessment_contract_version"] != 3
                or uuid_string(frozen["session_id"]) != observation["session_id"]
                or uuid_string(frozen["observation_id"]) != observation["observation_id"]
                or digest(frozen["reference_id"]) != observation["reference_id"]
                or digest(frozen["manifest_hash"]) != observation["manifest_sha256"]):
            raise ValueError("Context identity disagrees with observation")
        cutoff = number(frozen["source_cutoff_s"])
        integer(frozen["context_revision"])
        omitted = integer(frozen["omitted_resolved_history_count"])
        ids, history = set(), {}
        definitions = (
            ("open_issues", "issue_id", ("state", "initial_description", "latest_description",
                "first_assessment_cutoff_s", "latest_assessment_cutoff_s", "pending_repair", "resolution_condition")),
            ("resolved_history", "issue_id", ("description", "outcome", "first_assessment_cutoff_s", "terminal_assessment_cutoff_s")),
            ("pending_observations", "pending_id", ("description", "first_assessment_cutoff_s", "latest_assessment_cutoff_s", "recurrence_of")),
        )
        def text(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Context descriptions must be nonblank")
            value.encode("utf-8")

        for name, id_name, fields in definitions:
            rows = frozen[name]
            if not isinstance(rows, list):
                raise ValueError("Context collection must be a list")
            for row in rows:
                keys(row, (id_name,) + fields)
                identifier = uuid_string(row[id_name])
                if identifier in ids:
                    raise ValueError("Duplicate context ID")
                ids.add(identifier)
                first = number(row["first_assessment_cutoff_s"])
                last = number(row["terminal_assessment_cutoff_s" if name == "resolved_history" else "latest_assessment_cutoff_s"])
                if not first <= last <= cutoff:
                    raise ValueError("Context assessment times disagree")
                if name == "open_issues":
                    if row["state"] not in ("candidate", "active", "uncertain") or type(row["pending_repair"]) is not bool:
                        raise ValueError("Invalid open issue")
                    text(row["initial_description"])
                    text(row["latest_description"])
                    if row["resolution_condition"] is not None:
                        text(row["resolution_condition"])
                else:
                    text(row["description"])
                    if name == "resolved_history":
                        if row["outcome"] not in ("corrected", "retracted"):
                            raise ValueError("Invalid terminal outcome")
                        history[identifier] = row
                    elif row["recurrence_of"] is not None:
                        uuid_string(row["recurrence_of"])
        rows = frozen["resolved_history"]
        if rows != sorted(rows, key=lambda row: (-row["terminal_assessment_cutoff_s"], row["issue_id"])):
            raise ValueError("History must use projector ordering")
        for row in frozen["pending_observations"]:
            prior = row["recurrence_of"]
            if prior is not None:
                if prior in history:
                    if (history[prior]["outcome"] != "corrected" or history[prior]["terminal_assessment_cutoff_s"] > row["latest_assessment_cutoff_s"]):
                        raise ValueError("Invalid recurrence lineage")
                elif not omitted or prior in ids:
                    raise ValueError("Missing recurrence lineage")

        provenance = selected["jpeg_provenance"]
        registry = {role + "_s": [row["elapsed_s"] for row in provenance[role]["frames"]]
                    for role in ("reference", "execution")}
        if registry["execution_s"][-1] > cutoff:
            raise ValueError("Execution source frame exceeds source cutoff")
        parts = selected["parts"]
        position = 0
        for role in ("reference", "execution"):
            for elapsed in registry[role + "_s"]:
                parts[position]["text"] = role.upper() + " " + repr(elapsed) + " s"
                position += 2

        keys(request_template, ("model", "messages", "max_tokens", "n", "response_format"))
        if (not isinstance(request_template["model"], str) or not request_template["model"].strip()
                or type(request_template["n"]) is not int or request_template["n"] != 1
                or not isinstance(request_template["response_format"], dict)):
            raise ValueError("Invalid request template")
        integer(request_template["max_tokens"], 1)
        messages = request_template["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("Expected two messages")
        for message in messages:
            keys(message, ("role", "content"))
        if (messages[0]["role"] != "system" or not isinstance(messages[0]["content"], str)
                or messages[1]["role"] != "user" or messages[1]["content"] != []):
            raise ValueError("Invalid message template")
        for label, value in (("OBSERVATION", selected["header"]),
                             ("EVIDENCE_ANCHORS", {r: registry[r + "_s"] for r in ("reference", "execution")}),
                             ("ISSUE_CONTEXT", frozen)):
            parts.append({"type": "text", "text": label + "\n" + canonical(value).decode("utf-8")})
        request = deepcopy(request_template)
        request["messages"][1]["content"] = parts
        wire = canonical(request)
        if len(wire) > max_request_bytes:
            raise RuntimeError("JPEG request exceeds its complete encoded-byte capacity")
        return {"wire": wire, "request_sha256": hashlib.sha256(wire).hexdigest(),
                "jpeg_provenance": provenance, "frame_registry": registry, "context": frozen,
                "clock_mapping": "unavailable", "dispatchable": False,
                "observation_cutoff_s": observation["cutoff_s"]}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid contract-3 assessment JPEG request or observation") from exc
