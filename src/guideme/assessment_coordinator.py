"""Coordinate an offline contract-3 assessment using a simulated completion."""

from copy import deepcopy

from .frame_binding import build_assessment_binding
from .assessment import decode_assessment
from .assessment import build_assessment_request


def run_assessment(tracker, observation, *, source_cutoff_s, request_template,
        source_sha256, selected_indices, max_images, max_jpeg_bytes,
        max_request_bytes, max_resolved_history, max_context_bytes,
        max_response_bytes, clock_context, dispatched_at_s, completed_at_s,
        applied_at_s, replay_clock_assertion, completion):
    """Run a local simulated check against one frozen request and tracker state.

    The callback is for simulated replies only, not provider dispatch. The replay
    identity assertion is a caller-owned assumption, not proof of a clock map.
    Its numeric cutoff equality is necessary for the current tracker's binding
    contract but is not evidence of a physical camera or playback mapping.
    Invalid metadata or unavailable host timing stops before the callback.
    """
    observation, source_sha256, selected_indices, request_template, clock_context, assertion = deepcopy(
        (observation, source_sha256, selected_indices, request_template, clock_context, replay_clock_assertion))
    if not callable(completion):
        raise ValueError("A simulated completion callback is required")
    if (not isinstance(assertion, dict) or set(assertion) != {
            "kind", "session_id", "observation_id", "execution_source_sha256"}
            or assertion["kind"] != "replay_identity_asserted"
            or assertion["session_id"] != observation["session_id"]
            or assertion["observation_id"] != observation["observation_id"]
            or assertion["execution_source_sha256"] != source_sha256["execution"]
            or type(source_cutoff_s) not in (int, float)
            or source_cutoff_s != observation["cutoff_s"]):
        raise ValueError("A matching replay identity assumption is required")
    snapshot = tracker.issue_snapshot()
    if (snapshot["session_id"] != observation["session_id"]
            or snapshot["reference_id"] != observation["reference_id"]
            or snapshot["execution_source_sha256"] != source_sha256["execution"]
            or snapshot["clock_id"] != clock_context.get("clock_id")
            or snapshot["timing_basis"] != clock_context.get("timing_basis")):
        raise ValueError("Tracker, clock and observation must agree")
    context = tracker.issue_context(observation_id=observation["observation_id"],
        manifest_hash=observation["manifest_sha256"], source_cutoff_s=source_cutoff_s,
        max_resolved_history=max_resolved_history, max_context_bytes=max_context_bytes)
    request = build_assessment_request(observation, context,
        request_template=request_template, source_sha256=source_sha256,
        selected_indices=selected_indices, max_images=max_images,
        max_jpeg_bytes=max_jpeg_bytes, max_request_bytes=max_request_bytes)
    binding = build_assessment_binding(observation, jpeg_provenance=request["jpeg_provenance"],
        request_sha256=request["request_sha256"], dispatched_at_s=dispatched_at_s,
        completed_at_s=completed_at_s, clock_context=clock_context)
    if binding["timing_status"] not in ("mapped", "bounded"):
        raise ValueError("Simulated assessment requires a declared frame clock mapping")
    if (snapshot["context_available_at_s"] is not None
            and snapshot["context_available_at_s"] > dispatched_at_s
            or snapshot["latest_applied_source_cutoff_s"] is not None
            and source_cutoff_s <= snapshot["latest_applied_source_cutoff_s"]
            or completed_at_s > applied_at_s):
        raise ValueError("Context, source cutoff or application time is stale")
    raw = completion(request["wire"])
    assessment = decode_assessment(raw, context=request["context"],
        frame_registry=request["frame_registry"], max_response_bytes=max_response_bytes)
    result = tracker.apply_assessment(assessment, binding, context=request["context"], applied_at_s=applied_at_s)
    return {"request": request, "binding": binding, "assessment": assessment,
            "tracker_result": result, "clock_assumption": assertion,
            "mode": "simulated"}


def run_assessment_provider(tracker, observation, *, source_cutoff_s, provider,
        selected_indices, max_resolved_history, max_context_bytes, clock_context,
        replay_clock_assertion, host_clock):
    """Run offline file replay; source-logical and measured host times stay separate.

    The replay identity is a caller-owned file-replay convention, not a camera
    capture mapping. Logical source-time binding cannot measure live freshness;
    transport and host timings must never be subtracted from source time.
    """
    observation, selected_indices, clock_context, assertion = deepcopy(
        (observation, selected_indices, clock_context, replay_clock_assertion))
    snapshot = tracker.issue_snapshot()
    if (not callable(host_clock) or type(source_cutoff_s) not in (int, float)
            or source_cutoff_s != observation['cutoff_s']
            or snapshot['session_id'] != observation['session_id']
            or snapshot['reference_id'] != observation['reference_id']
            or not isinstance(assertion, dict)
            or assertion.get('execution_source_sha256') != snapshot['execution_source_sha256']
            or not isinstance(clock_context.get('latest_frame'), dict)
            or clock_context['latest_frame'].get('source_sha256') != snapshot['execution_source_sha256']
            or snapshot['clock_id'] != clock_context.get('clock_id')
            or snapshot['timing_basis'] != 'ideal_content_cutoff'
            or clock_context.get('timing_basis') != 'ideal_content_cutoff'
            or snapshot['context_available_at_s'] is not None
                and snapshot['context_available_at_s'] > source_cutoff_s
            or snapshot['latest_applied_source_cutoff_s'] is not None
                and source_cutoff_s <= snapshot['latest_applied_source_cutoff_s']):
        raise ValueError('Offline tracker, source cutoff or clock disagrees')
    context = tracker.issue_context(observation_id=observation['observation_id'],
        manifest_hash=observation['manifest_sha256'], source_cutoff_s=source_cutoff_s,
        max_resolved_history=max_resolved_history, max_context_bytes=max_context_bytes)
    prepared = provider.prepare(observation['model_input'], context,
        observation=observation, selected_indices=selected_indices)
    descriptor = provider.describe_prepared(prepared)
    binding = build_assessment_binding(observation, jpeg_provenance=descriptor['jpeg_provenance'],
        request_sha256=descriptor['request_sha256'], dispatched_at_s=source_cutoff_s,
        completed_at_s=source_cutoff_s, clock_context=clock_context)
    if binding['timing_status'] != 'mapped':
        raise ValueError('Offline frame timing is unavailable')
    import math
    def measured():
        value = host_clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid host clock reading')
        return value
    dispatched = measured()
    ticket = provider.admit(prepared, replay_clock_assertion=assertion)
    response = provider.complete(ticket, provider.perform(ticket))
    completed = measured()
    if completed < dispatched:
        raise ValueError('Host clock moved backward')
    if (response['outcome'] != 'completed' or response['observation_id'] != observation['observation_id']
            or response['request_sha256'] != descriptor['request_sha256']
            or response['jpeg_provenance'] != descriptor['jpeg_provenance']):
        raise ValueError('Provider response or frozen request disagrees')
    applied = measured()
    if applied < completed:
        raise ValueError('Host clock moved backward')
    tracker_result = tracker.apply_assessment(response['assessment'], binding,
        context=context, applied_at_s=source_cutoff_s)
    return {'request': descriptor, 'binding': binding, 'assessment': response['assessment'],
        'tracker_result': tracker_result, 'clock_assumption': assertion, 'mode': 'offline_provider',
        'host_timing': {'dispatched_at_s': dispatched, 'completed_at_s': completed, 'applied_at_s': applied},
        'transport_timing': {'started_at_s': response['transport_started_at_s'],
                             'received_at_s': response['transport_received_at_s']},
        'context': context}
