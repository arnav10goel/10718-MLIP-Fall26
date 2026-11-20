"""Shared causal replay, with optional bounded browser-driven concurrency."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import sys
from threading import Event, Lock, Thread
import time
from uuid import UUID, uuid4

from .completion_scheduler import CompletionScheduler
from .media import FileFrameSource, TimestampFrameSampler, encode_frame_jpeg
from .native_video_completion import ProviderAdmissionRefused
from .observations import build_observation


class _PreparationFailure(Exception):
    def __init__(self, code, stage):
        self.code, self.stage = code, stage
        super().__init__(code)


def run_content_replay(reference, execution_path, *, run_id, session_id,
                       completion_plan, sender, trace, config, data_root=None,
                       manifest=None, callback_errors_fatal=False, runtime=None,
                       scheduling=None, completion_provider=None,
                       execution_buffer=None, buffer_seal_interval_s=1.0,
                       assessment_contract_version=1, input_representation="native_video",
                       compact_selection=None, pair_provider=None, mock_assessment=None, assessment_provider=None,
                       simulated_assessment_delivery=False):
    """Coordinate bounded content replay through injected completion and delivery.

    Cutoffs are n * decision_interval_s, beginning at zero. Source capture and
    receipt both equal source elapsed PTS; all runtime completions/deliveries
    use the current cutoff. These are simulated content timings, never measured
    live latency. EOF is runtime provenance, not an observed task completion.

    completion_plan(model_input, prior_issues) receives only visual input and
    causal model-derived issue descriptions, states and last-update evidence,
    and returns exactly response, updates, and new_issue_candidate_indices.
    These are runtime beliefs, not a human checklist or evaluator labels.
    This adapter does not establish detection or semantic association quality.
    Every raw return is traced before strict validation against mock_predict's
    existing normalized-response boundary. A future real backend needs its own
    approved adapter. With callback_errors_fatal=True, unexpected callback and
    persistence errors propagate unchanged, leaving a partial trace. Such an
    adapter must return typed failed predictions for expected provider errors.
    The default retains scripted exception-to-failure behavior. The caller owns
    and closes the bound trace writer.

    With runtime supplied and scheduling absent, its blocking next_cutoff gate
    replaces the fixed grid. Only the latest observed execution prefix is consumed. Predictions
    become available at the current source playhead after callback completion,
    and delivery uses actual receiver acknowledgments. One invocation retains
    issue/state memory across the session; no inference queue is created here.

    Explicit scheduling selects the same engine's asynchronous browser mode.
    One preparation lane owns execution decoding; the coordinator alone owns
    issue state, provider admission/accounting, trace appends and notification
    outcomes. Completed requests are accounted even when their state proposal is
    superseded or the browser has stopped. Without scheduling the historical
    serial/offline path is retained.

    An optional caller-owned execution buffer selects an independent producer
    for asynchronous playback. Its committed source cutoff, rather than the
    advancing playhead, binds observations and native snapshots. The runtime
    must support concurrent pacing reads. The producer and provider workers
    are joined before return; the caller closes the buffer afterward.

    Contract 2 is explicitly opt-in and uses one assessment tracker and delivery
    ledger. Offline replay keeps ideal content clocks and separate transport
    diagnostics. Asynchronous browser replay freezes native-tail passage bounds
    before HTTP and uses the host session clock for context, transport, state and
    correlated receipt processing. Render intervals remain intervals; no point
    capture time or physical display is inferred.

    JPEG input uses the same preparation lane with a bounded full cadence pool
    and one transient actual endpoint. Compact selection limits the sent images;
    the full observation limits still apply independently. Terminal context
    refresh reuses the same source snapshot without extending its drain deadline
    or changing the selected frame's passage time.

    Contract 3 supports offline JPEG replay and scheduled browser JPEG replay.
    Offline callback execution is ordinary caller code, not a network sandbox;
    the execution hash is caller supplied and is not authenticated here. Offline
    capture and receipt times are logical video times. Browser replay uses the
    session clock and a frozen playback interval for the actual selected frame.
    Assessment simulated delivery is an explicit offline opt-in and does not measure
    live delivery timing.
    """
    if type(assessment_contract_version) is not int or assessment_contract_version not in (1, 2, 3):
        raise ValueError("Unsupported assessment contract version")
    assessment_two = assessment_contract_version == 2
    assessment_three = assessment_contract_version == 3
    live_tracking = assessment_three and runtime is not None
    versioned_live = (assessment_two or assessment_three) and runtime is not None
    if type(simulated_assessment_delivery) is not bool or simulated_assessment_delivery and not assessment_three:
        raise ValueError("Simulated assessment delivery requires contract 3")
    if assessment_three:
        mock_valid = (isinstance(mock_assessment, dict) and set(mock_assessment) == {"request_template", "max_request_bytes",
                    "max_response_bytes", "max_context_bytes", "max_resolved_history", "completion"}
                and callable(mock_assessment["completion"]))
        provider_valid = (isinstance(assessment_provider, dict) and set(assessment_provider) == {
                    "provider", "max_context_bytes", "max_resolved_history", "host_clock"}
                and callable(assessment_provider["host_clock"]) and all(callable(getattr(assessment_provider["provider"], name, None))
                    for name in ("prepare", "describe_prepared", "admit", "perform", "complete", "snapshot")))
        if live_tracking:
            if (input_representation != "jpeg_frames" or scheduling is None or completion_provider is None
                    or execution_buffer is not None or pair_provider is not None or mock_assessment is not None
                    or assessment_provider is not None or simulated_assessment_delivery):
                raise ValueError("Live contract 3 requires a scheduled JPEG split provider")
        elif (input_representation != "jpeg_frames" or scheduling is not None
                or completion_provider is not None or execution_buffer is not None or pair_provider is not None
                or mock_valid == provider_valid or mock_assessment is not None and not mock_valid
                or assessment_provider is not None and not provider_valid):
            raise ValueError("Contract 3 requires one offline JPEG completion path")
    elif mock_assessment is not None or assessment_provider is not None:
        raise ValueError("Simulated contract-3 settings require contract 3")
    if input_representation not in ("native_video", "jpeg_frames"):
        raise ValueError("Unsupported input representation")
    jpeg_input = input_representation == "jpeg_frames"
    pair_offline = pair_provider is not None and runtime is None
    pair_browser = pair_provider is not None and runtime is not None
    if pair_browser:
        if completion_provider is not None:
            raise ValueError("Pair provider cannot be combined with another completion provider")
        completion_provider = pair_provider
        if not assessment_two or not jpeg_input or scheduling is None or execution_buffer is not None:
            raise ValueError("Two-stage browser replay requires scheduled contract-2 JPEG input")
    if pair_offline and (not assessment_two or not jpeg_input or runtime is not None
                         or scheduling is not None or execution_buffer is not None
                         or completion_provider is not None):
        raise ValueError("Two-stage pair replay requires offline contract-2 JPEG input")
    if pair_offline and not all(callable(getattr(pair_provider, name, None))
                                for name in ("prepare", "describe_prepared", "admit", "perform", "complete")):
        raise ValueError("Two-stage replay requires a split pair provider")
    if jpeg_input and not pair_offline and not assessment_three and (not assessment_two or runtime is None or scheduling is None
                                             or execution_buffer is not None):
        raise ValueError("JPEG monitoring requires scheduled contract-2 playback without native buffering")
    if not jpeg_input and compact_selection is not None:
        raise ValueError("Compact selection requires JPEG input")
    provenance_key = "jpeg_provenance" if jpeg_input else "native_provenance"
    provenance_frames = "frames" if jpeg_input else "frame_map"
    if assessment_two and runtime is None and (scheduling is not None or execution_buffer is not None or completion_provider is not None):
        raise ValueError("Offline assessment replay does not support browser scheduling or split providers")
    if assessment_two and runtime is not None and (scheduling is None or completion_provider is None):
        raise ValueError("Browser assessment replay requires asynchronous scheduling and a split provider")
    def keys(record, fields):
        if not isinstance(record, dict) or set(record) != set(fields):
            raise ValueError("Record has missing or unsupported fields")

    def positive(value):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Expected a finite positive number")
        return value

    def integer(value):
        if type(value) is not int or value <= 0:
            raise ValueError("Expected a positive integer limit")

    if jpeg_input:
        keys(compact_selection, ("coarse_interval_s", "recent_window_s"))
        compact_selection = deepcopy(compact_selection)
        for value in compact_selection.values():
            positive(value)
        from .compact_frames import select_compact_frames

    def finite_json(value, ancestors):
        if type(value) in (dict, list):
            if id(value) in ancestors:
                raise ValueError("Manifest cannot contain cycles")
            ancestors.add(id(value))
            try:
                if type(value) is dict:
                    if any(type(key) is not str for key in value):
                        raise ValueError("Manifest keys must be strings")
                    children = value.values()
                else:
                    children = value
                for child in children:
                    finite_json(child, ancestors)
            finally:
                ancestors.remove(id(value))
        elif value is not None and type(value) not in (str, bool, int, float):
            raise ValueError("Manifest must contain finite JSON values")

    keys(config, ("decision_interval_s", "execution_fps", "execution_max_frames",
                  "max_side", "jpeg_quality", "max_observations", "max_images",
                  "max_jpeg_bytes", "max_evidence_age_s", "state", "issues", "sink")
         + (("repair_confirmation_count",) if isinstance(config, dict)
            and "repair_confirmation_count" in config else ()))
    config = deepcopy(config)
    repair_confirmation_count = config.get("repair_confirmation_count", 1)
    if (type(repair_confirmation_count) is not int or repair_confirmation_count not in (1, 2)
            or repair_confirmation_count == 2 and not (assessment_two or assessment_three)):
        raise ValueError("Repair confirmation count must be 1, or 2 for assessment contract 2")
    interval = positive(config["decision_interval_s"])
    positive(config["max_evidence_age_s"])
    for field in ("execution_max_frames", "max_side", "jpeg_quality", "max_observations",
                  "max_images", "max_jpeg_bytes"):
        integer(config[field])
    if config["jpeg_quality"] > 100:
        raise ValueError("JPEG quality must be from 1 to 100")
    keys(config["state"], ("max_records", "max_metadata_bytes"))
    keys(config["issues"], ("max_issues", "max_observations", "max_metadata_bytes"))
    keys(config["sink"], ("max_attempts", "max_metadata_bytes"))
    try:
        valid_id = isinstance(run_id, str) and str(UUID(run_id)) == run_id
    except ValueError:
        valid_id = False
    if not valid_id:
        raise ValueError("Run ID must be a canonical UUID string")
    if ((not assessment_three and not callable(completion_plan)) or not callable(sender)
            or not callable(getattr(trace, "append", None))):
        raise ValueError("Completion plan, sender and trace append must be callable")
    if type(callback_errors_fatal) is not bool:
        raise ValueError("callback_errors_fatal must be a boolean")
    asynchronous = scheduling is not None
    runtime_methods = ("source_time", "elapsed_s", "active") + (
        ("poll_cutoff", "opportunity_state", "publish", "delivery_outcome",
         "notification_ready", "retire_notification") if asynchronous else
        ("next_cutoff", "sender"))
    if runtime is not None and (getattr(runtime, "session_id", None) != session_id
            or not all(callable(getattr(runtime, name, None)) for name in runtime_methods)):
        raise ValueError("Runtime must provide the bound session, playhead and delivery interface")
    if versioned_live and (
            not isinstance(getattr(runtime, "clock_id", None), str) or not runtime.clock_id.strip()
            or not all(callable(getattr(runtime, name, None)) for name in
                       ("source_passage_interval", "close_receipts", "timing_snapshot"))
            or not callable(getattr(completion_provider, "describe_prepared", None))):
        raise ValueError("Browser assessment replay requires explicit clock, timing and receipt interfaces")
    if asynchronous:
        keys(scheduling, ("max_in_flight", "notification_min_visible_s",
                          "drain_timeout_s", "poll_interval_s"))
        scheduling = deepcopy(scheduling)
        integer(scheduling["max_in_flight"])
        positive(scheduling["poll_interval_s"])
        for name in ("notification_min_visible_s", "drain_timeout_s"):
            value = scheduling[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Scheduling durations must be finite and nonnegative")
        if runtime is None or not all(callable(getattr(completion_provider, name, None))
                for name in ("prepare", "admit", "perform", "complete", "cancel_unsent", "snapshot")):
            raise ValueError("Asynchronous scheduling requires a runtime and split completion provider")
    positive(buffer_seal_interval_s)
    if execution_buffer is not None and (not asynchronous or not all(
            callable(getattr(execution_buffer, name, None)) for name in ("append", "seal", "snapshot"))):
        raise ValueError("Execution buffering requires asynchronous playback and append/seal/snapshot")
    finite_json(manifest, set())
    try:
        json.dumps(manifest, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Manifest must contain finite UTF-8 JSON values") from exc
    manifest = deepcopy(manifest)
    reference = deepcopy(reference)
    if jpeg_input and len(reference["frames"]) + config["execution_max_frames"] + 1 > config["max_images"]:
        raise ValueError("Full observation pool cannot hold its reference, cadence and endpoint bounds")
    # Validate the complete prepared reference and fixed budgets without opening
    # execution media. Helpers enforce their own identity and capacity ranges.
    initial = build_observation(reference, [], session_id=session_id,
        observation_id=str(uuid4()), cutoff_s=0, max_images=config["max_images"],
        max_jpeg_bytes=config["max_jpeg_bytes"])
    reference_id = initial["reference_id"]
    sampler = TimestampFrameSampler(fps=config["execution_fps"],
                                    max_frames=config["execution_max_frames"])
    assessment_clock_id = runtime.clock_id if versioned_live else "content:" + session_id
    assessment_timing_basis = "session_monotonic" if runtime is not None else "ideal_content_cutoff"
    execution_source_sha256 = None
    if assessment_two or assessment_three:
        from .frame_binding import build_assessment_binding
        from .notification_selection import choose_assessment_notification
        from .notification_delivery import AssessmentNotificationSink
        from .issue_memory import AssessmentIssueTracker
        execution_source_sha256 = manifest.get("execution_source_sha256") if isinstance(manifest, dict) else None
        if (not isinstance(execution_source_sha256, str) or len(execution_source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in execution_source_sha256)):
            raise ValueError("Assessment replay requires the frozen execution source SHA-256")
        tracker = AssessmentIssueTracker(session_id, reference_id, execution_source_sha256,
            clock_id=assessment_clock_id, timing_basis=assessment_timing_basis,
            repair_confirmation_count=repair_confirmation_count, **config["issues"])
        state = None
    else:
        # The legacy tracker is used only by contract 1. Keep these modules out
        # of the assessment-monitor startup path without removing that contract.
        from .issue_tracker import IssueTracker
        from .issue_updates import validate_issue_updates
        from .mock_predictor import mock_predict
        from .notification_policy import choose_notification
        from .notification_sink import NotificationSink
        from .session_state import SessionState

        state = SessionState(session_id, reference_id, **config["state"])
        tracker = IssueTracker(session_id, reference_id, **config["issues"])
    now_s = 0
    send_trace_error = None

    def current_time():
        nonlocal now_s
        if runtime is not None:
            value = runtime.elapsed_s() if (assessment_two or live_tracking) else runtime.source_time()
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < now_s):
                raise ValueError("Runtime event clock must be finite and nondecreasing")
            now_s = value
        return now_s

    if pair_offline or pair_browser:
        from .assessment_pair_coordinator import AssessmentPairCoordinator
        pair_coordinator = AssessmentPairCoordinator(pair_provider, tracker, clock=current_time)

    def traced_sender(envelope):
        nonlocal send_trace_error
        try:
            emit("notification_attempt", {**deepcopy(envelope), "status": "pending",
                "completed_at_s": None, "delivered_at_s": None, "error_code": None})
        except BaseException as exc:
            # The sink catches ordinary sender exceptions as unknown delivery.
            # Preserve trace errors separately so the runner can rethrow them.
            send_trace_error = exc
            raise
        return sender(envelope)

    if assessment_two or assessment_three:
        sink = AssessmentNotificationSink(session_id, reference_id, execution_source_sha256,
            clock_id=assessment_clock_id, timing_basis=assessment_timing_basis,
            delivery_mode="display_ack" if runtime is not None else "simulated",
            assessment_contract_version=3 if (simulated_assessment_delivery or live_tracking) else 2, **config["sink"])
    else:
        sink = NotificationSink(session_id, reference_id, sender=traced_sender,
            clock=current_time, delivery_mode="simulated" if runtime is None else "display_ack",
            **config["sink"])
    observations = predictions = failures = skipped_assessments = 0
    status, reason = "incomplete", "max_observations"
    source = None
    samples = []
    latest_frame = endpoint_sample = None
    latest_ready_source = None
    scheduling_report = {}
    buffering_report = {}

    def emit(event_type, payload):
        # Trace failures propagate rather than becoming runtime failures. The
        # sender wrapper separately preserves exceptions caught by the sink.
        if assessment_two or assessment_three:
            version = 3 if assessment_three else 2
            if payload.get("assessment_contract_version", version) != version:
                raise ValueError("Cannot mix assessment contracts in one replay")
            payload = {"assessment_contract_version": version, **payload}
        trace.append(event_type, payload, at_s=current_time())

    def failure(code, stage, observation_id=None):
        nonlocal failures
        failures += 1
        payload = {"code": code, "stage": stage}
        if observation_id is not None:
            payload["observation_id"] = observation_id
        emit("failure", payload)

    def failed_prediction(observation, code):
        return {"schema_version": 1, "session_id": session_id,
            "observation_id": observation["observation_id"], "status": "failed",
            "candidate_issues": [], "error": {"type": code,
                "message": "Completion failed validation or execution"}}

    def empty_updates(observation):
        return {"schema_version": 1, "session_id": session_id,
            "observation_id": observation["observation_id"], "reference_id": reference_id,
            "candidate_links": [], "issue_updates": []}

    def issue_context(at_s):
        if assessment_three:
            snapshot = tracker.issue_snapshot()
            prior = {"assessment_contract_version": 3, "revision": snapshot["revision"],
                     "context_available_at_s": snapshot["context_available_at_s"],
                     "source_cutoffs_s": [row["latest_binding"]["observation_cutoff_s"] for row in snapshot["issues"]]}
            return snapshot, prior, None
        if assessment_two:
            snapshot = tracker.snapshot()
            prior = {"assessment_contract_version": 2, "revision": snapshot["revision"],
                     "context_available_at_s": snapshot["context_available_at_s"],
                     "source_cutoffs_s": [row["latest_binding"]["observation_cutoff_s"] for row in snapshot["issues"]]}
            if prior["context_available_at_s"] is not None and prior["context_available_at_s"] > at_s:
                raise ValueError("Assessment context was unavailable on the dispatch clock")
            pending_repairs = tracker.pending_repairs()
            resolution_conditions = tracker.resolution_conditions()
            context = {"assessment_contract_version": 2, "session_id": session_id,
                "reference_id": reference_id, "issues": [{"issue_id": row["issue_id"],
                    "state": row["state"], "initial_description": row["initial_description"],
                    "latest_description": row["latest_description"],
                    "repair_pending": row["issue_id"] in pending_repairs,
                    **({"resolution_condition": resolution_conditions[row["issue_id"]]}
                       if row["issue_id"] in resolution_conditions else {})}
                    for row in snapshot["issues"]]}
            return snapshot, prior, context
        snapshot = tracker.snapshot(at_s)
        prior = snapshot["prior_issues"]
        context = deepcopy(prior)
        records = {row["issue_id"]: row for row in snapshot["issues"]}
        for row in context["issues"]:
            issue = records[row["issue_id"]]
            row["description"] = issue["latest_candidate"]["description"]
            for name in ("execution_evidence_s", "reference_evidence_s"):
                row[name] = deepcopy(issue["latest_update"][name])
        return snapshot, prior, context

    def prepare_observation(cutoff, observation_id):
        nonlocal latest_frame, endpoint_sample, latest_ready_source
        # Source, sampler and samples have one owner, including in worker mode.
        while True:
            try:
                frame = source.read(cutoff)
            except Exception as exc:
                raise _PreparationFailure("source_failure", "source") from exc
            if frame is None:
                break
            if jpeg_input:
                latest_frame = frame
            try:
                selected = sampler.consider(frame, cutoff)
                if selected is not None:
                    image = encode_frame_jpeg(selected["frame"],
                        max_side=config["max_side"], jpeg_quality=config["jpeg_quality"])
                    samples.append({"session_id": session_id,
                        "reference_id": reference_id, "captured_at_s": frame["elapsed_s"],
                        "received_at_s": frame["elapsed_s"], "image": image,
                        "target_s": selected["target_s"],
                        "skipped_targets": selected["skipped_targets"]})
            except Exception as exc:
                raise _PreparationFailure("sampling_failure", "sampling") from exc
        ready_samples = samples
        if jpeg_input:
            ready_samples = list(samples)
            if latest_frame is not None and (not samples or
                    samples[-1]["image"]["frame_index"] != latest_frame["frame_index"]):
                if (endpoint_sample is None or
                        endpoint_sample["image"]["frame_index"] != latest_frame["frame_index"]):
                    try:
                        image = encode_frame_jpeg(latest_frame, max_side=config["max_side"],
                                                  jpeg_quality=config["jpeg_quality"])
                    except Exception as exc:
                        raise _PreparationFailure("sampling_failure", "sampling") from exc
                    endpoint_sample = {"session_id": session_id, "reference_id": reference_id,
                        "captured_at_s": latest_frame["elapsed_s"], "received_at_s": latest_frame["elapsed_s"],
                        "image": image, "target_s": latest_frame["elapsed_s"], "skipped_targets": 0}
                ready_samples.append(endpoint_sample)
        try:
            observation = build_observation(reference, ready_samples,
                session_id=session_id, observation_id=observation_id, cutoff_s=cutoff,
                max_images=config["max_images"], max_jpeg_bytes=config["max_jpeg_bytes"])
        except Exception as exc:
            raise _PreparationFailure("observation_failure", "observation") from exc
        if jpeg_input and observation["status"] == "ready":
            latest_ready_source = {"cutoff_s": cutoff, "samples": tuple(deepcopy(ready_samples)),
                "source_eof": source.eof,
                "endpoint_regular": samples[-1]["image"]["frame_index"] == latest_frame["frame_index"]}
        return observation, source.eof

    def normalize_result(observation, prior, dispatched_at, raw, completion_error=None):
        nonlocal predictions
        response = updates = admissions = None
        if completion_error is None:
            try:
                keys(raw, ("response", "updates", "new_issue_candidate_indices"))
                if not isinstance(raw["response"], dict):
                    raise ValueError("A normalized response is required")
                response = mock_predict(observation["model_input"], response=raw["response"])
                updates = validate_issue_updates(response, raw["updates"],
                    observation=observation, prior_issues=prior, dispatched_at_s=dispatched_at)
                admissions = raw["new_issue_candidate_indices"]
                if not isinstance(admissions, list):
                    raise ValueError("Admissions must be an explicit list")
                targets = {link["candidate_index"]: link["issue_id"]
                           for link in updates["candidate_links"]}
                seen = set()
                for index in admissions:
                    if (type(index) is not int or index in seen
                            or index not in targets or targets[index] is not None):
                        raise ValueError("Admit unique null-linked candidates only")
                    seen.add(index)
                admissions = deepcopy(admissions)
            except Exception:
                completion_error = "invalid_completion"
        if completion_error is not None:
            response = failed_prediction(observation, completion_error)
            updates = validate_issue_updates(response, empty_updates(observation),
                observation=observation, prior_issues=prior, dispatched_at_s=dispatched_at)
            admissions = []
            failure(completion_error, "completion", observation["observation_id"])
        elif response["status"] == "failed":
            failure("prediction_failed", "completion", observation["observation_id"])
        predictions += 1
        emit("prediction", {"response": response, "updates": updates,
                            "new_issue_candidate_indices": admissions})
        return response, updates, admissions

    def apply_result(observation, dispatched_at, response, updates, admissions):
        completed_at = current_time()
        receipt = tracker.apply_completed(response, updates,
            new_issue_candidate_indices=admissions, completed_at_s=completed_at)
        state.record_prediction(response, reference_id=reference_id,
            observation_cutoff_s=observation["cutoff_s"], dispatched_at_s=dispatched_at,
            completed_at_s=completed_at)
        issue_snapshot = tracker.snapshot(current_time())
        state_snapshot = state.snapshot(now_s)
        emit("issue_state", {"issue_snapshot": issue_snapshot,
            "state_snapshot": state_snapshot, "receipt": receipt})
        return issue_snapshot, receipt

    recoverable_assessment_skip = object()

    def consume_assessment(observation, prior, dispatched_at, raw, completion_error, *, frozen=None, apply_state=True):
        """Bind one completed core reply; model localization never supplies clocks."""
        nonlocal predictions, skipped_assessments
        code = completion_error
        assessment = binding = None
        recoverable_checked = False
        admissions = []
        if code is None:
            try:
                keys(raw, ("assessment_contract_version", "outcome", "assessment", "error",
                    "observation_id", "request_sha256", provenance_key,
                    "transport_started_at_s", "transport_received_at_s"))
                if (type(raw["assessment_contract_version"]) is not int or raw["assessment_contract_version"] != assessment_contract_version
                        or raw["observation_id"] != observation["observation_id"]):
                    raise ValueError("Completion identity or assessment contract disagrees")
                if raw["outcome"] == "not_assessed":
                    if observation["cutoff_s"] != 0 or any(raw[name] is not None for name in ("assessment", "error", "request_sha256",
                            provenance_key, "transport_started_at_s", "transport_received_at_s")):
                        raise ValueError("Unassessed observations cannot claim a provider result")
                elif raw["outcome"] in ("failed", "completed"):
                    if raw["outcome"] == "failed":
                        keys(raw["error"], ("type", "message"))
                        if raw["assessment"] is not None or any(
                                not isinstance(value, str) or not value.strip() for value in raw["error"].values()):
                            raise ValueError("Provider failure requires a typed error and no core assessment")
                        code = raw["error"]["type"]
                    elif raw["error"] is not None or not isinstance(raw["assessment"], dict):
                        raise ValueError("Completed provider result requires an assessment")
                    if code is not None and code != "recoverable_invalid_completion":
                        predictions += 1
                        failure(code, "completion", observation["observation_id"])
                        return False
                    transport = [raw[name] for name in ("transport_started_at_s", "transport_received_at_s")]
                    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in transport)
                            or transport[1] < transport[0]):
                        raise ValueError("Provider transport clocks disagree")
                    native = raw[provenance_key]
                    if native["execution"]["source_sha256"] != execution_source_sha256:
                        raise ValueError("Completion execution source differs from the frozen input")
                    latest = native["execution"][provenance_frames][-1]
                    clock_context = {"session_id": session_id, "observation_id": observation["observation_id"],
                        "clock_id": assessment_clock_id, "timing_basis": assessment_timing_basis}
                    binding_dispatch, binding_complete = dispatched_at, current_time()
                    if runtime is not None:
                        descriptor = frozen["descriptor"]
                        if (descriptor["observation_id"] != raw["observation_id"]
                                or descriptor["request_sha256"] != raw["request_sha256"]
                                or descriptor[provenance_key] != native
                                or not frozen["mapping_frozen_at_s"] <= frozen["admitted_at_s"] <= transport[0]):
                            raise ValueError("Completion differs from the pre-dispatch descriptor or chronology")
                        clock_context["playback_mapping"] = deepcopy(frozen["playback_mapping"])
                        binding_dispatch, binding_complete = transport
                    else:
                        # Ideal content timing is explicit and never a camera
                        # capture or browser presentation measurement.
                        clock_context["latest_frame"] = {"source_sha256": execution_source_sha256,
                            "frame_index": latest["frame_index"], "source_elapsed_s": latest["elapsed_s"],
                            "captured_at_s": latest["elapsed_s"], "received_at_s": latest["elapsed_s"]}
                    binding = build_assessment_binding(observation, **{provenance_key: native},
                        request_sha256=raw["request_sha256"], dispatched_at_s=binding_dispatch,
                        completed_at_s=binding_complete, clock_context=clock_context)
                    recoverable_checked = code == "recoverable_invalid_completion"
                    if code is None:
                        assessment = raw["assessment"]
                        admissions = ([] if live_tracking else [index for index, issue in enumerate(assessment["issues"]) if issue["link"] == "new"])
                else:
                    raise ValueError("Unsupported assessment outcome")
            except (KeyError, TypeError, ValueError, OverflowError, IndexError):
                code = "invalid_completion"
        if code is not None:
            predictions += 1
            if recoverable_checked:
                skipped_assessments += 1
                emit("assessment_skipped", {"observation_id": observation["observation_id"], "reason": code})
                return recoverable_assessment_skip
            failure(code, "completion", observation["observation_id"])
            return False
        if assessment is None:
            emit("assessment_skipped", {"observation_id": observation["observation_id"], "reason": "not_assessed"})
            return True
        predictions += 1
        emit("prediction", {"observation_id": observation["observation_id"], "assessment": assessment,
            "binding": binding, "new_issue_candidate_indices": admissions})
        if not apply_state:
            return assessment, binding, admissions
        return apply_assessment(observation, prior, assessment, binding, admissions, notify=True)

    def apply_assessment(observation, prior, assessment, binding, admissions, *, notify=False):
        try:
            receipt = tracker.apply_completed(assessment, binding, prior_revision=prior["revision"],
                new_issue_candidate_indices=admissions, applied_at_s=current_time())
        except (ValueError, RuntimeError):
            failure("state_failure", "state", observation["observation_id"])
            return False
        snapshot = tracker.snapshot()
        emit("issue_state", {"issue_snapshot": snapshot, "state_snapshot": snapshot, "receipt": receipt})
        if not notify:
            return True
        return deliver_assessment(observation, snapshot)

    def deliver_assessment(observation, snapshot):
        delivery = sink.snapshot()
        decision = choose_assessment_notification(snapshot,
            delivered_issue_ids=delivery["delivered_issue_ids"], pending_issue_ids=delivery["pending_issue_ids"],
            at_s=current_time(), clock_id=assessment_clock_id, max_input_age_s=config["max_evidence_age_s"])
        emit("policy_decision", decision)
        current_snapshot = tracker.issue_snapshot() if assessment_three else tracker.snapshot()
        envelope = sink.reserve(decision, current_snapshot, at_s=current_time())
        if envelope is None:
            return True
        emit("notification_attempt", sink.snapshot()["attempts"][-1])
        try:
            outcome = sender(deepcopy(envelope))
        except Exception:
            outcome = {"status": "unknown"}
        except BaseException:
            emit("notification_attempt", sink.complete(envelope, {"status": "unknown"}, at_s=current_time()))
            raise
        attempt = sink.complete(envelope, outcome, at_s=current_time())
        emit("notification_attempt", attempt)
        if attempt["status"] != "acknowledged":
            failure("delivery_" + attempt["status"], "notification", observation["observation_id"])
        return True

    def execute_offline_assessment(observation, cutoff):
        # This path owns FileFrameSource and builds its samples above using
        # source elapsed time as the simulated capture/receipt clock. It is
        # not a camera clock or evidence of live delivery timing.
        nonlocal predictions
        from .assessment_coordinator import run_assessment, run_assessment_provider
        rows = {}
        for role in ("reference", "execution"):
            rows[role] = [{"frame_index": row["frame_index"], "elapsed_s": row["elapsed_s"],
                "selection_reason": row["selection_reason"] if role == "reference" else "cadence"}
                for row in observation["manifest"][role + "_frames"]]
        if not latest_ready_source["endpoint_regular"]:
            rows["execution"][-1]["selection_reason"] = "current_endpoint"
        selection = select_compact_frames(rows["reference"], rows["execution"],
            source_cutoff_s=cutoff, **compact_selection)
        tail = observation["manifest"]["execution_frames"][selection["selected_indices"]["execution"][-1]]
        clock_context = {"session_id": session_id, "observation_id": observation["observation_id"],
            "clock_id": assessment_clock_id, "timing_basis": assessment_timing_basis,
            "latest_frame": {"source_sha256": execution_source_sha256, "frame_index": tail["frame_index"],
                "source_elapsed_s": tail["elapsed_s"], "captured_at_s": tail["captured_at_s"],
                "received_at_s": tail["received_at_s"]}}
        assertion = {"kind": "replay_identity_asserted", "session_id": session_id,
            "observation_id": observation["observation_id"], "execution_source_sha256": execution_source_sha256}
        emit("compact_selection", {"observation_id": observation["observation_id"], "audit": selection["audit"]})
        # Capture the provider ledger before this check so a prior failed
        # attempt cannot be mistaken for this observation's result.
        provider_before = (assessment_provider["provider"].snapshot()
            if assessment_provider is not None and callable(getattr(assessment_provider["provider"], "snapshot", None))
            else None)
        try:
            if assessment_provider is not None:
                assertion["kind"] = "offline_replay_source_elapsed"
                result = run_assessment_provider(tracker, observation, source_cutoff_s=cutoff,
                    provider=assessment_provider['provider'], selected_indices=selection['selected_indices'],
                    max_context_bytes=assessment_provider['max_context_bytes'],
                    max_resolved_history=assessment_provider['max_resolved_history'], clock_context=clock_context,
                    replay_clock_assertion=assertion, host_clock=assessment_provider['host_clock'])
            else:
                result = run_assessment(tracker, observation, source_cutoff_s=cutoff,
                request_template=mock_assessment["request_template"], source_sha256={
                    "reference": observation["manifest"]["reference_source_sha256"],
                    "execution": execution_source_sha256}, selected_indices=selection["selected_indices"],
                max_images=config["max_images"], max_jpeg_bytes=config["max_jpeg_bytes"],
                max_request_bytes=mock_assessment["max_request_bytes"], max_response_bytes=mock_assessment["max_response_bytes"],
                max_context_bytes=mock_assessment["max_context_bytes"], max_resolved_history=mock_assessment["max_resolved_history"],
                clock_context=clock_context, dispatched_at_s=cutoff, completed_at_s=cutoff,
                applied_at_s=cutoff, replay_clock_assertion=assertion, completion=mock_assessment["completion"])
        except Exception as exc:
            recoverable = False
            if (assessment_provider is not None and provider_before is not None
                    and type(exc) is ValueError
                    and str(exc) == "Provider response or frozen request disagrees"):
                after = assessment_provider["provider"].snapshot()
                before_attempts, attempts = provider_before.get("attempts"), after.get("attempts")
                if (isinstance(before_attempts, list) and isinstance(attempts, list)
                        and len(attempts) == len(before_attempts) + 1
                        and after.get("generation_calls") == provider_before.get("generation_calls", -2) + 1
                        and after.get("accounting_mode") == "settled_usage"
                        and after.get("stopped") is False):
                    attempt = attempts[-1]
                    recoverable = (attempt.get("observation_id") == observation["observation_id"]
                        and attempt.get("cutoff_s") == cutoff and attempt.get("status") == "failed"
                        and attempt.get("code") == "invalid_completion" and attempt.get("cost_complete") is True)
            if recoverable:
                predictions += 1
                failure("recoverable_invalid_completion", "completion", observation["observation_id"])
                return True
            if callback_errors_fatal:
                raise
            predictions += 1
            failure("assessment_completion_failed", "completion", observation["observation_id"])
            return False
        predictions += 1
        request = result["request"]
        emit("assessment_completion", {"observation_id": observation["observation_id"],
            "request_sha256": request["request_sha256"], "jpeg_provenance": request["jpeg_provenance"],
            "frame_registry": request.get("frame_registry") if assessment_provider is None else {
                role + "_s": [row["elapsed_s"] for row in request["jpeg_provenance"][role]["frames"]]
                for role in ("reference", "execution")},
            "context": result.get("context", request.get("context")),
            "binding": result["binding"], "assessment": result["assessment"],
            **({"host_timing": result["host_timing"], "transport_timing": result["transport_timing"]}
               if assessment_provider is not None else {})})
        snapshot = tracker.issue_snapshot()
        emit("issue_state", {"issue_snapshot": snapshot, "state_snapshot": snapshot})
        return deliver_assessment(observation, snapshot) if simulated_assessment_delivery else True

    def execute_offline_pair(observation, cutoff):
        nonlocal predictions
        rows = {}
        for role in ("reference", "execution"):
            rows[role] = [{"frame_index": row["frame_index"], "elapsed_s": row["elapsed_s"],
                "selection_reason": row["selection_reason"] if role == "reference" else "cadence"}
                for row in observation["manifest"][role + "_frames"]]
        if not latest_ready_source["endpoint_regular"]:
            rows["execution"][-1]["selection_reason"] = "current_endpoint"
        selection = select_compact_frames(rows["reference"], rows["execution"],
            source_cutoff_s=cutoff, **compact_selection)
        tail_index = selection["selected_indices"]["execution"][-1]
        tail = observation["manifest"]["execution_frames"][tail_index]
        clock_context = {"session_id": session_id, "observation_id": observation["observation_id"],
            "clock_id": assessment_clock_id, "timing_basis": assessment_timing_basis,
            "latest_frame": {"source_sha256": execution_source_sha256, "frame_index": tail["frame_index"],
                "source_elapsed_s": tail["elapsed_s"], "captured_at_s": tail["captured_at_s"],
                "received_at_s": tail["received_at_s"]}}
        emit("compact_selection", {"observation_id": observation["observation_id"], "audit": selection["audit"]})
        try:
            result = pair_coordinator.run(observation, selection["selected_indices"], clock_context=clock_context)
        except Exception:
            if callback_errors_fatal:
                raise
            predictions += 1
            failure("pair_completion_failed", "completion", observation["observation_id"])
            return False
        predictions += 1
        emit("pair_completion", {"observation_id": observation["observation_id"], "result": result})
        snapshot = tracker.snapshot()
        emit("issue_state", {"issue_snapshot": snapshot, "state_snapshot": snapshot, "receipt": result["receipt"]})
        return deliver_assessment(observation, snapshot)

    def run_scheduled():
        nonlocal observations
        revision, latest_applied, last_offered, offers = 0, None, None, 0
        last_offered_revision = None
        priorities, requests, held = {}, {}, []
        source_ended = stopped = deadline_logged = False
        ending_reason = finish_deadline = None
        pending_envelope = card = None
        poll_interval = scheduling["poll_interval_s"]
        reading_time = scheduling["notification_min_visible_s"]
        context_cache = issue_context(current_time() if (assessment_two or live_tracking) else 0)[1:] if execution_buffer is not None else None
        receipt_envelopes, receipt_revisions, terminal_receipts = {}, {}, set()
        receipts_closed_at = None
        producer_stop, producer_lock = Event(), Lock()
        producer_state = {"publication": None, "generation": 0, "frames_appended": 0,
                          "seals": 0, "done": False, "error": None}
        producer = None
        seen_generation, context_wait = 0, None
        terminal_context_skipped = False
        ready_cache = None
        pair_executor = ThreadPoolExecutor(max_workers=1) if pair_browser else None
        pair_future = pair_ticket = pair_observation = None

        def submit_pair_transport():
            """Admit and enqueue while the caller holds the session activity guard."""
            ticket = pair_coordinator.admit()
            try:
                # An admission callback can stop a session reentrantly.
                if not runtime.active():
                    pair_coordinator.transport_failed(ticket)
                    return None
                return ticket, pair_executor.submit(completion_provider.perform, ticket)
            except BaseException:
                pair_coordinator.transport_failed(ticket)
                raise

        def producer_view():
            with producer_lock:
                return dict(producer_state)

        def produce():
            # This thread is the sole owner of decoding, sampling and buffer
            # writing. No issue state, provider accounting or trace is mutated.
            local_source = None
            history, pending, last_pts, last_sealed = [], False, None, 0

            def seal(eof=False):
                nonlocal pending, last_sealed
                if not pending and not eof:
                    return
                try:
                    execution_buffer.seal(source_eof=eof)
                except Exception as exc:
                    raise _PreparationFailure("buffer_failure", "buffer_seal") from exc
                publication = {"cutoff_s": last_pts, "samples": tuple(deepcopy(history)),
                    "source_eof": eof, "committed_at_source_s": runtime.source_time(),
                    "committed_at_wall_s": runtime.elapsed_s()}
                with producer_lock:
                    producer_state["publication"] = publication
                    producer_state["generation"] += 1
                    producer_state["seals"] += int(pending)
                pending = False
                last_sealed = last_pts

            try:
                try:
                    local_source = FileFrameSource(execution_path, data_root=data_root)
                except Exception as exc:
                    raise _PreparationFailure("source_failure", "source") from exc
                while not producer_stop.is_set():
                    pacing = runtime.opportunity_state()
                    if not pacing["active"]:
                        break
                    if not pacing["ready"] or not (pacing["playing"] or pacing["ended"]):
                        seal()
                        producer_stop.wait(poll_interval)
                        continue
                    watermark = pacing["source_time_s"]
                    while not producer_stop.is_set():
                        latest = runtime.opportunity_state()
                        if not latest["active"] or not (latest["playing"] or latest["ended"]):
                            break
                        try:
                            frame = local_source.read(watermark)
                        except Exception as exc:
                            raise _PreparationFailure("source_failure", "source") from exc
                        if frame is None or producer_stop.is_set() or not runtime.active():
                            break
                        try:
                            execution_buffer.append(frame, available_through_s=watermark)
                        except Exception as exc:
                            raise _PreparationFailure("buffer_failure", "buffer_append") from exc
                        pending, last_pts = True, frame["elapsed_s"]
                        with producer_lock:
                            producer_state["frames_appended"] += 1
                        try:
                            selected = sampler.consider(frame, watermark)
                            if selected is not None:
                                image = encode_frame_jpeg(selected["frame"],
                                    max_side=config["max_side"], jpeg_quality=config["jpeg_quality"])
                                # Preserve the existing file-replay receipt
                                # convention; actual commit clocks are separate.
                                history.append({"session_id": session_id, "reference_id": reference_id,
                                    "captured_at_s": frame["elapsed_s"], "received_at_s": frame["elapsed_s"],
                                    "image": image, "target_s": selected["target_s"],
                                    "skipped_targets": selected["skipped_targets"]})
                        except Exception as exc:
                            raise _PreparationFailure("sampling_failure", "sampling") from exc
                        if last_pts >= last_sealed + buffer_seal_interval_s:
                            seal()
                    if producer_stop.is_set() or not runtime.active():
                        break
                    if local_source.eof:
                        seal(eof=True)
                        break
                    if pacing["ended"]:
                        seal()
                        break
                    producer_stop.wait(poll_interval)
            except BaseException as exc:
                with producer_lock:
                    producer_state["error"] = exc
            finally:
                if local_source is not None:
                    try:
                        local_source.close()
                    except BaseException:
                        with producer_lock:
                            if producer_state["error"] is None:
                                producer_state["error"] = _PreparationFailure("source_failure", "source_cleanup")
                with producer_lock:
                    producer_state["done"] = True

        def prepare(cutoff, frozen):
            if jpeg_input and "ready_source" in frozen:
                saved = frozen["ready_source"]
                observation = build_observation(reference, list(saved["samples"]),
                    session_id=session_id, observation_id=frozen["observation_id"], cutoff_s=cutoff,
                    max_images=config["max_images"], max_jpeg_bytes=config["max_jpeg_bytes"])
                eof = saved["source_eof"]
            elif execution_buffer is None:
                observation, eof = prepare_observation(cutoff, frozen["observation_id"])
            else:
                try:
                    observation = build_observation(reference, list(frozen["buffer_samples"]),
                        session_id=session_id, observation_id=frozen["observation_id"], cutoff_s=cutoff,
                        max_images=config["max_images"], max_jpeg_bytes=config["max_jpeg_bytes"])
                except Exception as exc:
                    raise _PreparationFailure("observation_failure", "observation") from exc
                eof = frozen["buffer_source_eof"]
            prepared = None
            if observation["status"] == "ready":
                if jpeg_input:
                    rows = {}
                    for role in ("reference", "execution"):
                        rows[role] = [{"frame_index": row["frame_index"], "elapsed_s": row["elapsed_s"],
                            "selection_reason": row["selection_reason"] if role == "reference" else "cadence"}
                            for row in observation["manifest"][role + "_frames"]]
                    saved = frozen.get("ready_source", latest_ready_source)
                    if not saved["endpoint_regular"]:
                        rows["execution"][-1]["selection_reason"] = "current_endpoint"
                    selected = select_compact_frames(rows["reference"], rows["execution"],
                        source_cutoff_s=cutoff, **compact_selection)
                    if not live_tracking:
                        prepared = completion_provider.prepare(observation["model_input"],
                            None if pair_browser else frozen["context"], observation=observation,
                            selected_indices=selected["selected_indices"],
                            **({"stage": "observation"} if pair_browser else {}))
                else:
                    prepared = completion_provider.prepare(observation["model_input"], frozen["context"])
            value = {"observation": observation, "prepared": prepared, "source_eof": eof}
            if jpeg_input and (prepared is not None or live_tracking and observation["status"] == "ready"):
                value.update(ready_source=saved, selection_audit=selected["audit"],
                    **({"selected_indices": selected["selected_indices"]} if pair_browser or live_tracking else {}))
            if (assessment_two or live_tracking) and prepared is not None and observation["cutoff_s"] > 0:
                value["descriptor"] = completion_provider.describe_prepared(prepared)
            return value

        scheduler = CompletionScheduler(max_in_flight=scheduling["max_in_flight"],
            prepare=prepare, perform=completion_provider.perform, clock=runtime.elapsed_s)

        def discard(observation, reason, event="prediction_discarded"):
            emit(event, {"observation_id": observation["observation_id"],
                "observation_cutoff_s": observation["cutoff_s"], "reason": reason})

        def stale(observation, frozen):
            if latest_applied is not None and observation["cutoff_s"] <= latest_applied:
                return "superseded"
            if frozen["revision"] != revision:
                return "stale_issue_context"
            return None

        def poll_pair(*, allow_state):
            nonlocal pair_future, pair_ticket, pair_observation, revision, latest_applied
            nonlocal stopped, ending_reason, predictions
            if pair_future is None or not pair_future.done():
                return
            future, ticket = pair_future, pair_ticket
            pair_future = pair_ticket = None
            try:
                outcome = future.result()
            except BaseException as exc:
                unsent = pair_coordinator.transport_failed(ticket)
                emit("completion_unavailable", {"observation_id": pair_observation["observation_id"],
                    "observation_cutoff_s": pair_observation["cutoff_s"],
                    "provably_unsent": unsent, "code": "transport_exception"})
                if not isinstance(exc, Exception) or callback_errors_fatal:
                    raise
                if not stopped:
                    predictions += 1
                    failure("pair_completion_failed", "completion", pair_observation["observation_id"])
                    ending_reason = "prediction_failed"
                    scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                    stopped = True
                pair_observation = None
                return
            try:
                granted, result = (runtime.run_if_active(lambda: pair_coordinator.complete(
                    ticket, outcome, allow_continue=True, allow_apply=True, is_active=runtime.active))
                    if allow_state and not stopped else (False, None))
                if not granted:
                    # Settle any started request even if Stop won the guard.
                    result = pair_coordinator.complete(ticket, outcome,
                        allow_continue=False, allow_apply=False)
                if result["status"] == "applied":
                    predictions += 1
                    pair_result = result["result"]
                    emit("pair_completion", {"observation_id": pair_observation["observation_id"],
                                             "result": pair_result})
                    snapshot = tracker.snapshot()
                    emit("issue_state", {"issue_snapshot": snapshot, "state_snapshot": snapshot,
                                         "receipt": pair_result["receipt"]})
                    revision, latest_applied = snapshot["revision"], pair_observation["cutoff_s"]
                    pair_observation = None
                elif result["status"] == "discarded":
                    discard(pair_observation, "session_stopped")
                    pair_observation = None
            except Exception:
                if callback_errors_fatal:
                    raise
                if not stopped:
                    predictions += 1
                    failure("pair_completion_failed", "completion", pair_observation["observation_id"])
                    ending_reason = "prediction_failed"
                    scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                    stopped = True
                pair_observation = None

        def consume_result(completed, *, allow_apply):
            nonlocal revision, latest_applied, ending_reason, stopped, context_cache
            request = requests.pop(id(completed.ticket))
            observation, frozen = request["observation"], completed.context
            if completed.error is not None:
                # Submission/worker failure does not prove that HTTP was unsent.
                cancelled = completion_provider.cancel_unsent(completed.ticket)
                emit("completion_unavailable", {"observation_id": observation["observation_id"],
                    "observation_cutoff_s": observation["cutoff_s"],
                    "provably_unsent": cancelled, "code": "transport_exception"})
                if callback_errors_fatal:
                    raise completed.error
                raw = {"error": {"type": "callback_exception"}}
                completion_error = "callback_exception"
            else:
                raw = completion_provider.complete(completed.ticket, completed.value)
                completion_error = None
            emit("completion_raw", {"observation_id": observation["observation_id"], "result": raw})
            if assessment_two or live_tracking:
                normalized = consume_assessment(observation, frozen["prior"], request["dispatched_at_s"],
                    raw, completion_error, frozen=request, apply_state=False)
                if normalized is False:
                    ending_reason = ending_reason or "prediction_failed"
                    if not stopped:
                        scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                        stopped = True
                    discard(observation, "prediction_failed")
                    return
                if not allow_apply or not completed.allow_state or not runtime.active():
                    discard(observation, "session_stopped")
                    return
                obsolete = stale(observation, frozen)
                if obsolete is not None:
                    discard(observation, obsolete)
                    return
                if normalized is recoverable_assessment_skip:
                    return
                if live_tracking and isinstance(normalized, tuple):
                    assessment, binding, _ = normalized
                    try:
                        receipt = tracker.apply_assessment(assessment, binding, context=request["assessment_context"], applied_at_s=current_time())
                    except (ValueError, RuntimeError):
                        failure("state_failure", "state", observation["observation_id"])
                        normalized = False
                    if normalized is not False:
                        current = tracker.issue_snapshot()
                        # The viewer's optional issue list is a read-only view,
                        # separate from the notification delivery ledger.
                        update_view = getattr(runtime, "update_tracked_issues", None)
                        if callable(update_view):
                            update_view(current)
                        emit("issue_state", {"issue_snapshot": current, "state_snapshot": current, "receipt": receipt})
                if not isinstance(normalized, tuple) or (not live_tracking and not apply_assessment(
                        observation, frozen["prior"], *normalized)):
                    ending_reason = "prediction_failed"
                    scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                    stopped = True
                    return
                current = tracker.issue_snapshot() if live_tracking else tracker.snapshot()
                revision, latest_applied = current["revision"], observation["cutoff_s"]
                if execution_buffer is not None:
                    context_cache = issue_context(current_time())[1:]
                return
            response, updates, admissions = normalize_result(observation, frozen["prior"],
                request["dispatched_at_s"], raw, completion_error)
            if response["status"] == "failed":
                ending_reason = ending_reason or "prediction_failed"
                if not stopped:
                    scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                    stopped = True
                discard(observation, "prediction_failed")
                return
            if not allow_apply or not completed.allow_state or not runtime.active():
                discard(observation, "session_stopped")
                return
            obsolete = stale(observation, frozen)
            if obsolete is not None:
                discard(observation, obsolete)
                return
            before = issue_context(current_time())[2]["issues"]
            _, receipt = apply_result(observation, request["dispatched_at_s"],
                response, updates, admissions)
            latest_applied = observation["cutoff_s"]
            _, after_prior, after_context = issue_context(current_time())
            after = after_context["issues"]
            if before != after:
                revision += 1
                if execution_buffer is not None:
                    context_cache = after_prior, after_context
            for created in sorted(receipt["created_issues"], key=lambda row: row["candidate_index"]):
                priorities[created["issue_id"]] = len(priorities)

        def collect_assessment_receipts():
            for attempt_id, envelope in receipt_envelopes.items():
                if attempt_id in terminal_receipts:
                    continue
                while True:
                    previous = receipt_revisions[attempt_id]
                    wrapper = runtime.delivery_outcome(envelope, after_revision=previous)
                    if wrapper is None:
                        break
                    keys(wrapper, ("assessment_contract_version", "attempt_id", "revision", "available_at_s", "outcome"))
                    available = wrapper["available_at_s"]
                    if (type(wrapper["assessment_contract_version"]) is not int or wrapper["assessment_contract_version"] != assessment_contract_version
                            or wrapper["attempt_id"] != attempt_id or type(wrapper["revision"]) is not int
                            or wrapper["revision"] != previous + 1
                            or type(available) not in (int, float) or not math.isfinite(available)
                            or not 0 <= available <= current_time()):
                        raise ValueError("Receiver transition identity, revision or availability disagrees")
                    outcome = wrapper["outcome"]
                    if isinstance(outcome, dict) and outcome.get("status") == "acknowledged":
                        if outcome.get("render_mapping", {}).get("received_at_s") != available:
                            raise ValueError("ACK availability differs from actual HTTP receipt")
                    emit("receiver_outcome", wrapper)
                    attempt = sink.complete(envelope, outcome, at_s=current_time())
                    emit("notification_attempt", attempt)
                    # Persist both records before advancing the correlation cursor.
                    receipt_revisions[attempt_id] = wrapper["revision"]
                    if attempt["status"] in ("acknowledged", "not_sent"):
                        terminal_receipts.add(attempt_id)
                        break

        def close_assessment_receipts():
            nonlocal receipts_closed_at
            if receipts_closed_at is None:
                closure = runtime.close_receipts()
                keys(closure, ("assessment_contract_version", "session_id", "clock_id", "closed_at_s"))
                if (type(closure["assessment_contract_version"]) is not int or closure["assessment_contract_version"] != assessment_contract_version
                        or closure["session_id"] != session_id or closure["clock_id"] != assessment_clock_id):
                    raise ValueError("Receiver closure identity disagrees")
                receipts_closed_at = closure["closed_at_s"]
                if (type(receipts_closed_at) not in (int, float) or not math.isfinite(receipts_closed_at)
                        or not 0 <= receipts_closed_at <= current_time()):
                    raise ValueError("Invalid receiver coverage cutoff")
                collect_assessment_receipts()
                emit("receipt_coverage_closed", {"closed_at_s": receipts_closed_at,
                    "unresolved_attempt_ids": [key for key in receipt_envelopes if key not in terminal_receipts]})

        def service_assessment_notifications():
            nonlocal card
            collect_assessment_receipts()
            current = tracker.issue_snapshot() if live_tracking else tracker.snapshot()
            if card is not None:
                issue = next((row for row in current["issues"] if row["issue_id"] == card["issue_id"]), None)
                attempt = next(row for row in sink.snapshot()["attempts"] if row["attempt_id"] == card["attempt_id"])
                interval = card["binding"].get("playback_mapping", {}).get("host_interval_s")
                retain_unknown = (attempt["status"] == "unknown" and live_tracking
                                  and runtime.snapshot().get("replay_clock_mode") == "host")
                if (stopped or issue is None or issue["state"] in ("resolved", "uncertain")
                        or live_tracking and card["issue_id"] in current["pending_repairs"]
                        or issue["latest_binding"] != card["binding"] and not retain_unknown or interval is None
                        or current_time() - interval[0] > card["max_input_age_s"]
                        or attempt["status"] == "not_sent"
                        or attempt["status"] == "unknown" and not retain_unknown):
                    runtime.retire_notification(card)
                    card = None
                    collect_assessment_receipts()
            if stopped or receipts_closed_at is not None or not runtime.active():
                return False
            delivery = sink.snapshot()
            decision = choose_assessment_notification(current,
                delivered_issue_ids=delivery["delivered_issue_ids"], pending_issue_ids=delivery["pending_issue_ids"],
                at_s=current_time(), clock_id=assessment_clock_id, max_input_age_s=config["max_evidence_age_s"])
            eligible = decision["notification"] is not None
            if not eligible or not runtime.notification_ready(reading_time):
                return eligible
            emit("policy_decision", decision)
            envelope = sink.reserve(decision, tracker.issue_snapshot() if live_tracking else tracker.snapshot(), at_s=current_time())
            if envelope is None:
                return False
            receipt_envelopes[envelope["attempt_id"]] = deepcopy(envelope)
            receipt_revisions[envelope["attempt_id"]] = 0
            emit("notification_attempt", sink.snapshot()["attempts"][-1])
            try:
                outcome = runtime.publish(envelope, min_visible_s=reading_time)
                if outcome is not None:
                    raise ValueError("Browser assessment outcomes must use the correlated receiver interface")
            except BaseException:
                emit("notification_attempt", sink.complete(envelope, {"status": "unknown"}, at_s=current_time()))
                raise
            card = envelope
            collect_assessment_receipts()
            return True

        def service_notifications():
            nonlocal pending_envelope, card
            if assessment_two or live_tracking:
                return service_assessment_notifications()
            current = tracker.snapshot(current_time())
            if card is not None:
                issue = next((row for row in current["issues"]
                              if row["issue_id"] == card["issue_id"]), None)
                if (stopped or issue is None or issue["state"] in ("resolved", "uncertain")
                        or issue["last_observation_cutoff_s"] != current["latest_observation_cutoff_s"]
                        or now_s - max(card["execution_evidence_s"]) > card["max_evidence_age_s"]):
                    runtime.retire_notification(card)
            if pending_envelope is not None:
                outcome = runtime.delivery_outcome(pending_envelope)
                if outcome is not None:
                    attempt = sink.complete(pending_envelope, outcome)
                    emit("notification_attempt", attempt)
                    pending_envelope = None
            if pending_envelope is not None or stopped or not runtime.active():
                return False
            delivery = sink.snapshot()
            decision = choose_notification(current,
                delivered_issue_ids=delivery["delivered_issue_ids"],
                pending_issue_ids=delivery["pending_issue_ids"], at_s=current["at_s"],
                max_evidence_age_s=config["max_evidence_age_s"], issue_priority=priorities)
            eligible = decision["notification"] is not None
            if not eligible or not runtime.notification_ready(reading_time):
                return eligible
            emit("policy_decision", decision)
            notification = decision["notification"]

            def support_expired():
                return (current_time() - max(notification["execution_evidence_s"])
                        > decision["max_evidence_age_s"])

            try:
                envelope = sink.reserve(decision)
            except ValueError:
                if not support_expired() and runtime.active():
                    raise
                emit("notification_suppressed", {"issue_id": notification["issue_id"],
                    "observation_id": notification["observation_id"],
                    "reason": "stale_before_reservation" if runtime.active() else "session_stopped"})
                return False
            emit("notification_attempt", {**deepcopy(envelope), "status": "pending",
                "completed_at_s": None, "delivered_at_s": None, "error_code": None})
            try:
                try:
                    outcome = runtime.publish(envelope, min_visible_s=reading_time)
                except ValueError:
                    if not support_expired():
                        raise
                    # The concrete publication adapter rejects expired support
                    # before publishing. This is a definite unsent reservation.
                    outcome = {"status": "not_sent"}
                    emit("notification_suppressed", {"issue_id": envelope["issue_id"],
                        "observation_id": envelope["observation_id"],
                        "reason": "stale_before_publication"})
            except BaseException:
                attempt = sink.complete(envelope, {"status": "unknown"})
                emit("notification_attempt", attempt)
                raise
            if outcome is None:
                card = envelope
                pending_envelope = envelope
            else:
                emit("notification_attempt", sink.complete(envelope, outcome))
            return outcome is None

        def drain_ready():
            while True:
                ready = scheduler.take_prepared()
                if ready is None:
                    break
                held.append(ready)
            while True:
                ready = scheduler.take_completed()
                if ready is None:
                    break
                consume_result(ready, allow_apply=not stopped)

        def record_deadline():
            nonlocal deadline_logged
            snapshot = scheduler.snapshot()
            pair_unknown = (pair_browser and pair_future is not None and not pair_future.done()
                            and finish_deadline is not None and runtime.elapsed_s() >= finish_deadline)
            unknown = snapshot["unknown_outstanding"] + int(bool(pair_unknown))
            if unknown and not deadline_logged:
                deadline_logged = True
                emit("drain_incomplete", {"unknown_outstanding": unknown,
                    "provider_accounting": completion_provider.snapshot(),
                    "reason": "transport_not_cancelled"})

        def record_terminal_gap(reason):
            if jpeg_input:
                if (ready_cache is not None and (source_ended or runtime.opportunity_state()["ended"])
                        and (latest_applied is None or latest_applied < ready_cache["cutoff_s"])):
                    emit("jpeg_terminal_unassessed", {"committed_cutoff_s": ready_cache["cutoff_s"],
                        "latest_applied_source_cutoff_s": latest_applied, "reason": reason})
                    return True
                return False
            if not assessment_two or execution_buffer is None:
                return False
            final = producer_view()
            publication = final["publication"]
            cutoff = None if publication is None else publication["cutoff_s"]
            if (not final["done"] or cutoff is None or cutoff <= 0
                    or latest_applied is not None and latest_applied >= cutoff):
                return False
            emit("buffer_terminal_unassessed", {"committed_cutoff_s": cutoff,
                "latest_applied_source_cutoff_s": latest_applied, "reason": reason})
            return True

        try:
            if execution_buffer is not None:
                producer = Thread(target=produce, name="guideme-execution-producer")
                producer.start()
            while True:
                opportunity = runtime.opportunity_state()
                current_time()
                buffer_state = producer_view() if execution_buffer is not None else None
                if buffer_state is not None:
                    publication = buffer_state["publication"]
                    if publication is not None and buffer_state["generation"] != seen_generation:
                        emit("execution_buffer_progress", {key: value for key, value in publication.items()
                            if key != "samples"} | {"frames_appended": buffer_state["frames_appended"],
                                "seals": buffer_state["seals"],
                                "coalesced_publications": buffer_state["generation"] - seen_generation - 1})
                        seen_generation = buffer_state["generation"]
                    error = buffer_state["error"]
                    if error is not None and not stopped:
                        if not isinstance(error, Exception):
                            raise error
                        code, stage = ((error.code, error.stage) if isinstance(error, _PreparationFailure)
                                       else ("buffer_failure", "buffer_producer"))
                        failure(code, stage)
                        ending_reason = code
                        producer_stop.set()
                        scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                        stopped = True
                if (not stopped and finish_deadline is not None
                        and runtime.elapsed_s() >= finish_deadline):
                    scheduler.stop(drain_timeout_s=0)
                    stopped = True
                    ending_reason = "drain_timeout"
                if (assessment_two or live_tracking) and finish_deadline is not None and runtime.elapsed_s() >= finish_deadline:
                    close_assessment_receipts()
                if not opportunity["active"] or ending_reason == "prediction_failed":
                    if not stopped:
                        scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                        stopped = True
                        ending_reason = ending_reason or "session_stopped"
                if (assessment_two or live_tracking) and stopped and finish_deadline is None:
                    finish_deadline = runtime.elapsed_s() + scheduling["drain_timeout_s"]
                if stopped:
                    producer_stop.set()
                if not stopped:
                    scheduler.set_playing(opportunity["playing"] or opportunity["ended"])
                scheduler.poll()
                drain_ready()
                if pair_browser:
                    poll_pair(allow_state=not stopped)
                    if (not stopped and pair_future is None and pair_observation is not None
                            and pair_coordinator.stage == "association" and runtime.active()):
                        try:
                            granted, submitted = runtime.run_if_active(submit_pair_transport)
                            if not granted or submitted is None:
                                if pair_coordinator.stage != "idle":
                                    pair_coordinator.abort()
                                discard(pair_observation, "session_stopped")
                                pair_observation = None
                            else:
                                pair_ticket, pair_future = submitted
                        except Exception as exc:
                            if pair_ticket is not None:
                                pair_coordinator.transport_failed(pair_ticket)
                                pair_ticket = None
                            else:
                                pair_coordinator.abort()
                            expected_refusal = (isinstance(exc, ProviderAdmissionRefused)
                                and exc.reason in ("budget_reservation_unavailable", "generation_call_limit_reached"))
                            if callback_errors_fatal and not expected_refusal:
                                raise
                            if not expected_refusal:
                                failure("pair_admission_failed", "completion", pair_observation["observation_id"])
                            ending_reason = exc.reason if expected_refusal else "prediction_failed"
                            scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                            stopped = True
                            pair_observation = None
                for ready in held[:]:
                    if ready.error is not None:
                        scheduler.discard_prepared(ready)
                        held.remove(ready)
                        if stopped:
                            continue
                        if isinstance(ready.error, _PreparationFailure):
                            failure(ready.error.code, ready.error.stage)
                            ending_reason = ready.error.code
                        elif callback_errors_fatal:
                            raise ready.error
                        else:
                            failure("observation_failure", "preparation")
                            ending_reason = "observation_failure"
                        scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                        stopped = True
                        continue
                    value, frozen = ready.value, ready.context
                    observation = value["observation"]
                    if jpeg_input and "ready_source" in value:
                        ready_cache = value["ready_source"]
                    if live_tracking and observation["status"] == "ready" and value["prepared"] is None:
                        if stale(observation, frozen) is not None:
                            scheduler.discard_prepared(ready)
                            held.remove(ready)
                            discard(observation, "stale_issue_context", "observation_discarded")
                            continue
                        frozen["context"] = tracker.issue_context(observation_id=observation["observation_id"],
                            manifest_hash=observation["manifest_sha256"], source_cutoff_s=observation["cutoff_s"],
                            max_resolved_history=32, max_context_bytes=config["issues"]["max_metadata_bytes"])
                        value["prepared"] = completion_provider.prepare(observation["model_input"], frozen["context"],
                            observation=observation, selected_indices=value["selected_indices"])
                        value["descriptor"] = completion_provider.describe_prepared(value["prepared"])
                    if not frozen.get("traced"):
                        observations += 1
                        emit("observation", {"observation": observation,
                            "prior_issues": frozen["prior"], "issue_context": frozen["context"],
                            **({"compact_selection": value["selection_audit"]} if jpeg_input and "selection_audit" in value else {}),
                            **({"context_frozen_at_s": frozen["context_frozen_at_s"]}
                               if execution_buffer is not None else {})})
                        # The owner-side handle context is not used by its worker.
                        frozen["traced"] = True
                        if execution_buffer is None:
                            source_ended = source_ended or value["source_eof"]
                    obsolete = "session_stopped" if stopped else stale(observation, frozen)
                    if obsolete is not None or observation["status"] != "ready":
                        scheduler.discard_prepared(ready)
                        held.remove(ready)
                        discard(observation, obsolete or "insufficient_observation", "observation_discarded")
                        continue
                    if pair_browser and pair_observation is not None:
                        # A pair in progress blocks admission even while paused.
                        # Keep only the latest unsent opportunity in that case.
                        frozen["admission_blocked"] = True
                    if (frozen.get("admission_blocked") and last_offered is not None
                            and observation["cutoff_s"] < last_offered):
                        # Only replace work delayed by admission backpressure.
                        # Replacing every slow preparation would starve dispatch
                        # whenever preparation takes longer than the source gate.
                        scheduler.discard_prepared(ready)
                        held.remove(ready)
                        discard(observation, "superseded_unsent", "observation_discarded")
                        continue
                    latest = runtime.opportunity_state()
                    if not latest["active"] or not (latest["playing"] or latest["ended"]):
                        continue
                    if pair_browser and pair_observation is not None:
                        continue
                    if (assessment_two or live_tracking) and "mapping_frozen_at_s" not in frozen:
                        descriptor = value["descriptor"]
                        keys(descriptor, ("assessment_contract_version", "observation_id", "request_sha256", provenance_key))
                        if descriptor["assessment_contract_version"] != assessment_contract_version or descriptor["observation_id"] != observation["observation_id"]:
                            raise ValueError("Prepared descriptor identity disagrees")
                        freeze_time = current_time()
                        tail = descriptor[provenance_key]["execution"][provenance_frames][-1]
                        frozen.update(descriptor=deepcopy(descriptor), mapping_frozen_at_s=freeze_time,
                            playback_mapping=deepcopy(runtime.source_passage_interval(tail["elapsed_s"], at_s=freeze_time)))
                    if live_tracking and frozen["playback_mapping"]["status"] != "mapped":
                        # Wait for additional browser samples without reserving or
                        # dispatching; a later poll recomputes the passage interval.
                        frozen.pop("mapping_frozen_at_s")
                        frozen.pop("playback_mapping")
                        continue
                    if pair_browser:
                        clock_context = {"session_id": session_id, "observation_id": observation["observation_id"],
                            "clock_id": assessment_clock_id, "timing_basis": assessment_timing_basis,
                            "playback_mapping": deepcopy(frozen["playback_mapping"])}
                        try:
                            pair_coordinator.begin(observation, value["selected_indices"], clock_context=clock_context,
                                prepared=value["prepared"], expected_revision=frozen["revision"])
                            granted, submitted = runtime.run_if_active(submit_pair_transport)
                            if not granted or submitted is None:
                                if pair_coordinator.stage != "idle":
                                    pair_coordinator.abort()
                                discard(observation, "session_stopped")
                            else:
                                pair_ticket, pair_future = submitted
                        except Exception as exc:
                            if pair_ticket is not None:
                                pair_coordinator.transport_failed(pair_ticket)
                                pair_ticket = None
                            elif pair_coordinator.stage != "idle":
                                pair_coordinator.abort()
                            expected_refusal = (isinstance(exc, ProviderAdmissionRefused)
                                and exc.reason in ("budget_reservation_unavailable", "generation_call_limit_reached"))
                            if callback_errors_fatal and not expected_refusal:
                                raise
                            if not expected_refusal:
                                failure("pair_admission_failed", "completion", observation["observation_id"])
                            ending_reason = exc.reason if expected_refusal else "prediction_failed"
                            scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                            stopped = True
                        else:
                            if pair_ticket is not None:
                                pair_observation = observation
                        scheduler.discard_prepared(ready)
                        held.remove(ready)
                        continue
                    try:
                        if live_tracking:
                            tail = frozen["descriptor"][provenance_key]["execution"][provenance_frames][-1]
                            admission_time = current_time()
                            witness = {"session_id":session_id, "observation_id":observation["observation_id"],
                                "reference_id":reference_id, "execution_source_sha256":execution_source_sha256,
                                "clock_id":assessment_clock_id, "timing_basis":assessment_timing_basis,
                                "request_sha256":frozen["descriptor"]["request_sha256"], "tail_frame_index":tail["frame_index"],
                                "tail_source_elapsed_s":tail["elapsed_s"], "mapping_frozen_at_s":frozen["mapping_frozen_at_s"],
                                "admitted_at_s":admission_time, "playback_mapping":deepcopy(frozen["playback_mapping"])}
                            ticket = completion_provider.admit(value["prepared"], browser_playback_witness=witness)
                        else:
                            ticket = completion_provider.admit(value["prepared"])
                    except RuntimeError as exc:
                        if completion_provider.snapshot().get("stopped", False):
                            raise
                        expected_refusal = (isinstance(exc, ProviderAdmissionRefused)
                            and exc.reason in ("budget_reservation_unavailable", "generation_call_limit_reached"))
                        if not expected_refusal and callback_errors_fatal:
                            raise
                        if requests:
                            frozen["admission_blocked"] = True
                            continue  # Settling an outstanding hold may restore capacity.
                        ending_reason = exc.reason if expected_refusal else "provider_admission_refused"
                        scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                        stopped = True
                        break
                    try:
                        dispatched_at = current_time()
                        if assessment_two or live_tracking:
                            frozen["admitted_at_s"] = admission_time if live_tracking else dispatched_at
                            emit("assessment_dispatch", {"observation_id": observation["observation_id"],
                                "observation_cutoff_s": observation["cutoff_s"], "descriptor": frozen["descriptor"],
                                "playback_mapping": frozen["playback_mapping"],
                                "mapping_frozen_at_s": frozen["mapping_frozen_at_s"], "admitted_at_s": frozen["admitted_at_s"],
                                "context_available_at_s": frozen["prior"]["context_available_at_s"]})
                        if execution_buffer is not None:
                            emit("buffer_dispatch", {"observation_id": observation["observation_id"],
                                "observation_cutoff_s": observation["cutoff_s"],
                                "context_frozen_at_s": frozen["context_frozen_at_s"],
                                "context_available_at_s": (frozen["prior"]["context_available_at_s"] if assessment_two
                                                           else frozen["context"]["available_at_s"]),
                                "dispatched_at_s": dispatched_at})
                        requests[id(ticket)] = {"observation": observation,
                                                 "dispatched_at_s": dispatched_at,
                            **({"assessment_context": deepcopy(frozen["context"])} if live_tracking else {}),
                            **({key: deepcopy(frozen[key]) for key in ("descriptor", "playback_mapping",
                                "mapping_frozen_at_s", "admitted_at_s")} if (assessment_two or live_tracking) else {})}
                        scheduler.dispatch(ready, ticket)
                    except BaseException:
                        try:
                            completion_provider.cancel_unsent(ticket)
                        finally:
                            requests.pop(id(ticket), None)
                        raise
                    held.remove(ready)
                if stopped:
                    for ready in held:
                        scheduler.discard_prepared(ready)
                    held.clear()
                    if pair_browser and pair_future is None and pair_observation is not None:
                        pair_coordinator.abort()
                        discard(pair_observation, "session_stopped")
                        pair_observation = None
                    record_deadline()
                eligible = service_notifications()
                opportunity = runtime.opportunity_state()
                if (not stopped and (not source_ended or (assessment_two or live_tracking) and (execution_buffer is not None or jpeg_input))
                        and ending_reason is None and offers < config["max_observations"]):
                    frozen = None
                    refresh_context = False
                    if execution_buffer is None:
                        cutoff = runtime.poll_cutoff(last_offered, interval)
                        if jpeg_input and ready_cache is not None and (source_ended or opportunity["ended"]):
                            refresh_context = (ready_cache["cutoff_s"] == last_offered
                                and last_offered_revision is not None and revision > last_offered_revision
                                and (latest_applied is None or ready_cache["cutoff_s"] > latest_applied))
                            if refresh_context:
                                cutoff = ready_cache["cutoff_s"]
                        if cutoff is not None:
                            if not (assessment_two or live_tracking):
                                cutoff = max(cutoff, current_time())
                            _, prior, context = issue_context(current_time() if (assessment_two or live_tracking) else cutoff)
                            frozen = {"prior": prior, "context": context}
                            if jpeg_input and ready_cache is not None and cutoff == ready_cache["cutoff_s"]:
                                frozen["ready_source"] = ready_cache
                            if (assessment_two or live_tracking) and any(value > cutoff for value in prior["source_cutoffs_s"]):
                                cutoff = None
                    else:
                        buffer_state = producer_view()
                        publication = buffer_state["publication"]
                        cutoff = None if publication is None else publication["cutoff_s"]
                        refresh_context = (assessment_two and cutoff is not None and cutoff == last_offered
                            and last_offered_revision is not None and revision > last_offered_revision
                            and (latest_applied is None or cutoff > latest_applied))
                        if (cutoff is None or cutoff <= 0
                                or not (opportunity["playing"] or opportunity["ended"])
                                or (last_offered is not None and cutoff <= last_offered and not refresh_context)
                                or (not refresh_context and not buffer_state["done"] and cutoff < (last_offered or 0) + interval
                                    and not math.isclose(cutoff, (last_offered or 0) + interval,
                                                         rel_tol=1e-12, abs_tol=1e-9))):
                            cutoff = None
                        if cutoff is not None:
                            prior, context = context_cache
                            available = prior["context_available_at_s"] if (assessment_two or live_tracking) else context["available_at_s"]
                            unavailable = ((available is not None and available > current_time())
                                or any(value > cutoff for value in prior["source_cutoffs_s"])) if (assessment_two or live_tracking) else (
                                    available > cutoff or any(row["last_observation_cutoff_s"] > cutoff for row in prior["issues"]))
                            if unavailable:
                                waiting = available, cutoff
                                if context_wait != waiting:
                                    emit("buffer_context_wait", {"committed_cutoff_s": cutoff,
                                        "context_available_at_s": available, "revision": revision})
                                    context_wait = waiting
                                if buffer_state["done"] and not terminal_context_skipped:
                                    terminal_context_skipped = True
                                    emit("buffer_terminal_unassessed", {"committed_cutoff_s": cutoff,
                                        "context_available_at_s": available, "revision": revision})
                                cutoff = None
                            else:
                                context_wait = None
                                frozen = {"prior": prior, "context": context,
                                    "buffer_samples": publication["samples"],
                                    "buffer_source_eof": publication["source_eof"],
                                    "context_frozen_at_s": current_time()}
                    if cutoff is not None:
                        if last_offered is not None and cutoff <= last_offered and not refresh_context:
                            raise ValueError("Runtime cutoffs must strictly advance")
                        coalesced = scheduler.snapshot()["queued_cutoff_s"]
                        if coalesced is not None:
                            emit("observation_coalesced", {"skipped_cutoff_s": coalesced,
                                "replacement_cutoff_s": cutoff, "reason": "latest_only_capacity"})
                        scheduler.offer(cutoff, {**frozen, "revision": revision,
                                                "observation_id": str(uuid4())},
                            **({"context_revision": revision} if (assessment_two or live_tracking) and (execution_buffer is not None or jpeg_input) else {}))
                        last_offered, last_offered_revision, offers = cutoff, revision, offers + 1
                snapshot = scheduler.snapshot()
                if execution_buffer is None:
                    exhausted = source_ended or offers >= config["max_observations"] or (
                        opportunity["ended"] and last_offered is not None
                        and last_offered >= opportunity["source_time_s"])
                else:
                    buffer_state = producer_view()
                    # Closing the source can fail after this iteration's first
                    # producer read. Handle that error before claiming a clean
                    # horizon/EOF termination from a newer done publication.
                    if buffer_state["error"] is not None and not stopped:
                        continue
                    publication = buffer_state["publication"]
                    final_cutoff = None if publication is None else publication["cutoff_s"]
                    final_offered = (final_cutoff is None or final_cutoff <= 0
                        or (last_offered is not None and last_offered >= final_cutoff)
                        or terminal_context_skipped)
                    source_ended = (buffer_state["done"] and final_offered and publication is not None
                                    and publication["source_eof"] and not terminal_context_skipped)
                    exhausted = offers >= config["max_observations"] or buffer_state["done"] and final_offered
                    if offers >= config["max_observations"]:
                        producer_stop.set()
                if exhausted and finish_deadline is None:
                    finish_deadline = runtime.elapsed_s() + scheduling["drain_timeout_s"]
                if (not snapshot["outstanding"] and snapshot["queued_cutoff_s"] is None
                        and (not pair_browser or pair_observation is None)):
                    waiting_receipts = (assessment_two or live_tracking) and receipts_closed_at is None and any(
                        key not in terminal_receipts for key in receipt_envelopes)
                    if stopped:
                        if waiting_receipts and runtime.elapsed_s() < finish_deadline:
                            time.sleep(poll_interval)
                            continue
                        if assessment_two or live_tracking:
                            close_assessment_receipts()
                        record_terminal_gap(ending_reason)
                        return "incomplete", ending_reason
                    if exhausted and pending_envelope is None and not eligible:
                        if waiting_receipts and runtime.elapsed_s() < finish_deadline:
                            time.sleep(poll_interval)
                            continue
                        if (assessment_two or live_tracking) and card is not None and not runtime.notification_ready(reading_time):
                            # Normal EOF must preserve the acknowledged card's
                            # reading interval. Explicit Stop/deadline takes the
                            # separate stopped branch above.
                            time.sleep(poll_interval)
                            continue
                        if assessment_two or live_tracking:
                            close_assessment_receipts()
                        terminal_reason = ("max_observations" if offers >= config["max_observations"]
                            else "terminal_context_unavailable" if terminal_context_skipped or source_ended
                            else "session_horizon")
                        if record_terminal_gap(terminal_reason):
                            return "incomplete", terminal_reason
                        if terminal_context_skipped:
                            return "incomplete", "terminal_context_unavailable"
                        if source_ended:
                            emit("source_exhausted", {"cutoff_s": last_offered})
                            return "complete", "source_exhausted"
                        return "incomplete", ("max_observations" if offers >= config["max_observations"]
                                               else "session_horizon")
                time.sleep(poll_interval)
        finally:
            # The command may close its provider client/trace only after every
            # worker is finished. Deadline expiry records unknown holds but is
            # not proof of HTTP cancellation and cannot justify closing early.
            interrupted = sys.exc_info()[0] is not None
            cleanup_error = None
            producer_stop.set()
            try:
                scheduler.stop(drain_timeout_s=scheduling["drain_timeout_s"])
                if pair_browser:
                    if pair_future is not None:
                        try:
                            pair_future.result()
                        except BaseException:
                            pass  # poll_pair performs conservative ticket accounting.
                        try:
                            poll_pair(allow_state=False)
                        except BaseException as exc:
                            cleanup_error = cleanup_error or exc
                    if pair_observation is not None and pair_coordinator.stage != "idle":
                        pair_coordinator.abort()
                        pair_observation = None
                    pair_executor.shutdown(wait=True)
                while scheduler.snapshot()["outstanding"]:
                    scheduler.poll()
                    while True:
                        ready = scheduler.take_prepared()
                        if ready is None:
                            break
                        held.append(ready)
                    for ready in held:
                        scheduler.discard_prepared(ready)
                    held.clear()
                    while True:
                        ready = scheduler.take_completed()
                        if ready is None:
                            break
                        try:
                            consume_result(ready, allow_apply=False)
                        except BaseException as exc:
                            cleanup_error = cleanup_error or exc
                    if assessment_two or live_tracking:
                        try:
                            collect_assessment_receipts()
                            if finish_deadline is not None and runtime.elapsed_s() >= finish_deadline:
                                close_assessment_receipts()
                        except BaseException as exc:
                            cleanup_error = cleanup_error or exc
                    try:
                        record_deadline()
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
                    if scheduler.snapshot()["outstanding"]:
                        time.sleep(poll_interval)
                scheduler.close()
            finally:
                if producer is not None and producer.ident is not None:
                    producer.join()
                if execution_buffer is not None:
                    final = producer_view()
                    publication = final["publication"]
                    error = final["error"]
                    buffering_report.update(enabled=True, seal_interval_s=buffer_seal_interval_s,
                        frames_appended=final["frames_appended"], seals=final["seals"],
                        committed_cutoff_s=None if publication is None else publication["cutoff_s"],
                        source_eof=False if publication is None else publication["source_eof"],
                        producer_done=final["done"], producer_joined=producer is not None and not producer.is_alive(),
                        error_code=None if error is None else (error.code if isinstance(error, _PreparationFailure)
                                                             else "buffer_failure"))
            scheduling_report.update(scheduler.snapshot(), decision_opportunities=offers,
                                     drain_deadline_reported=deadline_logged)
            if assessment_two or live_tracking:
                try:
                    close_assessment_receipts()
                    emit("timing_provenance", runtime.timing_snapshot())
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None and not interrupted:
                raise cleanup_error

    emit("run_started", {"mode": "content" if runtime is None else "browser",
                         "delivery_mode": ("display_ack" if live_tracking else "simulated" if simulated_assessment_delivery else "not_connected") if assessment_three else "simulated" if runtime is None else "display_ack",
                         **({"assessment_contract_version": 2, "timing_basis": assessment_timing_basis,
                             "repair_confirmation_count": repair_confirmation_count,
                             "clock_id": assessment_clock_id, "execution_source_sha256": execution_source_sha256}
                            if assessment_two else {"timing_basis": assessment_timing_basis} if simulated_assessment_delivery else {}),
                         "config": config, "manifest": manifest,
                         **({"input_representation": input_representation,
                             "compact_selection": compact_selection,
                             "endpoint_encoding": "FileFrameSource RGB through encode_frame_jpeg; "
                                 "byte identity with historical OpenCV-seek diagnostic endpoints is not assumed"}
                            if jpeg_input else {}),
                         **({"two_stage_pair": True} if pair_offline or pair_browser else {}),
                         "callback_errors_fatal": callback_errors_fatal,
                         **({"scheduling": deepcopy(scheduling)} if asynchronous else {}),
                         **({"execution_buffer": {"seal_interval_s": buffer_seal_interval_s}}
                            if execution_buffer is not None else {})})
    try:
        if execution_buffer is None:
            try:
                source = FileFrameSource(execution_path, data_root=data_root)
            except Exception:
                reason = "source_failure"
        if execution_buffer is not None:
            status, reason = run_scheduled()
        elif source is None:
            failure(reason, "source")
        elif asynchronous:
            status, reason = run_scheduled()
        else:
            previous_cutoff = None
            for opportunity in range(config["max_observations"]):
                try:
                    if runtime is None:
                        cutoff = opportunity * interval
                    else:
                        cutoff = runtime.next_cutoff(previous_cutoff, interval)
                        if cutoff is None or not runtime.active():
                            reason = "session_horizon" if runtime.active() else "session_stopped"
                            current_time()
                            break
                        prior_time = now_s
                        if (type(cutoff) not in (int, float) or cutoff < 0
                                or cutoff > current_time()
                                or (previous_cutoff is not None and cutoff <= previous_cutoff)
                                or cutoff < prior_time):
                            raise ValueError("Runtime cutoffs must advance within the available source time")
                    if (not math.isfinite(cutoff)
                            or (runtime is None and opportunity and cutoff <= now_s)):
                        raise ValueError("Cutoff grid is not representable")
                except (OverflowError, ValueError):
                    reason = "cutoff_failure"
                    failure(reason, "cutoff")
                    break
                previous_cutoff = cutoff
                if runtime is None:
                    now_s = cutoff
                observations += 1
                try:
                    observation, _ = prepare_observation(cutoff, str(uuid4()))
                    _, prior, context = issue_context(cutoff)
                except _PreparationFailure as exc:
                    reason = exc.code
                    failure(reason, exc.stage)
                    break
                except Exception:
                    reason = "observation_failure"
                    failure(reason, "observation")
                    break
                emit("observation", {"observation": observation, "prior_issues": prior,
                                     "issue_context": context})
                if observation["status"] == "ready":
                    if (assessment_two or assessment_three) and cutoff == 0:
                        emit("assessment_skipped", {"observation_id": observation["observation_id"],
                            "reason": "waiting_for_positive_duration_execution"})
                        if source.eof:
                            emit("source_exhausted", {"cutoff_s": cutoff})
                            status, reason = "complete", "source_exhausted"
                            break
                        continue
                    if assessment_three:
                        if not execute_offline_assessment(observation, cutoff):
                            reason = "prediction_failed"
                            break
                        if source.eof:
                            emit("source_exhausted", {"cutoff_s": cutoff})
                            status, reason = "complete", "source_exhausted"
                            break
                        continue
                    if pair_offline:
                        if not execute_offline_pair(observation, cutoff):
                            reason = "prediction_failed"
                            break
                        if source.eof:
                            emit("source_exhausted", {"cutoff_s": cutoff})
                            status, reason = "complete", "source_exhausted"
                            break
                        continue
                    if runtime is not None and not runtime.active():
                        reason = "session_stopped"
                        break
                    dispatched_at = current_time()
                    completion_error = None
                    try:
                        raw = completion_plan(deepcopy(observation["model_input"]), deepcopy(context))
                    except Exception:
                        if callback_errors_fatal:
                            raise
                        completion_error = "callback_exception"
                        raw = {"error": {"type": completion_error}}
                    emit("completion_raw", {"observation_id": observation["observation_id"],
                                            "result": raw})
                    if runtime is not None and not runtime.active():
                        reason = "session_stopped"
                        emit("prediction_discarded", {"observation_id": observation["observation_id"],
                            "reason": "session_stopped"})
                        break
                    if assessment_two:
                        if not consume_assessment(observation, prior, dispatched_at, raw, completion_error):
                            reason = "prediction_failed"
                            break
                    else:
                        response, updates, admissions = normalize_result(
                            observation, prior, dispatched_at, raw, completion_error)
                        try:
                            issue_snapshot, _ = apply_result(observation, dispatched_at,
                                response, updates, admissions)
                        except Exception:
                            reason = "state_failure"
                            failure(reason, "state", observation["observation_id"])
                            break
                        delivery = sink.snapshot()
                        try:
                            decision = choose_notification(issue_snapshot,
                                delivered_issue_ids=delivery["delivered_issue_ids"],
                                pending_issue_ids=delivery["pending_issue_ids"], at_s=now_s,
                                max_evidence_age_s=config["max_evidence_age_s"])
                        except Exception:
                            reason = "notification_failure"
                            failure(reason, "policy", observation["observation_id"])
                            break
                        emit("policy_decision", decision)
                        try:
                            attempt = sink.deliver(decision)
                        except Exception:
                            if send_trace_error is not None:
                                raise send_trace_error
                            reason = "notification_failure"
                            failure(reason, "notification", observation["observation_id"])
                            break
                        if send_trace_error is not None:
                            raise send_trace_error
                        if attempt is not None:
                            emit("notification_attempt", attempt)
                            if attempt["status"] != "acknowledged":
                                failure("delivery_" + attempt["status"], "notification",
                                        observation["observation_id"])
                        if runtime is not None and response["status"] == "failed":
                            reason = "prediction_failed"
                            break
                # Source EOF is inspected after the causal checkpoint; it is
                # never added to the predictor input or interpreted as success.
                if source.eof:
                    emit("source_exhausted", {"cutoff_s": cutoff})
                    status, reason = "complete", "source_exhausted"
                    break
    finally:
        if source is not None:
            if sys.exc_info()[0] is not None:
                try:
                    source.close()
                except BaseException:
                    pass  # Cleanup must not replace the original interruption.
            else:
                try:
                    source.close()
                except Exception:
                    status, reason = "incomplete", "source_failure"
                    failure(reason, "source_cleanup")
    current_time()
    delivery = sink.snapshot()
    report = {"schema_version": 1, "run_id": run_id, "session_id": session_id,
        "reference_id": reference_id, "status": status, "reason": reason,
        "observations": observations, "predictions": predictions, "failures": failures,
        "notifications": sum(row["status"] == "acknowledged" for row in delivery["attempts"]),
        "issue_snapshot": tracker.issue_snapshot() if assessment_three else tracker.snapshot() if assessment_two else tracker.snapshot(now_s), "delivery_snapshot": delivery,
        "state_snapshot": tracker.issue_snapshot() if assessment_three else tracker.snapshot() if assessment_two else state.snapshot(now_s)}
    if assessment_three:
        report.update(assessment_contract_version=3, delivery_mode="display_ack" if live_tracking else "simulated" if simulated_assessment_delivery else "not_connected",
            timing_basis=assessment_timing_basis, clock_id=assessment_clock_id,
            execution_source_sha256=execution_source_sha256,
            limitations=(["Offline source-logical timing and simulated delivery; no measured live latency."]
                if simulated_assessment_delivery else ["Offline source-logical timing; no notification delivery or live timing."]))
    if live_tracking:
        report.update(evaluation_projection_status="unsupported_live_interval_timing", wall_elapsed_s=runtime.elapsed_s(),
            limitations=["Live playback timing uses passage/render intervals, not physical capture or display proof."])
    if assessment_two:
        report["skipped_assessments"] = skipped_assessments
        report.update(assessment_contract_version=2, delivery_mode="display_ack" if runtime is not None else "simulated",
            timing_basis=assessment_timing_basis,
            clock_id=assessment_clock_id, execution_source_sha256=execution_source_sha256,
            limitations=(["Live timing retains causal passage/render intervals, not point capture or physical-display proof.",
                "Scalar live interval evaluation is unsupported; unknown receipts remain explicit after bounded coverage."]
                if runtime is not None else ["Offline logical content timing is simulated, not measured live delivery latency.",
                "Unknown send outcomes remain pending; this synchronous adapter has no receiver polling or automatic resend."]))
        if pair_offline or pair_browser:
            report["pending_observations"] = tracker.pending_observations()
            emit("pending_observations_unresolved", {"pending_observations": report["pending_observations"]})
        if runtime is not None:
            report.update(evaluation_projection_status="unsupported_live_interval_timing", wall_elapsed_s=runtime.elapsed_s())
    if runtime is not None and not (assessment_two or live_tracking):
        report.update(delivery_mode="display_ack", timing_basis="browser_source_playhead",
                      wall_elapsed_s=runtime.elapsed_s())
    if asynchronous:
        report.update(scheduling=deepcopy(scheduling), scheduler_snapshot=scheduling_report)
    if execution_buffer is not None:
        report["execution_buffer"] = buffering_report
    emit("run_finished", report)
    return report
