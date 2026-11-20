"""Run fixed-grid content replay with an explicit mock, Gemini or native-video backend.

Cutoffs start at zero. Mock predictions are fictional insufficient-evidence
responses; content timestamps are simulated and never establish live latency.
"""

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.guideme.content_replay import run_content_replay
from src.guideme.reference import prepare_reference
from src.guideme.replay_config import load_replay_config
from src.guideme.run_manifest import freeze_run_manifest
from src.guideme.run_trace import RunTraceWriter
from src.guideme.run_trace_reader import read_run_trace


def main(argv=None, *, runtime=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON settings file, relative to the working directory")
    parser.add_argument("--backend", required=True, choices=("mock", "gemini", "native-video"))
    parser.add_argument("--output-dir", required=True, help="New output directory, relative to the data root or absolute")
    parser.add_argument("--data-root", help="Absolute media/output root; overrides GUIDEME_DATA_ROOT")
    parser.add_argument("--assessment-contract-version", type=int, choices=(1, 2, 3), default=1,
                        help="Assessment contract; version 2 supports native-video replay")
    parser.add_argument("--two-stage", action="store_true",
                        help="Use two-stage assessment for scheduled JPEG browser monitoring")
    args = parser.parse_args(argv)
    live_assessment = args.assessment_contract_version in (2, 3) and runtime is not None
    if args.assessment_contract_version in (2, 3) and args.backend != "native-video":
        print("Assessment contracts 2 and 3 require the native-video backend.", file=sys.stderr)
        return 1
    if args.assessment_contract_version == 3 and runtime is None:
        print("Contract 3 command requires the scheduled browser runtime.", file=sys.stderr)
        return 1
    if args.two_stage and (args.assessment_contract_version != 2 or not live_assessment or args.backend != "native-video"):
        print("Two-stage monitoring requires the contract-2 native-video browser runtime.", file=sys.stderr)
        return 1

    replay_config = load_replay_config(args, runtime=runtime, repo=REPO, parser=parser)
    if replay_config is None:
        return 1
    root = replay_config.root
    output = replay_config.output
    settings_path = replay_config.settings_path
    settings = replay_config.settings
    configured_artifacts = replay_config.configured_artifacts
    reference_path = replay_config.reference_path
    execution_path = replay_config.execution_path
    reference_config = replay_config.reference_config
    replay = replay_config.replay
    repair_confirmation_count = replay_config.repair_confirmation_count
    scheduling = replay_config.scheduling
    gemini = replay_config.gemini
    native = replay_config.native
    buffer_config = replay_config.buffer_config
    input_representation = replay_config.input_representation
    compact_selection = replay_config.compact_selection
    image_limits = replay_config.image_limits

    run_id = str(uuid4())
    session_id = str(uuid4()) if runtime is None else runtime.session_id
    delivery_mode = "simulated" if runtime is None else "display_ack"
    timing_basis = "ideal_content_cutoff" if runtime is None else ("session_monotonic" if live_assessment else "browser_source_playhead")
    live_settings = {}
    if live_assessment:
        viewer_path = REPO / "src/guideme/monitor_viewer.html"
        actual_viewer_sha256 = hashlib.sha256(viewer_path.read_bytes()).hexdigest()
        if getattr(runtime, "served_viewer_sha256", None) != actual_viewer_sha256:
            print("Served viewer bytes differ from this replay's viewer source.", file=sys.stderr)
            return 1
        live_settings = {"timing_limits": {"max_samples": 16384, "max_markers": 32768,
                "max_metadata_bytes": 16 * 1024 * 1024}, "receipt_limits": deepcopy(replay["sink"]),
            "receipt_drain_timeout_s": scheduling["drain_timeout_s"], "served_viewer_sha256": actual_viewer_sha256}
    artifact_limits = {"max_manifest_files": 128, "max_manifest_bytes": 512_000_000,
                       "max_trace_events": 10000, "max_trace_bytes": configured_artifacts["max_trace_bytes"]}
    files = {"reference_media": str(reference_path), "execution_media": str(execution_path),
             "settings": str(settings_path), "command": str(Path(__file__).resolve()),
             "lock": str(REPO / "uv.lock"), "pyproject": str(REPO / "pyproject.toml")}
    files.update({"code:" + path.name: str(path.resolve()) for path in (REPO / "src/guideme").glob("*.py")})
    if live_assessment:
        files.update(served_viewer=str(viewer_path), monitor_entrypoint=str(REPO / "scripts/monitor_videos.py"))
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=REPO)
    try:
        git_state = {"branch": git("branch", "--show-current").decode().strip(),
            "commit": git("rev-parse", "HEAD").decode().strip(), "status": git("status", "--porcelain").decode(),
            "tracked_worktree_diff_sha256": hashlib.sha256(git("diff", "--binary")).hexdigest(),
            "index_diff_sha256": hashlib.sha256(git("diff", "--cached", "--binary")).hexdigest()}
        manifest = freeze_run_manifest({"settings": settings, "backend": args.backend,
            **({"assessment_contract_version": args.assessment_contract_version, "repair_confirmation_count": repair_confirmation_count}
               if args.assessment_contract_version in (2, 3) else {}),
            **({"two_stage": True} if args.two_stage else {}),
            **live_settings,
            "run_id": run_id, "session_id": session_id, "data_root": str(root),
            "reference_path": str(reference_path), "execution_path": str(execution_path),
            "output_dir": str(output), "artifact_limits": artifact_limits,
            "delivery_mode": delivery_mode, "timing_basis": timing_basis},
            files, git_state=git_state, provenance={"gold_loaded": False},
            max_files=artifact_limits["max_manifest_files"], max_total_bytes=artifact_limits["max_manifest_bytes"])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error("Cannot freeze replay inputs: " + type(exc).__name__)

    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def verify_media(role, path):
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != manifest["files"][role]["sha256"]:
            raise RuntimeError("Media changed after the manifest was frozen")

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Output directory already exists")
    started = perf_counter()
    writer = client = guard = reference = report = projection = abort = execution_buffer = None
    acknowledged_attempts = set()
    reference_metadata = {"status": "not_prepared"}
    current_cutoff = 0
    stage = "manifest"
    try:
        save("manifest.json", manifest)
        stage = "reference_preparation"
        reference = prepare_reference(reference_path, data_root=root, **reference_config)
        reference_metadata = {key: value for key, value in reference.items() if key != "frames"}
        reference_metadata["frames"] = [{key: value for key, value in frame.items() if key != "jpeg_bytes"}
                                        for frame in reference["frames"]]
        stage = "reference_integrity"
        if reference["source_sha256"] != manifest["files"]["reference_media"]["sha256"]:
            raise RuntimeError("Prepared reference differs from the frozen source")
        if live_assessment:
            runtime.configure_assessment(assessment_contract_version=args.assessment_contract_version, reference_id=reference["reference_id"],
                execution_source_sha256=manifest["files"]["execution_media"]["sha256"],
                timing_limits=live_settings["timing_limits"], receipt_limits=live_settings["receipt_limits"])
        stage = "trace"
        writer = RunTraceWriter(output / "trace.jsonl", run_id=run_id, session_id=session_id,
            reference_id=reference["reference_id"], max_events=artifact_limits["max_trace_events"],
            max_bytes=artifact_limits["max_trace_bytes"],
            deduplicate_payloads=buffer_config is not None or input_representation == "jpeg_frames")
        writer.append("reference_preparation", reference, at_s=0)
        trace_target = writer
        if runtime is not None:
            class BrowserTrace:
                def append(self, kind, payload, *, at_s):
                    value = deepcopy(payload)
                    source_time = runtime.source_time()
                    value["browser_timing"] = {"source_time_s": source_time,
                                               "wall_elapsed_s": runtime.elapsed_s()}
                    if kind.startswith("provider_") or kind.startswith("budget_"):
                        value.setdefault("observation_cutoff_s", at_s)
                    receipt = writer.append(kind, value, at_s=at_s if live_assessment else source_time)
                    if kind == "notification_attempt" and value.get("status") == "acknowledged":
                        acknowledged_attempts.add(value["attempt_id"])
                    if kind != "timing_provenance":
                        runtime.emit(kind, deepcopy(value))
                    return receipt
            trace_target = BrowserTrace()
        def emit(kind, payload):
            trace_target.append(kind, payload, at_s=current_cutoff)
        split_provider = None
        if args.backend == "mock":
            from src.guideme.mock_predictor import mock_predict

            def completion(model_input, prior):
                return {"response": mock_predict(model_input), "updates": {
                    "schema_version": 1, "session_id": session_id,
                    "observation_id": model_input["observation_id"], "reference_id": reference["reference_id"],
                    "candidate_links": [], "issue_updates": []}, "new_issue_candidate_indices": []}
            if runtime is not None and scheduling is not None:
                # A fictional, stateless predictor exercises the same scheduler
                # without authentication, HTTP, or fabricated mistake alerts.
                split_provider = SimpleNamespace(
                    prepare=lambda model, prior: (deepcopy(model), deepcopy(prior)),
                    admit=lambda prepared: prepared,
                    perform=lambda ticket: completion(*ticket),
                    complete=lambda ticket, outcome: outcome,
                    cancel_unsent=lambda ticket: True,
                    snapshot=lambda: {"stopped": False})
        elif args.backend == "native-video":
            stage = "native_reference_preparation"
            from src.guideme.native_video import prepare_native_video
            from src.guideme.native_video_completion import NativeVideoCompletionPlan
            if input_representation == "native_video":
                native_reference = prepare_native_video(reference_path,
                    expected_source_sha256=manifest["files"]["reference_media"]["sha256"],
                    cutoff_s=None, data_root=root, **native["media"])
            if buffer_config is not None:
                stage = "native_execution_buffer"
                from src.guideme.native_video_buffer import NativeVideoBuffer
                execution_buffer = NativeVideoBuffer(
                    source_sha256=manifest["files"]["execution_media"]["sha256"],
                    source_size_bytes=manifest["files"]["execution_media"]["size_bytes"],
                    max_pending_bytes=buffer_config["max_pending_bytes"], **native["media"])
            stage = "provider_setup"
            import httpx
            key = os.environ.get("VLM_API_KEY")
            if not key:
                from dotenv import dotenv_values
                key = dotenv_values(REPO / ".env", interpolate=False).get("VLM_API_KEY")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Gateway credential unavailable")
            client = httpx.Client(base_url=native["base_url"], timeout=native["timeout_s"],
                headers={"Authorization": "Bearer " + key}, follow_redirects=False, trust_env=False)

            def resolve_assets(cutoff_s):
                if execution_buffer is None:
                    execution_asset = prepare_native_video(execution_path,
                        expected_source_sha256=manifest["files"]["execution_media"]["sha256"],
                        cutoff_s=cutoff_s, data_root=root, **native["media"])
                else:
                    execution_asset = execution_buffer.snapshot(cutoff_s)
                    if execution_asset is None or execution_asset["cutoff_s"] != cutoff_s:
                        raise RuntimeError("Native buffer snapshot differs from the committed observation cutoff")
                return {"reference": native_reference, "execution": execution_asset}

            plan = NativeVideoCompletionPlan(client=client, model=native["model"],
                reference_id=reference["reference_id"],
                reference_source_sha256=manifest["files"]["reference_media"]["sha256"],
                execution_source_sha256=manifest["files"]["execution_media"]["sha256"],
                resolve_assets=resolve_assets if input_representation == "native_video" else None,
                trace=trace_target, limits=native["limits"],
                timeout_s=native["timeout_s"], max_request_bytes=native["max_request_bytes"],
                max_response_bytes=native["max_response_bytes"],
                **({"fps": native["fps"]} if input_representation == "native_video" else
                   {"input_representation": input_representation, "image_limits": image_limits}),
                event_clock=runtime.elapsed_s if live_assessment else runtime.source_time if runtime is not None else None,
                **({"clock": runtime.elapsed_s} if live_assessment else {}),
                **({"assessment_contract_version": args.assessment_contract_version} if args.assessment_contract_version in (2, 3) else {}))
            guard = plan
            split_provider = plan

            def completion(model_input, prior):
                nonlocal current_cutoff
                current_cutoff = model_input["cutoff_s"]
                return plan(model_input, prior)
        else:
            stage = "provider_setup"
            from google import genai
            from src.guideme.bounded_gemini import BoundedGeminiClient
            from src.guideme.gemini_completion import GeminiCompletionPlan
            from src.guideme.visual_request import build_visual_request
            from src.guideme.visual_completion import decode_visual_completion
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                from dotenv import dotenv_values
                key = dotenv_values(REPO / ".env", interpolate=False).get("GEMINI_API_KEY")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Gemini credential unavailable")
            client = genai.Client(vertexai=False, api_key=key,
                http_options={"timeout": math.ceil(gemini["timeout_s"] * 1000), "retry_options": {"attempts": 1}})
            guard = BoundedGeminiClient(client, emit=emit, limits=gemini["limits"])
            plan = GeminiCompletionPlan(client=guard, model=gemini["model"], trace=trace_target,
                build_request=partial(build_visual_request, **gemini["request_limits"]),
                decode_response=partial(decode_visual_completion, max_response_bytes=gemini["max_response_bytes"]),
                max_calls=gemini["limits"]["max_token_checks"],
                max_total_input_bytes=gemini["max_total_input_bytes"], timeout_s=gemini["timeout_s"],
                max_output_tokens=gemini["limits"]["max_output_tokens"], media_resolution=gemini["media_resolution"])
            def completion(model_input, prior):
                nonlocal current_cutoff
                current_cutoff = model_input["cutoff_s"]
                return plan(model_input, prior)
        stage = "execution_integrity"
        verify_media("execution_media", execution_path)
        if runtime is not None:
            runtime.ready()
        stage = "replay"
        report = run_content_replay(reference, execution_path, run_id=run_id, session_id=session_id,
            completion_plan=completion, sender=runtime.sender if runtime is not None else lambda envelope: {
                "status": "acknowledged", "delivered_at_s": envelope["reserved_at_s"],
                **({"clock_id": envelope["clock_id"]} if args.assessment_contract_version in (2, 3) else {})},
            trace=trace_target, config=replay, data_root=root, manifest={"manifest_id": manifest["manifest_id"],
                **({"execution_source_sha256": manifest["files"]["execution_media"]["sha256"]}
                   if args.assessment_contract_version in (2, 3) else {})},
            callback_errors_fatal=True, runtime=runtime,
            **({"assessment_contract_version": args.assessment_contract_version} if args.assessment_contract_version in (2, 3) else {}),
            **({"input_representation": input_representation, "compact_selection": compact_selection}
               if input_representation == "jpeg_frames" else {}),
            **({"scheduling": scheduling,
                ("pair_provider" if args.two_stage else "completion_provider"): split_provider}
               if runtime is not None and scheduling is not None else {}),
            **({"execution_buffer": execution_buffer,
                "buffer_seal_interval_s": buffer_config["seal_interval_s"]}
               if execution_buffer is not None else {}))
        stage = "media_integrity"
        verify_media("reference_media", reference_path)
        verify_media("execution_media", execution_path)
    except BaseException as exc:
        abort = {"stage": stage, "exception_type": type(exc).__name__, "code": "replay_command_aborted"}
        if stage == "reference_preparation":
            reference_metadata = {"status": "failed", "exception_type": type(exc).__name__}
    finally:
        if execution_buffer is not None:
            try:
                execution_buffer.close()
            except BaseException as exc:
                abort = abort or {"stage": "native_execution_buffer_close",
                    "exception_type": type(exc).__name__, "code": "native_execution_buffer_close_failed"}
        if writer is not None:
            try:
                writer.close()
            except BaseException as exc:
                abort = abort or {"stage": "trace_close", "exception_type": type(exc).__name__, "code": "trace_close_failed"}
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                abort = abort or {"stage": "provider_close", "exception_type": type(exc).__name__, "code": "provider_close_failed"}
    if not (output / "trace.jsonl").exists():
        (output / "trace.jsonl").open("xb").close()
    if reference is not None and runtime is None:
        try:
            from src.guideme.content_trace import project_content_trace

            decoded = read_run_trace(output / "trace.jsonl", max_events=artifact_limits["max_trace_events"],
                                     max_bytes=artifact_limits["max_trace_bytes"])
            if any(row["event_type"] == "run_started" for row in decoded["records"]):
                projection = project_content_trace(decoded, run_id=run_id, session_id=session_id,
                                                    reference_id=reference["reference_id"])
        except BaseException as exc:
            abort = abort or {"stage": "projection", "exception_type": type(exc).__name__, "code": "trace_projection_failed"}
    success = (abort is None and report is not None and report["status"] == "complete"
               and report["failures"] == 0
               and (runtime is not None or (projection is not None and projection["run_complete"])))
    summary = {"schema_version": 1, "run_id": run_id, "session_id": session_id,
        **({"assessment_contract_version": args.assessment_contract_version} if args.assessment_contract_version in (2, 3) else {}),
        **({"evaluation_projection_status": "unsupported_live_interval_timing"} if live_assessment else {}),
        "manifest_id": manifest["manifest_id"], "backend": args.backend, "delivery_mode": delivery_mode,
        "timing_basis": timing_basis, "status": "complete" if success else "failed_or_incomplete",
        "run_report": report, "projection": projection,
        "provider_accounting": guard.snapshot() if guard is not None else None,
        "abort": abort, "reference_preparation": reference_metadata, "wall_s": perf_counter() - started,
        "real_ui_notifications_delivered": len(acknowledged_attempts),
        "gold_loaded": False,
        "limitations": ["Browser-reported playhead and display acknowledgment; wall latency recorded separately"
                        if runtime is not None else "Fixed cutoff grid starting at zero; content timing is idealized",
                        "Mock predictions are fictional insufficient-evidence responses" if args.backend == "mock"
                        else "Provider validity does not establish physical mistake-detection accuracy"]}
    save("summary.json", summary)
    if runtime is not None:
        runtime.finish(summary)
    print(json.dumps({"backend": args.backend, "status": summary["status"], "output_dir": str(output)}), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
