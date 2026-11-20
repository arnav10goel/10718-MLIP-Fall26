"""Play configurable local videos while the shared runner monitors execution."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from threading import Thread

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_content_replay import main as run_replay
from src.guideme.monitoring_session import MonitoringSession
from src.guideme.monitor_server import MonitorServer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing replay settings JSON")
    parser.add_argument("--backend", required=True, choices=("mock", "native-video"))
    parser.add_argument("--output-dir", required=True, help="New run directory, relative to the data root or absolute")
    parser.add_argument("--data-root", help="Absolute root; overrides GUIDEME_DATA_ROOT")
    parser.add_argument("--stop-at", type=float, help="Optional execution source-time limit in seconds")
    parser.add_argument("--port", type=int, default=8769, help="Local port; zero chooses an available port")
    parser.add_argument("--host-replay", action="store_true", help="Use a server-driven clock for prerecorded replay")
    parser.add_argument("--heartbeat-timeout", type=float, default=10,
                        help="Seconds without playback reports before ending a playing session (default: 10)")
    parser.add_argument("--assessment-contract-version", type=int, choices=(1, 2, 3), default=1,
                        help="Opt in to interval-aware native-video monitoring")
    parser.add_argument("--two-stage", action="store_true",
                        help="Use two-stage observation and issue association")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between zero and 65535")
    if args.stop_at is not None and (not math.isfinite(args.stop_at) or args.stop_at <= 0):
        parser.error("--stop-at must be finite and positive")
    if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
        parser.error("--heartbeat-timeout must be finite and positive")
    if args.two_stage and (args.assessment_contract_version != 2 or args.backend != "native-video"):
        parser.error("--two-stage requires contract 2 and the native-video backend")
    if args.host_replay and (args.assessment_contract_version != 3 or args.backend != "native-video"):
        parser.error("--host-replay requires contract 3 and the native-video backend")
    try:
        root = Path(args.data_root or os.environ.get("GUIDEME_DATA_ROOT") or REPO / "data").expanduser()
        if not root.is_absolute():
            raise ValueError("Data root must be absolute")
        root = root.resolve()
        config = Path(args.config).expanduser().resolve(strict=True)

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate configuration field")
                result[key] = value
            return result

        settings = json.loads(config.read_text(), object_pairs_hook=unique)
        if args.assessment_contract_version in (2, 3) and (args.backend != "native-video" or settings.get("scheduling") is None):
            raise ValueError("Assessment monitoring requires native-video asynchronous settings")
        native = settings.get("native_video")
        if args.assessment_contract_version == 3 and (not isinstance(native, dict) or native.get("input_representation") != "jpeg_frames"):
            raise ValueError("Contract 3 requires scheduled JPEG frame settings")
        if args.two_stage and (not isinstance(settings.get("scheduling"), dict)
                               or not isinstance(native, dict) or native.get("input_representation") != "jpeg_frames"):
            raise ValueError("Two-stage monitoring requires scheduled JPEG frame settings")
        paths = []
        for role in ("reference", "execution"):
            value = settings[role]
            if type(value) is not str or not value.strip():
                raise ValueError("Media paths must be nonblank strings")
            path = Path(value).expanduser()
            paths.append((path if path.is_absolute() else root / path).resolve(strict=True))
        session = MonitoringSession(stop_at_s=args.stop_at, heartbeat_timeout_s=args.heartbeat_timeout,
            **({"replay_clock_mode": "host"} if args.host_replay else {}),
            **({"assessment_contract_version": args.assessment_contract_version} if args.assessment_contract_version in (2, 3) else {}))
        server = MonitorServer(("127.0.0.1", args.port), session=session,
            reference_path=paths[0], execution_path=paths[1], backend=args.backend)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error("Cannot start monitoring: " + type(exc).__name__)

    replay_args = ["--config", str(config), "--backend", args.backend,
                   "--output-dir", args.output_dir, "--data-root", str(root)]
    if args.assessment_contract_version in (2, 3):
        replay_args.extend(("--assessment-contract-version", str(args.assessment_contract_version)))
    if args.two_stage:
        replay_args.append("--two-stage")

    def worker():
        try:
            run_replay(replay_args, runtime=session)
        except BaseException:
            # Validation/preparation failures can occur before a runner exists.
            # Keep exception values, filesystem paths and credentials off HTTP.
            session.finish({"abort": {"code": "monitoring_worker_failed"}})

    thread = Thread(target=worker, daemon=True)
    thread.start()
    print(f"GuideMe: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if args.assessment_contract_version in (2, 3):
            session.stop()
            # Stop new inference/UI effects, but keep receipt transport available
            # through the coordinator's bounded drain. Provider/accounting cleanup
            # must finish before closing its server or returning from the process.
            server.timeout = 0.05
            try:
                while thread.is_alive():
                    try:
                        server.handle_request()
                    except KeyboardInterrupt:
                        continue
            finally:
                thread.join()
                server.server_close()
        else:
            snapshot = session.snapshot()
            try:
                session.playback({"session_id": session.session_id,
                    "sequence": snapshot["sequence"] + 1,
                    "source_time_s": snapshot["source_time_s"], "state": "stopped"})
            except (ValueError, RuntimeError):
                pass
            server.server_close()
            thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
