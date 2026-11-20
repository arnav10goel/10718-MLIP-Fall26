"""Launch the existing GuideMe monitor with a local video pair."""

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.monitor_videos import main as monitor_main

TEMPLATE = REPO / "configs/demo_monitor.example.json"
PRESETS = {
    "blocks": ("raw/egooops/videos-processed-720p/tsumiki/S1750003.MP4",
               "raw/egooops/videos-processed-720p/tsumiki/S1750004.MP4"),
    "circuit": ("raw/egooops/videos-processed-720p/circuit/S1790005.MP4",
                "raw/egooops/videos-processed-720p/circuit/S1790010.MP4"),
    "ion": ("raw/egooops/videos-processed-720p/ion/S1730006.MP4",
            "raw/egooops/videos-processed-720p/ion/S1730002.MP4"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", choices=("blocks", "circuit", "ion", "custom"))
    parser.add_argument("--budget", required=True, help="Configured per-run USD allowance")
    parser.add_argument("--reference", help="Reference video path (overrides preset)")
    parser.add_argument("--execution", help="Execution video path (overrides preset)")
    parser.add_argument("--data-root", help="Absolute data root (default: GUIDEME_DATA_ROOT or repo data)")
    parser.add_argument("--stop-at", type=float, help="Optional source-time end in seconds")
    parser.add_argument("--drain-timeout", type=float, default=180,
                        help="Maximum seconds to finish outstanding work after playback ends (default: 180)")
    parser.add_argument("--heartbeat-timeout", type=float, default=120,
                        help="Seconds to allow between browser playback reports (default: 120)")
    parser.add_argument("--port", type=int, default=0, help="Local port; default chooses one automatically")
    parser.add_argument("--browser-replay", action="store_true", help="Use the browser playhead instead of background replay")
    args = parser.parse_args(argv)
    if args.pair == "custom":
        if not args.reference or not args.execution:
            parser.error("custom requires --reference and --execution")
    try:
        budget = Decimal(args.budget)
        if not budget.is_finite() or budget <= 0:
            raise ValueError("Budget must be positive")
        if not math.isfinite(args.drain_timeout) or args.drain_timeout < 0:
            raise ValueError("Drain timeout must be a finite nonnegative number")
        if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
            raise ValueError("Heartbeat timeout must be a finite positive number")
        root = Path(args.data_root or os.environ.get("GUIDEME_DATA_ROOT") or REPO / "data").expanduser()
        if not root.is_absolute():
            raise ValueError("Data root must be absolute")
        root = root.resolve()
        settings = json.loads(TEMPLATE.read_text())
        defaults = PRESETS.get(args.pair, (None, None))
        def media_path(value):
            path = Path(value).expanduser()
            return str((path if path.is_absolute() else root / path).resolve(strict=True))
        settings["reference"] = media_path(args.reference or defaults[0])
        settings["execution"] = media_path(args.execution or defaults[1])
        settings["native_video"]["limits"]["budget_usd"] = str(budget)
        settings["scheduling"]["drain_timeout_s"] = args.drain_timeout
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as exc:
        parser.error(f"Cannot prepare demo: {exc}")

    # The runner requires a new output directory; UUID prevents reuse or overwrite.
    output = f"processed/egooops/runs/demo-{args.pair}-{uuid4()}"
    with tempfile.TemporaryDirectory(prefix="guideme-demo-") as directory:
        config = Path(directory) / "config.json"
        config.write_text(json.dumps(settings))
        arguments = ["--config", str(config), "--backend", "native-video", "--output-dir", output,
                     "--data-root", str(root), "--port", str(args.port), "--assessment-contract-version", "3",
                     "--heartbeat-timeout", str(args.heartbeat_timeout)]
        if not args.browser_replay:
            arguments.append("--host-replay")
        if args.stop_at is not None:
            arguments.extend(("--stop-at", str(args.stop_at)))
        print(f"Run output: {root / output}", flush=True)
        print("The video starts paused; playing it starts paid model checks.", flush=True)
        if not args.browser_replay:
            print("Background replay continues when the browser is hidden; use Pause or Stop to halt it.", flush=True)
        return monitor_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
