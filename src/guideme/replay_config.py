"""Load and validate replay settings before preparing media or contacting a provider."""

from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit


def load_replay_config(args, *, runtime, repo, parser):
    """Resolve paths and validated provider settings for a replay command.

    Return None for an unsupported runtime combination already reported to stderr.
    Invalid configuration is reported through the CLI parser before any output or auth.
    """
    live_assessment = args.assessment_contract_version in (2, 3) and runtime is not None
    def keys(value, required, optional=()):
        if (not isinstance(value, dict) or not set(required) <= value.keys()
                or not value.keys() <= set(required) | set(optional)):
            raise ValueError("Settings have missing or unsupported fields")

    def positive(value, *, integer=False):
        try:
            valid = (type(value) is int if integer else type(value) in (int, float))
            valid = valid and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Limits must be finite positive numbers of the required type")

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate settings key")
            result[name] = value
        return result

    def reject_constant(value):
        raise ValueError("Settings cannot contain nonfinite numbers")

    def local_path(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Paths must be nonblank strings")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    try:
        root = Path(args.data_root or os.environ.get("GUIDEME_DATA_ROOT") or repo / "data").expanduser()
        if not root.is_absolute():
            raise ValueError("--data-root / GUIDEME_DATA_ROOT must be absolute")
        root = root.resolve()
        output_arg = Path(args.output_dir).expanduser()
        output = (output_arg if output_arg.is_absolute() else root / output_arg)
        output = output.parent.resolve() / output.name
        if not output_arg.is_absolute() and not output.is_relative_to(root):
            raise ValueError("Relative output directory must remain within the data root")
        if output.exists() or output.is_symlink():
            raise ValueError("Output directory already exists")
        settings_path = Path(args.config).expanduser().resolve(strict=True)
        settings = json.loads(settings_path.read_text(), object_pairs_hook=unique_object,
                              parse_constant=reject_constant)
        keys(settings, ("schema_version", "reference", "execution", "reference_config", "replay"),
             ("gemini", "native_video", "scheduling", "artifact_limits"))
        if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
            raise ValueError("Expected settings schema version 1")
        configured_artifacts = settings.get("artifact_limits", {"max_trace_bytes": 128_000_000})
        keys(configured_artifacts, ("max_trace_bytes",))
        positive(configured_artifacts["max_trace_bytes"], integer=True)
        if args.assessment_contract_version == 2 and runtime is None and (settings.get("scheduling") is not None
                or isinstance(settings.get("native_video"), dict) and "buffer" in settings["native_video"]):
            print("Offline assessment replay does not support browser scheduling or buffering.", file=sys.stderr)
            return None
        if live_assessment and (settings.get("scheduling") is None
                or not callable(getattr(runtime, "configure_assessment", None))
                or not callable(getattr(runtime, "elapsed_s", None))
                or not isinstance(getattr(runtime, "clock_id", None), str)):
            print("Browser assessment replay requires a configured asynchronous timing runtime.", file=sys.stderr)
            return None
        if runtime is not None:
            snapshot_method = getattr(runtime, "snapshot", None)
            if callable(snapshot_method):
                runtime_version = snapshot_method().get("assessment_contract_version", 1)
                if type(runtime_version) is not int or runtime_version != args.assessment_contract_version:
                    print("Browser replay requires a matching assessment runtime.", file=sys.stderr)
                    return None
            elif live_assessment:
                print("Browser assessment replay requires a versioned runtime snapshot.", file=sys.stderr)
                return None
        reference_path, execution_path = (local_path(settings[name]) for name in ("reference", "execution"))
        if not reference_path.is_file() or not execution_path.is_file():
            raise ValueError("Reference and execution must be existing regular files")
        reference_config, replay = settings["reference_config"], settings["replay"]
        keys(reference_config, ("fps", "max_frames", "max_side", "jpeg_quality", "timeout_s"))
        for name, value in reference_config.items():
            positive(value, integer=name in ("max_frames", "max_side", "jpeg_quality"))
        keys(replay, ("decision_interval_s", "execution_fps", "execution_max_frames", "max_side",
            "jpeg_quality", "max_observations", "max_images", "max_jpeg_bytes", "max_evidence_age_s",
            "state", "issues", "sink"), ("repair_confirmation_count",))
        repair_confirmation_count = replay.get("repair_confirmation_count", 1)
        if (type(repair_confirmation_count) is not int or repair_confirmation_count not in (1, 2)
                or repair_confirmation_count == 2 and args.assessment_contract_version not in (2, 3)):
            raise ValueError("Repair confirmation count must be 1, or 2 for assessment contracts 2 and 3")
        for name, value in replay.items():
            if name not in ("state", "issues", "sink"):
                positive(value, integer=name not in ("decision_interval_s", "execution_fps", "max_evidence_age_s"))
        if reference_config["jpeg_quality"] > 100 or replay["jpeg_quality"] > 100:
            raise ValueError("JPEG quality must not exceed 100")
        for name, fields in (("state", ("max_records", "max_metadata_bytes")),
                ("issues", ("max_issues", "max_observations", "max_metadata_bytes")),
                ("sink", ("max_attempts", "max_metadata_bytes"))):
            keys(replay[name], fields)
            for value in replay[name].values():
                positive(value, integer=True)
        scheduling = settings.get("scheduling")
        if scheduling is not None:
            keys(scheduling, ("max_in_flight", "notification_min_visible_s",
                              "drain_timeout_s", "poll_interval_s"))
            positive(scheduling["max_in_flight"], integer=True)
            positive(scheduling["poll_interval_s"])
            for name in ("notification_min_visible_s", "drain_timeout_s"):
                value = scheduling[name]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Scheduling durations must be finite nonnegative numbers")
            if runtime is not None and args.backend == "gemini":
                raise ValueError("Overlapping monitoring requires the native-video or mock backend")
        gemini = settings.get("gemini")
        if args.backend == "gemini" and gemini is None:
            raise ValueError("Gemini requires explicit provider, request and budget settings")
        if gemini is not None:
            keys(gemini, ("model", "timeout_s", "media_resolution", "max_total_input_bytes",
                          "max_response_bytes", "request_limits", "limits"))
            if not isinstance(gemini["model"], str) or not gemini["model"].strip():
                raise ValueError("Gemini model must be explicit and nonblank")
            positive(gemini["timeout_s"])
            positive(gemini["max_total_input_bytes"], integer=True)
            positive(gemini["max_response_bytes"], integer=True)
            if gemini["media_resolution"] not in (None, "LOW", "MEDIUM", "HIGH"):
                raise ValueError("Unsupported Gemini media resolution")
            keys(gemini["request_limits"], ("max_images", "max_jpeg_bytes", "max_text_bytes", "max_prior_issues"))
            for value in gemini["request_limits"].values():
                positive(value, integer=True)
            keys(gemini["limits"], ("max_token_checks", "max_generation_calls", "max_input_tokens",
                "max_output_tokens", "budget_usd", "input_usd_per_million", "output_usd_per_million"))
            for name, value in gemini["limits"].items():
                if name.startswith("max_"):
                    positive(value, integer=True)
                elif type(value) is not str or not Decimal(value).is_finite() or Decimal(value) < 0:
                    raise ValueError("Monetary limits must be finite nonnegative decimal strings")
        native = settings.get("native_video")
        buffer_config = None
        input_representation = "native_video"
        compact_selection = image_limits = None
        if args.backend == "native-video" and native is None:
            raise ValueError("Native video requires an explicit gateway, media and spending profile")
        if native is not None:
            if not isinstance(native, dict):
                raise ValueError("Native provider settings must be an object")
            input_representation = native.get("input_representation", "native_video")
            if input_representation not in ("native_video", "jpeg_frames"):
                raise ValueError("Unsupported native input representation")
            if input_representation == "jpeg_frames":
                keys(native, ("base_url", "model", "timeout_s", "max_request_bytes", "max_response_bytes",
                    "limits", "input_representation", "image_limits", "compact_selection"))
                if args.backend != "native-video" or not live_assessment or scheduling is None:
                    raise ValueError("JPEG input requires scheduled assessment browser monitoring")
                image_limits, compact_selection = native["image_limits"], native["compact_selection"]
                keys(image_limits, ("max_images", "max_jpeg_bytes"))
                for value in image_limits.values():
                    positive(value, integer=True)
                keys(compact_selection, ("coarse_interval_s", "recent_window_s"))
                for value in compact_selection.values():
                    positive(value)
                if replay["max_images"] < reference_config["max_frames"] + replay["execution_max_frames"] + 1:
                    raise ValueError("Full observation pool cannot hold its reference, cadence and endpoint bounds")
            else:
                keys(native, ("base_url", "model", "timeout_s", "fps", "max_request_bytes",
                              "max_response_bytes", "media", "limits"), ("buffer", "input_representation"))
            if "buffer" in native:
                buffer_config = native["buffer"]
                keys(buffer_config, ("seal_interval_s", "max_pending_bytes"))
                positive(buffer_config["seal_interval_s"])
                positive(buffer_config["max_pending_bytes"], integer=True)
                if args.backend != "native-video" or runtime is None or scheduling is None:
                    raise ValueError("Native buffering requires scheduled browser monitoring")
            if not isinstance(native["base_url"], str):
                raise ValueError("Gateway URL must be explicit")
            parsed = urlsplit(native["base_url"])
            parsed.port  # Validate an explicitly supplied port before creating output/auth.
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or not parsed.path.endswith("/v1/")):
                raise ValueError("Gateway must be an HTTPS /v1/ URL without embedded credentials")
            from src.guideme.native_video_completion import NATIVE_VIDEO_PROFILES
            if not isinstance(native["model"], str) or native["model"] not in NATIVE_VIDEO_PROFILES:
                raise ValueError("Native video requires a supported frozen model profile")
            profile = NATIVE_VIDEO_PROFILES[native["model"]]
            for name in ("timeout_s", "max_request_bytes", "max_response_bytes"):
                positive(native[name], integer=name in ("max_request_bytes", "max_response_bytes"))
            if input_representation == "native_video" and type(native["fps"]) is dict:
                keys(native["fps"], ("reference", "execution"))
                for rate in native["fps"].values():
                    positive(rate)
                    if rate > 24:
                        raise ValueError("Native FPS rates must be at most 24")
            elif input_representation == "native_video":
                positive(native["fps"])
                if native["fps"] != 1:
                    raise ValueError("Scalar native FPS must use the legacy one-FPS profile")
            if input_representation == "native_video":
                keys(native["media"], ("max_side", "max_frames", "max_bytes", "timeout_s"))
                for name, value in native["media"].items():
                    positive(value, integer=name != "timeout_s")
                if native["media"]["max_side"] < 2:
                    raise ValueError("Native MP4 dimensions require at least two pixels")
            keys(native["limits"], ("max_generation_calls", "max_input_tokens", "max_output_tokens",
                "budget_usd", "input_usd_per_million", "output_usd_per_million"))
            for name, value in native["limits"].items():
                if name == "max_generation_calls" and value is None:
                    continue
                if name.startswith("max_"):
                    positive(value, integer=True)
                elif (type(value) is not str or not value or value.strip() != value
                        or not Decimal(value).is_finite() or Decimal(value) < 0):
                    raise ValueError("Native monetary limits must be finite nonnegative decimal strings")
            if (native["limits"]["max_input_tokens"] != profile["max_input_tokens"]
                    or native["limits"]["max_output_tokens"] != profile["max_output_tokens"]
                    or Decimal(native["limits"]["input_usd_per_million"]) != Decimal(profile["input_usd_per_million"])
                    or Decimal(native["limits"]["output_usd_per_million"]) != Decimal(profile["output_usd_per_million"])):
                raise ValueError("Native context/output limits and rates must match the frozen profile")
        if args.assessment_contract_version == 3 and input_representation != "jpeg_frames":
            raise ValueError("Contract 3 requires scheduled JPEG frame input")
        if args.two_stage and (input_representation != "jpeg_frames" or scheduling is None):
            raise ValueError("Two-stage monitoring requires scheduled JPEG input")
    except (OSError, ValueError, TypeError, InvalidOperation) as exc:
        parser.error("Invalid replay configuration: " + type(exc).__name__)

    return SimpleNamespace(
        root=root,
        output=output,
        settings_path=settings_path,
        settings=settings,
        configured_artifacts=configured_artifacts,
        reference_path=reference_path,
        execution_path=execution_path,
        reference_config=reference_config,
        replay=replay,
        repair_confirmation_count=repair_confirmation_count,
        scheduling=scheduling,
        gemini=gemini,
        native=native,
        buffer_config=buffer_config,
        input_representation=input_representation,
        compact_selection=compact_selection,
        image_limits=image_limits,
    )
