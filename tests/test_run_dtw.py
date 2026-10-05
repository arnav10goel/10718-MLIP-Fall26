"""Thin CLI contracts; video decoding and matching have separate real tests."""

from contextlib import chdir, redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunDtwTests(unittest.TestCase):
    def test_report_preserves_inputs_and_duration_source(self):
        # Wrong root precedence, late hashing, dropped results, or ignored
        # duration override must change an observable saved report.
        from scripts import run_dtw

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            cli_root, env_root = base / "cli-data", base / "env-data"
            execution = base / "execution.mkv"
            execution.write_bytes(b"execution bytes")
            for root in (cli_root, env_root):
                root.mkdir()
                (root / "reference.mkv").write_bytes(root.name.encode())
                (root / "checklist.csv").write_bytes(b"authored checklist")
            expected_inputs = {
                role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for role, path in (
                    ("reference", cli_root / "reference.mkv"),
                    ("execution", execution), ("checklist", cli_root / "checklist.csv"),
                )
            }

            expected_result = {
                "method": "classical_subsequence_dtw",
                "reference_path": str(cli_root / "reference.mkv"),
                "execution_path": str(execution),
                "reference_annotations_path": str(cli_root / "checklist.csv"),
                "reference_duration_s": 2.0, "checkpoint_s": 1.0,
                "window_s": .5, "window_start_s": .5, "fps": 5.0,
                "frame_size": [64, 64], "reference_times_s": [0.0],
                "execution_times_s": [.5], "reference_time_s": 0.0,
                "tied_reference_times_s": [0.0], "candidate_step": None,
                "unknown_reason": "unmapped_reference_time",
                "alignment": {
                    "path": [[0, 0]], "start_reference_index": 0,
                    "end_reference_index": 0, "tied_endpoints": [0],
                    "total_cost": .25, "normalized_cost": .25,
                    "endpoint_costs": [.25],
                },
            }
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = lambda key: {
                run_dtw.cv2.CAP_PROP_FRAME_COUNT: 50.0,
                run_dtw.cv2.CAP_PROP_FPS: 25.0,
            }[key]

            def checkpoint(reference, *args, **kwargs):
                # Record observed input hashes before downstream processing.
                reference.write_bytes(b"changed during core")
                return expected_result

            arguments = [
                "--reference", "reference.mkv", "--execution", str(execution),
                "--checklist", "checklist.csv", "--end-s", "1",
                "--window-s", "0.5",
            ]
            with chdir(base), patch.dict(os.environ, {"GUIDEME_DATA_ROOT": str(env_root)}), \
                    patch.object(run_dtw.cv2, "VideoCapture", return_value=capture) as metadata, \
                    patch.object(run_dtw, "run_classical_dtw_checkpoint", side_effect=checkpoint) as core, \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(run_dtw.main(arguments + [
                    "--data-root", str(cli_root), "--output", "metadata.json",
                ]), 0)
                report = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
                core.assert_called_once_with(
                    cli_root / "reference.mkv", execution, cli_root / "checklist.csv",
                    reference_duration_s=2.0, checkpoint_s=1.0, window_s=.5, fps=5.0,
                )
                capture.release.assert_called_once()
                self.assertEqual(report["reference_duration"], {
                    "method": "opencv_frame_count_over_fps", "value_s": 2.0,
                    "frame_count": 50.0, "fps": 25.0,
                })
                self.assertEqual(report["inputs"], expected_inputs)
                self.assertEqual((report["schema_version"], report["status"], report["data_root"]),
                                 (1, "ok", str(cli_root)))
                self.assertEqual(report["checkpoint"], {"end_s": 1.0, "window_s": .5, "fps": 5.0})
                self.assertEqual(report["result"], expected_result)
                self.assertLessEqual(datetime.fromisoformat(report["started_at"]),
                                     datetime.fromisoformat(report["finished_at"]))
                self.assertEqual(report["implementation_sha256"]["uv.lock"],
                                 hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest())

                core.reset_mock()
                metadata.reset_mock()
                self.assertEqual(run_dtw.main(arguments + [
                    "--reference-duration-s", "3", "--output", "provided.json",
                ]), 0)
                provided = json.loads((base / "provided.json").read_text(encoding="utf-8"))
                self.assertEqual(provided["data_root"], str(env_root))
                self.assertEqual(provided["reference_duration"], {"method": "provided", "value_s": 3.0})
                core.assert_called_once_with(
                    env_root / "reference.mkv", execution, env_root / "checklist.csv",
                    reference_duration_s=3.0, checkpoint_s=1.0, window_s=.5, fps=5.0,
                )
                metadata.assert_not_called()

    def test_refuses_overwrite_and_saves_core_failure(self):
        # Removing output preflight must invoke forbidden work; swallowing a
        # core failure must produce the wrong exit code or report status.
        from scripts import run_dtw

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            for name in ("reference.mkv", "execution.mkv", "checklist.csv"):
                (base / name).write_bytes(b"fixture bytes")
            output = base / "report.json"
            output.write_bytes(b"keep this report")
            arguments = [
                "--reference", "reference.mkv", "--execution", "execution.mkv",
                "--checklist", "checklist.csv", "--end-s", "1",
                "--window-s", ".5", "--reference-duration-s", "2",
                "--data-root", str(base), "--output", str(output),
            ]
            with patch.object(run_dtw, "resolve_data_root", side_effect=AssertionError("root before preflight")), \
                    patch.object(run_dtw, "run_classical_dtw_checkpoint", side_effect=AssertionError("core before preflight")), \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                try:
                    exit_code = run_dtw.main(arguments)
                except SystemExit as error:
                    exit_code = error.code
                self.assertNotEqual(exit_code, 0)
            self.assertEqual(output.read_bytes(), b"keep this report")

            failed_output = base / "failed.json"
            with patch.object(run_dtw, "run_classical_dtw_checkpoint", side_effect=ValueError("private failure detail")), \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(run_dtw.main(arguments[:-1] + [str(failed_output)]), 1)
            report = json.loads(failed_output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error"], {"stage": "run_checkpoint", "type": "ValueError"})
            self.assertIsNone(report["result"])

    def test_clip_selection_records_local_loader_metadata_and_failures(self):
        # Wrong encoder dispatch, dropped provenance, or forwarding the
        # requested rather than loaded device must break saved CLI behavior.
        from scripts import run_dtw

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "data"
            root.mkdir()
            for name in ("reference.mkv", "execution.mkv", "checklist.csv", "weights.pt"):
                (root / name).write_bytes(b"local fixture")
            weights = root / "weights.pt"
            weight_hash = hashlib.sha256(b"local fixture").hexdigest()
            model = object()
            metadata = {
                "architecture": "ViT-B-32", "force_quick_gelu": True,
                "precision": "fp32", "device": "cpu", "requested_device": "cpu",
                "checkpoint": {"path": str(weights), "sha256": weight_hash, "size_bytes": 13},
                "runtime": {"torch": "2.14.1", "open_clip": "3.3.0"},
                "preprocessing": {
                    "frame_size": [224, 224], "input_color": "BGR", "model_color": "RGB",
                    "pixel_scale": 1 / 255, "mean": [.48145466, .4578275, .40821073],
                    "std": [.26862954, .26130258, .27577711], "resize": "caller",
                    "crop": None, "embedding_normalization": "L2",
                },
            }
            result = {
                "method": "clip_subsequence_dtw", "reference_path": str(root / "reference.mkv"),
                "execution_path": str(root / "execution.mkv"),
                "reference_annotations_path": str(root / "checklist.csv"),
                "reference_duration_s": 2.0, "checkpoint_s": 1.0, "window_s": .5,
                "window_start_s": .5, "fps": 5.0, "frame_size": [224, 224],
                "reference_times_s": [0.0], "execution_times_s": [.5],
                "reference_time_s": 0.0, "tied_reference_times_s": [0.0],
                "candidate_step": None, "unknown_reason": "unmapped_reference_time",
                "alignment": {"path": [[0, 0]], "start_reference_index": 0,
                              "end_reference_index": 0, "tied_endpoints": [0],
                              "total_cost": .25, "normalized_cost": .25, "endpoint_costs": [.25]},
            }
            arguments = [
                "--reference", "reference.mkv", "--execution", "execution.mkv",
                "--checklist", "checklist.csv", "--end-s", "1", "--window-s", ".5",
                "--reference-duration-s", "2", "--data-root", str(root),
            ]
            clip = ["--encoder", "clip", "--clip-weights", "weights.pt"]
            with chdir(base), patch.dict(os.environ, {"GUIDEME_DATA_ROOT": str(base / "wrong-root")}), \
                    patch.object(run_dtw, "load_clip_encoder") as loader, \
                    patch.object(run_dtw, "run_dtw_checkpoint", return_value=result) as core, \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                for extra, requested, actual, batch, output in (
                    ([], "cpu", "cpu", 32, "clip-default.json"),
                    (["--device", "cuda", "--batch-size", "3"], "cuda", "cuda:0", 3, "clip-device.json"),
                ):
                    metadata.update(device=actual, requested_device=requested)
                    weights.write_bytes(b"local fixture")

                    def load(path, *, device):
                        weights.write_bytes(b"changed during load")
                        return model, actual, metadata

                    loader.side_effect = load
                    self.assertEqual(run_dtw.main(arguments + clip + extra + ["--output", output]), 0)
                    loader.assert_called_once_with(weights, device=requested)
                    core.assert_called_once_with(
                        root / "reference.mkv", root / "execution.mkv", root / "checklist.csv",
                        reference_duration_s=2.0, checkpoint_s=1.0, window_s=.5, fps=5.0,
                        encoder="clip", clip_model=model, device=actual, batch_size=batch,
                    )
                    report = json.loads((base / output).read_text(encoding="utf-8"))
                    self.assertEqual(report["inputs"]["clip_weights"], {"path": str(weights), "sha256": weight_hash})
                    self.assertEqual(report["encoder"], {"name": "clip", "batch_size": batch, "model": metadata})
                    self.assertEqual((report["status"], report["result"]), ("ok", result))
                    self.assertEqual(report["implementation_sha256"]["src/guideme/features.py"],
                                     hashlib.sha256((REPO_ROOT / "src/guideme/features.py").read_bytes()).hexdigest())
                    loader.reset_mock()
                    core.reset_mock()

                loader.side_effect = RuntimeError("private checkpoint details")
                self.assertEqual(run_dtw.main(arguments + clip + ["--output", "load-error.json"]), 1)
                failure = json.loads((base / "load-error.json").read_text(encoding="utf-8"))
                self.assertEqual(failure["error"], {"stage": "load_encoder", "type": "RuntimeError"})
                self.assertEqual((failure["status"], failure["result"]), ("error", None))
                core.assert_not_called()

            with patch.object(run_dtw, "resolve_data_root", side_effect=AssertionError("preflight skipped")), \
                    redirect_stderr(StringIO()):
                for extra in (["--encoder", "clip"], ["--batch-size", "32"], clip + ["--batch-size", "0"]):
                    with self.subTest(rejected=extra), self.assertRaises(SystemExit) as error:
                        run_dtw.main(arguments + extra + ["--output", str(base / "rejected.json")])
                    self.assertEqual(error.exception.code, 2)
            self.assertFalse((base / "rejected.json").exists())

    def test_warp_penalty_dispatch_preserves_zero_and_records_positive_failures(self):
        # Dropping a positive penalty or changing zero dispatch must alter
        # the pipeline arguments or saved checkpoint configuration.
        from scripts import run_dtw

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            for name in ("reference.mkv", "execution.mkv", "checklist.csv", "weights.pt"):
                (base / name).write_bytes(b"fixture")
            model = object()
            metadata = {
                "architecture": "ViT-B-32", "force_quick_gelu": True, "precision": "fp32",
                "device": "cuda:0", "requested_device": "cuda",
                "checkpoint": {"path": str(base / "weights.pt"), "size_bytes": 7,
                               "sha256": hashlib.sha256(b"fixture").hexdigest()},
                "runtime": {"torch": "2.14.1", "open_clip": "3.3.0"},
                "preprocessing": {"frame_size": [224, 224], "input_color": "BGR",
                                  "model_color": "RGB", "pixel_scale": 1 / 255,
                                  "mean": [.48145466, .4578275, .40821073],
                                  "std": [.26862954, .26130258, .27577711], "resize": "caller",
                                  "crop": None, "embedding_normalization": "L2"},
            }
            arguments = [
                "--reference", "reference.mkv", "--execution", "execution.mkv",
                "--checklist", "checklist.csv", "--end-s", "1", "--window-s", ".5",
                "--reference-duration-s", "2", "--data-root", str(base),
            ]
            clip = ["--encoder", "clip", "--clip-weights", "weights.pt", "--device", "cuda", "--batch-size", "3"]
            common = dict(reference_duration_s=2.0, checkpoint_s=1.0, window_s=.5, fps=5.0)
            with patch.object(run_dtw, "load_clip_encoder", return_value=(model, "cuda:0", metadata)), \
                    patch.object(run_dtw, "run_classical_dtw_checkpoint") as legacy, \
                    patch.object(run_dtw, "run_dtw_checkpoint") as shared, \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                for encoder, options in (("classical", []), ("clip", clip)):
                    for penalty in (0.0, .5):
                        with self.subTest(encoder=encoder, penalty=penalty):
                            result = {
                                "method": f"{encoder}_subsequence_dtw", "reference_path": str(base / "reference.mkv"),
                                "execution_path": str(base / "execution.mkv"),
                                "reference_annotations_path": str(base / "checklist.csv"),
                                **common, "window_start_s": .5,
                                "frame_size": [224, 224] if encoder == "clip" else [64, 64],
                                "reference_times_s": [0.0], "execution_times_s": [.5],
                                "reference_time_s": 0.0, "tied_reference_times_s": [0.0],
                                "candidate_step": None, "unknown_reason": "unmapped_reference_time",
                                "alignment": {"path": [[0, 0]], "start_reference_index": 0,
                                              "end_reference_index": 0, "tied_endpoints": [0],
                                              "total_cost": .25, "normalized_cost": .25, "endpoint_costs": [.25]},
                            }
                            expected = dict(common)
                            if encoder == "clip":
                                expected.update(encoder="clip", clip_model=model, device="cuda:0", batch_size=3)
                            if penalty:
                                expected["warp_penalty"] = result["warp_penalty"] = penalty
                            legacy.return_value = shared.return_value = result
                            output = base / f"{encoder}-{penalty}.json"
                            self.assertEqual(run_dtw.main(arguments + options + [
                                "--warp-penalty", str(penalty), "--output", str(output),
                            ]), 0)
                            selected, unused = (legacy, shared) if encoder == "classical" and penalty == 0 else (shared, legacy)
                            selected.assert_called_once_with(base / "reference.mkv", base / "execution.mkv",
                                                             base / "checklist.csv", **expected)
                            unused.assert_not_called()
                            report = json.loads(output.read_text(encoding="utf-8"))
                            self.assertEqual(report["checkpoint"], {
                                "end_s": 1.0, "window_s": .5, "fps": 5.0,
                                **({"warp_penalty": .5} if penalty else {}),
                            })
                            self.assertEqual(report["result"], result)
                            legacy.reset_mock()
                            shared.reset_mock()

                shared.side_effect = ValueError("private details")
                failed = base / "failed-penalty.json"
                self.assertEqual(run_dtw.main(arguments + ["--warp-penalty", ".5", "--output", str(failed)]), 1)
                report = json.loads(failed.read_text(encoding="utf-8"))
                self.assertEqual(report["checkpoint"]["warp_penalty"], .5)
                self.assertEqual((report["status"], report["error"], report["result"]),
                                 ("error", {"stage": "run_checkpoint", "type": "ValueError"}, None))

            rejected = base / "rejected-penalty.json"
            with patch.object(run_dtw, "resolve_data_root", side_effect=AssertionError("root before preflight")), \
                    redirect_stderr(StringIO()):
                for invalid in ("-1", "nan", "inf", "nonnumeric"):
                    with self.subTest(invalid=invalid), self.assertRaises(SystemExit) as error:
                        run_dtw.main(arguments + ["--warp-penalty", invalid, "--output", str(rejected)])
                    self.assertEqual(error.exception.code, 2)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
