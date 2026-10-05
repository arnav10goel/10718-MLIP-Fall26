"""Checkpoint pipeline tests using real color frames and hand-authored labels."""

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np

from src.guideme.baselines.dtw import run_classical_dtw_checkpoint
from src.guideme.video import sample_video_frames


def _write_color_video(path, colors):
    """Encode lossless BGR frames at 4 FPS; timestamps are 0, .25, .5, ... ."""
    frames = np.stack([
        np.full((12, 16, 3), color, dtype=np.uint8) for color in colors
    ])
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "16x12",
            "-framerate", "4", "-i", "pipe:0", "-c:v", "ffv1",
            "-pix_fmt", "bgr0", str(path),
        ],
        input=frames.tobytes(), capture_output=True, check=True,
    )


def _write_annotations(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step_id", "action", "start_s", "end_s", "requirements"])
        writer.writerows(rows)


class ClassicalDtwCheckpointTests(unittest.TestCase):
    def test_replay_keeps_causal_results_and_continues_after_one_checkpoint_failure(self):
        from src.guideme.baselines.dtw import run_dtw_checkpoint, run_dtw_replay
        from src.guideme.evaluation import dtw_prediction_from_report

        red, green, blue, yellow = (0, 0, 240), (0, 240, 0), (240, 0, 0), (0, 240, 240)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference, execution, checklist = (base / name for name in (
                "reference.mkv", "execution.mkv", "checklist.csv",
            ))
            _write_color_video(reference, [red, green, blue, yellow])
            _write_color_video(execution, [red, green, blue, yellow, red])
            _write_annotations(checklist, [
                ["s001", "Prepare", 0, .5, ""], ["s002", "Pour", .5, 1, ""],
            ])
            checkpoints = [.25, .75, 1]
            original_bytes = [path.read_bytes() for path in (reference, execution, checklist)]

            def run_with_one_failure(*args, **kwargs):
                if kwargs["checkpoint_s"] == .75:
                    raise RuntimeError("Do not persist this exception message")
                return run_dtw_checkpoint(*args, **kwargs)

            with patch("src.guideme.baselines.dtw.run_dtw_checkpoint",
                       side_effect=run_with_one_failure) as runner:
                reports = run_dtw_replay(
                    reference, execution, checklist, reference_duration_s=1,
                    checkpoint_times_s=checkpoints, window_s=.5, fps=4, warp_penalty=.25,
                )
            self.assertEqual([call.kwargs["checkpoint_s"] for call in runner.call_args_list], [.25, .75, 1.0])
            for call in runner.call_args_list:
                self.assertEqual(call.args, (reference, execution, checklist))
                for name, expected in (("reference_duration_s", 1.0), ("window_s", .5),
                                       ("fps", 4.0), ("warp_penalty", .25)):
                    self.assertEqual(call.kwargs[name], expected)
                    self.assertIsInstance(call.kwargs[name], float)
            self.assertEqual([report["status"] for report in reports], ["ok", "error", "ok"])
            for report, end in zip(reports, [.25, .75, 1.0]):
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["encoder"], {"name": "classical"})
                self.assertEqual(report["checkpoint"], {
                    "end_s": end, "window_s": .5, "fps": 4.0, "warp_penalty": .25,
                })
            for report, expected_times, step_id in ((reports[0], [0.0], "s001"),
                                                   (reports[2], [.5, .75], "s002")):
                self.assertEqual(set(report), {"schema_version", "status", "encoder", "checkpoint", "result"})
                result = report["result"]
                self.assertEqual(result["reference_times_s"], [0.0, .25, .5, .75])
                self.assertEqual(result["execution_times_s"], expected_times)
                self.assertTrue(all(time < report["checkpoint"]["end_s"] for time in expected_times))
                self.assertEqual(result["candidate_step"]["step_id"], step_id)
                self.assertAlmostEqual(result["alignment"]["normalized_cost"], 0.0)
            self.assertIsNone(reports[1]["result"])
            self.assertEqual(reports[1]["error"], {"stage": "run_checkpoint", "type": "RuntimeError"})
            self.assertNotIn("Do not persist", json.dumps(reports, allow_nan=False))
            self.assertEqual(dtw_prediction_from_report(reports[1]), {
                "checkpoint_s": .75, "predicted_step_id": None,
            })
            self.assertEqual(checkpoints, [.25, .75, 1])
            self.assertIs(type(checkpoints[-1]), int)
            self.assertEqual([path.read_bytes() for path in (reference, execution, checklist)], original_bytes)

    def test_replay_validates_the_entire_grid_and_configuration_before_running(self):
        from src.guideme.baselines.dtw import run_dtw_replay

        missing = Path("nonexistent-replay-input.mkv")
        options = {"reference_duration_s": 1, "window_s": .5, "fps": 4}
        with patch("src.guideme.baselines.dtw.run_dtw_checkpoint") as runner:
            self.assertEqual(run_dtw_replay(missing, missing, missing,
                                           checkpoint_times_s=[], **options), [])
            for invalid in ((.25,), [.25, True], [.25, .25], [.75, .25], [.25, float("inf")]):
                with self.subTest(checkpoints=invalid):
                    with self.assertRaises(ValueError):
                        run_dtw_replay(missing, missing, missing,
                                       checkpoint_times_s=invalid, **options)
            for overrides in ({"window_s": 0}, {"encoder": "clip", "device": "cpu"}, {"device": "cpu"}):
                with self.subTest(configuration=overrides):
                    with self.assertRaises(ValueError):
                        run_dtw_replay(missing, missing, missing, checkpoint_times_s=[],
                                       **{**options, **overrides})
            runner.assert_not_called()

    def test_warp_penalty_reaches_both_encoders_and_records_only_nonzero_settings(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("requires optional clip dependencies")
        from src.guideme.baselines.dtw import run_dtw_checkpoint

        class ColorEncoder(torch.nn.Module):
            def encode_image(self, images):
                return images[:, :, 0, 0]

        # Identical red frames yield equal directions, so a positive penalty
        # favors diagonals; zero has tied endpoints spanning both labels.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference, execution, checklist = (base / name for name in (
                "reference.mkv", "execution.mkv", "checklist.csv",
            ))
            _write_color_video(reference, [(0, 0, 240)] * 4)
            _write_color_video(execution, [(0, 0, 240)] * 3)
            _write_annotations(checklist, [
                ["s001", "First action", 0, .25, ""],
                ["s002", "Second action", .25, 1, ""],
            ])
            arguments = dict(reference_duration_s=1, checkpoint_s=.75, window_s=.75, fps=4)
            for options in (
                {"encoder": "classical"},
                {"encoder": "clip", "clip_model": ColorEncoder().eval(), "device": "cpu"},
            ):
                with self.subTest(encoder=options["encoder"]):
                    default = run_dtw_checkpoint(reference, execution, checklist, **arguments, **options)
                    zero = run_dtw_checkpoint(reference, execution, checklist,
                                              **arguments, **options, warp_penalty=0.0)
                    self.assertEqual(zero, default)
                    self.assertNotIn("warp_penalty", default)
                    self.assertEqual(default["alignment"]["path"], [(0, 0), (1, 0), (2, 0)])
                    self.assertIsNone(default["candidate_step"])
                    self.assertEqual(default["unknown_reason"], "tied_reference_steps")

                    penalized = run_dtw_checkpoint(reference, execution, checklist,
                                                   **arguments, **options, warp_penalty=.5)
                    self.assertEqual(penalized["warp_penalty"], .5)
                    self.assertEqual(penalized["alignment"]["path"], [(0, 0), (1, 1), (2, 2)])
                    self.assertEqual(penalized["alignment"]["tied_endpoints"], [2, 3])
                    self.assertAlmostEqual(penalized["alignment"]["total_cost"], 0.0)
                    self.assertEqual(penalized["candidate_step"]["step_id"], "s002")
                    self.assertEqual(penalized["execution_times_s"], [0.0, .25, .5])
                    json.dumps(penalized, allow_nan=False)

            missing = base / "missing"
            with patch("src.guideme.baselines.dtw.load_reference_annotations",
                       side_effect=AssertionError("file access before penalty validation")), \
                    patch("src.guideme.baselines.dtw.sample_video_frames",
                          side_effect=AssertionError("sampling before penalty validation")):
                for invalid in (True, "0.5", -1, float("nan"), float("inf"), 10**400):
                    with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                        run_dtw_checkpoint(missing, missing, missing, **arguments, warp_penalty=invalid)

    def test_clip_checkpoint_reuses_causal_matching_and_preserves_classical_results(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("requires optional clip dependencies")
        from src.guideme.baselines.dtw import run_dtw_checkpoint

        class ColorEncoder(torch.nn.Module):
            def __init__(self, error=None):
                super().__init__()
                self.shapes = []
                self.error = error

            def encode_image(self, images):
                self.shapes.append(tuple(images.shape))
                if self.error is not None:
                    raise self.error
                # Distinct normalized RGB pixels yield distinct real features;
                # equal source colors retain equal directions across batches.
                return images[:, :, 0, 0]

        red, green, blue, yellow = (0, 0, 240), (0, 240, 0), (240, 0, 0), (0, 240, 240)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference, execution = base / "reference.mkv", base / "execution.mkv"
            annotations = base / "checklist.csv"
            _write_color_video(reference, [red, green, blue, yellow])
            _write_color_video(execution, [red, green, blue, yellow, red])
            _write_annotations(annotations, [
                ["s001", "Prepare ingredients", 0, .5, "Keep original text"],
                ["s002", "Pour mixture", .5, 1, ""],
            ])
            arguments = dict(reference_duration_s=1, checkpoint_s=.75, window_s=.5, fps=4)
            model = ColorEncoder().eval()
            result = run_dtw_checkpoint(
                reference, execution, annotations, **arguments,
                encoder="clip", clip_model=model, device="cpu", batch_size=3,
            )
            self.assertEqual(result["method"], "clip_subsequence_dtw")
            self.assertEqual(result["frame_size"], [224, 224])
            self.assertEqual(model.shapes, [
                (3, 3, 224, 224), (1, 3, 224, 224), (2, 3, 224, 224),
            ])
            self.assertFalse(model.training)
            self.assertEqual(result["reference_times_s"], [0, .25, .5, .75])
            self.assertEqual(result["execution_times_s"], [.25, .5])
            self.assertEqual(result["window_start_s"], .25)
            self.assertEqual(result["alignment"]["path"], [(0, 1), (1, 2)])
            self.assertEqual(result["alignment"]["tied_endpoints"], [2])
            self.assertAlmostEqual(result["alignment"]["normalized_cost"], 0.0)
            self.assertEqual(result["reference_time_s"], .5)
            self.assertEqual(result["tied_reference_times_s"], [.5])
            self.assertEqual(result["candidate_step"], {
                "step_id": "s002", "action": "Pour mixture", "start_s": .5,
                "end_s": 1.0, "requirements": "",
            })
            self.assertIsNone(result["unknown_reason"])
            json.dumps(result, allow_nan=False)

            legacy = run_classical_dtw_checkpoint(reference, execution, annotations, **arguments)
            self.assertEqual(set(result), set(legacy))
            self.assertEqual(run_dtw_checkpoint(reference, execution, annotations,
                                               **arguments, encoder="classical"), legacy)
            self.assertEqual(run_dtw_checkpoint(reference, execution, annotations,
                                               **arguments), legacy)

            _write_color_video(base / "repeated.mkv", [red, blue, red])
            _write_color_video(base / "red.mkv", [red])
            for rows, reason in (
                ([
                    ["s001", "First action", 0, .25, ""],
                    ["s002", "Second action", .5, .75, ""],
                ], "tied_reference_steps"),
                ([["s001", "First action", 0, .25, ""]], "unmapped_reference_time"),
            ):
                with self.subTest(unknown_reason=reason):
                    _write_annotations(annotations, rows)
                    tied = run_dtw_checkpoint(
                        base / "repeated.mkv", base / "red.mkv", annotations,
                        reference_duration_s=.75, checkpoint_s=.25, window_s=1, fps=4,
                        encoder="clip", clip_model=ColorEncoder().eval(), device="cpu",
                    )
                    self.assertEqual(tied["execution_times_s"], [0])
                    self.assertEqual(tied["alignment"]["tied_endpoints"], [0, 2])
                    self.assertEqual(tied["tied_reference_times_s"], [0, .5])
                    self.assertIsNone(tied["candidate_step"])
                    self.assertEqual(tied["unknown_reason"], reason)

            missing = base / "missing"
            for overrides in (
                {"encoder": "unsupported"}, {"clip_model": None}, {"device": None},
                {"clip_model": ColorEncoder()}, {"batch_size": 0}, {"batch_size": True},
            ):
                with self.subTest(preflight=overrides):
                    options = dict(encoder="clip", clip_model=model, device="cpu", batch_size=3)
                    options.update(overrides)
                    with self.assertRaises(ValueError):
                        run_dtw_checkpoint(missing, missing, missing, **arguments, **options)

            error = RuntimeError("actual encoder failure")
            with self.assertRaises(RuntimeError) as caught:
                run_dtw_checkpoint(
                    reference, execution, annotations, **arguments,
                    encoder="clip", clip_model=ColorEncoder(error).eval(), device="cpu",
                )
            self.assertIs(caught.exception, error)

    def test_real_frames_match_reference_step_without_future_execution_frames(self):
        # Distinct saturated colors have different hue bins and one shared
        # saturation bin: equal colors cost zero and differing colors cost .5.
        red, green, blue, yellow = (0, 0, 240), (0, 240, 0), (240, 0, 0), (0, 240, 240)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference = base / "arbitrary-reference.mkv"
            execution = base / "arbitrary-execution.mkv"
            annotations = base / "reference-checklist.csv"
            _write_color_video(reference, [red, green, blue, yellow])
            _write_color_video(execution, [red, green, blue, yellow, red])
            _write_annotations(annotations, [
                ["s001", "Prepare ingredients", 0, .5, "Keep the original text"],
                ["s002", "Pour mixture", .5, 1, ""],
            ])
            with patch(
                "src.guideme.baselines.dtw.sample_video_frames",
                wraps=sample_video_frames,
            ) as sampler:
                result = run_classical_dtw_checkpoint(
                    reference, execution, annotations,
                    reference_duration_s=1, checkpoint_s=.75, window_s=.5, fps=4,
                )

            # The full reference remains available; execution frames at .75
            # and later must not affect this checkpoint.
            self.assertEqual(len(sampler.call_args_list), 2)
            requested = []
            for call in sampler.call_args_list:
                supplied = dict(zip(("source_path", "start_s", "end_s"), call.args))
                supplied.update(call.kwargs)
                requested.append((
                    Path(supplied["source_path"]).resolve(), supplied["start_s"],
                    supplied["end_s"], supplied["fps"], supplied["size"],
                ))
            self.assertEqual(requested, [
                (reference.resolve(), 0, 1, 4, (64, 64)),
                (execution.resolve(), .25, .75, 4, (64, 64)),
            ])
            self.assertEqual(result["method"], "classical_subsequence_dtw")
            self.assertEqual(result["reference_path"], str(reference.resolve()))
            self.assertEqual(result["execution_path"], str(execution.resolve()))
            self.assertEqual(result["reference_annotations_path"], str(annotations.resolve()))
            for name, expected in (
                ("reference_duration_s", 1.0), ("checkpoint_s", .75),
                ("window_s", .5), ("window_start_s", .25), ("fps", 4.0),
            ):
                self.assertEqual(result[name], expected)
                self.assertIsInstance(result[name], float)
            self.assertEqual(result["frame_size"], [64, 64])
            self.assertEqual(result["reference_times_s"], [0.0, .25, .5, .75])
            self.assertEqual(result["execution_times_s"], [.25, .5])
            match = result["alignment"]
            self.assertEqual(match["path"], [(0, 1), (1, 2)])
            self.assertEqual(match["start_reference_index"], 1)
            self.assertEqual(match["end_reference_index"], 2)
            self.assertEqual(match["tied_endpoints"], [2])
            self.assertAlmostEqual(match["total_cost"], 0.0)
            self.assertAlmostEqual(match["normalized_cost"], 0.0)
            np.testing.assert_allclose(match["endpoint_costs"], [.5, .25, 0, .25], atol=1e-12)
            self.assertEqual(result["reference_time_s"], .5)
            self.assertEqual(result["tied_reference_times_s"], [.5])
            self.assertEqual(result["candidate_step"], {
                "step_id": "s002", "action": "Pour mixture", "start_s": .5,
                "end_s": 1.0, "requirements": "",
            })
            self.assertIsNone(result["unknown_reason"])
            json.dumps(result, allow_nan=False)

            clipped = run_classical_dtw_checkpoint(
                reference, execution, annotations,
                reference_duration_s=1, checkpoint_s=.25, window_s=2, fps=4,
            )
            self.assertEqual(clipped["window_start_s"], 0.0)
            self.assertEqual(clipped["execution_times_s"], [0.0])
            self.assertEqual(clipped["reference_time_s"], 0.0)
            self.assertEqual(clipped["candidate_step"]["step_id"], "s001")

    def test_all_best_endpoints_control_unknowns_and_failures_are_preserved(self):
        red, blue, green = (0, 0, 240), (240, 0, 0), (0, 240, 0)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            reference, execution = base / "reference.mkv", base / "execution.mkv"
            annotations = base / "checklist.csv"
            _write_color_video(reference, [red, blue, red])
            _write_color_video(execution, [red])
            cases = (
                ("same step", [["s001", "One action", 0, .75, "authored note"]], "s001", None),
                ("different steps", [
                    ["s001", "First action", 0, .25, ""],
                    ["s002", "Second action", .5, .75, ""],
                ], None, "tied_reference_steps"),
                ("one gap", [["s001", "First action", 0, .25, ""]], None, "unmapped_reference_time"),
                ("overlap", [
                    ["s001", "First action", 0, .75, ""],
                    ["s002", "Second action", .5, .75, ""],
                ], None, "unmapped_reference_time"),
            )
            for name, rows, step_id, reason in cases:
                with self.subTest(case=name):
                    _write_annotations(annotations, rows)
                    result = run_classical_dtw_checkpoint(
                        reference, execution, annotations,
                        reference_duration_s=.75, checkpoint_s=.25, window_s=1, fps=4,
                    )
                    self.assertEqual(result["alignment"]["tied_endpoints"], [0, 2])
                    self.assertEqual(result["reference_time_s"], 0.0)
                    self.assertEqual(result["tied_reference_times_s"], [0.0, .5])
                    self.assertEqual(result["unknown_reason"], reason)
                    if step_id is None:
                        self.assertIsNone(result["candidate_step"])
                    else:
                        self.assertEqual(result["candidate_step"], {
                            "step_id": "s001", "action": "One action", "start_s": 0.0,
                            "end_s": .75, "requirements": "authored note",
                        })
                    json.dumps(result, allow_nan=False)

            # Even a nonzero-cost raw match stays a candidate when every tied
            # endpoint maps to the same step; confidence policy is not applied.
            _write_annotations(annotations, [["s001", "One action", 0, .75, ""]])
            green_execution = base / "green.mkv"
            _write_color_video(green_execution, [green])
            poor_match = run_classical_dtw_checkpoint(
                reference, green_execution, annotations,
                reference_duration_s=.75, checkpoint_s=.25, window_s=1, fps=4,
            )
            self.assertAlmostEqual(poor_match["alignment"]["normalized_cost"], .5)
            self.assertEqual(poor_match["alignment"]["tied_endpoints"], [0, 1, 2])
            self.assertEqual(poor_match["candidate_step"]["step_id"], "s001")
            self.assertIsNone(poor_match["unknown_reason"])

            missing = base / "missing"
            for parameter, invalid in (
                ("reference_duration_s", True), ("reference_duration_s", 10**400),
                ("checkpoint_s", "1"), ("window_s", float("inf")), ("fps", 0),
            ):
                with self.subTest(parameter=parameter, invalid=repr(invalid)):
                    arguments = dict(reference_duration_s=.75, checkpoint_s=.25, window_s=1, fps=4)
                    arguments[parameter] = invalid
                    with self.assertRaises(ValueError):
                        run_classical_dtw_checkpoint(missing, missing, missing, **arguments)

            for name, rows, duration in (
                ("empty checklist", [], .75),
                ("interval beyond supplied duration", [["s001", "Action", 0, .75, ""]], .5),
            ):
                with self.subTest(case=name):
                    _write_annotations(annotations, rows)
                    with self.assertRaises(ValueError):
                        run_classical_dtw_checkpoint(
                            missing, missing, annotations,
                            reference_duration_s=duration, checkpoint_s=.25, window_s=1,
                        )

            _write_annotations(annotations, [["s001", "Action", 0, .75, ""]])
            # An empty causal sample is a sampling error, not an unknown step.
            with self.assertRaises(ValueError):
                run_classical_dtw_checkpoint(
                    reference, execution, annotations,
                    reference_duration_s=.75, checkpoint_s=2, window_s=.25, fps=4,
                )


if __name__ == "__main__":
    unittest.main()
