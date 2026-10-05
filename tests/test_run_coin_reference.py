"""End-to-end behavior tests for the repeatable COIN DTW command."""

import hashlib
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import imageio_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = "ExampleTask"


def make_tiny_video(path: Path) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", "16x16",
            "-framerate", "5",
            "-i", "pipe:0",
            "-frames:v", "10",
            "-c:v", "mpeg4",
            str(path),
        ],
        input=bytes([255, 0, 0]) * 16 * 16 * 10,
        capture_output=True,
        check=True,
    )


def coin_entry(video_id: str, subset: str, step_ids: tuple[int, ...]) -> dict:
    return {
        "class": TASK,
        "recipe_type": 99,
        "subset": subset,
        "duration": 2.0,
        "start": 0.0,
        "end": 2.0,
        "video_url": f"https://example.org/{video_id}",
        "annotation": [
            {
                "id": step_id,
                "label": f"step {index + 1}",
                "segment": [float(index), float(index + 1)],
            }
            for index, step_id in enumerate(step_ids)
        ],
    }


def make_fixture(root: Path, *, include_comparison: bool = True) -> tuple[Path, Path, str]:
    raw_dir = root / "raw"
    video_dir = root / "videos" / TASK
    raw_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    reference_video = video_dir / "ref.mp4"
    make_tiny_video(reference_video)
    media_bytes = reference_video.read_bytes()
    if include_comparison:
        (video_dir / "compare.mp4").write_bytes(media_bytes)

    database = {
        "ref": coin_entry("ref", "training", (10, 11)),
        "missing": coin_entry("missing", "testing", (10, 11)),
        "different": coin_entry("different", "testing", (10, 12)),
    }
    if include_comparison:
        database["compare"] = coin_entry("compare", "testing", (10, 11))
    coin_path = raw_dir / "COIN.json"
    coin_path.write_text(json.dumps({"database": database}) + "\n")
    return coin_path, reference_video, hashlib.sha256(media_bytes).hexdigest()


def run_command(*args: str, data_root_env: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("GUIDEME_DATA_ROOT", None)
    if data_root_env is not None:
        env["GUIDEME_DATA_ROOT"] = data_root_env
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_coin_reference", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class RunCoinReferenceTests(unittest.TestCase):
    def test_clip_encoder_uses_selected_cohort_and_writes_method_metadata(self):
        from scripts import run_coin_reference

        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir).resolve()
            root = base / "coin-data"
            make_fixture(root)
            output = base / "clip-results.json"
            clip_adapter = ModuleType("scripts.score_coin_clip")

            def score_clip(rows, reference_id):
                self.assertEqual(reference_id, "ref")
                self.assertEqual([row["youtube_id"] for row in rows], ["ref", "compare"])
                self.assertEqual([row["subset"] for row in rows], ["training", "testing"])
                self.assertEqual([row["video_path"] for row in rows], [
                    str(root / "videos" / TASK / "ref.mp4"),
                    str(root / "videos" / TASK / "compare.mp4"),
                ])
                return {
                    "task": TASK,
                    "reference_id": "ref",
                    "reference_frames": 2,
                    "steps": [{"id": 10, "label": "step 1"}, {"id": 11, "label": "step 2"}],
                    "scores": [{
                        "video_id": "compare",
                        "split": "testing",
                        "sampled_frames": 2,
                        "global_distance": 0.25,
                        "step_distances": [0.2, 0.3],
                        "step_path_counts": [
                            {"reference": 1, "comparison": 1},
                            {"reference": 1, "comparison": 1},
                        ],
                        "step_issues": [],
                    }],
                    "failures": [],
                }

            clip_adapter.score_clip_reference_cohort = score_clip
            stdout = StringIO()
            with patch.dict(sys.modules, {"scripts.score_coin_clip": clip_adapter}), \
                    patch.object(run_coin_reference, "score_reference_cohort", side_effect=AssertionError("classical scorer called")), \
                    redirect_stdout(stdout):
                result = run_coin_reference.main([
                    "--task", TASK,
                    "--reference-id", "ref",
                    "--data-root", str(root),
                    "--encoder", "clip",
                    "--output", str(output),
                ])

            self.assertEqual(result, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["method"], {
                "name": "clip_reference_dtw",
                "encoder": "clip",
                "model": "ViT-B-32",
                "pretrained": "openai",
                "fps": 5.0,
                "feature_cache": False,
                "uses_comparison_annotations": True,
                "hash_check": "observed_only",
            })
            self.assertEqual(report["summary"], {
                "annotated_videos": 4,
                "usable_media": 2,
                "matching_training": 1,
                "matching_testing": 1,
                "comparisons_scored": 1,
                "comparison_failures": 0,
                "cohort_exclusions": 2,
            })
            self.assertEqual(report["scoring"]["scores"][0]["global_distance"], 0.25)
            self.assertIn("Wrote 1 scores", stdout.getvalue())

    def test_clip_scoring_failure_does_not_write_report(self):
        from scripts import run_coin_reference

        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir).resolve()
            root = base / "coin-data"
            make_fixture(root)
            output = base / "clip-results.json"
            clip_adapter = ModuleType("scripts.score_coin_clip")

            def fail_scoring(rows, reference_id):
                raise RuntimeError("synthetic CLIP failure")

            clip_adapter.score_clip_reference_cohort = fail_scoring
            with patch.dict(sys.modules, {"scripts.score_coin_clip": clip_adapter}), \
                    patch.object(run_coin_reference, "score_reference_cohort", side_effect=AssertionError("classical scorer called")), \
                    redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "synthetic CLIP failure"):
                    run_coin_reference.main([
                        "--task", TASK,
                        "--reference-id", "ref",
                        "--data-root", str(root),
                        "--encoder", "clip",
                        "--output", str(output),
                    ])

            self.assertFalse(output.exists())

    def test_command_saves_scores_hashes_and_all_task_dispositions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir).resolve()
            root = base / "coin-data"
            coin_path, _, media_hash = make_fixture(root)
            output = base / "results.json"

            completed = run_command(
                "--task", TASK,
                "--reference-id", "ref",
                "--data-root", str(root),
                "--output", str(output),
                data_root_env=str(base / "wrong-root"),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(
                output.read_text(),
                parse_constant=lambda value: self.fail(f"Nonfinite JSON value: {value}"),
            )
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["task"], TASK)
            self.assertEqual(report["reference_id"], "ref")
            self.assertEqual(report["data_root"], str(root))
            self.assertEqual(report["annotation_source"], {
                "path": str(coin_path),
                "sha256": hashlib.sha256(coin_path.read_bytes()).hexdigest(),
            })
            self.assertEqual(report["method"], {
                "name": "classical_reference_dtw",
                "fps": 5.0,
                "feature_cache": False,
                "uses_comparison_annotations": True,
                "hash_check": "observed_only",
            })
            self.assertEqual(report["summary"], {
                "annotated_videos": 4,
                "usable_media": 2,
                "matching_training": 1,
                "matching_testing": 1,
                "comparisons_scored": 1,
                "comparison_failures": 0,
                "cohort_exclusions": 2,
            })
            records = {record["video_id"]: record for record in report["inventory_records"]}
            self.assertEqual(set(records), {"ref", "compare", "missing", "different"})
            for video_id in ("ref", "compare"):
                self.assertTrue(records[video_id]["usable"])
                self.assertEqual(records[video_id]["sha256"], media_hash)
                self.assertIsNone(records[video_id]["expected_sha256"])
            self.assertEqual(records["missing"]["issues"], ["missing_media"])
            self.assertEqual(records["different"]["issues"], ["missing_media"])
            self.assertEqual(report["cohort"]["training_ids"], ["ref"])
            self.assertEqual(report["cohort"]["testing_ids"], ["compare"])
            self.assertEqual(
                {(item["video_id"], item["reason"]) for item in report["cohort"]["excluded"]},
                {("missing", "unavailable_media"), ("different", "different_steps")},
            )
            self.assertEqual(report["scoring"]["failures"], [])
            self.assertEqual(len(report["scoring"]["scores"]), 1)
            score = report["scoring"]["scores"][0]
            self.assertEqual(score["video_id"], "compare")
            self.assertAlmostEqual(score["global_distance"], 0.0, places=6)
            for distance in score["step_distances"]:
                self.assertAlmostEqual(distance, 0.0, places=6)

    def test_environment_data_root_and_zero_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir).resolve()
            root = base / "coin-data"
            make_fixture(root, include_comparison=False)
            output = base / "reference-only.json"

            completed = run_command(
                "--task", TASK,
                "--reference-id", "ref",
                "--output", str(output),
                data_root_env=str(root),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["data_root"], str(root))
            self.assertEqual(report["summary"]["comparisons_scored"], 0)
            self.assertEqual(report["scoring"]["scores"], [])
            self.assertEqual(report["cohort"]["training_ids"], ["ref"])

    def test_existing_report_is_preserved_before_loading_annotations(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir).resolve()
            root = base / "coin-data"
            root.mkdir()
            output = base / "results.json"
            output.write_text("sentinel")

            completed = run_command(
                "--task", TASK,
                "--reference-id", "ref",
                "--data-root", str(root),
                "--output", str(output),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(output.read_text(), "sentinel")
            self.assertIn("exist", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
