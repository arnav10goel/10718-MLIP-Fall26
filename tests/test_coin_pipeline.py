"""Focused checks for COIN selection, media, and offline scoring."""

import hashlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np

from scripts.coin_tasks import select_matching_cohort
from scripts.prepare_coin import cohort_manifest_rows
from scripts.score_coin_clip import score_clip_reference_cohort
from scripts.score_coin_reference import score_reference_cohort
from scripts.validate_coin_media import inventory_task_media


TASK = "ExampleTask"


def coin_entry(step_ids, subset="training"):
    return {
        "class": TASK,
        "recipe_type": 99,
        "subset": subset,
        "duration": 3.0,
        "start": 0.0,
        "end": 3.0,
        "video_url": "https://example.org/video",
        "annotation": [
            {"id": step_id, "label": f"step {index}", "segment": [index, index + 1]}
            for index, step_id in enumerate(step_ids)
        ],
    }


def score_row(video_id, path, subset="training"):
    return {
        "youtube_id": video_id,
        "task": TASK,
        "subset": subset,
        "video_path": str(path),
        "steps": [
            {"id": 10, "label": "first", "start": 0.0, "end": 1.0},
            {"id": 11, "label": "second", "start": 1.0, "end": 2.0},
        ],
    }


def make_tiny_video(path):
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "16x16", "-framerate", "1",
            "-i", "pipe:0", "-frames:v", "2", "-c:v", "mpeg4", str(path),
        ],
        input=bytes([255, 0, 0]) * 16 * 16 * 2,
        capture_output=True,
        check=True,
    )


class CoinPipelineTests(unittest.TestCase):
    def test_cohort_uses_ordered_steps_and_keeps_exclusions(self):
        database = {
            "ref": coin_entry([10, 11, 11]),
            "test": coin_entry([10, 11, 11], "testing"),
            "missing": coin_entry([10, 11, 11]),
            "different": coin_entry([10, 11], "testing"),
        }

        cohort = select_matching_cohort(
            database, TASK, "ref", {"ref", "test", "different"}
        )

        self.assertEqual(cohort["step_ids"], ["10", "11", "11"])
        self.assertEqual(cohort["training_ids"], ["ref"])
        self.assertEqual(cohort["testing_ids"], ["test"])
        self.assertEqual(cohort["excluded"], [
            {"video_id": "different", "reason": "different_steps"},
            {"video_id": "missing", "reason": "unavailable_media"},
        ])

    def test_inventory_decodes_media_and_rejects_bad_hash_or_video(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            video_dir = root / "videos" / TASK
            video_dir.mkdir(parents=True)
            good = video_dir / "good.mp4"
            make_tiny_video(good)
            (video_dir / "bad.mp4").write_bytes(b"not a video")
            expected_hash = hashlib.sha256(good.read_bytes()).hexdigest()
            database = {
                "good": coin_entry([10, 11]),
                "bad": coin_entry([10, 11], "testing"),
                "missing": coin_entry([10, 11], "testing"),
            }

            inventory = inventory_task_media(database, TASK, root, {"good": expected_hash})

            self.assertEqual(inventory["usable_paths"], {"good": good})
            records = {record["video_id"]: record for record in inventory["records"]}
            self.assertEqual(set(records), set(database))
            self.assertEqual(records["good"]["sha256"], expected_hash)
            self.assertEqual(records["good"]["decoded_frame_count"], 2)
            self.assertIn("decode_error", records["bad"]["issues"])
            self.assertEqual(records["missing"]["issues"], ["missing_media"])

            mismatched = inventory_task_media(database, TASK, root, {"good": "0" * 64})
            self.assertNotIn("good", mismatched["usable_paths"])
            self.assertIn("checksum_mismatch", next(
                record["issues"] for record in mismatched["records"]
                if record["video_id"] == "good"
            ))

    def test_manifest_uses_each_videos_own_annotation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            reference = coin_entry([10, 11])
            comparison = coin_entry([10, 11], "testing")
            comparison["annotation"][0]["segment"] = [0.25, 0.75]
            database = {"ref": reference, "test": comparison}
            cohort = {
                "training_ids": ["ref"], "testing_ids": ["test"]
            }

            rows = cohort_manifest_rows(database, cohort, {
                "ref": root / "ref.mp4", "test": root / "test.mp4"
            })

            self.assertEqual([row["youtube_id"] for row in rows], ["ref", "test"])
            self.assertEqual(rows[0]["steps"][0]["start"], 0)
            self.assertEqual(rows[1]["steps"][0]["start"], 0.25)
            self.assertEqual(rows[1]["steps"][0]["end"], 0.75)

    def test_scorer_reports_step_costs_and_continues_after_sampling_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            rows = [
                score_row("ref", root / "ref.mp4"),
                score_row("bad", root / "bad.mp4", "testing"),
                score_row("test", root / "test.mp4", "testing"),
            ]

            def sample(item):
                if item["youtube_id"] == "bad":
                    raise RuntimeError("cannot sample")
                vectors = [[1.0, 0.0], [1.0, 0.0]]
                if item["youtube_id"] == "test":
                    vectors[1] = [-1.0, 0.0]
                return np.array(vectors), np.array([0.25, 1.25])

            result = score_reference_cohort(rows, "ref", sample_features=sample)

            self.assertEqual([score["video_id"] for score in result["scores"]], ["test"])
            self.assertEqual(result["scores"][0]["global_distance"], 1.0)
            self.assertEqual(result["scores"][0]["step_distances"], [0.0, 2.0])
            self.assertEqual(result["failures"], [{
                "video_id": "bad", "split": "testing",
                "reason": "sampling_error", "detail": "cannot sample",
            }])

    def test_uncovered_step_is_reported_as_null(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            rows = [
                score_row("ref", root / "ref.mp4"),
                score_row("test", root / "test.mp4", "testing"),
            ]
            for row in rows:
                for step in row["steps"]:
                    step["start"], step["end"] = 0.0, 2.0

            result = score_reference_cohort(
                rows, "ref",
                sample_features=lambda _: (
                    np.array([[1.0, 0.0], [1.0, 0.0]]),
                    np.array([0.25, 1.25]),
                ),
            )

            score = result["scores"][0]
            self.assertEqual(score["step_distances"], [None, 0.0])
            self.assertEqual(score["step_issues"], [
                {"step_index": 0, "reason": "no_labeled_frames"}
            ])

    def test_clip_adapter_loads_once_and_uses_embeddings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            rows = [
                score_row("ref", root / "ref.mp4"),
                score_row("test", root / "test.mp4", "testing"),
            ]
            calls = []
            model = object()

            def load_clip():
                calls.append("load")
                return model, "cpu"

            def sample_frames(item):
                calls.append(("sample", item["youtube_id"]))
                return item["youtube_id"], np.array([0.25, 1.25])

            def embed_frames(actual_model, device, frames):
                self.assertIs(actual_model, model)
                self.assertEqual(device, "cpu")
                calls.append(("embed", frames))
                vectors = [[1.0, 0.0], [1.0, 0.0]]
                if frames == "test":
                    vectors[1] = [-1.0, 0.0]
                return np.array(vectors)

            clip = SimpleNamespace(
                load_clip=load_clip,
                sample_frames=sample_frames,
                embed_frames=embed_frames,
            )
            with patch("scripts.score_coin_clip.import_module", return_value=clip):
                result = score_clip_reference_cohort(rows, "ref")

            self.assertEqual(calls, [
                "load", ("sample", "ref"), ("embed", "ref"),
                ("sample", "test"), ("embed", "test"),
            ])
            self.assertEqual(result["scores"][0]["global_distance"], 1.0)
            self.assertEqual(result["scores"][0]["step_distances"], [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
