"""The shared runner selects a task cohort and scores every pair."""

import unittest

import numpy as np

from scripts.task_runner import summarize_pairs
from scripts.task_spec import TaskSpec, matching_rows


def _unit_rows() -> np.ndarray:
    features = np.eye(2, dtype=np.float32)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


class MatchingRowsTests(unittest.TestCase):
    def test_keeps_exact_step_order(self):
        spec = TaskSpec(
            task="ReplaceSIMCard",
            step_labels=("open", "insert"),
            expected_videos=2,
            artifact_stem="sim",
        )
        rows = [
            {"youtube_id": "b", "task": "ReplaceSIMCard", "steps": [{"label": "open"}, {"label": "insert"}]},
            {"youtube_id": "a", "task": "ReplaceSIMCard", "steps": [{"label": "open"}, {"label": "insert"}]},
            {"youtube_id": "c", "task": "ReplaceSIMCard", "steps": [{"label": "insert"}, {"label": "open"}]},
            {"youtube_id": "d", "task": "BoilNoodles", "steps": [{"label": "open"}, {"label": "insert"}]},
        ]

        matched = matching_rows(rows, spec)

        self.assertEqual([row["youtube_id"] for row in matched], ["a", "b"])

    def test_rejects_the_wrong_count(self):
        spec = TaskSpec("ReplaceSIMCard", ("open",), 48, "sim")
        with self.assertRaises(RuntimeError):
            matching_rows([], spec)


class PairwiseCutoffTests(unittest.TestCase):
    def test_identical_videos_have_zero_distance(self):
        features = [_unit_rows(), _unit_rows()]
        labels = [np.array([0, 1], dtype=np.int8), np.array([0, 1], dtype=np.int8)]

        summary, matrix = summarize_pairs(features, labels, ("open", "insert"), ["a", "b"], "classical")

        self.assertEqual(summary["n_pairs"], 1)
        self.assertAlmostEqual(summary["global"]["mean"], 0.0)
        self.assertAlmostEqual(summary["steps"]["insert"]["mean"], 0.0)
        self.assertAlmostEqual(matrix[0, 1], 0.0)

    def test_replace_simcard_paths_match_the_existing_files(self):
        from scripts.replace_simcard import SPEC

        self.assertEqual(SPEC.summary_path("classical").name, "sim_classical_dtw_summary.json")
        self.assertEqual(SPEC.summary_path("clip").name, "sim_clip_dtw_summary.json")
        self.assertEqual(SPEC.reference_cache("classical").name, "sim_classical_features.npz")
        self.assertEqual(SPEC.expected_videos, 48)
        self.assertEqual(len(SPEC.step_labels), 3)
