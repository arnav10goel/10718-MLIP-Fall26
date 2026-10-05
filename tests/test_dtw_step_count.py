"""Behavior tests for the number of steps in classical pairwise DTW."""

import unittest

import numpy as np

from scripts.classical_pairwise_dtw import dtw_pair, open_end_alignment


class DtwStepCountTests(unittest.TestCase):
    def test_two_steps_keep_unannotated_frames_out_of_step_totals(self):
        cost = np.array(
            [[1.0, 100.0, 100.0], [100.0, 2.0, 100.0], [100.0, 100.0, 3.0]]
        )
        labels_a = np.array([-1, 0, 1], dtype=np.int8)
        labels_b = np.array([0, -1, 1], dtype=np.int8)

        global_mean, sum_a, count_a, sum_b, count_b = dtw_pair(
            cost, labels_a, labels_b, n_steps=2
        )

        self.assertEqual(global_mean, 2.0)
        np.testing.assert_array_equal(sum_a, [2.0, 3.0])
        np.testing.assert_array_equal(count_a, [1.0, 1.0])
        np.testing.assert_array_equal(sum_b, [1.0, 3.0])
        np.testing.assert_array_equal(count_b, [1.0, 1.0])

    def test_four_steps_return_four_totals_per_side(self):
        cost = np.full((4, 4), 100.0)
        np.fill_diagonal(cost, [1.0, 2.0, 3.0, 4.0])
        labels = np.array([0, 1, 2, 3], dtype=np.int8)

        global_mean, sum_a, count_a, sum_b, count_b = dtw_pair(
            cost, labels, labels, n_steps=4
        )

        self.assertEqual(global_mean, 2.5)
        for sums, counts in ((sum_a, count_a), (sum_b, count_b)):
            np.testing.assert_array_equal(sums, [1.0, 2.0, 3.0, 4.0])
            np.testing.assert_array_equal(counts, [1.0, 1.0, 1.0, 1.0])

    def test_legacy_three_argument_call_uses_three_steps(self):
        cost = np.array(
            [[1.0, 100.0, 100.0], [100.0, 2.0, 100.0], [100.0, 100.0, 3.0]]
        )
        labels = np.array([0, 1, 2], dtype=np.int8)

        global_mean, sum_a, count_a, sum_b, count_b = dtw_pair(cost, labels, labels)

        self.assertEqual(global_mean, 2.0)
        for sums, counts in ((sum_a, count_a), (sum_b, count_b)):
            np.testing.assert_array_equal(sums, [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(counts, [1.0, 1.0, 1.0])

    def test_rejects_nonpositive_step_count(self):
        cost = np.array([[1.0]])
        labels = np.array([0], dtype=np.int8)

        for n_steps in (0, -1):
            with self.subTest(n_steps=n_steps):
                with self.assertRaises(ValueError):
                    dtw_pair(cost, labels, labels, n_steps=n_steps)

    def test_rejects_label_outside_requested_step_count_on_either_side(self):
        cost = np.array([[1.0]])
        valid = np.array([0], dtype=np.int8)
        invalid = np.array([2], dtype=np.int8)

        for labels_a, labels_b in ((invalid, valid), (valid, invalid)):
            with self.subTest(labels_a=labels_a[0], labels_b=labels_b[0]):
                with self.assertRaises(ValueError):
                    dtw_pair(cost, labels_a, labels_b, n_steps=2)


class OpenEndAlignmentTests(unittest.TestCase):
    def test_prefix_stops_before_the_unmatched_reference_suffix(self):
        cost = np.array(
            [
                [0.0, 5.0, 5.0],
                [5.0, 0.0, 5.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1], dtype=np.int8)

        path_mean, step_sum, step_count, end_index, last_step = open_end_alignment(
            cost, labels, n_steps=2
        )

        self.assertEqual(end_index, 1)
        self.assertEqual(last_step, 0)
        self.assertEqual(path_mean, 0.0)
        self.assertEqual(step_count[1], 0.0)
        self.assertGreater(step_count[0], 0.0)


if __name__ == "__main__":
    unittest.main()
