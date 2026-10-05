"""Behavior tests for matching an execution window inside a reference."""

import unittest

import numpy as np

from src.guideme.alignment import align_feature_window, subsequence_dtw


class SubsequenceDtwTests(unittest.TestCase):
    def test_middle_reference_match_and_exact_endpoint_ties(self):
        cost = np.array([
            [9, 1, 8, 8, 9],
            [9, 8, 2, 8, 9],
            [9, 8, 8, 3, 9],
        ])

        match = subsequence_dtw(cost)

        # The matched diagonal costs 1 + 2 + 3; reference ends cost
        # 27, 17, 11, 6, 15 before normalization by three execution rows.
        self.assertEqual(match["path"], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(match["start_reference_index"], 1)
        self.assertEqual(match["end_reference_index"], 3)
        self.assertEqual(match["total_cost"], 6.0)
        self.assertEqual(match["normalized_cost"], 2.0)
        self.assertEqual(match["endpoint_costs"], [9.0, 17 / 3, 11 / 3, 2.0, 5.0])
        self.assertEqual(match["tied_endpoints"], [3])

        tied = subsequence_dtw([[0, 0], [0, 0]])
        self.assertEqual(tied["tied_endpoints"], [0, 1])
        self.assertEqual(tied["end_reference_index"], 0)
        self.assertEqual(tied["path"], [(0, 0), (1, 0)])
        self.assertEqual(tied["endpoint_costs"], [0.0, 0.0])

        diagonal_tie = subsequence_dtw([[0, 0, 9], [9, 0, 9]])
        self.assertEqual(diagonal_tie["path"], [(0, 0), (1, 1)])

    def test_invalid_cost_matrices_raise_value_error(self):
        cases = (
            ("empty", []),
            ("zero rows", np.empty((0, 2))),
            ("zero columns", np.empty((2, 0))),
            ("one dimensional", [0, 1]),
            ("three dimensional", np.zeros((1, 1, 1))),
            ("negative", [[-1]]),
            ("NaN", [[float("nan")]]),
            ("infinity", [[float("inf")]]),
            ("accumulated overflow", [[1e308], [1e308], [1e308]]),
        )
        for name, cost in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    subsequence_dtw(cost)

    def test_feature_window_uses_cosine_cost_without_changing_inputs(self):
        reference = np.array([[-2.0, 0], [4.0, 0], [0, 7.0], [0, -11.0]])
        execution = np.array([[3.0, 0], [0, 5.0]])
        expected = {
            "path": [(0, 1), (1, 2)],
            "start_reference_index": 1,
            "end_reference_index": 2,
            "total_cost": 0.0,
            "normalized_cost": 0.0,
            "endpoint_costs": [1.5, 0.5, 0.0, 1.0],
            "tied_endpoints": [2],
        }
        # Unit directions produce cosine rows [2, 0, 1, 1] and [1, 1, 0, 2].
        for reference_scale, execution_scale in ((1, 1), (1e300, 1e-300)):
            with self.subTest(scales=(reference_scale, execution_scale)):
                reference_input = reference * reference_scale
                execution_input = execution * execution_scale
                reference_before = reference_input.copy()
                execution_before = execution_input.copy()

                self.assertEqual(
                    align_feature_window(reference_input, execution_input), expected
                )
                np.testing.assert_array_equal(reference_input, reference_before)
                np.testing.assert_array_equal(execution_input, execution_before)

    def test_invalid_feature_windows_raise_value_error(self):
        valid = np.array([[1.0, 0]])
        cases = (
            ("empty", np.empty((0, 2))),
            ("one dimensional", [1, 0]),
            ("nonfinite", [[float("nan"), 1]]),
            ("zero row", [[0, 0]]),
            ("complex", [[1j, 0]]),
        )
        for name, invalid in cases:
            for role in ("reference", "execution"):
                with self.subTest(case=name, role=role):
                    reference = invalid if role == "reference" else valid
                    execution = invalid if role == "execution" else valid
                    with self.assertRaises(ValueError):
                        align_feature_window(reference, execution)
        with self.subTest(case="different feature dimensions"):
            with self.assertRaises(ValueError):
                align_feature_window(valid, [[1, 0, 0]])

    def test_warp_penalty_favors_diagonal_and_charges_actual_warp_moves(self):
        # Omitting either warp charge or charging entry/diagonal moves changes
        # these hand-derived paths and objective values.
        cost = np.array([
            [9, 0, .2, 9, 9],
            [9, 0, 9, .2, 9],
            [9, 0, 9, 9, .2],
        ])
        before = cost.copy()
        default = subsequence_dtw(cost)
        self.assertEqual(default["path"], [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(subsequence_dtw(cost, warp_penalty=0.0), default)

        penalized = subsequence_dtw(cost, warp_penalty=.5)
        self.assertEqual(penalized["path"], [(0, 2), (1, 3), (2, 4)])
        self.assertAlmostEqual(penalized["total_cost"], .6)
        self.assertAlmostEqual(penalized["normalized_cost"], .2)
        self.assertAlmostEqual(penalized["endpoint_costs"][1], 1 / 3)
        np.testing.assert_array_equal(cost, before)

        vertical = subsequence_dtw([[1], [2], [3]], warp_penalty=.5)
        self.assertEqual(vertical["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(vertical["total_cost"], 7.0)  # 1+2+3 and two up moves.
        self.assertEqual(vertical["normalized_cost"], 7 / 3)
        horizontal = subsequence_dtw([
            [0, 5, 5, 5], [5, 0, 0, 5], [5, 5, 5, 0],
        ], warp_penalty=.5)
        self.assertEqual(horizontal["path"], [(0, 0), (1, 1), (1, 2), (2, 3)])
        self.assertEqual(horizontal["total_cost"], .5)  # One left move only.
        self.assertEqual(horizontal["normalized_cost"], .5 / 3)

    def test_warp_penalty_preserves_free_boundaries_ties_and_rejects_invalid_values(self):
        free_start = subsequence_dtw([[9, 9, 2, 0]], warp_penalty=10)
        self.assertEqual(free_start["path"], [(0, 3)])
        self.assertEqual(free_start["total_cost"], 0.0)
        tied = subsequence_dtw([[0, 0, 0], [0, 0, 0]], warp_penalty=.5)
        self.assertEqual(tied["tied_endpoints"], [1, 2])
        self.assertEqual(tied["path"], [(0, 0), (1, 1)])
        predecessor_tie = subsequence_dtw([[.5, 0, 9], [9, 0, 9]], warp_penalty=.5)
        self.assertEqual(predecessor_tie["path"], [(0, 0), (1, 1)])
        with self.subTest(case="unused transition overflow"):
            finite = subsequence_dtw([[0, 0], [0, 0]], warp_penalty=1e308)
            self.assertEqual(finite["path"], [(0, 0), (1, 1)])
            self.assertEqual(finite["total_cost"], 0.0)
            self.assertEqual(finite["endpoint_costs"], [5e307, 0.0])

        for invalid in (True, "0.5", -1, float("nan"), float("inf"), 10**400, None):
            with self.subTest(penalty=repr(invalid)), self.assertRaises(ValueError):
                subsequence_dtw([[0]], warp_penalty=invalid)
        with self.assertRaises(ValueError):
            subsequence_dtw([[0], [0], [0]], warp_penalty=1e308)

    def test_feature_window_forwards_warp_penalty_without_changing_features(self):
        # Identical directions give all-zero cosine costs. Dropping the
        # penalty would retain a vertical path rather than this diagonal.
        reference = np.array([[1.0, 0]] * 4)
        execution = np.array([[1.0, 0]] * 3)
        reference_before, execution_before = reference.copy(), execution.copy()
        default = align_feature_window(reference, execution)
        self.assertEqual(default["path"], [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(default["tied_endpoints"], [0, 1, 2, 3])
        self.assertEqual(align_feature_window(reference, execution, warp_penalty=0.0), default)
        self.assertEqual(align_feature_window(reference, execution, warp_penalty=.5), {
            "path": [(0, 0), (1, 1), (2, 2)],
            "start_reference_index": 0, "end_reference_index": 2,
            "total_cost": 0.0, "normalized_cost": 0.0,
            "endpoint_costs": [1 / 3, .5 / 3, 0.0, 0.0],
            "tied_endpoints": [2, 3],
        })
        with self.assertRaises(ValueError):
            align_feature_window(reference, execution, warp_penalty=-1)
        np.testing.assert_array_equal(reference, reference_before)
        np.testing.assert_array_equal(execution, execution_before)


if __name__ == "__main__":
    unittest.main()
