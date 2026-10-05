"""Current-step scoring with hand-derived groups and explicit unknowns."""

from copy import deepcopy
import unittest


class CurrentStepEvaluationTests(unittest.TestCase):
    def test_dtw_report_adapter_preserves_known_unknown_and_failed_checkpoints(self):
        from src.guideme.evaluation import dtw_prediction_from_report

        known = {
            "schema_version": 1, "status": "ok", "checkpoint": {"end_s": 5.0},
            "result": {
                "method": "classical_subsequence_dtw", "checkpoint_s": 5,
                "candidate_step": {"step_id": " step-7 ", "action": "Fold"},
            },
            "runtime": {"unused_metadata": ["Preserve this"]},
        }
        clip = {**known, "result": {
            "method": "clip_subsequence_dtw", "checkpoint_s": 5.0,
            "candidate_step": {"step_id": "step-2"},
        }}
        unknown = {**known, "result": {**known["result"], "candidate_step": None}}
        failed = {
            "schema_version": 1, "status": "error", "checkpoint": {"end_s": 6.0},
            "result": None, "error": {"stage": "run_checkpoint", "type": "ValueError"},
        }
        cases = (
            ("classical known", known, {"checkpoint_s": 5.0, "predicted_step_id": " step-7 "}),
            ("clip known", clip, {"checkpoint_s": 5.0, "predicted_step_id": "step-2"}),
            ("successful unknown", unknown, {"checkpoint_s": 5.0, "predicted_step_id": None}),
            ("failed checkpoint", failed, {"checkpoint_s": 6.0, "predicted_step_id": None}),
        )
        for name, report, expected in cases:
            with self.subTest(case=name):
                before = deepcopy(report)
                self.assertEqual(dtw_prediction_from_report(report), expected)
                self.assertEqual(report, before)

        invalid_cases = (
            ("missing candidate is not unknown", {**known, "result": {
                "method": "classical_subsequence_dtw", "checkpoint_s": 5.0,
            }}),
            ("inconsistent checkpoint", {**known, "result": {**known["result"], "checkpoint_s": 6}}),
            ("unfinished run", {**known, "status": "running"}),
            ("blank candidate ID", {**known, "result": {
                **known["result"], "candidate_step": {"step_id": " "},
            }}),
            ("boolean times", {**known, "checkpoint": {"end_s": True},
                               "result": {**known["result"], "checkpoint_s": True}}),
            ("failure still has a result", {**failed, "result": known["result"]}),
            ("boolean schema version", {**known, "schema_version": True}),
        )
        for name, report in invalid_cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    dtw_prediction_from_report(report)

    def test_checkpoint_pairing_preserves_order_unknowns_and_balanced_scores(self):
        from src.guideme.evaluation import evaluate_current_steps

        reference_id = " refA "
        execution_rows = [
            {"occurrence_id": "first", "reference_step_id": "s001", "action": "Fold",
             "start_s": 0.0, "end_s": 2.0, "evidence_type": "visible_action", "notes": ""},
            {"occurrence_id": "second", "reference_step_id": "s002", "action": "Attach",
             "start_s": 3.0, "end_s": 4.0, "evidence_type": "visible_action", "notes": ""},
            {"occurrence_id": "result", "reference_step_id": "s002", "action": "Attached result",
             "start_s": 5.0, "end_s": 6.0, "evidence_type": "result", "notes": "Action is unseen."},
        ]
        predictions = [
            {"checkpoint_s": 3.5, "predicted_step_id": "s001", "metadata": {"ignored": True}},
            {"checkpoint_s": .5, "predicted_step_id": "s001"},
            {"checkpoint_s": 5.5, "predicted_step_id": "s002"},
            {"checkpoint_s": 1.5, "predicted_step_id": None},
            {"checkpoint_s": 2.5, "predicted_step_id": "s001"},
        ]
        predictions_before, rows_before = deepcopy(predictions), deepcopy(execution_rows)

        result = evaluate_current_steps(reference_id, predictions, execution_rows)

        # s001 has one of two correct; s002 has zero of one. Their macro
        # accuracy is .25, whereas pooling checkpoints would give 1/3.
        self.assertEqual(result, {
            "scores": {
                "balanced_step_accuracy": .25, "evaluated_count": 3, "excluded_count": 2,
                "per_step": [
                    {"reference_id": " refA ", "step_id": "s001", "correct_count": 1,
                     "evaluated_count": 2, "accuracy": .5},
                    {"reference_id": " refA ", "step_id": "s002", "correct_count": 0,
                     "evaluated_count": 1, "accuracy": 0.0},
                ],
            },
            "checkpoints": [
                {"reference_id": " refA ", "checkpoint_s": 3.5,
                 "true_step_id": "s002", "predicted_step_id": "s001"},
                {"reference_id": " refA ", "checkpoint_s": .5,
                 "true_step_id": "s001", "predicted_step_id": "s001"},
                {"reference_id": " refA ", "checkpoint_s": 5.5,
                 "true_step_id": None, "predicted_step_id": "s002"},
                {"reference_id": " refA ", "checkpoint_s": 1.5,
                 "true_step_id": "s001", "predicted_step_id": None},
                {"reference_id": " refA ", "checkpoint_s": 2.5,
                 "true_step_id": None, "predicted_step_id": "s001"},
            ],
        })
        self.assertEqual(predictions, predictions_before)
        self.assertEqual(execution_rows, rows_before)
        self.assertEqual(evaluate_current_steps(reference_id, [], execution_rows), {
            "scores": {"balanced_step_accuracy": None, "evaluated_count": 0,
                       "excluded_count": 0, "per_step": []},
            "checkpoints": [],
        })

        cases = (
            ("blank reference with empty predictions", " ", []),
            ("missing prediction", reference_id, [{"checkpoint_s": .5}]),
            ("duplicate int and float time", reference_id, [
                {"checkpoint_s": 1, "predicted_step_id": None},
                {"checkpoint_s": 1.0, "predicted_step_id": "s001"},
            ]),
            ("unknown checkpoint time", reference_id, [{"checkpoint_s": None, "predicted_step_id": None}]),
            ("blank prediction", reference_id, [{"checkpoint_s": .5, "predicted_step_id": " "}]),
        )
        for name, supplied_reference, supplied_predictions in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    evaluate_current_steps(supplied_reference, supplied_predictions, execution_rows)

    def test_balanced_groups_preserve_reference_identity_and_exact_step_ids(self):
        from src.guideme.evaluation import score_current_steps

        records = [
            {"reference_id": "refB", "true_step_id": "s001", "predicted_step_id": None},
            {"reference_id": "refA", "true_step_id": "s001", "predicted_step_id": "s001",
             "metadata": {"note": "Preserve extra fields"}},
            {"reference_id": "refA", "true_step_id": None, "predicted_step_id": "s001"},
            {"reference_id": "refA", "true_step_id": "s002", "predicted_step_id": "s002"},
            {"reference_id": "refA", "true_step_id": "s001", "predicted_step_id": " s001 "},
            {"reference_id": "refA", "true_step_id": "s001", "predicted_step_id": "s001"},
        ]
        before = deepcopy(records)

        result = score_current_steps(records)

        # Groups score 2/3, 1, and 0: their equal-weight mean is 5/9.
        # Pooling rows would instead yield 3/5; merging reference IDs would
        # incorrectly combine the two different s001 groups.
        self.assertEqual(set(result), {
            "balanced_step_accuracy", "evaluated_count", "excluded_count", "per_step",
        })
        self.assertIs(type(result["balanced_step_accuracy"]), float)
        self.assertAlmostEqual(result["balanced_step_accuracy"], 5 / 9)
        self.assertEqual(result["evaluated_count"], 5)
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["per_step"], [
            {"reference_id": "refA", "step_id": "s001", "correct_count": 2,
             "evaluated_count": 3, "accuracy": 2 / 3},
            {"reference_id": "refA", "step_id": "s002", "correct_count": 1,
             "evaluated_count": 1, "accuracy": 1.0},
            {"reference_id": "refB", "step_id": "s001", "correct_count": 0,
             "evaluated_count": 1, "accuracy": 0.0},
        ])
        self.assertEqual(records, before)

    def test_empty_and_unknown_truth_have_no_measurable_accuracy(self):
        from src.guideme.evaluation import score_current_steps

        for records, excluded in (
            ([], 0),
            ([
                {"reference_id": " refA ", "true_step_id": None, "predicted_step_id": None},
                {"reference_id": "refA", "true_step_id": None, "predicted_step_id": "s001"},
            ], 2),
        ):
            with self.subTest(excluded=excluded):
                self.assertEqual(score_current_steps(records), {
                    "balanced_step_accuracy": None, "evaluated_count": 0,
                    "excluded_count": excluded, "per_step": [],
                })

    def test_malformed_records_fail_instead_of_becoming_unknowns(self):
        from src.guideme.evaluation import score_current_steps

        valid = {"reference_id": "refA", "true_step_id": "s001", "predicted_step_id": None}
        cases = (
            ("non-list", (valid,)),
            ("non-dict row", [None]),
            ("missing prediction", [{"reference_id": "refA", "true_step_id": "s001"}]),
            ("blank reference", [{**valid, "reference_id": " "}]),
            ("non-string truth", [{**valid, "true_step_id": 42}]),
            ("blank prediction", [{**valid, "predicted_step_id": ""}]),
        )
        for name, records in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    score_current_steps(records)


if __name__ == "__main__":
    unittest.main()
