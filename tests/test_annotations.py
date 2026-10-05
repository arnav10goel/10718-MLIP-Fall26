"""Behavior tests for manually authored reference annotation CSV files."""

from pathlib import Path
import tempfile
import unittest

from src.guideme.annotations import load_reference_annotations, reference_step_at_time


class ReferenceAnnotationTests(unittest.TestCase):
    def test_execution_lookup_requires_one_matched_visible_action_and_returns_a_copy(self):
        from src.guideme.annotations import execution_step_at_time

        first = {
            "occurrence_id": "occ-7", "reference_step_id": "step-7", "action": "Fold",
            "start_s": 0.0, "end_s": 2.0, "evidence_type": "visible_action", "notes": "Keep edges aligned.",
        }
        second = {
            "occurrence_id": "occ-2", "reference_step_id": "step-2", "action": "Mix",
            "start_s": 2.0, "end_s": 4.0, "evidence_type": "visible_action", "notes": "",
        }
        unmatched = {
            "occurrence_id": "occ-9", "reference_step_id": None, "action": "Adjust stand",
            "start_s": 6.0, "end_s": 7.0, "evidence_type": "visible_action", "notes": "",
        }
        result = {
            "occurrence_id": "occ-3", "reference_step_id": "step-7", "action": "Folded result",
            "start_s": 8.0, "end_s": 10.0, "evidence_type": "result", "notes": "Action is not shown.",
        }
        rows = [first, second, unmatched, result]
        before = [row.copy() for row in rows]
        cases = (
            ("included start", rows, 0.0, first),
            ("adjacent boundary", rows, 2.0, second),
            ("excluded end", rows, 4.0, None),
            ("gap", rows, 5.0, None),
            ("unmatched visible action", rows, 6.0, None),
            ("result alone", rows, 8.0, None),
            ("outside", rows, 11.0, None),
            ("giant outside integer", rows, 10**400, None),
            ("unknown time", rows, None, None),
            ("empty rows", [], 1.0, None),
            ("same-step visible overlap", [first, {
                **first, "occurrence_id": "repeat", "start_s": 1.0, "end_s": 3.0,
            }], 1.5, None),
            ("matched plus unmatched overlap", [first, {
                **unmatched, "start_s": 1.0, "end_s": 2.0,
            }], 1.5, None),
            ("result does not obscure sole action", [first, {
                **result, "start_s": 1.0, "end_s": 2.0,
            }], 1.5, first),
        )
        for name, input_rows, time_s, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(execution_step_at_time(input_rows, time_s), expected)

        match = execution_step_at_time(rows, 1.0)
        self.assertIsNot(match, first)
        match["notes"] = "Changed by caller"
        self.assertEqual(rows, before)
        for invalid in (True, "1", float("nan"), -1.0):
            with self.subTest(invalid_time=invalid):
                with self.assertRaises(ValueError):
                    execution_step_at_time(rows, invalid)

    def test_execution_occurrences_preserve_repeats_order_unknowns_and_evidence(self):
        from src.guideme.annotations import load_execution_annotations

        header = "occurrence_id,reference_step_id,action,start_s,end_s,evidence_type,notes\n"
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir)
            reference = base / "reference.csv"
            reference.write_text(
                "step_id,action,start_s,end_s,requirements\n"
                "step-7,Fold,0,1,Source notes\n"
                " step-2 ,Inspect,1,3,\n", encoding="utf-8",
            )
            reference_rows = load_reference_annotations(reference, 10.0)
            reference_before = [row.copy() for row in reference_rows]
            reference_bytes = reference.read_bytes()
            path = base / "execution.csv"
            path.write_text(
                header
                + "occ-9,step-7, Fold again ,4,5,visible_action, Keep notes. \n"
                + "occ-2,step-7,Fold,1,2,visible_action,\n"
                + ' occ-7 , step-2 ,"Inspect, gently",1.5,3,result,Only the result is visible.\n'
                + "occ-3,,Unmatched action,7,10,visible_action,\n",
                encoding="utf-8",
            )
            before = path.read_bytes()

            rows = load_execution_annotations(path, 10.0, reference_rows)

            self.assertEqual(rows, [
                {"occurrence_id": "occ-9", "reference_step_id": "step-7",
                 "action": " Fold again ", "start_s": 4.0, "end_s": 5.0,
                 "evidence_type": "visible_action", "notes": " Keep notes. "},
                {"occurrence_id": "occ-2", "reference_step_id": "step-7",
                 "action": "Fold", "start_s": 1.0, "end_s": 2.0,
                 "evidence_type": "visible_action", "notes": ""},
                {"occurrence_id": " occ-7 ", "reference_step_id": " step-2 ",
                 "action": "Inspect, gently", "start_s": 1.5, "end_s": 3.0,
                 "evidence_type": "result", "notes": "Only the result is visible."},
                {"occurrence_id": "occ-3", "reference_step_id": None,
                 "action": "Unmatched action", "start_s": 7.0, "end_s": 10.0,
                 "evidence_type": "visible_action", "notes": ""},
            ])
            self.assertEqual(reference_rows, reference_before)
            self.assertEqual(reference.read_bytes(), reference_bytes)
            self.assertEqual(path.read_bytes(), before)
            for row in rows:
                self.assertIsInstance(row["start_s"], float)
                self.assertIsInstance(row["end_s"], float)
            path.write_text(header, encoding="utf-8")
            self.assertEqual(load_execution_annotations(str(path), 10.0, reference_rows), [])

    def test_invalid_execution_csv_and_duration_raise_value_error(self):
        from src.guideme.annotations import load_execution_annotations

        reference_rows = [{
            "step_id": "step-7", "action": "Fold", "start_s": 0.0,
            "end_s": 1.0, "requirements": "",
        }]
        header = "occurrence_id,reference_step_id,action,start_s,end_s,evidence_type,notes\n"
        valid_row = "occ-1,step-7,Fold,0,1,visible_action,\n"
        cases = (
            ("missing header field", header.replace(",notes", "") + "occ-1,step-7,Fold,0,1,visible_action\n"),
            ("incomplete row", header + "occ-1,step-7,Fold,0,1,visible_action\n"),
            ("duplicate occurrence", header + valid_row + valid_row),
            ("unknown padded reference ID", header + "occ-1, step-7 ,Fold,0,1,visible_action,\n"),
            ("evidence must be exact", header + "occ-1,step-7,Fold,0,1,result ,\n"),
            ("nonfinite timestamp", header + "occ-1,step-7,Fold,nan,1,visible_action,\n"),
            ("timestamp beyond video", header + "occ-1,step-7,Fold,0,11,visible_action,\n"),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "execution.csv"
            for name, text in cases:
                with self.subTest(case=name):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_execution_annotations(path, 10.0, reference_rows)
            path.write_text(header + valid_row, encoding="utf-8")
            for invalid in (True, float("inf")):
                with self.subTest(duration=invalid):
                    with self.assertRaises(ValueError):
                        load_execution_annotations(path, invalid, reference_rows)

    def test_valid_rows_preserve_authored_fields_order_and_gaps(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "reference.csv"
            path.write_text(
                "step_id,action,start_s,end_s,requirements\n"
                "step-7,Finish folding,1,2,\n"
                'step-2,"Mix, gently",4,5,Use a spoon.\n',
                encoding="utf-8",
            )

            rows = load_reference_annotations(path, 10.0)

            self.assertEqual(rows, [
                {
                    "step_id": "step-7", "action": "Finish folding",
                    "start_s": 1.0, "end_s": 2.0, "requirements": "",
                },
                {
                    "step_id": "step-2", "action": "Mix, gently",
                    "start_s": 4.0, "end_s": 5.0, "requirements": "Use a spoon.",
                },
            ])
            for row in rows:
                self.assertIsInstance(row["start_s"], float)
                self.assertIsInstance(row["end_s"], float)

    def test_invalid_rows_and_headers_raise_value_error(self):
        header = "step_id,action,start_s,end_s,requirements\n"
        cases = (
            ("duplicate ID", header + "a,Fold,0,1,\na,Mix,2,3,\n"),
            ("missing header field", "step_id,action,start_s,end_s\na,Fold,0,1\n"),
            ("missing row field", header + "a,Fold,0,1\n"),
            ("empty action", header + "a,,0,1,\n"),
            ("nonfinite timestamp", header + "a,Fold,nan,1,\n"),
            ("timestamp beyond video", header + "a,Fold,0,11,\n"),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "reference.csv"
            for name, csv_text in cases:
                with self.subTest(case=name):
                    path.write_text(csv_text, encoding="utf-8")

                    with self.assertRaises(ValueError):
                        load_reference_annotations(path, 10.0)

    def test_reference_step_lookup_preserves_unknowns_boundaries_and_rows(self):
        first = {
            "step_id": "step-7", "action": "Fold",
            "start_s": 0.0, "end_s": 2.0, "requirements": "Use paper.",
        }
        second = {
            "step_id": "step-2", "action": "Mix",
            "start_s": 2.0, "end_s": 4.0, "requirements": "",
        }
        last = {
            "step_id": "step-9", "action": "Finish",
            "start_s": 6.0, "end_s": 8.0, "requirements": "",
        }
        rows = [first, second, last]
        before = [row.copy() for row in rows]
        cases = (
            ("interval start", rows, 0, first),
            ("adjacent boundary", rows, 2.0, second),
            ("interval end", rows, 4.0, None),
            ("gap", rows, 5.0, None),
            ("outside", rows, 9.0, None),
            ("large outside integer", rows, 10**400, None),
            ("empty", [], 1.0, None),
            ("unknown time", rows, None, None),
            ("overlap", [first, {**second, "start_s": 1.0}], 1.5, None),
        )
        for name, input_rows, time_s, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(reference_step_at_time(input_rows, time_s), expected)

        match = reference_step_at_time(rows, 1.0)
        self.assertIsNot(match, first)
        match["action"] = "Changed by caller"
        self.assertEqual(rows, before)

        for invalid in (True, -1.0, float("nan")):
            with self.subTest(invalid_time=invalid):
                with self.assertRaises(ValueError):
                    reference_step_at_time(rows, invalid)


if __name__ == "__main__":
    unittest.main()
