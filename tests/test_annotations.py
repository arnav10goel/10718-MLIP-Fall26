"""Behavior tests for manually authored reference annotation CSV files."""

from pathlib import Path
import tempfile
import unittest

from src.guideme.annotations import load_reference_annotations


class ReferenceAnnotationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
