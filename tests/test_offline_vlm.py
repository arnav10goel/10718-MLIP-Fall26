"""Offline tests for the whole-execution Gemini alignment (no network)."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import imageio_ffmpeg

from src.guideme.annotations import load_reference_annotations
from src.guideme.offline_vlm import (
    OFFLINE_SCHEMA, coin_checklist_rows, cut_segments, predict_offline_alignment,
    video_duration_s, write_checklist_csv,
)


ROWS = [
    {"step_id": "s001", "action": "fold the paper", "start_s": 0.0, "end_s": 5.0, "requirements": ""},
    {"step_id": "s002", "action": "cut the edges", "start_s": 5.0, "end_s": 9.0, "requirements": ""},
    {"step_id": "s003", "action": "pin to stick", "start_s": 9.0, "end_s": 12.0, "requirements": ""},
]


def fake_client(prediction, finish_reason="STOP"):
    text = json.dumps(prediction) if not isinstance(prediction, str) else prediction
    raw = {"candidates": [{"finish_reason": finish_reason,
                           "content": {"role": "model", "parts": [{"text": text}]}}]}
    response = SimpleNamespace(text=text, model_dump=Mock(return_value=raw))
    generate = Mock(return_value=response)
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate)), generate


def step(step_id, status, start=None, end=None):
    return {"reference_step_id": step_id, "status": status, "execution_start_s": start,
            "execution_end_s": end, "evidence": "seen", "confidence": 0.8}


def good_prediction():
    return {
        "steps": [step("s001", "done", 0, 4), step("s002", "skipped"), step("s003", "done", 5, 8)],
        "deviations": [{"type": "skipped_step", "reference_step_id": "s002", "execution_time_s": None,
                        "evidence": "edges never cut", "confidence": 0.7,
                        "message": "Cut along the folds before pinning."}],
        "summary": "Cutting was skipped.",
    }


class PredictTests(unittest.TestCase):
    def test_valid_prediction_and_request_shape(self):
        client, generate = fake_client(good_prediction())
        result = predict_offline_alignment(client, "uri://ref", "uri://exe", ROWS,
                                           execution_duration_s=10.0, fps=1, model="m")
        self.assertIsNone(result["validation_error"])
        self.assertEqual(result["prediction"]["steps"][1]["status"], "skipped")
        request = generate.call_args.kwargs
        self.assertEqual(request["model"], "m")
        self.assertEqual(request["config"]["response_json_schema"], OFFLINE_SCHEMA)
        parts = request["contents"][0]["parts"]
        self.assertEqual([p.get("text") for p in parts[:3:2]], ["REFERENCE", "EXECUTION"])
        self.assertEqual(parts[1]["file_data"]["file_uri"], "uri://ref")
        self.assertEqual(parts[3]["file_data"]["file_uri"], "uri://exe")
        texts = [p["text"] for p in parts if "text" in p]
        checklist = json.loads(next(t for t in texts if t.startswith("CHECKLIST")).split("\n", 1)[1])
        self.assertEqual([r["step_id"] for r in checklist], ["s001", "s002", "s003"])
        self.assertTrue(any(t.startswith("EXECUTION_DURATION_S") for t in texts))

    def test_invalid_predictions_are_rejected_with_reason(self):
        cases = {}
        p = good_prediction(); p["steps"] = p["steps"][:2]
        cases["missing step"] = p
        p = good_prediction(); p["steps"][2]["reference_step_id"] = "s001"
        cases["duplicate step"] = p
        p = good_prediction(); p["deviations"][0]["reference_step_id"] = "s999"
        cases["unknown deviation step"] = p
        p = good_prediction(); p["deviations"][0]["message"] = " "
        cases["empty message"] = p
        p = good_prediction(); p["steps"][0]["status"] = "maybe"
        cases["bad status"] = p
        for name, prediction in cases.items():
            with self.subTest(name):
                client, _ = fake_client(prediction)
                result = predict_offline_alignment(client, "u1", "u2", ROWS, execution_duration_s=10.0)
                self.assertIsNone(result["prediction"])
                self.assertTrue(result["validation_error"])
        for name, (text, reason) in {"not json": ("nope", "STOP"),
                                     "unfinished": (json.dumps(good_prediction()), "MAX_TOKENS")}.items():
            with self.subTest(name):
                client, _ = fake_client(text, finish_reason=reason)
                result = predict_offline_alignment(client, "u1", "u2", ROWS)
                self.assertIsNone(result["prediction"])

    def test_timing_problems_are_repaired_with_warnings_not_rejected(self):
        p = good_prediction()
        p["steps"][0] = step("s001", "done", 5, 3)          # start after end
        p["steps"][1] = step("s002", "skipped", 1, 2)       # skipped with times
        p["steps"][2] = step("s003", "done", 5, 99)         # far past the end
        client, _ = fake_client(p)
        result = predict_offline_alignment(client, "u1", "u2", ROWS, execution_duration_s=10.0)
        self.assertIsNone(result["validation_error"])
        steps = result["prediction"]["steps"]
        self.assertEqual([s["status"] for s in steps], ["done", "skipped", "done"])
        self.assertEqual([s["execution_start_s"] for s in steps], [None, None, None])
        self.assertIsNone(steps[2]["execution_end_s"])
        self.assertEqual(len(result["warnings"]), 6)

    def test_small_overshoot_past_the_end_is_clamped(self):
        p = good_prediction()
        p["steps"][2] = step("s003", "done", 8, 12.5)
        p["deviations"][0]["execution_time_s"] = 11.0
        client, _ = fake_client(p)
        result = predict_offline_alignment(client, "u1", "u2", ROWS, execution_duration_s=10.0)
        self.assertIsNone(result["validation_error"])
        self.assertEqual(result["prediction"]["steps"][2]["execution_end_s"], 10.0)
        self.assertEqual(result["prediction"]["deviations"][0]["execution_time_s"], 10.0)

    def test_bad_arguments_raise(self):
        client, _ = fake_client(good_prediction())
        for kwargs in ({"fps": 0}, {"model": " "}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                predict_offline_alignment(client, "u1", "u2", ROWS, **kwargs)
        with self.assertRaises(ValueError):
            predict_offline_alignment(client, "", "u2", ROWS)
        with self.assertRaises(ValueError):
            predict_offline_alignment(client, "u1", "u2", [])


class HelperTests(unittest.TestCase):
    def test_coin_rows_shift_and_load_as_reference_checklist(self):
        info = {"annotation": [{"label": "fold", "segment": [12.0, 20.0]},
                               {"label": "cut", "segment": [21.5, 30.0]}]}
        rows = coin_checklist_rows(info, offset_s=10.0)
        self.assertEqual([(r["step_id"], r["start_s"], r["end_s"]) for r in rows],
                         [("s001", 2.0, 10.0), ("s002", 11.5, 20.0)])
        with tempfile.TemporaryDirectory() as directory:
            path = write_checklist_csv(rows, Path(directory) / "c.csv")
            loaded = load_reference_annotations(path, 25.0)
            self.assertEqual([r["action"] for r in loaded], ["fold", "cut"])
            with self.assertRaises(FileExistsError):
                write_checklist_csv(rows, path)

    def test_cut_segments_reorders_drops_audio_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=6",
                "-f", "lavfi", "-i", "sine=duration=6",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
            ], check=True)
            out = cut_segments(source, [(4.0, 6.0), (0.0, 1.0)], Path(directory) / "cut.mp4")
            self.assertAlmostEqual(video_duration_s(out), 3.0, delta=0.25)
            probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(out)],
                                   capture_output=True, text=True)
            self.assertNotIn("Audio:", probe.stderr)
            with self.assertRaises(FileExistsError):
                cut_segments(source, [(0.0, 1.0)], out)
            with self.assertRaises(ValueError):
                cut_segments(source, [(2.0, 1.0)], Path(directory) / "bad.mp4")


if __name__ == "__main__":
    unittest.main()
