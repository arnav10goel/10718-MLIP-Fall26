"""Offline tests for alignment through an OpenAI-compatible chat API (no network)."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import imageio_ffmpeg
import numpy as np

from src.guideme.offline_openai import (
    build_request, predict_offline_alignment_openai, redact_images, sample_frames,
)
from src.guideme.offline_vlm import OFFLINE_SCHEMA
from tests.test_offline_vlm import ROWS, good_prediction


def make_video(path: Path, seconds: int = 6, size: str = "1280x720") -> Path:
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi",
        "-i", f"testsrc=size={size}:rate=10:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True)
    return path


def fake_client(content, finish_reason="stop"):
    raw = {"choices": [{"finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": content}}]}
    response = SimpleNamespace(model_dump=Mock(return_value=raw))
    create = Mock(return_value=response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


class FrameTests(unittest.TestCase):
    def test_frames_are_timed_shrunk_and_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            video = make_video(Path(directory) / "v.mp4")
            frames = sample_frames(video, 1, max_side=320)
            self.assertEqual([t for t, _ in frames], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            self.assertTrue(all(jpeg[:2] == b"\xff\xd8" for _, jpeg in frames))
            image = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(max(image.shape[:2]), 320)
            capped = sample_frames(video, 2, max_frames=3)
            self.assertEqual([t for t, _ in capped], [0.0, 2.0, 4.0])
            with self.assertRaises(ValueError):
                sample_frames(video, 0)


class RequestTests(unittest.TestCase):
    def test_request_order_matches_the_gemini_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            ref = make_video(Path(directory) / "ref.mp4", seconds=4, size="64x48")
            exe = make_video(Path(directory) / "exe.mp4", seconds=3, size="64x48")
            request, times = build_request(ref, exe, ROWS, model="m", execution_duration_s=3.0,
                                           disable_thinking=True)
        self.assertEqual(times, {"reference": [0.0, 1.0, 2.0, 3.0], "execution": [0.0, 1.0, 2.0]})
        self.assertEqual(request["response_format"]["json_schema"]["schema"], OFFLINE_SCHEMA)
        self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        content = request["messages"][0]["content"]
        texts = [p["text"] for p in content if p["type"] == "text"]
        self.assertTrue(texts[0].startswith("REFERENCE: 4 frames"))
        self.assertTrue(texts[5].startswith("EXECUTION: 3 frames"))
        self.assertTrue(texts[-3].startswith("You are watching a livestream"))
        self.assertTrue(texts[-2].startswith("CHECKLIST"))
        self.assertTrue(texts[-1].startswith("EXECUTION_DURATION_S"))
        self.assertEqual(sum(p["type"] == "image_url" for p in content), 7)
        redacted = redact_images(request)
        self.assertNotIn("base64,", json.dumps(redacted))
        self.assertIn("base64,", content[2]["image_url"]["url"])


class PredictTests(unittest.TestCase):
    REQUEST = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
    STEP_IDS = [row["step_id"] for row in ROWS]

    def test_valid_answer_with_think_block_and_fence(self):
        text = "<think>look at frames</think>\n```json\n" + json.dumps(good_prediction()) + "\n```"
        client, create = fake_client(text)
        result = predict_offline_alignment_openai(client, self.REQUEST, self.STEP_IDS, 10.0)
        self.assertIsNone(result["validation_error"])
        self.assertEqual(result["prediction"]["current_step_id"], "s002")
        self.assertEqual(create.call_args.kwargs["model"], "m")

    def test_bad_answers_are_rejected_with_reason(self):
        for name, (text, reason) in {
            "not json": ("nope", "stop"),
            "cut off": (json.dumps(good_prediction()), "length"),
            "empty": ("", "stop"),
            "wrong steps": (json.dumps({**good_prediction(), "steps": []}), "stop"),
        }.items():
            with self.subTest(name):
                client, _ = fake_client(text, reason)
                result = predict_offline_alignment_openai(client, self.REQUEST, self.STEP_IDS, 10.0)
                self.assertIsNone(result["prediction"])
                self.assertTrue(result["validation_error"])


if __name__ == "__main__":
    unittest.main()
