"""Offline behavior tests for the VLM deviation request and response."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from google import genai
from google.genai import types
import httpx
import imageio_ffmpeg

from src.guideme.video import prepare_execution_prefix
from src.guideme.vlm import predict_deviation, upload_video


def fake_response(text):
    raw = {
        "response_id": "response-test",
        "model_version": "gemini-3.5-flash-lite",
        "candidates": [{
            "finish_reason": "STOP",
            "content": {"role": "model", "parts": [{"text": text}]},
        }],
        "usage_metadata": {"prompt_token_count": 12, "candidates_token_count": 8},
    }
    response = SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
        model_dump=Mock(return_value=raw),
    )
    return response, raw


class VlmTests(unittest.TestCase):
    def test_valid_predictions_keep_request_and_raw_response(self):
        warning = {
            "current_step": "Attach the blades",
            "reference_step_id": None,
            "reference_time_s": 12.0,
            "problem": "skipped_step",
            "evidence": "The blades are attached before the center is folded.",
            "message": "Fold the center before attaching the blades.",
            "confidence": 0.8,
        }
        silent = {
            "current_step": "Fold the center",
            "reference_step_id": None,
            "reference_time_s": None,
            "problem": "none",
            "evidence": "The center is being folded as shown.",
            "message": "",
            "confidence": 0.7,
        }
        for prediction, checklist in ((warning, "Fold the center first."), (silent, None)):
            with self.subTest(problem=prediction["problem"]):
                response, raw = fake_response(json.dumps(prediction))
                generate = Mock(return_value=response)
                client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

                result = predict_deviation(
                    client, "gs://test/reference.mp4", "gs://test/prefix.mp4",
                    checklist=checklist, fps=2,
                )

                generate.assert_called_once()
                request = generate.call_args.kwargs
                self.assertEqual(result["request"], request)
                self.assertEqual(request["model"], "gemini-3.5-flash-lite")
                self.assertEqual(len(request["contents"]), 1)
                self.assertEqual(request["contents"][0]["role"], "user")
                parts = request["contents"][0]["parts"]
                self.assertEqual(parts[:4], [
                    {"text": "REFERENCE"},
                    {
                        "file_data": {"file_uri": "gs://test/reference.mp4", "mime_type": "video/mp4"},
                        "video_metadata": {"fps": 2},
                        "media_resolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                        "media_processing": "STATIC",
                    },
                    {"text": "EXECUTION"},
                    {
                        "file_data": {"file_uri": "gs://test/prefix.mp4", "mime_type": "video/mp4"},
                        "video_metadata": {"fps": 2},
                        "media_resolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                        "media_processing": "STATIC",
                    },
                ])
                expected_prompt = Path(__file__).resolve().parents[1].joinpath(
                    "src/guideme/prompts/deviation.txt",
                ).read_text(encoding="utf-8")
                self.assertEqual(parts[4], {"text": expected_prompt})
                self.assertEqual(len(parts), 6 if checklist else 5)
                if checklist:
                    self.assertEqual(parts[5], {"text": f"CHECKLIST\n{checklist}"})
                    self.assertNotIn(checklist, parts[4]["text"])
                config = request["config"]
                self.assertEqual(config["response_mime_type"], "application/json")
                self.assertEqual(set(config["response_json_schema"]["required"]), set(prediction))
                self.assertEqual(result["prediction"], prediction)
                self.assertIsNone(result["validation_error"])
                self.assertEqual(result["response"], raw)
                self.assertGreaterEqual(result["request_duration_s"], 0)
                json.dumps(request, allow_nan=False)
                response.model_dump.assert_called_once_with(mode="json")

    def test_invalid_predictions_preserve_raw_response(self):
        invalid_confidence = json.dumps({
            "current_step": "Fold the center",
            "reference_step_id": None,
            "reference_time_s": None,
            "problem": "none",
            "evidence": "The center is being folded.",
            "message": "",
            "confidence": 1.5,
        })
        for text in ("{malformed JSON", invalid_confidence):
            with self.subTest(text=text):
                response, raw = fake_response(text)
                client = SimpleNamespace(models=SimpleNamespace(
                    generate_content=Mock(return_value=response),
                ))

                result = predict_deviation(client, "gs://test/ref.mp4", "gs://test/prefix.mp4")

                self.assertIsNone(result["prediction"])
                self.assertIsInstance(result["validation_error"], str)
                self.assertTrue(result["validation_error"].strip())
                self.assertEqual(result["response"], raw)

    def test_reference_step_id_uses_exact_supplied_ids_and_preserves_unknowns(self):
        structured = '[{"step_id":"missing-step","action":"Fold center"},' \
                     '{"step_id":"  observed/id  ","action":"Attach blades"}]'
        base = {
            "current_step": "Attach the blades", "reference_time_s": 12.0,
            "problem": "skipped_step", "evidence": "Blades precede the center fold.",
            "message": "Fold the center first.", "confidence": .8,
        }
        cases = (
            ("supplied observed ID", structured, {"reference_step_id": "  observed/id  "}, True),
            ("unknown observed ID", structured, {"reference_step_id": None}, True),
            ("ID must not be trimmed", structured, {"reference_step_id": "observed/id"}, False),
            ("plain notes have no IDs", "Fold first.", {"reference_step_id": "missing-step"}, False),
            ("malformed JSON remains notes", "[{unfinished", {"reference_step_id": None}, True),
            ("nonstring ID", structured, {"reference_step_id": 42}, False),
            ("blank ID", structured, {"reference_step_id": " "}, False),
            ("missing required ID", structured, {}, False),
        )
        for name, checklist, fields, valid in cases:
            with self.subTest(case=name):
                candidate = {**base, **fields}
                response, raw = fake_response(json.dumps(candidate))
                generate = Mock(return_value=response)
                client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
                result = predict_deviation(client, "gs://test/ref.mp4", "gs://test/prefix.mp4",
                                           checklist=checklist)
                generate.assert_called_once()
                self.assertEqual(result["response"], raw)
                self.assertEqual(result["request"]["contents"][0]["parts"][-1],
                                 {"text": f"CHECKLIST\n{checklist}"})
                if valid:
                    self.assertEqual(result["prediction"], candidate)
                    self.assertIsNone(result["validation_error"])
                else:
                    self.assertIsNone(result["prediction"])
                    self.assertTrue(result["validation_error"].strip())

    def test_api_errors_propagate(self):
        error = RuntimeError("synthetic API failure")
        client = SimpleNamespace(models=SimpleNamespace(
            generate_content=Mock(side_effect=error),
        ))

        with self.assertRaises(RuntimeError) as raised:
            predict_deviation(client, "gs://test/ref.mp4", "gs://test/prefix.mp4")

        self.assertIs(raised.exception, error)

    def test_actual_sdk_serializes_requests_and_rejects_incomplete_candidates_offline(self):
        prediction = {
            "current_step": "Fold the center", "reference_time_s": None,
            "reference_step_id": None,
            "problem": "none", "evidence": "The center is being folded.",
            "message": "", "confidence": .8,
        }
        candidate = {
            "content": {"role": "model", "parts": [{"text": json.dumps(prediction)}]},
            "finishReason": "STOP",
        }
        replies = [
            {"candidates": [candidate], "modelVersion": "served-default"},
            {"candidates": [candidate], "modelVersion": "served-override"},
            {"candidates": [{**candidate, "finishReason": "MAX_TOKENS"}]},
            {"promptFeedback": {"blockReason": "SAFETY"}},
            {"candidates": [candidate, candidate]},
        ]
        sent = []

        def respond(request):
            sent.append((request.url.path, json.loads(request.content)))
            self.assertTrue(request.url.path.endswith(":generateContent"))
            return httpx.Response(200, json=replies.pop(0))

        transport = httpx.MockTransport(respond)
        with genai.Client(
            vertexai=False, api_key="offline-test-key",
            http_options={"client_args": {"transport": transport}, "retry_options": {"attempts": 1}},
        ) as client:
            for index, model in enumerate((None, "override-model", None, None, None)):
                with self.subTest(response=index):
                    options = {} if model is None else {"model": model}
                    result = predict_deviation(
                        client, "gs://test/reference.mp4", "gs://test/prefix.mp4",
                        checklist="Fold first.", fps=3, **options,
                    )
                    self.assertEqual(len(sent), index + 1)
                    selected = model or "gemini-3.5-flash-lite"
                    self.assertEqual(result["request"]["model"], selected)
                    path, wire = sent[-1]
                    self.assertEqual(path, f"/v1beta/models/{selected}:generateContent")
                    self.assertEqual(len(wire["contents"]), 1)
                    self.assertEqual(wire["contents"][0]["role"], "user")
                    parts = wire["contents"][0]["parts"]
                    self.assertEqual(parts[0], {"text": "REFERENCE"})
                    self.assertEqual(parts[2], {"text": "EXECUTION"})
                    for part, uri in ((parts[1], "gs://test/reference.mp4"),
                                      (parts[3], "gs://test/prefix.mp4")):
                        self.assertEqual(part, {
                            "fileData": {"file_uri": uri, "mime_type": "video/mp4"},
                            "videoMetadata": {"fps": 3},
                            "mediaResolution": {"level": "MEDIA_RESOLUTION_HIGH"},
                            "mediaProcessing": "STATIC",
                        })
                    self.assertEqual(parts[5], {"text": "CHECKLIST\nFold first."})
                    self.assertEqual(wire["generationConfig"]["responseMimeType"], "application/json")
                    self.assertEqual(set(wire["generationConfig"]["responseJsonSchema"]["required"]),
                                     set(prediction))
                    json.dumps(result, allow_nan=False)
                    if index < 2:
                        self.assertEqual(result["prediction"], prediction)
                        self.assertIsNone(result["validation_error"])
                        self.assertEqual(result["response"]["model_version"],
                                         "served-default" if index == 0 else "served-override")
                    else:
                        self.assertIsNone(result["prediction"])
                        self.assertTrue(result["validation_error"].strip())
                        if index == 2:
                            self.assertEqual(result["response"]["candidates"][0]["finish_reason"], "MAX_TOKENS")
                        elif index == 3:
                            self.assertEqual(result["response"]["prompt_feedback"]["block_reason"], "SAFETY")
                        else:
                            self.assertEqual(len(result["response"]["candidates"]), 2)
            for invalid in (None, "", " ", 42):
                with self.subTest(model=invalid):
                    with self.assertRaises(ValueError):
                        predict_deviation(client, "gs://test/ref.mp4", "gs://test/prefix.mp4",
                                          model=invalid)
            self.assertEqual(len(sent), 5)

    def test_upload_video_polls_until_active_or_failed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir).resolve() / "prefix.mp4"
            path.write_bytes(b"nonempty upload fixture")
            uploading = types.File(name="files/video-test", state=types.FileState.PROCESSING)
            for state in (types.FileState.ACTIVE, types.FileState.FAILED):
                with self.subTest(state=state):
                    ready = types.File(
                        name="files/video-test", uri="https://example.org/video-test", state=state,
                    )
                    client = SimpleNamespace(
                        files=SimpleNamespace(
                            upload=Mock(return_value=uploading), get=Mock(return_value=ready),
                        ),
                        interactions=SimpleNamespace(create=Mock()),
                    )
                    with patch("src.guideme.vlm.perf_counter", side_effect=range(20)), \
                            patch("src.guideme.vlm.sleep") as sleep:
                        if state == types.FileState.FAILED:
                            with self.assertRaises(RuntimeError):
                                upload_video(client, path)
                        else:
                            result = upload_video(client, path)
                            self.assertEqual(result["uri"], ready.uri)
                            self.assertEqual(result["name"], ready.name)
                            self.assertGreaterEqual(result["upload_duration_s"], 0)
                            self.assertGreaterEqual(result["processing_duration_s"], 0)

                    client.files.upload.assert_called_once_with(
                        file=str(path), config={"mime_type": "video/mp4"},
                    )
                    client.files.get.assert_called_once_with(name="files/video-test")
                    sleep.assert_called_once()
                    client.interactions.create.assert_not_called()

    def test_execution_prefix_excludes_future_frames_audio_and_overwrite(self):
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir).resolve() / "source.mp4"
            output = source.with_name("prefix.mp4")
            frames = bytes([255, 0, 0]) * 16 * 16 * 4 + bytes([0, 0, 255]) * 16 * 16 * 6
            subprocess.run(
                [
                    ffmpeg, "-loglevel", "error",
                    "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "16x16", "-framerate", "10", "-i", "pipe:0",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(source),
                ],
                input=frames, capture_output=True, check=True,
            )
            original = source.read_bytes()
            subprocess.run(
                [ffmpeg, "-loglevel", "error", "-i", str(source), "-map", "0:a:0", "-f", "null", "-"],
                capture_output=True, check=True,
            )

            result = prepare_execution_prefix(source, 0.4, output)

            self.assertEqual(result, output)
            self.assertIsInstance(result, Path)
            decoded = subprocess.run(
                [
                    ffmpeg, "-loglevel", "error", "-i", str(output),
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                ],
                capture_output=True, check=True,
            ).stdout
            self.assertEqual(len(decoded), 4 * 16 * 16 * 3)
            self.assertGreater(min(decoded[0::3]), 240)
            self.assertLess(max(decoded[2::3]), 16)
            audio_check = subprocess.run(
                [ffmpeg, "-loglevel", "error", "-i", str(output), "-map", "0:a:0", "-f", "null", "-"],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(audio_check.returncode, 0)
            self.assertIn("matches no streams", audio_check.stderr)
            self.assertEqual(source.read_bytes(), original)
            prefix_bytes = output.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_execution_prefix(source, 0.4, output)
            self.assertEqual(output.read_bytes(), prefix_bytes)

    def test_command_prepares_uploads_and_records_checkpoint_without_overwrite(self):
        from scripts import run_vlm

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir).resolve()
            data_root = root / "data"
            data_root.mkdir()
            reference = data_root / "reference.mp4"
            execution = data_root / "execution.mp4"
            prefix = data_root / "prefix.mp4"
            output = root / "report.json"
            env_file = root / "fake.env"
            checklist = root / "checklist.txt"
            reference.write_bytes(b"reference fixture")
            execution.write_bytes(b"full execution fixture")
            checklist.write_text("Fold the center first.")
            client = SimpleNamespace(close=Mock())
            reference_upload = {
                "uri": "https://example.org/reference", "name": "files/reference",
                "upload_duration_s": 1.0, "processing_duration_s": 2.0,
            }
            prefix_upload = {
                "uri": "https://example.org/prefix", "name": "files/prefix",
                "upload_duration_s": 1.0, "processing_duration_s": 2.0,
            }
            prediction_result = {
                "prediction": {
                    "current_step": "Fold the center", "reference_time_s": 0.2,
                    "reference_step_id": None,
                    "problem": "none", "evidence": "The center is being folded.",
                    "message": "", "confidence": 0.8,
                },
                "validation_error": None,
                "request": {"model": "gemini-3.5-flash-lite"},
                "response": {
                    "candidates": [{"finish_reason": "STOP", "content": {
                        "parts": [{"text": "prediction fixture"}],
                    }}], "usage_metadata": {},
                },
                "request_duration_s": 3.0,
            }

            def prepare(source_path, end_s, output_path):
                Path(output_path).write_bytes(b"prepared prefix fixture")
                return Path(output_path)

            argv = [
                "--reference", "reference.mp4", "--execution", "execution.mp4",
                "--end-s", "0.4", "--prefix-output", "prefix.mp4",
                "--output", str(output), "--data-root", str(data_root),
                "--checklist", str(checklist), "--env-file", str(env_file),
            ]
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(run_vlm, "dotenv_values", return_value={"GEMINI_API_KEY": "fake-local-key"}) as load_env, \
                    patch.object(run_vlm.genai, "Client", return_value=client) as make_client, \
                    patch.object(run_vlm, "prepare_execution_prefix", side_effect=prepare) as prepare_prefix, \
                    patch.object(run_vlm, "upload_video", side_effect=[reference_upload, prefix_upload]) as upload, \
                    patch.object(run_vlm, "predict_deviation", return_value=prediction_result) as predict:
                self.assertEqual(run_vlm.main(argv), 0)

                prepare_prefix.assert_called_once_with(execution, 0.4, prefix)
                self.assertEqual(upload.call_args_list, [
                    call(client, reference), call(client, prefix),
                ])
                predict.assert_called_once_with(
                    client, reference_upload["uri"], prefix_upload["uri"],
                    checklist="Fold the center first.", fps=2, model="gemini-3.5-flash-lite",
                )
                self.assertEqual(Path(load_env.call_args.args[0]), env_file)
                self.assertEqual(load_env.call_args.kwargs, {"interpolate": False})
                make_client.assert_called_once_with(
                    api_key="fake-local-key",
                    http_options={"timeout": 120000, "retry_options": {"attempts": 1}},
                )
                client.close.assert_called_once()
                saved = output.read_text()
                report = json.loads(saved)
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["status"], "ok")
                self.assertEqual(report["requested_model"], "gemini-3.5-flash-lite")
                self.assertEqual(report["data_root"], str(data_root))
                for role, path in (("reference", reference), ("execution", execution), ("prefix", prefix)):
                    self.assertEqual(report["inputs"][role], {
                        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    })
                self.assertEqual(report["checkpoint_s"], 0.4)
                self.assertEqual(report["uploads"], {
                    "reference": reference_upload, "execution": prefix_upload,
                })
                self.assertEqual(report["result"], prediction_result)
                self.assertNotIn("fake-local-key", saved)

                for mock in (load_env, make_client, prepare_prefix, upload, predict):
                    mock.reset_mock()
                with self.assertRaises(FileExistsError):
                    run_vlm.main(argv)
                for mock in (load_env, make_client, prepare_prefix, upload, predict):
                    mock.assert_not_called()
                self.assertEqual(output.read_text(), saved)

    def test_command_model_override_survives_results_and_errors_and_rejects_blanks(self):
        from scripts import run_vlm

        selected = "  caller-selected-model  "
        for status in ("ok", "error"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                reference, execution = root / "reference.mp4", root / "execution.mp4"
                prefix, output = root / "prefix.mp4", root / "report.json"
                reference.write_bytes(b"reference fixture")
                execution.write_bytes(b"execution fixture")
                client = SimpleNamespace(close=Mock())
                result = {
                    "prediction": {
                        "current_step": "Fold", "reference_time_s": None, "problem": "none",
                        "reference_step_id": None,
                        "evidence": "Folding", "message": "", "confidence": .8,
                    },
                    "validation_error": None if status == "ok" else "incomplete candidate",
                    "request": {"model": selected}, "response": {}, "request_duration_s": 0,
                }

                def prepare(source_path, end_s, output_path):
                    Path(output_path).write_bytes(b"prepared prefix fixture")
                    return Path(output_path)

                argv = [
                    "--reference", str(reference), "--execution", str(execution),
                    "--end-s", "0.4", "--prefix-output", str(prefix),
                    "--output", str(output), "--data-root", str(root), "--model", selected,
                    "--env-file", str(root / "unused.env"),
                ]
                with patch.dict(os.environ, {}, clear=True), \
                        patch.object(run_vlm, "dotenv_values", return_value={"GEMINI_API_KEY": "offline-key"}) as load_env, \
                        patch.object(run_vlm.genai, "Client", return_value=client) as make_client, \
                        patch.object(run_vlm, "prepare_execution_prefix", side_effect=prepare) as prepare_prefix, \
                        patch.object(run_vlm, "upload_video", return_value={"uri": "https://example.org/video"}) as upload, \
                        patch.object(run_vlm, "predict_deviation", return_value=result,
                                     side_effect=RuntimeError("prediction failed") if status == "error" else None) as predict:
                    self.assertEqual(run_vlm.main(argv), 0 if status == "ok" else 1)
                    predict.assert_called_once_with(
                        client, "https://example.org/video", "https://example.org/video",
                        checklist=None, fps=2, model=selected,
                    )
                    saved = output.read_text()
                    report = json.loads(saved)
                    self.assertEqual(report["requested_model"], selected)
                    self.assertEqual(report["status"], status)
                    self.assertEqual(report["result"], None if status == "error" else result)
                    if status == "error":
                        self.assertEqual(report["error"], {
                            "stage": "predict", "type": "RuntimeError", "code": None,
                        })
                    self.assertNotIn("offline-key", saved)
                    client.close.assert_called_once()

                    if status == "error":
                        for mock in (load_env, make_client, prepare_prefix, upload, predict):
                            mock.reset_mock()
                        # Missing inputs expose an accidentally late model guard.
                        for blank in ("", " "):
                            invalid = [
                                "--reference", str(root / "missing-reference.mp4"),
                                "--execution", str(root / "missing-execution.mp4"),
                                "--end-s", "0.4", "--prefix-output", str(root / "unused-prefix.mp4"),
                                "--output", str(root / "unused-report.json"),
                                "--data-root", str(root), "--model", blank,
                            ]
                            with self.subTest(model=blank):
                                with self.assertRaisesRegex(ValueError, "model"):
                                    run_vlm.main(invalid)
                                self.assertFalse((root / "unused-report.json").exists())
                        for mock in (load_env, make_client, prepare_prefix, upload, predict):
                            mock.assert_not_called()

    def test_csv_checklist_is_validated_before_calls_and_serialized_with_metadata(self):
        import cv2
        from scripts import run_vlm

        expected_rows = [
            {"step_id": "a", "action": "Fold, carefully", "start_s": 0.0,
             "end_s": 2.0, "requirements": "Align edges"},
            {"step_id": "b", "action": "Fasten", "start_s": 3.0,
             "end_s": 5.0, "requirements": ""},
        ]
        for valid, last_end in ((True, 5), (False, 11)):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as temporary_dir:
                root = Path(temporary_dir).resolve()
                reference, execution = root / "reference.mp4", root / "execution.mp4"
                prefix, output = root / "prefix.mp4", root / "report.json"
                checklist = root / "steps.CSV"
                reference.write_bytes(b"reference fixture")
                execution.write_bytes(b"execution fixture")
                checklist.write_text(
                    "step_id,action,start_s,end_s,requirements\n"
                    'a,"Fold, carefully",0,2,Align edges\n'
                    f"b,Fasten,3,{last_end},\n", encoding="utf-8",
                )
                capture = Mock()
                capture.isOpened.return_value = True
                capture.get.side_effect = lambda field: {
                    cv2.CAP_PROP_FRAME_COUNT: 100, cv2.CAP_PROP_FPS: 10,
                }[field]

                def prepare(source_path, end_s, output_path):
                    Path(output_path).write_bytes(b"prepared prefix fixture")
                    return Path(output_path)

                prediction_result = {
                    "prediction": {"current_step": "Fold", "reference_time_s": 1.0,
                                   "reference_step_id": "a",
                                   "problem": "none", "evidence": "Folding", "message": "", "confidence": 0.8},
                    "validation_error": None, "request": {}, "response": {}, "request_duration_s": 0.0,
                }
                argv = [
                    "--reference", "reference.mp4", "--execution", "execution.mp4",
                    "--end-s", "0.4", "--prefix-output", "prefix.mp4",
                    "--output", str(output), "--data-root", str(root),
                    "--checklist", str(checklist), "--env-file", str(root / "fake.env"),
                ]
                with patch.dict(os.environ, {}, clear=True), \
                        patch.object(cv2, "VideoCapture", return_value=capture) as open_capture, \
                        patch.object(run_vlm, "dotenv_values", return_value={"GEMINI_API_KEY": "fake-local-key"}) as load_env, \
                        patch.object(run_vlm.genai, "Client", return_value=Mock()) as make_client, \
                        patch.object(run_vlm, "prepare_execution_prefix", side_effect=prepare) as prepare_prefix, \
                        patch.object(run_vlm, "upload_video", return_value={"uri": "https://example.org/video"}) as upload, \
                        patch.object(run_vlm, "predict_deviation", return_value=prediction_result) as predict:
                    if valid:
                        self.assertEqual(run_vlm.main(argv), 0)
                        sent = predict.call_args.kwargs["checklist"]
                        try:
                            rows = json.loads(sent)
                        except json.JSONDecodeError:
                            self.fail("CSV checklist must be serialized as JSON annotation rows")
                        self.assertEqual(rows, expected_rows)
                        report = json.loads(output.read_text())
                        self.assertEqual(report["checklist"], {
                            "path": str(checklist),
                            "sha256": hashlib.sha256(checklist.read_bytes()).hexdigest(),
                            "format": "csv", "format_version": 1, "step_count": 2,
                            "reference_duration_s": 10.0,
                            "duration_method": "opencv_frame_count_over_fps",
                            "reference_frame_count": 100, "reference_fps": 10.0,
                        })
                    else:
                        with self.assertRaises(ValueError):
                            run_vlm.main(argv)
                        for mock in (load_env, make_client, prepare_prefix, upload, predict):
                            mock.assert_not_called()
                        self.assertFalse(prefix.exists())
                        self.assertFalse(output.exists())
                    self.assertEqual(Path(open_capture.call_args.args[0]), reference)
                    open_capture.assert_called_once()
                    capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
