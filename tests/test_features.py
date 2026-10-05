"""CLIP frame-encoding behavior without model weights or downloads."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.guideme.features import encode_clip_frames

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


if torch is not None:
    class PixelEncoder(torch.nn.Module):
        """Use actual preprocessed pixels as features at the model boundary."""

        def __init__(self, outputs=None):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))
            self.inputs = []
            self.modes = []
            self.outputs = outputs

        def encode_image(self, images):
            self.inputs.append(images.detach().clone())
            self.modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
            if self.outputs is not None:
                output = self.outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return output
            return images[:, :, 0, 0] * self.scale


@unittest.skipIf(torch is None, "requires optional clip dependencies")
class ClipFrameTests(unittest.TestCase):
    def test_local_loader_restores_strict_weights_and_records_checkpoint_provenance(self):
        try:
            import open_clip
        except ModuleNotFoundError as error:
            if error.name != "open_clip":
                raise
            self.skipTest("requires optional open-clip dependency")
        from src.guideme.features import load_clip_encoder

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.logit_bias = None

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint = base / "local-weights.pt"
            torch.save({"weight": torch.tensor([3.0, 4.0])}, checkpoint)
            before = checkpoint.read_bytes()
            model = TinyModel()
            # Only replace expensive ViT construction; serialization, OpenCLIP
            # checkpoint conversion, strict Torch restore and device moves run.
            with patch("open_clip.create_model", return_value=model) as constructor:
                loaded, device, metadata = load_clip_encoder(checkpoint)
                constructor.assert_called_once_with(
                    "ViT-B-32", pretrained=None, precision="fp32", device="cpu",
                    force_quick_gelu=True, pretrained_image=False, pretrained_text=False,
                )
            self.assertIs(loaded, model)
            np.testing.assert_array_equal(loaded.weight.detach().numpy(), [3.0, 4.0])
            self.assertFalse(loaded.training)
            self.assertEqual(str(loaded.weight.device), "cpu")
            self.assertEqual(device, "cpu")
            self.assertEqual(metadata, {
                "architecture": "ViT-B-32", "force_quick_gelu": True,
                "precision": "fp32", "device": "cpu", "requested_device": "cpu",
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": hashlib.sha256(before).hexdigest(), "size_bytes": len(before),
                },
                "runtime": {"torch": torch.__version__, "open_clip": open_clip.__version__},
                "preprocessing": {
                    "frame_size": [224, 224], "input_color": "BGR", "model_color": "RGB",
                    "pixel_scale": 1 / 255,
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "std": [0.26862954, 0.26130258, 0.27577711],
                    "resize": "caller", "crop": None, "embedding_normalization": "L2",
                },
            })
            json.dumps(metadata, allow_nan=False)
            self.assertEqual(checkpoint.read_bytes(), before)

            empty = base / "empty.pth"
            empty.touch()
            unsupported = base / "numpy-branch.npz"
            unsupported.write_bytes(b"not a supported checkpoint format")
            with patch("open_clip.create_model", return_value=TinyModel()) as constructor:
                for invalid in (base / "missing.pt", empty, unsupported):
                    with self.subTest(checkpoint=invalid.name):
                        with self.assertRaises(ValueError):
                            load_clip_encoder(invalid)
                for invalid in (None, "", "not-a-device"):
                    with self.subTest(device=invalid):
                        with self.assertRaises(ValueError):
                            load_clip_encoder(checkpoint, device=invalid)
                constructor.assert_not_called()

            incompatible = base / "incompatible.pt"
            torch.save({"wrong_weight": torch.tensor([3.0, 4.0])}, incompatible)
            with patch("open_clip.create_model", return_value=TinyModel()):
                with self.assertRaises(RuntimeError):
                    load_clip_encoder(incompatible)

    def test_preprocessing_and_normalized_features_preserve_frame_order(self):
        # A strided, read-only view: red, green, blue, white, then black in BGR.
        storage = np.empty((5, 224, 448, 3), dtype=np.uint8)
        storage[:] = np.array([
            [0, 0, 255], [0, 255, 0], [255, 0, 0],
            [255, 255, 255], [0, 0, 0],
        ], dtype=np.uint8)[:, None, None, :]
        frames = storage[:, :, ::2, :]
        frames.flags.writeable = False
        before = storage.copy()
        model = PixelEncoder().eval()

        result = encode_clip_frames(model, torch.device("cpu"), frames, batch_size=2)

        # Hand-derived (RGB/255 - CLIP mean)/std and unit directions. The
        # encoder's factor of two cancels only when feature rows normalize.
        expected_pixels = np.array([
            [1.93033625, -1.75209713, -1.48021977],
            [-1.79226253, 2.07488384, -1.48021977],
            [-1.79226253, -1.75209713, 2.14589699],
            [1.93033625, 2.07488384, 2.14589699],
            [-1.79226253, -1.75209713, -1.48021977],
        ])
        expected_features = np.array([
            [0.64390730, -0.58445161, -0.49376077],
            [-0.57521186, 0.66591683, -0.47506431],
            [-0.54318660, -0.53101354, 0.65036370],
            [0.54303080, 0.58369408, 0.60367108],
            [-0.61571603, -0.60191756, -0.50851649],
        ])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, expected_features, rtol=1e-6, atol=1e-6)
        self.assertEqual([tuple(batch.shape) for batch in model.inputs],
                         [(2, 3, 224, 224), (2, 3, 224, 224), (1, 3, 224, 224)])
        observed = torch.cat(model.inputs).numpy()
        np.testing.assert_allclose(observed[:, :, 0, 0], expected_pixels, atol=1e-6)
        np.testing.assert_allclose(observed[:, :, -1, -1], expected_pixels, atol=1e-6)
        self.assertEqual(model.modes, [(True, False)] * 3)
        self.assertFalse(model.training)
        self.assertIsNone(model.scale.grad)
        self.assertFalse(frames.flags.writeable)
        np.testing.assert_array_equal(storage, before)

    def test_invalid_inputs_and_encoder_outputs_fail_without_hiding_runtime_errors(self):
        frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        model = PixelEncoder().eval()
        for invalid in ([], frames[:0], frames[:, :223], frames.astype(np.float32)):
            with self.subTest(frames_shape=getattr(invalid, "shape", None)):
                with self.assertRaises(ValueError):
                    encode_clip_frames(model, "cpu", invalid)
        for batch_size in (0, True, 1.5, np.int64(2)):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    encode_clip_frames(model, "cpu", frames, batch_size=batch_size)
        training = PixelEncoder()
        with self.assertRaises(ValueError):
            encode_clip_frames(training, "cpu", frames)
        self.assertTrue(training.training)

        cases = (
            ("not tensor", [np.ones((2, 3))], 2),
            ("wrong rank", [torch.ones(2)], 2),
            ("wrong rows", [torch.ones(1, 3)], 2),
            ("empty width", [torch.ones(2, 0)], 2),
            ("nonfinite", [torch.full((2, 3), float("nan"))], 2),
            ("zero norm", [torch.zeros(2, 3)], 2),
            ("complex", [torch.ones(2, 3, dtype=torch.complex64)], 2),
            ("changing width", [torch.ones(1, 3), torch.ones(1, 4)], 1),
        )
        for name, outputs, batch_size in cases:
            with self.subTest(output=name):
                with self.assertRaises(ValueError):
                    encode_clip_frames(PixelEncoder(outputs).eval(), "cpu", frames,
                                       batch_size=batch_size)

        # Finite nonzero rows remain valid even when an unscaled norm or an
        # early float32 conversion would overflow.
        for value, dtype in ((1e20, torch.float32), (1e300, torch.float64)):
            with self.subTest(finite_large_output=value):
                outputs = [torch.full((2, 2), value, dtype=dtype)]
                result = encode_clip_frames(PixelEncoder(outputs).eval(), "cpu", frames)
                np.testing.assert_allclose(result, [[0.70710678, 0.70710678]] * 2,
                                           rtol=1e-6, atol=1e-6)
                self.assertEqual(result.dtype, np.float32)

        error = RuntimeError("encoder failed")
        with self.assertRaises(RuntimeError) as caught:
            encode_clip_frames(PixelEncoder([error]).eval(), "cpu", frames)
        self.assertIs(caught.exception, error)
        with self.assertRaises(RuntimeError):
            encode_clip_frames(model, "invalid-device", frames)


if __name__ == "__main__":
    unittest.main()
