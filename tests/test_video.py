"""Behavior tests for sampling actual video frames and elapsed timestamps."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

from src.guideme.video import sample_video_frames


def _write_video(path, timestamps, colors):
    """Create a lossless temporary video with hand-authored PTS and BGR colors."""
    frames = np.stack([
        np.full((12, 16, 3), color, dtype=np.uint8) for color in colors
    ])
    pts = str(timestamps[-1])
    for index in range(len(timestamps) - 2, -1, -1):
        pts = f"if(eq(N,{index}),{timestamps[index]},{pts})"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "16x12",
            "-framerate", "8", "-i", "pipe:0",
            "-vf", "setpts=" + pts.replace(",", "\\,") + "/TB",
            "-fps_mode", "passthrough", "-c:v", "ffv1", "-pix_fmt", "bgr0",
            str(path),
        ],
        input=frames.tobytes(), capture_output=True, check=True,
    )


class VideoSamplingTests(unittest.TestCase):
    def test_samples_source_frames_inside_window_with_actual_elapsed_pts(self):
        colors = [
            (0, 0, 240), (0, 240, 0), (240, 0, 0), (40, 80, 120),
            (180, 60, 20), (20, 100, 200), (100, 200, 20), (220, 40, 160),
        ]
        cases = (
            (
                "constant rate", [0, .125, .25, .375, .5, .625, .75, .875],
                .25, .75, [.25, .5], [2, 4],
            ),
            (
                "variable rate and source offset", [2, 2.125, 2.375, 2.5, 2.875, 3.125],
                .1, 1.125, [.125, .375, .875], [1, 2, 4],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, pts, start, end, expected_times, indices in cases:
                with self.subTest(case=name):
                    source = Path(directory) / (name.replace(" ", "-") + ".mkv")
                    _write_video(source, pts, colors[:len(pts)])
                    original = source.read_bytes()

                    frames, times = sample_video_frames(
                        source, start, end, fps=4.0, size=(12, 10),
                    )

                    self.assertEqual(frames.shape, (len(indices), 10, 12, 3))
                    self.assertEqual(frames.dtype, np.uint8)
                    self.assertEqual(times.dtype, np.float64)
                    np.testing.assert_allclose(times, expected_times, rtol=0, atol=1e-6)
                    for frame, index in zip(frames, indices):
                        np.testing.assert_array_equal(
                            frame, np.full((10, 12, 3), colors[index], dtype=np.uint8),
                        )
                    self.assertEqual(source.read_bytes(), original)

    def test_rejects_invalid_parameters_sources_and_windows_without_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mkv"
            _write_video(source, [0, .125], [(0, 0, 240), (240, 0, 0)])
            invalid = (
                {"start_s": -1},
                {"end_s": .25},
                {"end_s": float("inf")},
                {"fps": True},
                {"fps": 0},
                {"size": (0, 10)},
                {"size": (True, 10)},
                {"size": (12.0, 10)},
            )
            for overrides in invalid:
                with self.subTest(parameters=overrides):
                    parameters = {"start_s": .25, "end_s": 1.0}
                    parameters.update(overrides)
                    with self.assertRaises(ValueError):
                        sample_video_frames(source, **parameters)

            for start, end in ((.01, .1), (1, 2)):
                with self.subTest(empty_window=(start, end)):
                    with self.assertRaises(ValueError):
                        sample_video_frames(source, start, end)

            empty = Path(directory) / "empty.mkv"
            empty.touch()
            corrupt = Path(directory) / "corrupt.mkv"
            corrupt.write_bytes(b"this is not a video")
            for invalid_source in (empty, corrupt, Path(directory) / "missing.mkv", Path(directory)):
                with self.subTest(source=invalid_source.name):
                    # Decoder errors may be reported as ValueError or RuntimeError.
                    with self.assertRaises((ValueError, RuntimeError)):
                        sample_video_frames(invalid_source, 0, 1)


if __name__ == "__main__":
    unittest.main()
