"""CLIP frame embeddings for the shared cosine DTW.

Frames are sampled at the same 5 fps as the classical features. Each frame
becomes an L2-normalized CLIP ViT-B/32 image embedding (OpenAI weights, 512-D).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_pairwise_dtw import FPS  # noqa: E402
from coin_tasks import REPO_ROOT  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sample_frames(row: dict) -> tuple[np.ndarray, np.ndarray]:
    path = REPO_ROOT / row["video_path"]
    duration = max(row["roi_end"] - row["roi_start"], 1.0 / FPS)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-ss",
        f"{row['roi_start']:.3f}",
        "-i",
        str(path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"fps={FPS},scale=224:224:flags=bicubic",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = 224 * 224 * 3
    count = len(raw) // frame_bytes
    if count < 2:
        raise RuntimeError(f"{row['youtube_id']} produced {count} frames")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, 224, 224, 3)
    times = row["roi_start"] + np.arange(count, dtype=np.float64) / FPS
    return frames, times


def load_clip():
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


@torch.inference_mode()
def embed_frames(model, device: str, frames_bgr: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(frames_bgr[:, :, :, ::-1])
    batch = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    batch = (batch.to(device) - mean) / std
    parts = []
    for start in range(0, len(batch), 32):
        features = model.encode_image(batch[start : start + 32])
        # Scale before the float32 norm so a float16 embedding cannot overflow.
        scale = features.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        features = (features / scale).float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        parts.append(features.cpu().numpy())
    return np.concatenate(parts).astype(np.float32)
