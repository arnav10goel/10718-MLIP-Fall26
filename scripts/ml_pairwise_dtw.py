"""ML baseline: the same 1,128 DTW pairs, with CLIP frame embeddings.

Same 48 ReplaceSIMCard videos, same 5 fps sampling, same cosine DTW as
scripts/classical_pairwise_dtw.py. The only change is the frame descriptor:
an L2-normalized CLIP ViT-B/32 image embedding (OpenAI weights, 512-D).

Outputs:
    data/prepared/sim_clip_dtw_summary.json
    data/prepared/sim_clip_dtw_matrix.npy
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_pairwise_dtw import (  # noqa: E402
    FPS,
    STEP_LABELS,
    canonical_videos,
    dtw_pair,
    label_frames,
    step_distance,
)
from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402

SUMMARY_PATH = PREPARED_DIR / "sim_clip_dtw_summary.json"
MATRIX_PATH = PREPARED_DIR / "sim_clip_dtw_matrix.npy"
FEATURE_CACHE = PREPARED_DIR / "sim_clip_features.npz"
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
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        parts.append(features.float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def load_or_extract(rows: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ids = [row["youtube_id"] for row in rows]
    if FEATURE_CACHE.exists():
        cached = np.load(FEATURE_CACHE, allow_pickle=True)
        if list(cached["ids"]) == ids:
            print(f"loaded cached CLIP features for {len(ids)} videos", flush=True)
            return list(cached["features"]), list(cached["labels"])

    model, device = load_clip()
    print(f"CLIP ViT-B/32 on {device}", flush=True)
    features = []
    labels = []
    for index, row in enumerate(rows, start=1):
        frames, times = sample_frames(row)
        features.append(embed_frames(model, device, frames))
        labels.append(label_frames(times, row["steps"]))
        print(f"[{index}/{len(rows)}] {row['youtube_id']} {len(frames)} frames", flush=True)
    np.savez_compressed(
        FEATURE_CACHE,
        ids=np.array(ids),
        features=np.array(features, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    return features, labels


def stats(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    rows = canonical_videos()
    features, labels = load_or_extract(rows)
    n = len(rows)
    n_pairs = n * (n - 1) // 2
    print(f"aligning {n_pairs} pairs", flush=True)
    dtw_pair(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(2, dtype=np.int8),
        np.zeros(2, dtype=np.int8),
    )

    matrix = np.zeros((n, n), dtype=np.float64)
    global_costs = np.empty(n_pairs, dtype=np.float64)
    step_costs = np.empty((n_pairs, 3), dtype=np.float64)
    pair_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            cost = 1.0 - features[i] @ features[j].T
            np.clip(cost, 0.0, 2.0, out=cost)
            path_mean, sum_a, count_a, sum_b, count_b = dtw_pair(
                cost.astype(np.float64), labels[i], labels[j]
            )
            matrix[i, j] = matrix[j, i] = path_mean
            global_costs[pair_index] = path_mean
            for step_index in range(3):
                step_costs[pair_index, step_index] = step_distance(
                    sum_a, count_a, sum_b, count_b, step_index
                )
            pair_index += 1
        print(f"finished reference {i + 1}/{n}", flush=True)

    summary = {
        "n_videos": n,
        "n_pairs": n_pairs,
        "fps": FPS,
        "model": "CLIP ViT-B/32, OpenAI weights",
        "feature": "L2-normalized 512-D image embedding",
        "distance": "mean cosine distance along the full DTW path",
        "video_ids": [row["youtube_id"] for row in rows],
        "global": stats(global_costs),
        "steps": {
            label: stats(step_costs[:, step_index])
            for step_index, label in enumerate(STEP_LABELS)
        },
    }
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MATRIX_PATH, matrix)
    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"pairs: {n_pairs}")
    print(
        f"global mean {summary['global']['mean']:.4f} "
        f"std {summary['global']['std']:.4f}"
    )
    for label, step_stats in summary["steps"].items():
        print(f"{label}: mean {step_stats['mean']:.4f} std {step_stats['std']:.4f}")
    print(f"wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
