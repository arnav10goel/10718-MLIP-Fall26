"""Shared classical frame features and cosine DTW.

Each annotated region is sampled at 5 fps. A frame becomes a hue histogram, a
saturation histogram, and a 4x4 grid of gradient-orientation histograms.
Descriptors are L2-normalized. The distance between two videos is the mean
cosine distance along the full DTW path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coin_tasks import REPO_ROOT  # noqa: E402

FPS = 5.0


def frame_feature(frame_bgr: np.ndarray) -> np.ndarray:
    small = frame_bgr
    if small.shape[0] != 64 or small.shape[1] != 64:
        small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = cv2.calcHist([hsv], [0], None, [16], [0, 180]).ravel()
    sat = cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    magnitude_cells = magnitude.reshape(4, 16, 4, 16).swapaxes(1, 2).reshape(16, 256)
    angle_cells = angle.reshape(4, 16, 4, 16).swapaxes(1, 2).reshape(16, 256)
    bin_index = np.clip((angle_cells / 20.0).astype(np.int32), 0, 8)
    hog_cells = np.zeros((16, 9), np.float32)
    for bin_id in range(9):
        hog_cells[:, bin_id] = (magnitude_cells * (bin_index == bin_id)).sum(axis=1)
    hog = hog_cells.ravel()
    # Normalize each block first. Raw gradient sums are far larger than the
    # colour counts and would otherwise hold ~99% of the vector.
    blocks = []
    for block in (hue, sat, hog):
        block = block.astype(np.float32)
        block_norm = float(np.linalg.norm(block))
        blocks.append(block / block_norm if block_norm > 0 else block)
    feature = np.concatenate(blocks)
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return feature


def sample_video(row: dict) -> tuple[np.ndarray, np.ndarray]:
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
        f"fps={FPS},scale=64:64",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = 64 * 64 * 3
    if len(raw) < frame_bytes * 2:
        raise RuntimeError(f"{row['youtube_id']} produced {len(raw) // frame_bytes} frames")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 64, 64, 3)
    features = np.stack([frame_feature(frame) for frame in frames])
    times = row["roi_start"] + np.arange(len(frames), dtype=np.float64) / FPS
    return features, times


def label_frames(times: np.ndarray, steps: list[dict]) -> np.ndarray:
    labels = np.full(len(times), -1, dtype=np.int8)
    for step_index, step in enumerate(steps):
        inside = (times >= step["start"]) & (times < step["end"])
        labels[inside] = step_index
        if not np.any(labels == step_index):
            middle = 0.5 * (step["start"] + step["end"])
            labels[int(np.argmin(np.abs(times - middle)))] = step_index
    return labels


@njit
def dtw_alignment(
    cost: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, n_steps: int = 3
):
    """Global DTW, plus the side-A frames matched to each side-B step."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    for label in labels_a:
        if label < -1 or label >= n_steps:
            raise ValueError("labels_a contains a step outside n_steps")
    for label in labels_b:
        if label < -1 or label >= n_steps:
            raise ValueError("labels_b contains a step outside n_steps")
    n, m = cost.shape
    inf = 1e18
    accumulated = np.empty((n + 1, m + 1))
    pointer = np.empty((n, m), np.uint8)
    accumulated[0, 0] = 0.0
    for i in range(1, n + 1):
        accumulated[i, 0] = inf
    for j in range(1, m + 1):
        accumulated[0, j] = inf
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = accumulated[i - 1, j - 1]
            up = accumulated[i - 1, j]
            left = accumulated[i, j - 1]
            if diagonal <= up and diagonal <= left:
                best = diagonal
                choice = np.uint8(0)
            elif up <= left:
                best = up
                choice = np.uint8(1)
            else:
                best = left
                choice = np.uint8(2)
            pointer[i - 1, j - 1] = choice
            accumulated[i, j] = cost[i - 1, j - 1] + best

    i = n - 1
    j = m - 1
    path_sum = 0.0
    path_count = 0
    step_sum_a = np.zeros(n_steps)
    step_count_a = np.zeros(n_steps)
    step_sum_b = np.zeros(n_steps)
    step_count_b = np.zeros(n_steps)
    first_a_for_b = np.full(n_steps, n, dtype=np.int32)
    last_a_for_b = np.full(n_steps, -1, dtype=np.int32)
    while True:
        local = cost[i, j]
        path_sum += local
        path_count += 1
        label_a = labels_a[i]
        label_b = labels_b[j]
        if label_a >= 0:
            step_sum_a[label_a] += local
            step_count_a[label_a] += 1
        if label_b >= 0:
            step_sum_b[label_b] += local
            step_count_b[label_b] += 1
            if i < first_a_for_b[label_b]:
                first_a_for_b[label_b] = i
            if i > last_a_for_b[label_b]:
                last_a_for_b[label_b] = i
        if i == 0 and j == 0:
            break
        choice = pointer[i, j]
        if choice == 0:
            i -= 1
            j -= 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    return (
        path_sum / path_count,
        step_sum_a,
        step_count_a,
        step_sum_b,
        step_count_b,
        first_a_for_b,
        last_a_for_b,
    )


@njit
def dtw_pair(
    cost: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, n_steps: int = 3
):
    path_mean, sum_a, count_a, sum_b, count_b, _, _ = dtw_alignment(
        cost, labels_a, labels_b, n_steps
    )
    return path_mean, sum_a, count_a, sum_b, count_b


def step_distance(sum_a, count_a, sum_b, count_b, step_index: int) -> float:
    sides = []
    if count_a[step_index] > 0:
        sides.append(sum_a[step_index] / count_a[step_index])
    if count_b[step_index] > 0:
        sides.append(sum_b[step_index] / count_b[step_index])
    return float(np.mean(sides))
