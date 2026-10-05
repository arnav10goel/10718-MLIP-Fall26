"""Non-ML baseline: pairwise DTW on the 48 canonical ReplaceSIMCard videos.

A video is included when it has exactly three steps, in order:

    use the needle to open the SIM card slot
    put the SIM card into the SIM card slot
    press the SIM card slot back

Each ROI is sampled at 5 fps. A frame becomes a classical descriptor: a hue
histogram, a saturation histogram, and a 4x4 grid of gradient-orientation
histograms. Descriptors are L2-normalized. The distance between two videos is
the mean cosine distance along the full DTW path, so clips of different
lengths stay comparable.

There are 48 choose 2 = 1128 pairs. The script also records, for each pair,
the mean path cost inside each of the three steps.

Outputs:
    data/prepared/sim_classical_dtw_summary.json
    data/prepared/sim_classical_dtw_matrix.npy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402

FPS = 5.0
STEP_LABELS = (
    "use the needle to open the SIM card slot",
    "put the SIM card into the SIM card slot",
    "press the SIM card slot back",
)
SUMMARY_PATH = PREPARED_DIR / "sim_classical_dtw_summary.json"
MATRIX_PATH = PREPARED_DIR / "sim_classical_dtw_matrix.npy"
FEATURE_CACHE = PREPARED_DIR / "sim_classical_features.npz"


def canonical_videos() -> list[dict]:
    rows = []
    with (PREPARED_DIR / "manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["task"] != "ReplaceSIMCard" or len(row["steps"]) != 3:
                continue
            if [step["label"] for step in row["steps"]] != list(STEP_LABELS):
                continue
            rows.append(row)
    rows.sort(key=lambda row: row["youtube_id"])
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 canonical videos, found {len(rows)}")
    return rows


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
    feature = np.concatenate([hue, sat, hog]).astype(np.float32)
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
def dtw_pair(
    cost: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, n_steps: int = 3
):
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
    )


def step_distance(sum_a, count_a, sum_b, count_b, step_index: int) -> float:
    sides = []
    if count_a[step_index] > 0:
        sides.append(sum_a[step_index] / count_a[step_index])
    if count_b[step_index] > 0:
        sides.append(sum_b[step_index] / count_b[step_index])
    return float(np.mean(sides))


def load_or_extract(rows: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ids = [row["youtube_id"] for row in rows]
    if FEATURE_CACHE.exists():
        cached = np.load(FEATURE_CACHE, allow_pickle=True)
        if list(cached["ids"]) == ids:
            features = list(cached["features"])
            labels = list(cached["labels"])
            print(f"loaded cached features for {len(ids)} videos")
            return features, labels
    features = []
    labels = []
    for index, row in enumerate(rows, start=1):
        frames, times = sample_video(row)
        features.append(frames)
        labels.append(label_frames(times, row["steps"]))
        print(
            f"[{index}/{len(rows)}] {row['youtube_id']} {len(frames)} frames",
            flush=True,
        )
    np.savez_compressed(
        FEATURE_CACHE,
        ids=np.array(ids),
        features=np.array(features, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    return features, labels


def main() -> None:
    rows = canonical_videos()
    features, labels = load_or_extract(rows)
    n = len(rows)
    n_pairs = n * (n - 1) // 2
    print(f"aligning {n_pairs} pairs")

    # Compile the DTW loop before the timed run.
    tiny = np.zeros((2, 2), dtype=np.float64)
    dtw_pair(tiny, np.zeros(2, dtype=np.int8), np.zeros(2, dtype=np.int8))

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
        print(f"finished reference {i + 1}/{n}")

    def stats(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }

    summary = {
        "n_videos": n,
        "n_pairs": n_pairs,
        "fps": FPS,
        "feature": (
            "L2-normalized 16-bin hue histogram, 8-bin saturation histogram, "
            "and a 4x4 grid of 9-bin gradient orientation histograms"
        ),
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
