"""Cohorts, cached frame features and reference-side DTW step costs."""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.classical_pairwise_dtw import dtw_pair, label_frames  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
EXPERIMENT = Path(__file__).resolve().parents[1]
FEATURE_DIR = DATA_DIR / "features"
HF_REPO_ID = "ttyue/COIN_Dataset"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def load_database() -> dict:
    return json.loads((DATA_DIR / "raw" / "COIN.json").read_text())["database"]


def load_mirror() -> dict[str, int]:
    """Map YouTube id -> file size for every video on the Hugging Face mirror."""
    cache = DATA_DIR / "raw" / "hf_mirror_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi

    mirror = {}
    for item in HfApi().list_repo_tree(
        HF_REPO_ID, repo_type="dataset", path_in_repo="videos", recursive=True
    ):
        path = getattr(item, "path", "")
        if path.endswith(".mp4"):
            mirror[path.rsplit("/", 1)[-1][:-4]] = int(item.size or 0)
    cache.write_text(json.dumps(mirror))
    return mirror


def expected_steps(canon: list[str], seq: list[str]) -> list[int] | None:
    """Canonical step indices a deviant should light up, or None if not a deviant.

    A deviant either drops steps (the rest stay in order, no repeats) or keeps
    the same steps in another order.
    """
    if seq == canon:
        return None
    if len(seq) < len(canon) and len(set(seq)) == len(seq) and all(s in canon for s in seq):
        if [s for s in canon if s in seq] == seq:
            return [i for i, s in enumerate(canon) if s not in seq]
    if len(seq) == len(canon) and sorted(seq) == sorted(canon) and len(set(seq)) == len(seq):
        return [i for i, s in enumerate(canon) if seq[i] != s]
    return None


def task_cohort(database: dict, task: str, mirror: dict[str, int]) -> dict:
    """Correct videos (most common exact step sequence) and natural deviants."""
    on_mirror = {
        vid: info for vid, info in database.items()
        if info["class"] == task and vid in mirror
    }
    sequences = collections.Counter(
        tuple(str(s["id"]) for s in info["annotation"]) for info in on_mirror.values()
    )
    canon_ids = list(sequences.most_common(1)[0][0])
    correct = sorted(
        vid for vid, info in on_mirror.items()
        if [str(s["id"]) for s in info["annotation"]] == canon_ids
    )
    # Every correct video carries the identical ordered step ids.
    assert all(
        [str(s["id"]) for s in on_mirror[vid]["annotation"]] == canon_ids for vid in correct
    )
    canon_labels = [s["label"] for s in on_mirror[correct[0]]["annotation"]]
    deviants = {}
    for vid, info in sorted(on_mirror.items()):
        seq = [s["label"] for s in info["annotation"]]
        expected = expected_steps(canon_labels, seq)
        if expected is not None:
            kind = "reorder" if len(seq) == len(canon_labels) else "skip"
            deviants[vid] = {"kind": kind, "expected": expected, "steps": seq}
    return {
        "task": task,
        "canon_ids": canon_ids,
        "canon_labels": canon_labels,
        "correct": correct,
        "deviants": deviants,
    }


def video_path(task: str, vid: str) -> Path:
    return DATA_DIR / "videos" / task / f"{vid}.mp4"


def download(task: str, vid: str, info: dict) -> Path:
    dest = video_path(task, vid)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    import shutil
    from huggingface_hub import hf_hub_download

    staging = DATA_DIR / "_hf_staging"
    src = Path(hf_hub_download(
        repo_id=HF_REPO_ID, repo_type="dataset",
        filename=f"videos/{info['recipe_type']}/{vid}.mp4", local_dir=staging,
    ))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dest)
    return dest


def row_for(database: dict, vid: str) -> dict:
    info = database[vid]
    return {
        "youtube_id": vid,
        "video_path": str(video_path(info["class"], vid)),
        "roi_start": info["start"],
        "roi_end": info["end"],
        "steps": [
            {"id": s["id"], "label": s["label"], "start": s["segment"][0], "end": s["segment"][1]}
            for s in info["annotation"]
        ],
    }


class Extractor:
    """Compute and cache L2-normalized frame features for one encoder."""

    def __init__(self, encoder: str):
        self.encoder = encoder
        self.model = None

    def __call__(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        cache = FEATURE_DIR / self.encoder / f"{row['youtube_id']}.npz"
        if cache.exists():
            data = np.load(cache)
            return data["features"], data["times"]
        if self.encoder == "classical":
            from scripts.classical_pairwise_dtw import sample_video

            features, times = sample_video(row)
        else:
            from scripts import ml_pairwise_dtw as clip

            if self.model is None:
                self.model, self.device = clip.load_clip()
            frames, times = clip.sample_frames(row)
            features = clip.embed_frames(self.model, self.device, frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=features, times=times)
        return features, times


def reference_step_costs(
    ref_features: np.ndarray, ref_labels: np.ndarray, query: np.ndarray, n_steps: int
) -> np.ndarray:
    """Mean path cost over the reference's frames in each step (NaN if none)."""
    cost = 1.0 - ref_features @ query.T
    np.clip(cost, 0.0, 2.0, out=cost)
    dummy = np.full(len(query), -1, dtype=np.int8)
    _, sum_a, count_a, _, _ = dtw_pair(cost.astype(np.float64), ref_labels, dummy, n_steps)
    out = np.full(n_steps, np.nan)
    seen = count_a > 0
    out[seen] = sum_a[seen] / count_a[seen]
    return out


def synthetic_deviants(features: np.ndarray, labels: np.ndarray, n_steps: int) -> dict:
    """Skip each step, and swap each pair of neighbouring steps, in one correct video."""
    blocks = {}
    for k in range(n_steps):
        where = np.flatnonzero(labels == k)
        blocks[k] = (int(where[0]), int(where[-1]) + 1)
    out = {}
    for k in range(n_steps):
        start, end = blocks[k]
        keep = np.r_[0:start, end:len(features)]
        if len(keep) >= 2:
            out[f"skip_{k}"] = (features[keep], [k])
    for k in range(n_steps - 1):
        (a0, a1), (b0, b1) = blocks[k], blocks[k + 1]
        if a1 > b0:
            continue
        order = np.r_[0:a0, b0:b1, a1:b0, a0:a1, b1:len(features)]
        out[f"swap_{k}"] = (features[order], [k, k + 1])
    return out


__all__ = [
    "Extractor", "label_frames", "log", "load_database", "load_mirror",
    "reference_step_costs", "row_for", "synthetic_deviants", "task_cohort", "download",
    "EXPERIMENT",
]


def step_prototypes(refs: dict, n_steps: int, skip_id: str | None = None) -> np.ndarray:
    """Unit-length mean embedding of each step, pooled over the references."""
    protos = []
    for k in range(n_steps):
        frames = np.concatenate([f[l == k] for vid, (f, l) in refs.items() if vid != skip_id])
        mean = frames.mean(axis=0)
        protos.append(mean / np.linalg.norm(mean))
    return np.stack(protos)


def prototype_profile(query: np.ndarray, protos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of query frames nearest each step, and their mean position (0 to 1).

    A step with no frames gets position NaN.
    """
    nearest = np.argmax(query @ protos.T, axis=1)
    position = np.linspace(0.0, 1.0, len(query))
    share = np.array([(nearest == k).mean() for k in range(len(protos))])
    where = [position[nearest == k] for k in range(len(protos))]
    mean_pos = np.array([w.mean() if len(w) else np.nan for w in where])
    return share, mean_pos


def step_separation(features: np.ndarray, labels: np.ndarray, n_steps: int) -> float:
    """Mean cosine distance between frames of different steps minus within the same step."""
    within, between = [], []
    for a in range(n_steps):
        fa = features[labels == a]
        for b in range(a, n_steps):
            fb = features[labels == b]
            d = float((1.0 - fa @ fb.T).mean())
            (within if a == b else between).append(d)
    return float(np.mean(between) - np.mean(within))
