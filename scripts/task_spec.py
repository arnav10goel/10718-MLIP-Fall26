"""Task settings for the shared DTW runner.

A task script builds one TaskSpec. The runner selects COIN videos whose step
labels match that spec, writes the all-pairs cutoff, and can score recordings.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402


@dataclass(frozen=True)
class TaskSpec:
    task: str
    step_labels: tuple[str, ...]
    expected_videos: int
    artifact_stem: str
    recording_dirname: str = "recorded"
    recording_count: int | None = None
    annotations_name: str = "recorded_annotations.json"
    annotated_output_name: str = "recorded_similarity.json"
    unlabeled_output_name: str = "recorded_similarity_unlabeled.json"
    recording_classical_cache_name: str = "recorded_classical_features.npz"
    recording_clip_cache_name: str = "recorded_clip_features.npz"

    def recording_dir(self) -> Path:
        return REPO_ROOT / "data" / "videos" / self.task / self.recording_dirname

    def annotations_path(self) -> Path:
        return PREPARED_DIR / self.annotations_name

    def annotated_output(self) -> Path:
        return PREPARED_DIR / self.annotated_output_name

    def unlabeled_output(self) -> Path:
        return PREPARED_DIR / self.unlabeled_output_name

    def reference_cache(self, encoder: str) -> Path:
        return PREPARED_DIR / f"{self.artifact_stem}_{encoder}_features.npz"

    def summary_path(self, encoder: str) -> Path:
        return PREPARED_DIR / f"{self.artifact_stem}_{encoder}_dtw_summary.json"

    def matrix_path(self, encoder: str) -> Path:
        return PREPARED_DIR / f"{self.artifact_stem}_{encoder}_dtw_matrix.npy"

    def recording_cache(self, encoder: str) -> Path:
        if encoder == "classical":
            name = self.recording_classical_cache_name
        elif encoder == "clip":
            name = self.recording_clip_cache_name
        else:
            raise ValueError(f"unknown encoder: {encoder}")
        return PREPARED_DIR / name


def load_manifest_rows() -> list[dict]:
    path = PREPARED_DIR / "manifest.jsonl"
    rows = []
    with path.open() as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def matching_rows(rows: list[dict], spec: TaskSpec) -> list[dict]:
    """Keep videos of this task whose steps are exactly spec.step_labels, in order."""
    wanted = list(spec.step_labels)
    matched = [
        row
        for row in rows
        if row.get("task") == spec.task
        and [step["label"] for step in row.get("steps", [])] == wanted
    ]
    matched.sort(key=lambda row: row["youtube_id"])
    if len(matched) != spec.expected_videos:
        raise RuntimeError(
            f"expected {spec.expected_videos} canonical {spec.task} videos, "
            f"found {len(matched)}"
        )
    return matched
