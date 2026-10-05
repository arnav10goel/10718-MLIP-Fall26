"""Score a COIN reference cohort with the teammate's CLIP frame features."""

from __future__ import annotations

from importlib import import_module

from scripts.score_coin_reference import score_reference_cohort


def score_clip_reference_cohort(rows: list[dict], reference_id: str) -> dict:
    """Load CLIP once, then score each video's embeddings without a cache."""
    clip = None
    model = None
    device = None

    def sample_features(row: dict):
        nonlocal clip, model, device
        if clip is None:
            clip = import_module("scripts.ml_pairwise_dtw")
            model, device = clip.load_clip()
        frames, times = clip.sample_frames(row)
        return clip.embed_frames(model, device, frames), times

    return score_reference_cohort(rows, reference_id, sample_features=sample_features)
