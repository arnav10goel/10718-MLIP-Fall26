"""Zero-shot step recognition: match each frame to the CLIP text vector of each step label.

Uses the cached CLIP frame features of every correct video scored in runs/*.json.
Reports per-frame accuracy against chance (1 / number of steps), and a synthetic
skip test: remove step k's frames from a correct video and check whether the
share of frames assigned to step k drops below the normal range.

Usage (repo root): python experiments/deviation_detection/scripts/text_steps.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EXPERIMENT, FEATURE_DIR, label_frames, load_database, log, row_for  # noqa: E402

TEMPLATES = ("a photo of a person {}.", "a video frame of someone who is about to {}.", "{}")


@torch.inference_mode()
def text_vectors(model, tokenizer, labels: list[str]) -> np.ndarray:
    out = []
    for label in labels:
        tokens = tokenizer([t.format(label) for t in TEMPLATES])
        v = model.encode_text(tokens).float()
        v = v / v.norm(dim=-1, keepdim=True)
        v = v.mean(dim=0)
        out.append((v / v.norm()).numpy())
    return np.stack(out)


def main() -> None:
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    database = load_database()
    results = {}
    seen = set()
    for path in sorted(glob.glob(str(EXPERIMENT / "runs" / "*.json"))):
        report = json.loads(Path(path).read_text())
        task = report["task"]
        if task in seen or report.get("encoder") != "clip" or "counts" not in report:
            continue
        seen.add(task)
        steps = report["steps"]
        n = len(steps)
        text = text_vectors(model, tokenizer, steps)
        vids = report["references"] + [h["video_id"] for h in report["held_out_correct"]]
        correct, total = 0, 0
        per_step = np.zeros((n, n))
        shares, skips = [], []
        for vid in vids:
            data = np.load(FEATURE_DIR / "clip" / f"{vid}.npz")
            feats, labels = data["features"], label_frames(data["times"], row_for(database, vid)["steps"])
            pred = np.argmax(feats @ text.T, axis=1)
            mask = labels >= 0
            correct += int((pred[mask] == labels[mask]).sum())
            total += int(mask.sum())
            for k in range(n):
                for j in range(n):
                    per_step[k, j] += int(((labels == k) & (pred == j)).sum())
            shares.append([(pred == k).mean() for k in range(n)])
            for k in range(n):
                keep = labels != k
                if keep.sum() >= 2:
                    skips.append((k, (pred[keep] == k).mean()))
        shares = np.array(shares)
        mean, sd = shares.mean(axis=0), shares.std(axis=0)
        low = mean - 2 * sd
        caught = [share < low[k] for k, share in skips]
        z_skip = [(share - mean[k]) / sd[k] for k, share in skips]
        acc = correct / total
        results[task] = {
            "steps": steps, "videos": len(vids), "frame_accuracy": round(acc, 3),
            "chance": round(1 / n, 3),
            "confusion_rows_true_cols_pred": per_step.astype(int).tolist(),
            "share_mean": mean.round(3).tolist(), "share_sd": sd.round(3).tolist(),
            "skip_caught": round(float(np.mean(caught)), 3),
            "skip_mean_z": round(float(np.mean(z_skip)), 2),
        }
        log(f"{task:24s} videos {len(vids):2d} accuracy {acc:.2f} (chance {1 / n:.2f}) "
            f"skip caught {np.mean(caught):.0%} mean z {np.mean(z_skip):.2f}")
    (EXPERIMENT / "runs" / "text_steps.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
