"""One contact sheet per task: a frame from the middle of each step, one row per video.

Also writes runs/video_info.json with resolution, task length and split per video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deviation_detection" / "scripts"))

from common import load_database, load_mirror, log, task_cohort, video_path  # noqa: E402

EXPERIMENT = Path(__file__).resolve().parents[1]
TASKS = ("UseRiceCookerToCookRice", "MakePaperWindMill", "PutOnQuiltCover")
W, H, LABEL_W = 240, 135, 230


def frame_at(cap: cv2.VideoCapture, seconds: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return np.zeros((H, W, 3), np.uint8)
    return cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)


def main() -> None:
    database, mirror = load_database(), load_mirror()
    info_out = {}
    for task in TASKS:
        cohort = task_cohort(database, task, mirror)
        n = len(cohort["canon_labels"])
        header = np.full((40, LABEL_W + n * W, 3), 255, np.uint8)
        for k, label in enumerate(cohort["canon_labels"]):
            cv2.putText(header, f"{k + 1}. {label[:34]}", (LABEL_W + k * W + 4, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        rows = [header]
        for vid in cohort["correct"]:
            info = database[vid]
            cap = cv2.VideoCapture(str(video_path(task, vid)))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = [frame_at(cap, sum(s["segment"]) / 2) for s in info["annotation"]]
            cap.release()
            roi = info["end"] - info["start"]
            label = np.full((H, LABEL_W, 3), 255, np.uint8)
            for i, text in enumerate([vid, f"{info['subset']}", f"{width}x{height}", f"task {roi:.0f}s"]):
                cv2.putText(label, text, (6, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 1, cv2.LINE_AA)
            rows.append(np.hstack([label] + frames))
            info_out.setdefault(task, {})[vid] = {
                "subset": info["subset"], "width": width, "height": height,
                "task_seconds": round(roi, 1),
                "step_seconds": [round(s["segment"][1] - s["segment"][0], 1) for s in info["annotation"]],
            }
        sheet = np.vstack(rows)
        out = EXPERIMENT / "contact_sheets" / f"{task}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log(f"{task}: {len(cohort['correct'])} videos -> {out}")
    (EXPERIMENT / "runs" / "video_info.json").write_text(json.dumps(info_out, indent=1) + "\n")


if __name__ == "__main__":
    main()
