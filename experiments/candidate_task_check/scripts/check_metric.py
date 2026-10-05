"""Sanity checks on the team's DTW step-cost metric (scripts/classical_pairwise_dtw.py).

1. dtw_pair against a slow, plain DTW on random cost matrices.
2. Different-task control: is a reference closer to videos of its own task than to
   videos of other tasks? Uses the cached CLIP features from experiments/deviation_detection.
3. Step specificity: is the cost of step k lower against the same step of another video
   than against a different step? And how strongly do a pair's step costs move together?
4. Histogram feature balance: share of the feature vector's energy in hue, saturation
   and gradient-orientation parts.

Usage (repo root): python experiments/candidate_task_check/scripts/check_metric.py
"""

from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "deviation_detection" / "scripts"))

from scripts.classical_pairwise_dtw import dtw_pair, frame_feature, label_frames, step_distance  # noqa: E402
from common import FEATURE_DIR, load_database, row_for  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "metric_checks.json"


def slow_dtw(cost):
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    # backtrack with the same tie order as dtw_pair: diagonal, up, left
    i, j, path = n, m, []
    while True:
        path.append((i - 1, j - 1))
        if i == 1 and j == 1:
            break
        options = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        best = min(o[0] for o in options)
        _, i, j = next(o for o in options if o[0] == best)
    return acc[n, m], path


def check_dtw(rng):
    worst = 0.0
    for _ in range(200):
        n, m = rng.integers(2, 30, size=2)
        cost = rng.random((n, m))
        la = rng.integers(-1, 3, size=n).astype(np.int8)
        lb = rng.integers(-1, 3, size=m).astype(np.int8)
        mean, sa, ca, sb, cb = dtw_pair(cost, la, lb, 3)
        total, path = slow_dtw(cost)
        worst = max(worst, abs(mean * len(path) - total))
        for k in range(3):
            ref = [cost[p] for p in path if la[p[0]] == k]
            if ref:
                worst = max(worst, abs(sa[k] / ca[k] - np.mean(ref)))
    return worst


def load_cached():
    """Cached CLIP features of correct videos, grouped by task, from the deviation screen."""
    database = load_database()
    by_task = {}
    for path in sorted(glob.glob(str(REPO / "experiments/deviation_detection/runs/*_clip_*.json"))):
        r = json.loads(Path(path).read_text())
        vids = r["references"] + [h["video_id"] for h in r["held_out_correct"]]
        videos = []
        for v in vids:
            d = np.load(FEATURE_DIR / "clip" / f"{v}.npz")
            videos.append((v, d["features"], label_frames(d["times"], row_for(database, v)["steps"])))
        by_task[r["task"]] = {"n_steps": len(r["steps"]), "videos": videos}
    return by_task


def pair_costs(fa, la, fb, lb, n):
    cost = np.clip(1.0 - fa @ fb.T, 0.0, 2.0).astype(np.float64)
    mean, sa, ca, sb, cb = dtw_pair(cost, la, lb, n)
    steps = [step_distance(sa, ca, sb, cb, k) if (ca[k] or cb[k]) else np.nan for k in range(n)]
    return mean, steps


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    out["dtw_max_abs_error_vs_slow"] = check_dtw(rng)
    print(f"1. dtw_pair vs slow DTW, 200 random cases: max error {out['dtw_max_abs_error_vs_slow']:.2e}")

    tasks = load_cached()
    names = sorted(tasks)
    same, other = [], []
    per_task = {}
    for t in names:
        vids = tasks[t]["videos"]
        n = tasks[t]["n_steps"]
        ref_v, ref_f, ref_l = vids[0]
        s = [pair_costs(ref_f, ref_l, f, l, n)[0] for v, f, l in vids[1:]]
        o = []
        for u in names:
            if u == t:
                continue
            for v, f, l in tasks[u]["videos"][:3]:
                # other task: label the comparison as unlabelled, so only the global path cost is used
                o.append(pair_costs(ref_f, ref_l, f, np.full(len(f), -1, np.int8), n)[0])
        per_task[t] = {"same_task_mean": float(np.mean(s)), "other_task_mean": float(np.mean(o)),
                       "same_task_sd": float(np.std(s)), "other_task_sd": float(np.std(o))}
        same += s
        other += o
    out["task_control"] = per_task
    print("2. whole-video cost, reference vs same task / vs other tasks (CLIP):")
    for t, d in per_task.items():
        print(f"   {t:24s} same {d['same_task_mean']:.3f}±{d['same_task_sd']:.3f}  other {d['other_task_mean']:.3f}±{d['other_task_sd']:.3f}")

    spec = {}
    for t in names:
        vids, n = tasks[t]["videos"], tasks[t]["n_steps"]
        same_step, diff_step, rows = [], [], []
        for a in range(len(vids)):
            for b in range(a + 1, len(vids)):
                _, fa, la = vids[a]
                _, fb, lb = vids[b]
                for k in range(n):
                    xa = fa[la == k]
                    for j in range(n):
                        xb = fb[lb == j]
                        if len(xa) and len(xb):
                            d = float((1.0 - xa @ xb.T).mean())
                            (same_step if k == j else diff_step).append(d)
                rows.append(pair_costs(fa, la, fb, lb, n)[1])
        rows = np.array(rows, dtype=float)
        rows = rows[~np.isnan(rows).any(axis=1)]
        corr = np.corrcoef(rows.T)
        off = corr[~np.eye(n, dtype=bool)].mean()
        # share of step-cost variance explained by the pair's mean step cost
        pair_mean = rows.mean(axis=1, keepdims=True)
        explained = 1 - ((rows - pair_mean) ** 2).sum() / ((rows - rows.mean(axis=0)) ** 2).sum()
        spec[t] = {"same_step": float(np.mean(same_step)), "different_step": float(np.mean(diff_step)),
                   "step_cost_correlation": float(off), "variance_explained_by_pair": float(explained)}
    out["step_specificity"] = spec
    print("3. step specificity (CLIP, all pairs):")
    for t, d in spec.items():
        print(f"   {t:24s} same-step {d['same_step']:.3f} diff-step {d['different_step']:.3f} "
              f"gap {d['different_step'] - d['same_step']:.3f} | step costs corr {d['step_cost_correlation']:.2f}, "
              f"pair explains {d['variance_explained_by_pair']:.0%}")

    import cv2  # noqa: E402

    shares = []
    for video in sorted(glob.glob(str(REPO / "data/videos/MakePaperWindMill/*.mp4")))[:5]:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        raw = frame_feature(frame)  # normalized; recover part energies
        e = raw ** 2
        shares.append([e[:16].sum(), e[16:24].sum(), e[24:].sum()])
    shares = np.mean(shares, axis=0)
    out["histogram_energy_share"] = {"hue": float(shares[0]), "saturation": float(shares[1]), "gradients": float(shares[2])}
    print(f"4. histogram feature energy: hue {shares[0]:.1%}, saturation {shares[1]:.1%}, gradients {shares[2]:.1%}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
