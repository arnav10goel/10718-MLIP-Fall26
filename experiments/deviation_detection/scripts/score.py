"""Score correct, synthetic-deviant and natural-deviant videos for one task.

Three per-step measures, each with its own normal range from the training
correct videos (leave-one-out):

    absolute   reference-side DTW cost of step k (median over references)
    relative   absolute cost of step k minus the mean over the query's steps,
               so a whole-video offset (another kitchen) cancels
    prototype  share of query frames nearest step k's mean embedding, and the
               mean position of those frames; a skipped step gets few frames,
               a swapped step sits in the wrong place

Usage (repo root):
    python experiments/deviation_detection/scripts/score.py --task MakeBurger --encoder clip
    add --download to fetch every correct and natural-deviant video from the mirror
    (otherwise only videos already in data/videos/<task>/ are used); add
    --max-correct N to cap the correct videos downloaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EXPERIMENT, Extractor, download, label_frames, load_database, load_mirror, log,
    prototype_profile, reference_step_costs, row_for, step_prototypes, step_separation,
    synthetic_deviants, task_cohort, video_path,
)

METHODS = ("absolute", "relative", "prototype")


def measure(query, refs, n_steps, skip_id=None) -> dict:
    costs = np.nanmedian(np.stack([
        reference_step_costs(f, l, query, n_steps)
        for vid, (f, l) in refs.items() if vid != skip_id
    ]), axis=0)
    share, pos = prototype_profile(query, step_prototypes(refs, n_steps, skip_id))
    return {
        "absolute": costs,
        "relative": costs - costs.mean(),
        "share": share,
        "position": pos,
    }


def normal_range(calib: list[dict]) -> dict:
    out = {}
    for key in ("absolute", "relative", "share", "position"):
        values = np.stack([c[key] for c in calib])
        out[key] = {"mean": np.nanmean(values, axis=0), "sd": np.nanstd(values, axis=0)}
    return out


def flags(m: dict, rng: dict) -> dict:
    """Which steps each method puts outside its normal range."""
    def z(key):
        return (m[key] - rng[key]["mean"]) / rng[key]["sd"]
    share_low = z("share") < -2
    pos_off = np.nan_to_num(np.abs(z("position")), nan=np.inf) > 2
    return {
        "absolute": (z("absolute") > 2).tolist(),
        "relative": (z("relative") > 2).tolist(),
        "prototype": (share_low | pos_off).tolist(),
        "z_absolute": np.round(z("absolute"), 2).tolist(),
        "z_relative": np.round(z("relative"), 2).tolist(),
        "z_share": np.round(z("share"), 2).tolist(),
        "z_position": np.round(np.nan_to_num(z("position"), nan=99.0), 2).tolist(),
    }


def judge(over: list[bool], expected: list[int]) -> dict:
    hit = all(over[k] for k in expected)
    others = [over[k] for k in range(len(over)) if k not in expected]
    return {"right_step": hit, "localized": hit and not any(others), "any_flag": any(over)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--encoder", choices=("clip", "classical"), default="clip")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-correct", type=int)
    parser.add_argument("--no-natural", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    database = load_database()
    mirror = load_mirror()
    cohort = task_cohort(database, args.task, mirror)
    n_steps = len(cohort["canon_ids"])
    annotated = sum(1 for info in database.values() if info["class"] == args.task)
    on_mirror = sum(1 for v, info in database.items() if info["class"] == args.task and v in mirror)
    counts = {
        "annotated": annotated, "on_mirror": on_mirror,
        "correct_on_mirror": len(cohort["correct"]),
        "natural_deviants_on_mirror": len(cohort["deviants"]),
    }
    log(f"{args.task}: {n_steps} steps {cohort['canon_labels']}")
    log(f"{len(cohort['correct'])} correct of {annotated} annotated ({on_mirror} on mirror); "
        f"{len(cohort['deviants'])} natural deviants")

    correct_pool = list(cohort["correct"])
    if args.max_correct:
        on_disk = [v for v in correct_pool if video_path(args.task, v).is_file()]
        # Smallest files first, to keep downloads small.
        rest = sorted(set(correct_pool) - set(on_disk), key=lambda v: mirror[v])
        correct_pool = sorted(on_disk + rest[: max(0, args.max_correct - len(on_disk))])
    wanted = correct_pool + ([] if args.no_natural else list(cohort["deviants"]))
    if args.download:
        for index, vid in enumerate(wanted, 1):
            download(args.task, vid, database[vid])
            log(f"[download {index}/{len(wanted)}] {vid}")
    present = {vid for vid in wanted if video_path(args.task, vid).is_file()}
    correct = [v for v in correct_pool if v in present]
    deviants = {v: d for v, d in cohort["deviants"].items() if v in present and v in wanted}

    extract = Extractor(args.encoder)
    feats, labels, failed = {}, {}, []
    todo = correct + list(deviants)
    for index, vid in enumerate(todo, 1):
        row = row_for(database, vid)
        try:
            feats[vid], times = extract(row)
        except Exception as exc:  # noqa: BLE001 - one bad video should not stop the run
            failed.append({"video_id": vid, "error": str(exc)[:300]})
            log(f"[features {index}/{len(todo)}] FAILED {vid}: {exc}")
            continue
        labels[vid] = label_frames(times, row["steps"])
        log(f"[features {index}/{len(todo)}] {vid} {len(times)} frames")
    # Overlapping annotations can leave a step with no sampled frames.
    empty = [v for v in correct if v in feats and any(not (labels[v] == k).any() for k in range(n_steps))]
    for vid in empty:
        failed.append({"video_id": vid, "error": "a step has no sampled frames"})
        log(f"dropped {vid}: a step has no sampled frames")
    correct = [v for v in correct if v in feats and v not in empty]
    deviants = {v: d for v, d in deviants.items() if v in feats}

    train = [v for v in correct if database[v]["subset"] == "training"]
    test = [v for v in correct if database[v]["subset"] == "testing"]
    refs = {v: (feats[v], labels[v]) for v in train}
    counts.update({"correct_used": len(correct), "references": len(train),
                   "held_out_correct": len(test), "natural_deviants_used": len(deviants)})
    log(f"used: {len(correct)} correct ({len(train)} training references, "
        f"{len(test)} held out), {len(deviants)} natural deviants")

    separation = [step_separation(feats[v], labels[v], n_steps) for v in correct]
    calib_m = {v: measure(feats[v], refs, n_steps, skip_id=v) for v in train}
    rng = normal_range(list(calib_m.values()))
    log(f"step separation (between - within, mean over videos): {np.mean(separation):.3f}; "
        f"absolute-cost sd per step: {np.round(rng['absolute']['sd'], 3).tolist()}")

    def record(vid, m, expected, **extra):
        f = flags(m, rng)
        return {
            "video_id": vid, **extra, "expected_steps": expected,
            "absolute": np.round(m["absolute"], 4).tolist(),
            "share": np.round(m["share"], 3).tolist(),
            "position": np.round(np.nan_to_num(m["position"], nan=-1), 3).tolist(),
            **f, **{f"judge_{k}": judge(f[k], expected) for k in METHODS},
        }

    calibration = [record(v, calib_m[v], []) for v in train]
    held_out = [record(v, measure(feats[v], refs, n_steps), []) for v in test]
    synthetic = []
    for vid in correct:
        skip = vid if vid in refs else None
        for name, (query, expected) in synthetic_deviants(feats[vid], labels[vid], n_steps).items():
            synthetic.append(record(vid, measure(query, refs, n_steps, skip), expected,
                                    kind=name.split("_")[0], variant=name))
    natural = [
        record(v, measure(feats[v], refs, n_steps), d["expected"], kind=d["kind"],
               their_steps=d["steps"], split=database[v]["subset"])
        for v, d in deviants.items()
    ]

    report = {
        "task": args.task, "encoder": args.encoder, "steps": cohort["canon_labels"],
        "counts": counts, "references": train,
        "step_separation": float(np.mean(separation)),
        "normal_range": {k: {s: np.round(v, 4).tolist() for s, v in r.items()}
                         for k, r in rng.items()},
        "calibration": calibration, "held_out_correct": held_out,
        "synthetic": synthetic, "natural": natural, "feature_failures": failed,
    }
    out = EXPERIMENT / "runs" / f"{args.task}_{args.encoder}{args.tag}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for group, items in (("held-out correct", held_out), ("synthetic", synthetic),
                         ("natural", natural)):
        if not items:
            continue
        for method in METHODS:
            j = [i[f"judge_{method}"] for i in items]
            log(f"{group:16s} {method:9s} n={len(items):3d} "
                f"any-flag {np.mean([x['any_flag'] for x in j]):4.0%} "
                f"right-step {np.mean([x['right_step'] for x in j]):4.0%} "
                f"localized {np.mean([x['localized'] for x in j]):4.0%}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
