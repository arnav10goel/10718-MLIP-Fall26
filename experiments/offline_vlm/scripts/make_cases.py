"""Build the execution test clips and their private answer keys from one COIN video."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.coin_tasks import load_database, resolve_data_root  # noqa: E402
from src.guideme.offline_vlm import cut_segments  # noqa: E402

TASK, EXECUTION_ID = "MakePaperWindMill", "e7p9QHRmd4k"
CASES = Path(__file__).resolve().parents[1] / "cases"


def build(name, info, order, drop):
    """order: step indices in the new order; drop: indices removed. Gaps stay with the step before."""
    start, end = float(info["start"]), float(info["end"])
    bounds = [tuple(map(float, s["segment"])) for s in info["annotation"]]
    n = len(bounds)
    # Block k runs from step k's start to step k+1's start (last: to the task end),
    # so the pause after a step moves with it. Lead-in before step 1 stays first.
    blocks = [(bounds[k][0], bounds[k + 1][0] if k + 1 < n else end) for k in range(n)]
    segments = [(start, bounds[0][0])] if bounds[0][0] > start else []
    truth, t = [], (bounds[0][0] - start if bounds[0][0] > start else 0.0)
    placed = {}
    for k in order:
        if k in drop:
            continue
        a, b = blocks[k]
        segments.append((a, b))
        step_len = bounds[k][1] - bounds[k][0]
        placed[k] = (round(t, 2), round(t + step_len, 2))
        t += b - a
    for k in range(n):
        if k in drop:
            status = "skipped"
        elif order.index(k) != k:
            status = "out_of_order"
        else:
            status = "done"
        times = placed.get(k, (None, None))
        truth.append({"reference_step_id": f"s{k + 1:03d}", "label": info["annotation"][k]["label"],
                      "status": status, "execution_start_s": times[0], "execution_end_s": times[1]})
    source = resolve_data_root() / "videos" / TASK / f"{EXECUTION_ID}.mp4"
    clip = cut_segments(source, segments, CASES / f"{name}.mp4")
    (CASES / f"{name}_truth.json").write_text(json.dumps(
        {"case": name, "source": EXECUTION_ID, "segments_s": segments, "steps": truth}, indent=1) + "\n")
    print(name, [(s["reference_step_id"], s["status"], s["execution_start_s"]) for s in truth], clip)


def main():
    info = resolve_data_root() / "raw" / "COIN.json"
    info = load_database(info)[EXECUTION_ID]
    build("correct", info, [0, 1, 2, 3], set())
    build("skip_s002", info, [0, 1, 2, 3], {1})
    build("swap_s003_s004", info, [0, 1, 3, 2], set())


if __name__ == "__main__":
    main()
