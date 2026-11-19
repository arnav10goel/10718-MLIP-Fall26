"""Zero-shot Gemini, GLM, and Muse on every recorded cut.

Both videos are frames sampled at the requested rate. GLM and Muse accept one
video file per request, so a video upload cannot set the execution frame rate.
Stapler has no recordings and is not included.

    .venv/bin/python scripts/run_vlm_grid.py
    .venv/bin/python scripts/run_vlm_grid.py --only glm muse
    .venv/bin/python scripts/run_vlm_grid.py --few-shot --only gemini
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coin_tasks import PREPARED_DIR, REPO_ROOT  # noqa: E402
from few_shot_examples import example_text  # noqa: E402
from perform_cpr import CORRECT_FP_CASES, SPEC as CPR  # noqa: E402

FPS_VALUES = (2.0,)
MODELS = (
    {"name": "gemini", "backend": "gemini", "model": "gemini-3.5-flash-lite", "max_tokens": 8192},
    {
        "name": "glm",
        "backend": "openai",
        "model": "zai-org/GLM-5.3-Flash",
        "base_url": "https://iwatts--ep-guideme-glm-server.us-west.modal.direct/v1",
        "max_tokens": 32768,
    },
    {
        "name": "muse",
        "backend": "openai",
        "model": "meta-models/Muse-Spark-Open-816B-A42B",
        "base_url": "https://iwatts--ep-guideme-muse-server.us-west.modal.direct/v1",
        "max_tokens": 32768,
    },
)
# Long enough that 5 fps is not quietly replaced by 256 evenly spaced frames.
MAX_FRAMES = 2000
OUTPUT_ROOT = PREPARED_DIR / "vlm_zero_shot"
FEW_SHOT_ROOT = PREPARED_DIR / "vlm_few_shot"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def midpoint(video: dict, step_index: int) -> float:
    step = next(item for item in video["steps"] if item["index"] == step_index)
    if step["start"] is None or step["end"] is None:
        raise ValueError(f"{video['file']} step {step_index} has no interval")
    return round(0.5 * (float(step["start"]) + float(step["end"])), 3)


def cut_for_deviation(video: dict) -> tuple[int, int, float]:
    """Return test step, cut step, and the mid-step cut time."""
    test_step = int(video["deviation_step"])
    cut = next(
        step for step in video["steps"]
        if step["index"] >= test_step and step["start"] is not None
    )
    return test_step, int(cut["index"]), midpoint(video, int(cut["index"]))


def recording_cases(spec, reference: dict, correct_step: int | None) -> list[dict]:
    payload = json.loads(spec.annotations_path().read_text())
    cases = []
    for video in payload["videos"]:
        if video["outcome"] == "correct":
            if correct_step is None:
                continue
            test_step = cut_step = correct_step
            end_s = midpoint(video, correct_step)
        else:
            test_step, cut_step, end_s = cut_for_deviation(video)
        cases.append({
            "task": spec.artifact_stem,
            "file": video["file"],
            "execution": str(spec.recording_dir() / video["file"]),
            "test_step": test_step,
            "cut_step": cut_step,
            "execution_end_s": end_s,
            **reference,
        })
    return cases


def smoothie_cases() -> list[dict]:
    from make_strawberry_smoothie import SPEC

    payload = json.loads((PREPARED_DIR / "smoothie_recorded_steps.json").read_text())
    seen = set()
    cases = []
    reference = {
        "reference": str(SPEC.recording_dir().parent / "reference.mp4"),
        "checklist": str(REPO_ROOT / "data/annotations/reference/MakeStrawberrySmoothie/reference.csv"),
    }
    for row in payload["results"]:
        if row.get("encoder") not in (None, "classical"):
            continue
        key = (row["file"], row["test_step"], row["cut_step"])
        if key in seen:
            continue
        seen.add(key)
        cases.append({
            "task": "smoothie",
            "file": row["file"],
            "execution": str(SPEC.recording_dir() / row["file"]),
            "test_step": row["test_step"],
            "cut_step": row["cut_step"],
            "execution_end_s": round(float(row["cut_time_s"]), 3),
            **reference,
        })
    return cases


def cpr_cases() -> list[dict]:
    payload = json.loads(CPR.annotations_path().read_text())
    by_file = {video["file"]: video for video in payload["videos"]}
    reference = {
        "reference": str(CPR.recording_dir().parent / "reference.mp4"),
        "checklist": str(REPO_ROOT / "data/annotations/reference/PerformCPR/reference.csv"),
    }
    cases = []
    for video in payload["videos"]:
        if video["outcome"] != "incorrect":
            continue
        test_step, cut_step, end_s = cut_for_deviation(video)
        cases.append({
            "task": "cpr",
            "file": video["file"],
            "execution": str(CPR.recording_dir() / video["file"]),
            "test_step": test_step,
            "cut_step": cut_step,
            "execution_end_s": end_s,
            **reference,
        })
    for item in CORRECT_FP_CASES:
        video = by_file[item["file"]]
        cases.append({
            "task": "cpr",
            "file": item["file"],
            "execution": str(CPR.recording_dir() / item["file"]),
            "test_step": item["step"],
            "cut_step": item["step"],
            "execution_end_s": midpoint(video, item["step"]),
            "check": item["check"],
            **reference,
        })
    return cases


def all_cases() -> list[dict]:
    from clean_toilet import SPEC as toilet
    from replace_battery import SPEC as battery
    from replace_simcard import SPEC as sim

    videos = REPO_ROOT / "data" / "videos"
    return [
        *recording_cases(sim, {
            "reference": str(videos / "ReplaceSIMCard/used/0b6joFdBwUw.mp4"),
            "coin_reference_id": "0b6joFdBwUw",
        }, 2),
        *smoothie_cases(),
        *recording_cases(toilet, {
            "reference": str(videos / "CleanToilet/used/d2gMl6jwEEc.mp4"),
            "coin_reference_id": "d2gMl6jwEEc",
        }, 1),
        *recording_cases(battery, {
            "reference": str(videos / "ReplaceBatteryOnTVControl/used/ROGBQhc9tCs.mp4"),
            "coin_reference_id": "ROGBQhc9tCs",
        }, 3),
        *cpr_cases(),
    ]


def output_path(model: str, fps: float, case: dict, root: Path) -> Path:
    stem = Path(case["file"]).stem.replace(" ", "_")
    name = f"{case['task']}_{stem}_s{case['test_step']}.json"
    return root / model / f"fps{fps:g}" / name


def command(model: dict, fps: float, case: dict, output: Path) -> list[str]:
    argv = [
        str(PYTHON), "-m", "scripts.run_offline_vlm",
        "--backend", model["backend"],
        "--model", model["model"],
        "--reference", case["reference"],
        "--execution", case["execution"],
        "--execution-end-s", str(case["execution_end_s"]),
        "--fps", str(fps),
        "--max-frames", str(MAX_FRAMES),
        "--max-tokens", str(model["max_tokens"]),
        "--output", str(output),
    ]
    if case.get("coin_reference_id"):
        argv.extend(["--coin-reference-id", case["coin_reference_id"]])
    else:
        argv.extend(["--checklist", case["checklist"]])
    if model["backend"] == "openai":
        argv.extend(["--base-url", model["base_url"]])
    if case.get("examples"):
        argv.extend(["--examples", case["examples"]])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="+", choices=[model["name"] for model in MODELS],
        help="Run these models only. Existing reports are skipped.",
    )
    parser.add_argument(
        "--few-shot", action="store_true",
        help="Add the other tasks' steps and annotated mistakes as text.",
    )
    args = parser.parse_args()
    selected = MODELS if args.only is None else tuple(
        model for model in MODELS if model["name"] in args.only
    )
    root = FEW_SHOT_ROOT if args.few_shot else OUTPUT_ROOT
    cases = all_cases()
    if args.few_shot:
        example_dir = root / "examples"
        example_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for case in cases:
            task = case["task"]
            if task not in written:
                path = example_dir / f"{task}.txt"
                path.write_text(example_text(task), encoding="utf-8")
                written[task] = str(path)
            case["examples"] = written[task]
    jobs = [
        (model, fps, case)
        for model in selected
        for fps in FPS_VALUES
        for case in cases
    ]
    print(f"{len(cases)} cuts, {len(jobs)} calls", flush=True)
    done = failed = 0
    for index, (model, fps, case) in enumerate(jobs, start=1):
        output = output_path(model["name"], fps, case, root)
        label = f"[{index}/{len(jobs)}] {model['name']} fps={fps:g} {output.name}"
        if output.exists():
            done += 1
            print(f"{label} exists", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command(model, fps, case, output),
            cwd=REPO_ROOT, text=True,
        )
        if completed.returncode == 0:
            done += 1
            print(f"{label} ok", flush=True)
        else:
            failed += 1
            print(f"{label} FAILED {completed.returncode}", flush=True)
    print(f"finished ok={done} failed={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
