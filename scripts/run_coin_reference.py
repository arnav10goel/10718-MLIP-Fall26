"""Run offline DTW with the selected encoder on one validated COIN cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.classical_pairwise_dtw import FPS
from scripts.coin_tasks import load_database, resolve_data_root, select_matching_cohort
from scripts.prepare_coin import cohort_manifest_rows
from scripts.score_coin_reference import score_reference_cohort
from scripts.validate_coin_media import inventory_task_media


def main(argv: list[str] | None = None) -> int:
    """Validate task media, score one reference cohort, and write a new report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Exact COIN class name")
    parser.add_argument("--reference-id", required=True, help="Training video ID")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--data-root", help="Absolute COIN data root")
    parser.add_argument(
        "--encoder", choices=("classical", "clip"), default="classical",
        help="Frame feature encoder (default: classical)",
    )
    args = parser.parse_args(argv)

    requested_output = Path(args.output).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"Report already exists: {requested_output}")
    output = requested_output.resolve()
    if not output.parent.is_dir():
        raise ValueError(f"Report parent directory does not exist: {output.parent}")

    root = resolve_data_root(args.data_root)
    annotation_path = root / "raw" / "COIN.json"
    annotation_sha256 = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    database = load_database(annotation_path)

    print(f"Validating media for {args.task} under {root}", flush=True)
    inventory = inventory_task_media(database, args.task, root)
    cohort = select_matching_cohort(
        database, args.task, args.reference_id, set(inventory["usable_paths"])
    )
    rows = cohort_manifest_rows(database, cohort, inventory["usable_paths"])
    print(f"Scoring {len(rows) - 1} comparisons against {args.reference_id}", flush=True)
    if args.encoder == "clip":
        from scripts.score_coin_clip import score_clip_reference_cohort

        scoring = score_clip_reference_cohort(rows, args.reference_id)
    else:
        scoring = score_reference_cohort(rows, args.reference_id)

    method = {
        "name": "classical_reference_dtw",
        "fps": FPS,
        "feature_cache": False,
        "uses_comparison_annotations": True,
        "hash_check": "observed_only",
    }
    if args.encoder == "clip":
        method.update({
            "name": "clip_reference_dtw",
            "encoder": "clip",
            "model": "ViT-B-32",
            "pretrained": "openai",
        })

    report = {
        "schema_version": 1,
        "task": args.task,
        "reference_id": args.reference_id,
        "data_root": str(root),
        "annotation_source": {
            "path": str(annotation_path),
            "sha256": annotation_sha256,
        },
        "method": method,
        "summary": {
            "annotated_videos": len(inventory["records"]),
            "usable_media": len(inventory["usable_paths"]),
            "matching_training": len(cohort["training_ids"]),
            "matching_testing": len(cohort["testing_ids"]),
            "comparisons_scored": len(scoring["scores"]),
            "comparison_failures": len(scoring["failures"]),
            "cohort_exclusions": len(cohort["excluded"]),
        },
        "inventory_records": inventory["records"],
        "cohort": cohort,
        "scoring": scoring,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(rendered)
    print(f"Wrote {len(scoring['scores'])} scores to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
