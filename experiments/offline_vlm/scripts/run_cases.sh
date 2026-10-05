#!/usr/bin/env bash
# Run run_offline_vlm on the three cases. Usage (worktree root):
#   CASES_DIR=experiments/offline_vlm/cases/<id> bash experiments/offline_vlm/scripts/run_cases.sh <tag> [--dry-run] [--model ID]
set -u
export UV_PROJECT_ENVIRONMENT=../.venv-online
export UV_CACHE_DIR=../.uv-cache
export GUIDEME_DATA_ROOT="${GUIDEME_DATA_ROOT:-/Users/arnav/Documents/10718-MLIP-Fall26/data}"
E=experiments/offline_vlm
CASES_DIR="${CASES_DIR:-$E/cases}"
tag=$1
shift
for case in correct skip_s002 swap_s003_s004; do
  echo "=== $case ($tag) $(date '+%F %T')"
  uv run --locked --extra vlm python -u -m scripts.run_offline_vlm \
    --reference videos/MakePaperWindMill/TULczOB5joI.mp4 --coin-reference-id TULczOB5joI \
    --execution "$PWD/$CASES_DIR/$case.mp4" \
    --output "$E/runs/${case}_${tag}.json" "$@" > "$E/logs/${case}_${tag}.log" 2>&1
  echo "exit $?"
  tail -1 "$E/logs/${case}_${tag}.log"
done
