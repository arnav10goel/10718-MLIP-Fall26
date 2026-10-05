#!/usr/bin/env bash
# Run score.py for one task and keep a log.
# Usage (repo root): bash experiments/deviation_detection/scripts/run.sh <Task> <tag> [score.py flags...]
set -u
export UV_PROJECT_ENVIRONMENT=../.venv-baselines
export UV_CACHE_DIR=../.uv-cache
export HF_HOME="$PWD/../.hf-cache"
E=experiments/deviation_detection
task=$1
tag=$2
shift 2
log="$E/logs/${task}_clip${tag}.log"
uv run --locked --extra clip python -u "$E/scripts/score.py" --task "$task" --encoder clip --tag "$tag" "$@" 2>&1 \
  | grep -v Warning > "$log"
grep -v "\[features \|\[download " "$log"
