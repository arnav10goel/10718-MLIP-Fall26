#!/usr/bin/env bash
# Run the collaborators' scripts.run_coin_reference (unchanged) on the three tasks, both encoders.
# Usage (repo root): bash experiments/candidate_task_check/scripts/run_all.sh
set -u
export UV_PROJECT_ENVIRONMENT=../.venv-baselines
export UV_CACHE_DIR=../.uv-cache
export HF_HOME="$PWD/../.hf-cache"
E=experiments/candidate_task_check

run() {
  local task=$1 ref=$2 enc=$3
  local out="$E/runs/${task}_${enc}.json" log="$E/logs/${task}_${enc}.log"
  local extra=()
  [ "$enc" = clip ] && extra=(--extra clip)
  echo "=== $task $enc start $(date '+%F %T')"
  uv run --locked ${extra[@]+"${extra[@]}"} python -u -m scripts.run_coin_reference \
    --task "$task" --reference-id="$ref" --encoder "$enc" --output "$out" > "$log" 2>&1
  echo "exit $? end $(date '+%F %T')"
  tail -1 "$log"
}

for enc in classical clip; do
  run UseRiceCookerToCookRice VUWV9rEme7c "$enc"
  run MakePaperWindMill 4ufBW5Cfpgw "$enc"
  run PutOnQuiltCover H52vAkIp80A "$enc"
done
