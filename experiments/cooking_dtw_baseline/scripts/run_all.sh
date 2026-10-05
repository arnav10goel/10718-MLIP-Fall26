#!/usr/bin/env bash
# Run the repo's DTW runner on both cooking tasks with both encoders.
# Usage (from the repo root): bash experiments/cooking_dtw_baseline/scripts/run_all.sh
set -u
export UV_PROJECT_ENVIRONMENT=../.venv-baselines
export UV_CACHE_DIR=../.uv-cache
export HF_HOME="$PWD/../.hf-cache"
E=experiments/cooking_dtw_baseline

run() {
  local task=$1 ref=$2 short=$3 enc=$4
  local out="$E/runs/${short}_${enc}.json" log="$E/logs/${short}_${enc}.log"
  local extra=()
  [ "$enc" = clip ] && extra=(--extra clip)
  echo "=== $task $enc start $(date '+%F %T')"
  uv run --locked ${extra[@]+"${extra[@]}"} python -u -m scripts.run_coin_reference \
    --task "$task" --reference-id="$ref" --encoder "$enc" --output "$out" \
    > "$log" 2>&1
  echo "exit $? end $(date '+%F %T')"
  tail -2 "$log"
}

run BoilNoodles krNVdmW0Nto boilnoodles classical
run BoilNoodles krNVdmW0Nto boilnoodles clip
run MakeBurger aogu4SX7ULg makeburger classical
run MakeBurger aogu4SX7ULg makeburger clip
