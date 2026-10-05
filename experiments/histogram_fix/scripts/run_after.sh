#!/usr/bin/env bash
# After the fix: SIM + smoothie histograms, the three candidate tasks' histograms, smoothie CLIP.
# Usage (repo root): bash experiments/histogram_fix/scripts/run_after.sh
set -u
export UV_PROJECT_ENVIRONMENT=../.venv-baselines
export UV_CACHE_DIR=../.uv-cache
export HF_HOME="$PWD/../.hf-cache"
E=experiments/histogram_fix

bash "$E/scripts/run_sim_and_smoothie.sh" after

run() {
  local task=$1 ref=$2 enc=$3 tag=$4
  local extra=()
  [ "$enc" = clip ] && extra=(--extra clip)
  echo "=== $task $enc ($tag) $(date '+%F %T')"
  uv run --locked ${extra[@]+"${extra[@]}"} python -u -m scripts.run_coin_reference \
    --task "$task" --reference-id="$ref" --encoder "$enc" \
    --output "$E/runs/${task}_${enc}_${tag}.json" > "$E/logs/${task}_${enc}_${tag}.log" 2>&1
  echo "exit $?"
  tail -1 "$E/logs/${task}_${enc}_${tag}.log"
}

run UseRiceCookerToCookRice VUWV9rEme7c classical after
run MakePaperWindMill 4ufBW5Cfpgw classical after
run PutOnQuiltCover H52vAkIp80A classical after
run MakeStrawberrySmoothie -3C-VGhs2mo clip same
echo "ALL DONE $(date '+%F %T')"
