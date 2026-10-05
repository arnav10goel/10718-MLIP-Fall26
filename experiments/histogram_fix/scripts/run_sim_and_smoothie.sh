#!/usr/bin/env bash
# Run the SIM all-pairs histogram script and the smoothie runner with whatever
# frame_feature is checked out. Usage (repo root): bash .../run_sim_and_smoothie.sh <before|after>
set -u
export UV_PROJECT_ENVIRONMENT=../.venv-baselines
export UV_CACHE_DIR=../.uv-cache
export HF_HOME="$PWD/../.hf-cache"
E=experiments/histogram_fix
tag=$1

echo "=== SIM classical all-pairs ($tag) $(date '+%F %T')"
rm -f data/prepared/sim_classical_features.npz   # never reuse features from the other version
uv run --locked python -u scripts/prepare_coin.py > "$E/logs/prepare_coin_$tag.log" 2>&1
uv run --locked python -u scripts/classical_pairwise_dtw.py > "$E/logs/sim_classical_$tag.log" 2>&1
echo "exit $?"
cp data/prepared/sim_classical_dtw_summary.json "$E/runs/sim_classical_$tag.json"
tail -5 "$E/logs/sim_classical_$tag.log"

echo "=== smoothie classical ($tag) $(date '+%F %T')"
uv run --locked python -u -m scripts.run_coin_reference --task MakeStrawberrySmoothie \
  --reference-id=-3C-VGhs2mo --encoder classical \
  --output "$E/runs/MakeStrawberrySmoothie_classical_$tag.json" > "$E/logs/smoothie_classical_$tag.log" 2>&1
echo "exit $?"
tail -1 "$E/logs/smoothie_classical_$tag.log"
