#!/usr/bin/env bash
# Wait until .env holds a GEMINI_API_KEY line, check the model, then run the three cases and score them.
cd "$(dirname "$0")/../../.."
until grep -q '^GEMINI_API_KEY=..*' .env 2>/dev/null; do sleep 3; done
echo "key found $(date '+%F %T')"
../.venv-online/bin/python experiments/offline_vlm/scripts/check_model.py gemini-3.5-flash-lite || exit 1
bash experiments/offline_vlm/scripts/run_cases.sh live
../.venv-online/bin/python experiments/offline_vlm/scripts/score.py live
