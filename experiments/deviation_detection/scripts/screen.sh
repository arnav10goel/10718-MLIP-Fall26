#!/usr/bin/env bash
# Screen candidate tasks: 10 correct videos each, synthetic deviants only.
# Usage (repo root): bash experiments/deviation_detection/scripts/screen.sh
for task in MakeOrangeJuice CookOmelet MakeStrawberrySmoothie MakeMatchaTea MakePickles Transplant ChangeTonerCartridge ReplaceSIMCard; do
  echo "=== $task $(date '+%F %T')"
  bash experiments/deviation_detection/scripts/run.sh "$task" _screen --download --max-correct 10 --no-natural | grep -E "correct of|used:|separation|synthetic|held-out"
done
