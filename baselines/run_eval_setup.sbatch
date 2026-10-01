#!/bin/bash
#SBATCH --job-name=eval-coin
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# COIN evaluation (eval_setup.py) for several step groups, then the cross-task summary (summarize_tasks.py).
# Hand-crafted features are computed and cached on first use; CLIP features must already be cached.
# Submit from the repo root with the environment activated:
#   sbatch baselines/run_eval_setup.sbatch ReplaceSIMCard-1 BoilNoodles-1 MakeBurger-1 ChangeBikeTires-1 ParkParallel-1 WashDish-1
cd "$SLURM_SUBMIT_DIR"
export PYTHONWARNINGS=ignore
for g in "$@"; do
  echo "===== $g"
  ${PYTHON:-python} -u baselines/eval_setup.py --group "$g" --workers "$SLURM_CPUS_PER_TASK"
done
${PYTHON:-python} -u baselines/summarize_tasks.py --groups "$@"
