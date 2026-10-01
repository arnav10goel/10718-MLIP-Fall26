#!/bin/bash
#SBATCH --job-name=extract-feats
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# CLIP feature extraction for COIN on a GPU (extract_features.py). Cached files are skipped, so a requeued job resumes.
# Submit from the repo root with the environment activated:  sbatch baselines/run_extract_features.sbatch
cd "$SLURM_SUBMIT_DIR"
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
${PYTHON:-python} -u baselines/extract_features.py --workers 6 "$@"
