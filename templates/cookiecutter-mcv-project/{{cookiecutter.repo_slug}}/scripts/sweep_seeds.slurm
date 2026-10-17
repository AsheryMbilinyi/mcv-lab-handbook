#!/bin/bash
#SBATCH --job-name=mcv_seed_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4

set -euo pipefail
cd ${SLURM_SUBMIT_DIR}
mkdir -p slurm_logs runs

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python train.py       dataset=rsna_trauma       model=resnet50       optim=adamw       seed=${SEED}       experiment.name="seed_sweep"
