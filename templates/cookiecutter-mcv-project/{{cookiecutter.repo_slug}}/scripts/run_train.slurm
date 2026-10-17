#!/bin/bash
#SBATCH --job-name=mcv_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd ${SLURM_SUBMIT_DIR}
mkdir -p slurm_logs runs

echo "JOB_ID: ${SLURM_JOB_ID}"
echo "NODELIST: ${SLURM_NODELIST}"
echo "START: $(date)"
nvidia-smi || true

python train.py       dataset=rsna_trauma       model=resnet50       optim=adamw       seed=0

echo "END: $(date)"
