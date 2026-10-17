#!/bin/bash
#SBATCH --job-name=mcv_ddp
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

set -euo pipefail
cd ${SLURM_SUBMIT_DIR}
mkdir -p slurm_logs runs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

torchrun --nproc_per_node=2 train.py       dataset=rsna_trauma       model=resnet50       optim=adamw       seed=0       experiment.name="ddp_run"
