# {{ cookiecutter.project_name }} (MCV Lab)

{{ cookiecutter.description }}

This repository follows the **MCV Lab Experimentation Standard**:
- no code edits for experiments
- configs drive experimentation (Hydra recommended)
- reproducibility artifacts saved per run
- SLURM scripts included for HPC

## Setup
Use Python 3.10 to 3.13. Hydra 1.3 fails at startup on Python 3.14.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code style
The lab coding standard is enforced with isort, black and flake8.
```bash
pip install -r requirements-dev.txt
pre-commit install   # run the checks on every commit
isort . && black .   # fix formatting and import order
flake8 .             # check style
mypy .               # check types
```

## Training
```bash
python train.py
python train.py model=resnet50 optim.lr=5e-5 seed=2
python train.py -m optim.lr=1e-3,1e-4,1e-5 seed=0,1,2
```

## Experiment tracking
Every run of `train.py` is logged to MLflow: the resolved config, the
seed, the git commit and the SLURM job IDs. Runs go to `mlflow.db` and
`mlartifacts/` in the directory you launch from, so launch from the
project root. To browse them:
```bash
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db"
```
See `docs/mlflow_guide.md` in the MCV Lab handbook for logging metrics,
viewing runs from the cluster and summarizing sweeps.

## SLURM
```bash
sbatch scripts/run_train.slurm
sbatch scripts/sweep_seeds.slurm
sbatch scripts/ddp_train.slurm
```
