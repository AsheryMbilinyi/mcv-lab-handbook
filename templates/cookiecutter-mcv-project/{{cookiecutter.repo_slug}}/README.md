# {{ cookiecutter.project_name }} (MCV Lab)

{{ cookiecutter.description }}

This repository follows the **MCV Lab Experimentation Standard**:
- no code edits for experiments
- configs drive experimentation (Hydra recommended)
- reproducibility artifacts saved per run
- SLURM scripts included for HPC

## Setup
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

## SLURM
```bash
sbatch scripts/run_train.slurm
sbatch scripts/sweep_seeds.slurm
sbatch scripts/ddp_train.slurm
```
