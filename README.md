# MCV Lab Handbook

Shared reference material for the Medical Computer Vision (MCV) Lab at the
University of Victoria: our coding and experimentation standards, a guide to
debugging ML experiments, runnable debugging drills, a guide to tracking
experiments with MLflow, and a project template.

## What is in this repository

| Path | What it is |
|---|---|
| `docs/standards/` | LaTeX sources of the lab standards (see below). Built PDFs are in `docs/standards/pdf/`. |
| `docs/debugging_guide.md` | A step-by-step process for debugging a model that will not train, with a symptom to cause cheat sheet. |
| `docs/mlflow_guide.md` | How to track experiments with MLflow in a project made from the template, locally and on SLURM. |
| `drills/pytorch_debug_drills.py` | Six runnable "spot the bug" scenarios that go with the debugging guide. |
| `templates/cookiecutter-mcv-project/` | Starter for a new experiment repository (Hydra configs, SLURM scripts, trainer skeleton). |

### The standards

- `coding_standards`: Python coding standards (PEP 8, naming, docstrings, tooling)
- `mcv_lab_experimentation_standard`: scalable, reproducible ML experimentation
- `argparse_lab_standard` and `hydra_lab_standards`: configuring experiments with argparse or Hydra
- `slurm_lab_standard`: SLURM scripts for HPC runs
- `mcv_lab_standard_docker_kubernetes`: containers and Kubernetes
- `mcv_lab_standard_cicd_containers_mlops`: CI/CD and MLOps
- `mcv_lab_standard_cloud_ml_deployment`: cloud deployment of ML models
- `mcv_lab_standard_api_design_scaling`: API design and scaling

## Getting started

To read the standards you need nothing: open the PDFs in
`docs/standards/pdf/`, which GitHub renders in the browser.

Everything else needs Python 3.10 or newer. Clone the repository and
create a virtual environment:

```bash
git clone https://github.com/AsheryMbilinyi/mcv-lab-handbook.git
cd mcv-lab-handbook
python -m venv .venv
source .venv/bin/activate
```

Then install only what you need:

```bash
# To run the debugging drills (NumPy and PyTorch)
pip install -r requirements.txt

# To contribute (adds the lint tools and the pre-commit hooks)
pip install -r requirements-dev.txt
pre-commit install
```

## Debugging your experiments

Read `docs/debugging_guide.md`, then run the drills and try to name each bug
from the printed symptom before reading the fix:

```bash
python drills/pytorch_debug_drills.py              # all scenarios
python drills/pytorch_debug_drills.py --scenario 4 # just the NaN-loss one
```

## Starting a new project

Create a new repository from the project template:

```bash
pip install cookiecutter
cookiecutter templates/cookiecutter-mcv-project
```

Generated projects need Python 3.10 to 3.13. They use Hydra 1.3, which
fails at startup on Python 3.14 (`ValueError: badly formed help string`).

The generated project already meets the minimum requirements in section 6.2
of the coding standard: a `README.md` with setup and usage, a
reproducible environment file, and tool configuration (`pyproject.toml`,
`.flake8` and pre-commit hooks for black, isort and flake8).

## Code style

The coding standard is enforced with isort, black and flake8, all configured
for the standard's 79 character line limit. Before opening a pull request:

```bash
make format   # fix formatting and import order
make lint     # the same checks that run in CI
```

## Rebuilding the standards PDFs

Requires a LaTeX installation with `latexmk`.

```bash
make docs
```

## Contributing

Propose changes to a standard through a pull request, and describe what
changed and why in the description. Keep the LaTeX source as the single
source of truth and rebuild the PDFs with `make docs` before committing.

## License

MIT, see `LICENSE`.
