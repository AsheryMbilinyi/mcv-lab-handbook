# Tracking Experiments with MLflow

The lab standards ask every run to log its configuration, its metrics, the
code version it ran, and its artifacts (see "Experiment Tracking" in
`mcv_lab_standard_cicd_containers_mlops`). This guide shows how to do that
with MLflow in a project made from `templates/cookiecutter-mcv-project`, on
your own machine and on SLURM. Everything here was tested with MLflow 3.17,
Hydra 1.3 and PyTorch 2.

## The pieces

| MLflow term | In a lab project |
|---|---|
| Experiment | One per project, named by `experiment.project` in the config |
| Run | One call of `train.py`, named by `experiment.name` |
| Params | The whole resolved Hydra config, flattened (`optim.lr`, `model.name`, …) |
| Metrics | Losses and scores, logged with a `step` so they can be plotted |
| Tags | Seed, git commit, SLURM job IDs, run directory |
| Artifacts | `config_resolved.yaml`, the best checkpoint, plots |
| Tracking store | Where all of the above is kept: a SQLite file, or a tracking server |

## Choose where runs are stored

**On your own, or a small sweep: a SQLite file in the project directory.**
No server to run, and the file sits on the cluster's shared filesystem, so
every compute node can reach it. This is the setup the rest of the guide
uses.

**Large sweeps, or several people sharing results: a tracking server.**
SQLite allows one writer at a time. In testing, eight parallel jobs writing
to one SQLite file on a local disk worked fine, but locking on network
filesystems is less reliable, and many jobs logging at once can fail with
`database is locked`. When that happens, run a server (see "Using a tracking
server" below) and point the jobs at it.

## Setup

Add MLflow to the project's `requirements.txt` and install it:

```bash
pip install mlflow
```

Keep the tracking store out of git. The template's `.gitignore` ignores
`mlflow.db`, `mlartifacts/` and `mlruns/`; in a project generated before
this guide existed, add the first two yourself.

**Create the store once, before your first sweep:**

```bash
mlflow db upgrade "sqlite:///$PWD/mlflow.db"
```

Run this from the project root. If you skip it, the first jobs of a sweep
all try to create the database tables at the same moment, and some of them
crash with `table experiments already exists` or `duplicate column name`.
Once the store exists, jobs starting together no longer race to create it.

## Wiring it into the template

### 1. Add a `tracking` section to `configs/config.yaml`

```yaml
tracking:
  uri: sqlite:///${hydra:runtime.cwd}/mlflow.db
  artifact_root: ${hydra:runtime.cwd}/mlartifacts
```

The paths must be absolute. The template sets `hydra.job.chdir: true`, so by
the time `train.py` runs, the working directory is the run's own folder under
`runs/`. A relative path such as MLflow's default, `sqlite:///mlflow.db`,
then gives every run its own separate database, and you can never compare
two runs. `${hydra:runtime.cwd}` is the directory you launched from, which
is the project root when you run `python train.py` there or use the SLURM
scripts (they `cd ${SLURM_SUBMIT_DIR}`).

### 2. Add `src/utils/tracking.py`

```python
"""MLflow experiment tracking helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf


def flatten(d: dict[Any, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary into dotted keys.

    Args:
        d: The dictionary to flatten, for example a resolved config.
        prefix: Prefix for every key. Used by the recursion.

    Returns:
        A flat dictionary, such as ``{"optim.lr": 1e-4}``.
    """
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def git_tags(repo_dir: Path) -> dict[str, str]:
    """Describe the code version the run was launched from.

    MLflow normally records the commit itself, but not when Hydra has
    changed the working directory into the (git-ignored) run directory,
    so the template records it explicitly.

    Args:
        repo_dir: Any directory inside the project's git repository.

    Returns:
        The commit hash and whether there were uncommitted changes, or an
        empty dictionary if ``repo_dir`` is not in a git repository.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"mlflow.source.git.commit": commit, "git_dirty": str(dirty)}


def start_run(cfg: DictConfig) -> mlflow.ActiveRun:
    """Point MLflow at the lab store and start a run for this config.

    Args:
        cfg: The composed Hydra config. Uses ``cfg.tracking`` and
            ``cfg.experiment``.

    Returns:
        The active run. Use it as a context manager so the run is ended,
        and marked failed, even if training crashes.
    """
    mlflow.set_tracking_uri(cfg.tracking.uri)

    name = cfg.experiment.project
    if mlflow.get_experiment_by_name(name) is None:
        root = cfg.tracking.artifact_root
        location = Path(root).as_uri() if root else None
        try:
            mlflow.create_experiment(name, artifact_location=location)
        except MlflowException:
            # Another job in the same sweep created it first.
            pass
    mlflow.set_experiment(name)

    run = mlflow.start_run(run_name=cfg.experiment.name)
    try:
        config = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(config, dict)
        mlflow.log_params(flatten(config))
        mlflow.set_tags(
            {
                "seed": cfg.seed,
                "run_dir": os.getcwd(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                # Shared by every task of an array job, so it groups a
                # sweep.
                "slurm_array_job_id": os.environ.get(
                    "SLURM_ARRAY_JOB_ID", "none"
                ),
                "slurm_array_task_id": os.environ.get(
                    "SLURM_ARRAY_TASK_ID", "none"
                ),
            }
        )
        # This file is src/utils/tracking.py, so parents[2] is the repo
        # root.
        mlflow.set_tags(git_tags(Path(__file__).resolve().parents[2]))
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
    return run
```

Three things in this file are there on purpose:

- **The git commit is recorded explicitly.** MLflow normally finds the
  commit itself, but it looks next to the script *from the current working
  directory*, and after Hydra's `chdir` that is a git-ignored folder under
  `runs/`. Without `git_tags` the commit is silently missing. `git_dirty`
  is `True` when the run started with uncommitted changes, so a result you
  cannot reproduce from its commit is easy to spot. Commit before you
  launch a sweep.
- **Every SLURM ID is recorded.** In an array job each task has its own
  `SLURM_JOB_ID`; the `SLURM_ARRAY_JOB_ID` is what they share, so it is the
  tag that pulls up one whole sweep.
- **Use the run as a context manager** (`with start_run(cfg):`). A run that
  raises is then marked `FAILED` instead of staying `RUNNING` forever.

### 3. Log from `train.py`

```python
import mlflow

from src.utils.tracking import start_run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(os.getcwd())
    (run_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )
    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))

    with start_run(cfg):
        mlflow.log_artifact(str(run_dir / "config_resolved.yaml"))

        step = 0
        for epoch in range(int(cfg.trainer.epochs)):
            for batch in train_loader:
                loss = train_step(model, batch)
                if step % int(cfg.trainer.log_every) == 0:
                    mlflow.log_metric("train/loss", loss.item(), step=step)
                step += 1

            val = evaluate(model, val_loader)
            mlflow.log_metrics(
                {"val/loss": val["loss"], "val/auroc": val["auroc"]},
                step=step,
            )

        mlflow.log_artifact(
            str(run_dir / "checkpoints" / "best.ckpt"),
            artifact_path="checkpoints",
        )
```

- **Log every `log_every` steps, not every step.** Each call is a database
  write; logging every step of a long run makes the store large and the UI
  slow.
- **Always pass `step`.** Without it every value is logged at step 0, and
  the curve collapses to a single point.
- **With DDP, log from rank 0 only** (`if dist.get_rank() == 0:` around the
  `start_run` block and every `mlflow.log_*` call). Otherwise each GPU
  starts its own run.
- **Upload the best checkpoint, not every checkpoint.** `last.ckpt` and the
  per-epoch checkpoints stay in the run directory; the `run_dir` tag tells
  you where.

## Running on SLURM

The template's SLURM scripts need no changes. Create the store once (see
Setup), commit your code, then submit as usual:

```bash
sbatch scripts/sweep_seeds.slurm
```

Each task becomes one run, tagged with its seed and SLURM IDs.

## Looking at results

Start the UI on the cluster's login node, from the project root:

```bash
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db" --port 5000
```

Then, on your laptop, forward the port and open http://localhost:5000:

```bash
ssh -L 5000:localhost:5000 <you>@<cluster>
```

If port 5000 is taken on the login node, pick another and use it on both
sides. Stop the UI with `Ctrl-C` when you are done; login nodes are shared.

In the UI, select several runs and click **Compare** to see their curves
and params side by side. Search with the same syntax as the code below, for
example `params.optim.lr = '0.0001'` or `tags.seed = '2'`.

## Summarizing a sweep

The experimentation standard recommends an automated summary of metrics
across runs. With MLflow that is a query:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(
    experiment_names=["mcv_lab"],
    filter_string="tags.slurm_array_job_id = '123456' and status = 'FINISHED'",
)
auroc = runs["metrics.val/auroc"]
print(runs[["tags.seed", "metrics.val/auroc"]].sort_values("tags.seed"))
print(f"val/auroc over {len(runs)} seeds: {auroc.mean():.4f} ± {auroc.std():.4f}")
```

Replace `123456` with the array job ID that `sbatch` printed. `search_runs`
returns a pandas DataFrame with one column per param (`params.…`), metric
(`metrics.…`) and tag (`tags.…`), holding each metric's last logged value.
Params and tags come back as strings.

## Using a tracking server

Start a server on a machine the compute nodes can reach (check your
cluster's documentation; some do not allow long-running processes on login
nodes):

```bash
mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts "<server-host>:5000,localhost:5000"
```

`--allowed-hosts` is needed whenever jobs reach the server by hostname: the
server otherwise rejects them with `403 Invalid Host header`. Each entry
needs the port, and the list replaces the default, so keep `localhost:5000`
in it if you also use the server through an SSH tunnel.

Then point the jobs at it on the command line. Leave `artifact_root` empty so
the server decides where artifacts go:

```bash
python train.py tracking.uri=http://<server-host>:5000 tracking.artifact_root=null
```

Started this way, the server has no login. Anyone who can reach the port
can read and change every run, so keep it inside the cluster or lab network and never
expose it to the internet.

## Registering a model (optional)

The MLOps standard separates *checkpoints* (training states) from *model
versions* (a model you have approved, with the run that produced it).
MLflow's model registry is one way to keep model versions:

```python
import numpy as np

mlflow.pytorch.log_model(
    model,
    name="model",
    input_example=np.zeros((1, 3, 224, 224), dtype=np.float32),
    registered_model_name="rsna-trauma-resnet50",
)
```

Use an input of the shape your model really takes. Current MLflow saves an
exported graph of the model, which needs an `input_example`, and the example
must be a NumPy array: a torch tensor gives an "Unsupported signature type"
error. Load a version with
`mlflow.pytorch.load_model("models:/rsna-trauma-resnet50/1")`. The loaded
object is for inference; to resume training, use your checkpoints.

## Quick reference: symptom → likely cause

| Symptom | Most likely cause |
|---|---|
| Each run has its own `mlflow.db` inside `runs/…`, and runs never appear together | Relative tracking URI combined with Hydra's `chdir`; use the absolute `tracking.uri` above |
| The first jobs of a sweep crash with `table experiments already exists` or `duplicate column name` | The store did not exist yet; run `mlflow db upgrade` once before submitting |
| No git commit shown for a run | Hydra's `chdir` hid the repository from MLflow; use `git_tags` from the helper |
| `database is locked` | Too many jobs writing to one SQLite file, often on a network filesystem; use a tracking server |
| `Changing param values is not allowed` | Params are fixed once logged; log values that change during training, such as the learning rate, as metrics |
| Runs stuck as `RUNNING` | The job was killed (time limit, out of memory) before MLflow could close the run; see below |
| One run per GPU in a DDP job | Logging from every rank; log from rank 0 only |
| UI slow, `mlflow.db` very large | Logging every step; use `trainer.log_every` |

SLURM kills a job without giving Python the chance to close its run. To
mark those runs, once their jobs are gone:

```python
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
stuck = mlflow.search_runs(
    experiment_names=["mcv_lab"], filter_string="status = 'RUNNING'"
)
for run_id in stuck.run_id:
    client.set_terminated(run_id, status="KILLED")
```

## Checklist

Before you trust a sweep's numbers:

- [ ] Code committed, so every run has a commit and `git_dirty` is `False`
- [ ] Tracking store created once with `mlflow db upgrade`
- [ ] All runs in one experiment, in one store
- [ ] Resolved config uploaded as an artifact of each run
- [ ] Metrics logged with `step`
- [ ] Best checkpoint uploaded; the rest findable from the `run_dir` tag
- [ ] Results summarized with `search_runs`, mean ± std over seeds
