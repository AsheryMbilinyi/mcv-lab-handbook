# Tracking Experiments with MLflow

The lab standards ask every run to log its configuration, its metrics, the
code version it ran, and its artifacts (see "Experiment Tracking" in
`mcv_lab_standard_cicd_containers_mlops`). This guide shows how to do that
with MLflow in a project made from `templates/cookiecutter-mcv-project`,
which comes with MLflow tracking built in, on your own machine and on SLURM. Everything here was tested with MLflow 3.17,
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
every compute node can reach it. This is the template's default, and the
setup the rest of the guide uses.

**Large sweeps, or several people sharing results: a tracking server.**
SQLite allows one writer at a time. In testing, eight parallel jobs writing
to one SQLite file on a local disk worked fine, but locking on network
filesystems is less reliable, and many jobs logging at once can fail with
`database is locked`. When that happens, run a server (see "Using a tracking
server" below) and point the jobs at it.

## What the template already does

Projects made from `templates/cookiecutter-mcv-project` track every run of
`train.py` with no setup beyond `pip install -r requirements.txt`, which
includes MLflow:

- `configs/config.yaml` has a `tracking` section that says where the store
  is.
- `src/utils/tracking.py` has `start_run(cfg)`, which connects to the
  store, starts a run, and logs the resolved config as params plus the
  seed, git commit and SLURM IDs as tags.
- `train.py` wraps the job in `with start_run(cfg):` and uploads
  `config_resolved.yaml`.

What is left for you is logging your metrics and your best checkpoint (see
"Logging from your training code" below). The rest of this section explains
the choices, so you know what to keep if you change them.

**The store paths are absolute.**

```yaml
tracking:
  uri: sqlite:///${hydra:runtime.cwd}/mlflow.db
  artifact_root: ${hydra:runtime.cwd}/mlartifacts
```

The template sets `hydra.job.chdir: true`, so by the time `train.py` runs,
the working directory is the run's own folder under `runs/`. A relative
path such as MLflow's default, `sqlite:///mlflow.db`, would then give every
run its own separate database, and you could never compare two runs.
`${hydra:runtime.cwd}` is the directory you launched from: the project root
when you run `python train.py` there or use the SLURM scripts (they
`cd ${SLURM_SUBMIT_DIR}`). Always launch from the project root.

**The git commit is recorded explicitly.** MLflow normally finds the
commit itself, but it looks next to the script *from the current working
directory*, and after Hydra's `chdir` that is a git-ignored folder under
`runs/`. Without the helper's `git_tags` the commit is silently missing.
`git_dirty` is `True` when the run started with uncommitted changes,
including new files git does not track yet, so a result you cannot
reproduce from its commit is easy to spot. Commit before you launch a
sweep.

**Jobs take turns connecting to a SQLite store.** The first connection to
a new store creates its tables. When the jobs of a sweep all do that at the
same moment, they can corrupt the new store, and then every job fails; in
testing this happened in two of three tries. `start_run` therefore holds a
lock, the directory `mlflow.db.lock`, only while it connects; training
itself runs in parallel as usual. If a job is killed while holding the
lock, the next jobs wait five minutes and then fail with a message telling
you to remove it. A tracking server needs no lock.

**Every SLURM ID is recorded.** In an array job each task has its own
`SLURM_JOB_ID`; the `SLURM_ARRAY_JOB_ID` is what they share, so it is the
tag that pulls up one whole sweep.

**The run is a context manager.** A run that raises inside
`with start_run(cfg):` is marked `FAILED` instead of staying `RUNNING`
forever.

**Only rank 0 tracks.** `torchrun` (as in `scripts/ddp_train.slurm`)
starts one `train.py` per GPU. `train.py` calls `start_run` only where
`is_main_process()` is true, so a DDP job is one run.

For a project made before the template had tracking, copy
`src/utils/tracking.py`, the `tracking` section of `configs/config.yaml`
and the `start_run` block of `train.py` from the template, add `mlflow` to
`requirements.txt`, and add `mlflow.db` and `mlartifacts/` to
`.gitignore`.

## Logging from your training code

Inside the `with start_run(cfg):` block in `train.py`:

```python
step = 0
for epoch in range(int(cfg.trainer.epochs)):
    for batch in train_loader:
        loss = train_step(model, batch)
        if main_process and step % int(cfg.trainer.log_every) == 0:
            mlflow.log_metric("train/loss", loss.item(), step=step)
        step += 1

    val = evaluate(model, val_loader)
    if main_process:
        mlflow.log_metrics(
            {"val/loss": val["loss"], "val/auroc": val["auroc"]},
            step=step,
        )

if main_process:
    mlflow.log_artifact(
        str(run_dir / "checkpoints" / "best.ckpt"),
        artifact_path="checkpoints",
    )
```

- **Guard every `mlflow` call with `main_process`.** A logging call made
  where no run is active, as on the other ranks of a DDP job, silently
  starts a new run. Without DDP, `main_process` is always true.
- **Log every `log_every` steps, not every step.** Each call is a database
  write; logging every step of a long run makes the store large and the UI
  slow.
- **Always pass `step`.** Without it every value is logged at step 0, and
  the curve collapses to a single point.
- **Upload the best checkpoint, not every checkpoint.** `last.ckpt` and the
  per-epoch checkpoints stay in the run directory; the `run_dir` tag tells
  you where.

## Running on SLURM

The template's SLURM scripts need no changes. Commit your code, then submit
as usual:

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
| The first jobs of a sweep crash with `table … already exists` or `duplicate column name` | Jobs created a new store at the same moment, without the template's lock. Delete the broken `mlflow.db` and use `start_run` from the template |
| `mlflow.db.lock is still locked after 300 s` | A job was killed while connecting and left the lock behind; check no job is starting, then `rmdir` it |
| No git commit shown for a run | Hydra's `chdir` hid the repository from MLflow; use `git_tags` from the helper |
| `database is locked` | Too many jobs writing to one SQLite file, often on a network filesystem; use a tracking server |
| `Changing param values is not allowed` | Params are fixed once logged; log values that change during training, such as the learning rate, as metrics |
| Runs stuck as `RUNNING` | The job was killed (time limit, out of memory) before MLflow could close the run; see below |
| One run per GPU in a DDP job, or extra runs with no params | An `mlflow` call on a rank other than 0; guard it with `is_main_process()` |
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
- [ ] All runs in one experiment, in one store
- [ ] Resolved config uploaded as an artifact of each run
- [ ] Metrics logged with `step`
- [ ] Best checkpoint uploaded; the rest findable from the `run_dir` tag
- [ ] Results summarized with `search_runs`, mean ± std over seeds
