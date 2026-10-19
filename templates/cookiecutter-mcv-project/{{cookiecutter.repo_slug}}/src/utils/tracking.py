"""MLflow experiment tracking helpers.

See ``docs/mlflow_guide.md`` in the MCV Lab handbook for how to use them.
"""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

SQLITE_PREFIX = "sqlite:///"


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


def is_main_process() -> bool:
    """Tell whether this process should log to MLflow.

    ``torchrun`` starts one process per GPU and sets ``RANK`` in each.
    Only rank 0 logs, so a DDP job is one run. A process without ``RANK``,
    such as a single-GPU job, counts as rank 0.

    Returns:
        True for rank 0.
    """
    return os.environ.get("RANK", "0") == "0"


@contextmanager
def store_lock(uri: str, timeout: float = 300.0) -> Iterator[None]:
    """Let only one job at a time connect to a SQLite tracking store.

    The first connection to a new SQLite store creates its tables. When
    several jobs of a sweep do that at the same moment they corrupt the
    new store, so connections are serialized with a lock directory next
    to the database file. Creating a directory is atomic, also on network
    filesystems. Other stores, such as a tracking server, need no lock.

    Args:
        uri: The tracking URI.
        timeout: Seconds to wait for the lock before giving up.

    Raises:
        TimeoutError: If the lock is still held after ``timeout`` seconds,
            usually because a job was killed while holding it.
    """
    if not uri.startswith(SQLITE_PREFIX):
        yield
        return

    lock = Path(uri[len(SQLITE_PREFIX) :] + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{lock} is still locked after {timeout:.0f} s. If no "
                    "other job is starting, a killed job left it behind; "
                    "remove it with: rmdir " + str(lock)
                ) from None
            time.sleep(0.5)
    try:
        yield
    finally:
        lock.rmdir()


def start_run(cfg: DictConfig) -> mlflow.ActiveRun:
    """Point MLflow at the lab store and start a run for this config.

    Args:
        cfg: The composed Hydra config. Uses ``cfg.tracking`` and
            ``cfg.experiment``.

    Returns:
        The active run. Use it as a context manager so the run is ended,
        and marked failed, even if training crashes.
    """
    uri = str(cfg.tracking.uri)
    mlflow.set_tracking_uri(uri)

    name = cfg.experiment.project
    with store_lock(uri):
        if mlflow.get_experiment_by_name(name) is None:
            root = cfg.tracking.artifact_root
            location = Path(root).as_uri() if root else None
            try:
                mlflow.create_experiment(name, artifact_location=location)
            except MlflowException:
                # Another job of the same sweep, logging to a tracking
                # server, created it first.
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
