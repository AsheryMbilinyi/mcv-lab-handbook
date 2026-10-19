#!/usr/bin/env python3
"""Hydra training entrypoint.

Run ``python train.py`` and override any config value on the command line,
for example ``python train.py model=resnet50 optim.lr=5e-5 seed=2``.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from src.utils.seed import set_seed
from src.utils.tracking import is_main_process, start_run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one training job described by a Hydra config.

    Args:
        cfg: The composed configuration, injected by Hydra.

    Side effects:
        Writes ``config_resolved.yaml`` and a placeholder checkpoint into
        the Hydra run directory, which is also the working directory, and
        logs the run to the MLflow store in ``cfg.tracking``.
    """
    # Hydra changes the working directory to the run directory because
    # hydra.job.chdir is true.
    run_dir = Path(os.getcwd())
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config. This is mandatory in the lab standard.
    (run_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )

    # Optionally save the command too. Hydra stores the overrides, but
    # logging the full command is still useful.
    # cmd_file = run_dir / "cmd.txt"
    # cmd_file.write_text("python train.py ...\n", encoding="utf-8")

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))

    print("=== RUN DIR ===")
    print(run_dir)

    print("\n=== CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # The run is marked failed if anything in this block raises. Under
    # DDP only rank 0 tracks, so the job is one run. Guard every mlflow
    # call with is_main_process() as well: a call made without an active
    # run silently starts a new one.
    main_process = is_main_process()
    with start_run(cfg) if main_process else nullcontext():
        if main_process:
            mlflow.log_artifact(str(run_dir / "config_resolved.yaml"))

        # Train here. Log metrics with their step, every
        # cfg.trainer.log_every steps:
        #   if main_process:
        #       mlflow.log_metric("train/loss", loss.item(), step=step)

        # Placeholder checkpoint write. Replace it with real checkpointing,
        # then upload the best checkpoint from rank 0:
        #   mlflow.log_artifact(str(best), artifact_path="checkpoints")
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        (ckpt_dir / "last.ckpt").write_text(
            "dummy checkpoint", encoding="utf-8"
        )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
