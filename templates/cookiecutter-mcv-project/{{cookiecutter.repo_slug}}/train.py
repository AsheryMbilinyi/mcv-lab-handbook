#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra changes CWD to run dir because hydra.job.chdir=true
    run_dir = Path(os.getcwd())
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config (mandatory lab standard)
    (run_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )

    # Optional: save command if desired (Hydra stores overrides; cmd logging is still useful)
    # (run_dir / "cmd.txt").write_text("python train.py ...\n", encoding="utf-8")

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))

    print("=== RUN DIR ===")
    print(run_dir)

    print("\n=== CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # Placeholder checkpoint write (replace with real checkpointing)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (ckpt_dir / "last.ckpt").write_text("dummy checkpoint", encoding="utf-8")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
