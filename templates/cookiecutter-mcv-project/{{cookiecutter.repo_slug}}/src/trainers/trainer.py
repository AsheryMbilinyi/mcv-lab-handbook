from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0


def train_loop(*, cfg: Any) -> None:
    """Placeholder training loop.

    Replace with your actual training loop (PyTorch Lightning or custom).
    """
    state = TrainState()
    for epoch in range(int(cfg.trainer.epochs)):
        state.epoch = epoch
        # TODO: iterate data loader, forward, backward, optimize, log, checkpoint
