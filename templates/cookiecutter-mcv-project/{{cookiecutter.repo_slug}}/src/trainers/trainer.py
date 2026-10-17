"""Training loop and training state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainState:
    """Counters that track progress through training.

    Attributes:
        epoch: Index of the current epoch.
        step: Number of optimizer steps taken so far.
    """

    epoch: int = 0
    step: int = 0


def train_loop(*, cfg: Any) -> None:
    """Run a placeholder training loop.

    Replace this with your actual training loop (PyTorch Lightning or
    custom).

    Args:
        cfg: The experiment config. Only ``cfg.trainer.epochs`` is used.
    """
    state = TrainState()
    for epoch in range(int(cfg.trainer.epochs)):
        state.epoch = epoch
        # TODO: Iterate the data loader, then run forward, backward,
        # optimize, log and checkpoint.
