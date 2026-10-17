"""Seeding for reproducible runs."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for reproducibility.

    Args:
        seed: Seed shared by Python, NumPy and, if installed, PyTorch.
        deterministic: If true, make cuDNN deterministic and turn off its
            autotuner. This can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
