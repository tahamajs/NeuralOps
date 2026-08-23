"""Reproducibility utilities for cross-platform deterministic RNG seeding."""

from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set consistent random seed across Python, NumPy, PyTorch CPU, CUDA, and Apple MPS.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
