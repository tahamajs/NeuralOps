"""Comprehensive seed setting for reproducibility."""

import random
import numpy as np
import torch


def set_all_seeds(seed: int):
    """
    Set all random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False