"""
Random Seed Utilities

Sets deterministic random seeds across PyTorch, NumPy, and Python standard libraries.

Author: Taha Majlesi
"""

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set the all seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
