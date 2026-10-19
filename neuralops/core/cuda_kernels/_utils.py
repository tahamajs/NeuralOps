# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CUDA Kernel Utilities

Helper functions for CUDA error checking, stream synchronization, and GPU device capability queries.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Return True if PyTorch detects a functional CUDA environment."""
    return torch.cuda.is_available()


def get_device_capability(device: int | torch.device = 0) -> Tuple[int, int]:
    """Return compute capability (major, minor) for target GPU."""
    if not torch.cuda.is_available():
        return (0, 0)
    try:
        dev_idx = device.index if isinstance(device, torch.device) else device
        if dev_idx is None:
            dev_idx = 0
        return torch.cuda.get_device_capability(dev_idx)
    except Exception as e:
        logger.debug(f"get_device_capability failed: {e}")
        return (0, 0)


def check_cuda_error() -> None:
    """Synchronize current stream and check for pending asynchronous CUDA errors."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
