"""
CUDA error checking, stream management, and device capability queries.
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
