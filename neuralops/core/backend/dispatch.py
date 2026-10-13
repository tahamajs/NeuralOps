"""
Backend Dispatch Registry and Dispatcher for Hardware-Specific Ops.
"""
from __future__ import annotations

import torch
from enum import Enum
from typing import Callable, Dict, Any, Optional


class BackendDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    TRITON = "triton"


def get_available_backend() -> BackendDevice:
    if torch.cuda.is_available():
        try:
            import triton
            return BackendDevice.TRITON
        except ImportError:
            return BackendDevice.CUDA
    return BackendDevice.CPU
