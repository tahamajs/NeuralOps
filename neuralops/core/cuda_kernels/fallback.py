"""
Hardware Kernel Dispatch and Fallback Management

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.cuda_kernels._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
    get_last_dispatch_backend,
    is_kernel_available,
    register_kernel,
)

__all__ = [
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "get_last_dispatch_backend",
    "is_kernel_available",
    "register_kernel",
]
