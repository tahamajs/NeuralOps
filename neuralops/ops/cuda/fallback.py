#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Kernel Dispatch and Fallback Management
"""

from __future__ import annotations

from neuralops.ops.cuda._fallback import (
    FORCE_PYTORCH_FALLBACK,
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
    get_last_dispatch_backend,
    get_last_dispatch_error,
    is_kernel_available,
    register_kernel,
)

__all__ = [
    "FORCE_PYTORCH_FALLBACK",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "get_last_dispatch_backend",
    "get_last_dispatch_error",
    "is_kernel_available",
    "register_kernel",
]
