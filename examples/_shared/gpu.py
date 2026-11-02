#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware and accelerator detection utilities for NeuralOps examples.
"""

from __future__ import annotations

import torch


def gpu_available() -> bool:
    """Check if a CUDA GPU accelerator is available."""
    return torch.cuda.is_available()


def gpu_name() -> str:
    """Return a descriptive name of the available hardware accelerator."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"CUDA: {device_name}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple Silicon MPS"
    return "CPU only"


def get_backend_preference() -> str:
    """Return the preferred device string ('cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_cuda_sync() -> None:
    """Synchronize CUDA stream if CUDA is available; no-op on CPU and MPS."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
