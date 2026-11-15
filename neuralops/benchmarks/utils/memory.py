#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware memory telemetry and tracking utilities for benchmarks.

Adheres strictly to AGENTS.md:
- Rule 2.4: Memory reporting (allocated, reserved, peak allocated)
- Rule 2.5: Safe device fallback for CPU, MPS, and CUDA
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import torch

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryStats:
    """Hardware memory statistics across active devices."""

    device: str
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    device_total_mb: float | None = None


def cleanup_memory(force: bool = True) -> None:
    """Perform garbage collection and allocator cache flushing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        if force and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def get_memory_telemetry(device: torch.device | str | None = None) -> MemoryStats:
    """Capture current memory consumption metrics per Rule 2.4.

    Args:
        device: Device to inspect. Defaults to cuda if available, else mps if available, else cpu.

    Returns:
        MemoryStats instance.
    """
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    if dev.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(dev) / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved(dev) / (1024.0 * 1024.0)
        peak = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
        try:
            props = torch.cuda.get_device_properties(dev)
            total = float(props.total_memory) / (1024.0 * 1024.0)
        except Exception:
            total = None
        return MemoryStats(
            device=str(dev),
            allocated_mb=round(allocated, 4),
            reserved_mb=round(reserved, 4),
            peak_mb=round(peak, 4),
            device_total_mb=round(total, 4) if total is not None else None,
        )

    if dev.type == "mps" and hasattr(torch, "mps"):
        try:
            allocated = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
            driver = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
            return MemoryStats(
                device="mps",
                allocated_mb=round(allocated, 4),
                reserved_mb=round(driver, 4),
                peak_mb=round(max(allocated, driver), 4),
                device_total_mb=None,
            )
        except Exception:
            pass

    return MemoryStats(
        device=str(dev),
        allocated_mb=0.0,
        reserved_mb=0.0,
        peak_mb=0.0,
        device_total_mb=None,
    )


__all__ = [
    "MemoryStats",
    "cleanup_memory",
    "get_memory_telemetry",
]
