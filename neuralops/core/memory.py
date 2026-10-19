# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
VRAM and Memory Allocation Profiler

GPU memory tracking, peak allocation measurement, and activation footprint analysis utilities.
"""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def cleanup_gpu_memory(force: bool = True) -> None:
    """
    Perform exhaustive GPU / MPS / CPU memory deallocation.
    Frees all cached blocks, calls cyclic garbage collector, and flushes allocators.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        if force and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except (
            AttributeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            ImportError,
        ):
            pass


def get_gpu_memory_stats() -> dict[str, float]:
    """
    Get detailed memory statistics in Megabytes (MB).
    """
    stats: dict[str, float] = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
    }

    if torch.cuda.is_available():
        stats["allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        stats["reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        stats["max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            stats["allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            if hasattr(torch.mps, "driver_allocated_memory"):
                stats["reserved_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024)
        except (
            AttributeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            ImportError,
        ):
            pass

    return stats


class GPUMemoryGuard:
    """Context manager for zero-leak memory safety during execution."""

    def __enter__(self):
        cleanup_gpu_memory(force=False)
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        cleanup_gpu_memory(force=True)


__all__ = ["cleanup_gpu_memory", "get_gpu_memory_stats", "GPUMemoryGuard"]
