#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Memory Validator

Tracks and profiles host RAM and GPU VRAM usage during model execution.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Tracks GPU VRAM allocation, reservation, and peak memory usage."""

    def __init__(self, device: Optional[Union[int, str, torch.device]] = None) -> None:
        self.device = self._resolve_device(device)
        self.peak: float = 0.0
        self.reset()

    @staticmethod
    def _resolve_device(device: Optional[Union[int, str, torch.device]] = None) -> Optional[torch.device]:
        if not torch.cuda.is_available():
            return None
        if device is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str):
            return torch.device(device if device.startswith("cuda") else f"cuda:{device}")
        return device

    def reset(self) -> None:
        if torch.cuda.is_available() and self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak = 0.0

    def update(self) -> float:
        if torch.cuda.is_available() and self.device is not None:
            current = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.peak = max(self.peak, current)
            return current
        return 0.0

    def get_stats(self) -> Dict[str, float]:
        """Return memory statistics dictionary per Rule 2.4."""
        if not torch.cuda.is_available() or self.device is None:
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "peak_mb": 0.0,
                "total_mb": 0.0,
            }
        dev = self.device
        allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(dev) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
        total_mb = torch.cuda.get_device_properties(dev).total_memory / (1024 * 1024)
        return {
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "peak_mb": peak_mb,
            "total_mb": total_mb,
        }

    def report(self) -> None:
        if torch.cuda.is_available() and self.device is not None:
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            ratio = (self.peak / total * 100.0) if total > 0 else 0.0
            print(f"  Peak VRAM: {self.peak:.2f} GB / {total:.2f} GB ({ratio:.1f}%)")
        else:
            print("  Peak VRAM: N/A (CPU mode)")

    def check_ok(self, threshold: float = 0.9) -> bool:
        if not torch.cuda.is_available() or self.device is None:
            return True
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        if total > 0 and (self.peak / total) > threshold:
            raise MemoryError(f"VRAM usage > {threshold * 100:.1f}%")
        return True


__all__ = ["MemoryTracker"]
