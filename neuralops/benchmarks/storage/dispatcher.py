#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""MultiGPUDispatcher and VRAMGovernor with strict lease and lifecycle tracking."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

LOG = logging.getLogger(__name__)


@dataclass
class DeviceLease:
    device_id: int
    task_id: str
    active: bool = True

    def release(self) -> None:
        self.active = False


class VRAMGovernor:
    """Manages memory limits and scheduling decisions based on VRAM availability."""

    def __init__(self, min_free_mb: int = 1024) -> None:
        self.min_free_mb = min_free_mb

    def can_schedule(self, device_id: int = 0) -> bool:
        """Decide whether a task requiring VRAM can be scheduled."""
        import torch

        if not torch.cuda.is_available():
            # Non-CUDA environment: permit scheduling
            return True

        if device_id < 0 or device_id >= torch.cuda.device_count():
            return False

        try:
            device = torch.device(f"cuda:{device_id}")
            props = torch.cuda.get_device_properties(device)
            total_mb = props.total_memory // (1024 * 1024)
            if total_mb == 0:
                return True
            allocated_mb = torch.cuda.memory_allocated(device) // (1024 * 1024)
            free_mb = total_mb - allocated_mb
            return free_mb >= self.min_free_mb
        except Exception as e:
            LOG.warning("Failed to query VRAM on cuda:%d: %s", device_id, e)
            return False


class MultiGPUDispatcher:
    """Dispatches tasks across available GPUs and tracks device leases."""

    def __init__(self, device_ids: Optional[List[int]] = None, governor: Optional[VRAMGovernor] = None) -> None:
        self.device_ids = device_ids or [0]
        self.governor = governor or VRAMGovernor()
        self._lock = threading.Lock()
        self._active_leases: Dict[int, Set[str]] = {dev: set() for dev in self.device_ids}

    def acquire_device(self, task_id: str, preferred_device: Optional[int] = None) -> Optional[DeviceLease]:
        with self._lock:
            candidates = [preferred_device] if preferred_device is not None else self.device_ids
            for dev in candidates:
                if dev in self._active_leases:
                    if self.governor.can_schedule(dev):
                        self._active_leases[dev].add(task_id)
                        return DeviceLease(device_id=dev, task_id=task_id)
            return None

    def release_device(self, lease: DeviceLease) -> None:
        if lease.device_id < 0:
            lease.release()
            return
        with self._lock:
            if lease.device_id in self._active_leases:
                self._active_leases[lease.device_id].discard(lease.task_id)
        lease.release()

    def __del__(self) -> None:
        for dev_id, leases in getattr(self, "_active_leases", {}).items():
            if leases:
                LOG.warning("Dispatcher destroyed with %d active leases on device %d", len(leases), dev_id)
