#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GPU Memory Leak & OOM Forensics Detector

Monitors VRAM trajectory across benchmark iterations, detects steady leaks,
and captures forensic diagnostics upon Out-Of-Memory events.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GPUMemoryForensics:
    """
    Monitors memory allocations across benchmark steps to identify
    memory leaks and diagnose Out-Of-Memory (OOM) errors.
    """

    def __init__(
        self,
        leak_threshold_mb: float = 5.0,
        leak_consecutive_steps: int = 3,
        snapshot_dir: Path | str | None = None,
    ) -> None:
        self.leak_threshold_mb = leak_threshold_mb
        self.leak_consecutive_steps = leak_consecutive_steps
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None

        self._history: list[dict[str, float]] = []
        self._leak_streak = 0
        self._leak_detected = False

    def step(self, step_idx: int) -> dict[str, Any]:
        """
        Record memory after an iteration step and check for leaks.
        """
        allocated_mb = 0.0
        reserved_mb = 0.0

        try:
            import torch

            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                reserved_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        except Exception:
            pass

        entry = {
            "step": float(step_idx),
            "allocated_mb": round(allocated_mb, 2),
            "reserved_mb": round(reserved_mb, 2),
        }

        # Check for growth compared to previous step
        warning = None
        if self._history:
            prev_allocated = self._history[-1]["allocated_mb"]
            growth = allocated_mb - prev_allocated
            if growth > self.leak_threshold_mb:
                self._leak_streak += 1
                if self._leak_streak >= self.leak_consecutive_steps:
                    self._leak_detected = True
                    warning = (
                        f"MEMORY_LEAK_WARNING: VRAM grew by {growth:.2f} MB across "
                        f"{self._leak_streak} consecutive steps (current: {allocated_mb:.2f} MB)."
                    )
                    logger.warning(warning)
            else:
                self._leak_streak = 0

        self._history.append(entry)
        return {
            "step": step_idx,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "leak_streak": self._leak_streak,
            "leak_detected": self._leak_detected,
            "warning": warning,
        }

    def handle_oom(self, exc: Exception, step_idx: int | None = None) -> dict[str, Any]:
        """
        Capture forensic metadata and dump memory snapshot upon Out-Of-Memory.
        """
        allocated_mb = 0.0
        reserved_mb = 0.0
        total_mb = 0.0
        snapshot_path = None

        try:
            import torch

            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                reserved_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                total_mb = props.total_memory / (1024.0 * 1024.0)

                # Attempt PyTorch memory snapshot dump if supported
                if self.snapshot_dir and hasattr(torch.cuda.memory, "_dump_snapshot"):
                    try:
                        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                        snap_file = self.snapshot_dir / f"oom_snapshot_step_{step_idx or 0}.pickle"
                        torch.cuda.memory._dump_snapshot(str(snap_file))
                        snapshot_path = str(snap_file)
                    except Exception as dump_err:
                        logger.warning("Could not dump PyTorch memory snapshot: %s", dump_err)
        except Exception:
            pass

        return {
            "status": "FAILED_OOM",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "step": step_idx,
            "allocated_mb": round(allocated_mb, 2),
            "reserved_mb": round(reserved_mb, 2),
            "total_mb": round(total_mb, 2),
            "snapshot_file": snapshot_path,
        }
