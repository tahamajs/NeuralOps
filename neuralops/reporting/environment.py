#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Environment and Hardware Telemetry Capture for Reproducibility.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

import neuralops


def _get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def capture_environment(seed: Optional[int] = None) -> Dict[str, Any]:
    """Capture snapshot of software, hardware, and runtime environment."""
    env = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "neuralops": {
            "version": getattr(neuralops, "__version__", "1.9.0"),
            "git_commit": _get_git_commit(),
        },
        "seed": seed,
    }

    # Triton version check
    try:
        import triton
        env["triton"] = {"version": triton.__version__}
    except ImportError:
        env["triton"] = {"version": None}

    return env


__all__ = ["capture_environment"]
