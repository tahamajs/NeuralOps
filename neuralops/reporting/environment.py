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

import platform
import socket
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
    except Exception:  # allowed-swallow: git detection failure fallback
        pass
    return "unknown"


def capture_environment(seed: Optional[int] = None) -> Dict[str, Any]:
    """Capture snapshot of software, hardware, and runtime environment.

    Schema:
    - neuralops: str
    - torch: str
    - transformers: Optional[str]
    - cuda: Optional[str]
    - gpu: Optional[str]
    - hostname: str
    - git_sha: str
    """
    transformers_version: Optional[str] = None
    try:
        import transformers  # type: ignore

        transformers_version = getattr(transformers, "__version__", None)
    except ImportError:
        transformers_version = None

    git_sha = _get_git_commit()
    hostname = socket.gethostname()

    cuda_avail = torch.cuda.is_available()
    cuda_version: Optional[str] = torch.version.cuda if cuda_avail else None
    gpu_name: Optional[str] = torch.cuda.get_device_name(0) if cuda_avail else None

    env: Dict[str, Any] = {
        # Top-level required keys per frozen Phase 3 Evidence Schema
        "neuralops": getattr(neuralops, "__version__", "1.9.0"),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda": cuda_version,
        "gpu": gpu_name,
        "hostname": hostname,
        "git_sha": git_sha,
        # Detailed diagnostics
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
        "seed": seed,
    }

    # Triton optional version check
    try:
        import triton  # type: ignore

        env["triton"] = getattr(triton, "__version__", None)
    except ImportError:
        env["triton"] = None

    return env


__all__ = ["capture_environment"]
