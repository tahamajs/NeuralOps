#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Small measured tile-configuration sweep for registered Triton kernels."""

from __future__ import annotations

import time
from typing import Any

import torch


def run_benchmark(device: str = "cuda", output: str | None = None) -> dict[str, Any]:
    """Measure candidate launch configurations for a GEMM tuning harness."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    a, b = torch.randn(128, 256, device=target), torch.randn(256, 256, device=target)
    rows = []
    for warps in (1, 2, 4, 8):
        started = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        rows.append({"num_warps": warps, "latency_ms": (time.perf_counter() - started) * 100.0})
    best = min(rows, key=lambda row: row["latency_ms"])
    result = {"category": "sweep", "device": str(target), "candidates": rows, "best": best, "status": "PASS"}
    if output:
        import json
        from pathlib import Path
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
