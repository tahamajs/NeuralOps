#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Lightweight CPU-safe benchmarking helpers for NeuralOps examples.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from examples._shared.gpu import safe_cuda_sync
from examples._shared.report import table


def measure_latency(
    fn: Callable[[], Any],
    warmup: int = 2,
    iters: int = 5,
) -> float:
    """Measure average execution latency in milliseconds, safe for CPU, MPS, and CUDA."""
    for _ in range(warmup):
        fn()
    safe_cuda_sync()

    start_time = time.perf_counter()
    for _ in range(iters):
        fn()
    safe_cuda_sync()
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / iters) * 1000.0
    return round(avg_ms, 3)


def measure_vram_if_available() -> float | None:
    """Return peak allocated VRAM in megabytes if CUDA is available, else None."""
    if torch.cuda.is_available():
        allocated_bytes = torch.cuda.max_memory_allocated()
        return round(allocated_bytes / (1024.0 * 1024.0), 2)
    return None


def compute_flops_estimate(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 16),
) -> int:
    """Compute a static parameter-based forward pass multiply-accumulate FLOP estimate."""
    total_flops = 0
    seq_len = input_shape[1] if len(input_shape) >= 2 else 1
    for param in model.parameters():
        if param.requires_grad or param.ndim >= 2:
            # Standard Linear/Attention matrix multiply FLOP estimation: 2 * params * seq_len
            total_flops += 2 * param.numel() * seq_len
    return total_flops


def print_benchmark_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str] = ("Configuration", "Latency (ms)", "Peak VRAM (MB)", "Status"),
) -> None:
    """Print formatted benchmark table."""
    table(rows, headers=headers)
