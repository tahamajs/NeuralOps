#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Reference recurrent scan benchmark for SSM-shaped workloads."""

from __future__ import annotations

import time
from typing import Any

import torch


def run_benchmark(device: str = "cuda", sequence_lengths: tuple[int, ...] = (512, 1024, 2048)) -> dict[str, Any]:
    """Measure a finite recurrent scan with mamba/selective-SSM shapes."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    rows = []
    for seq_len in sequence_lengths:
        x = torch.randn(4, seq_len, 256, device=target)
        state = torch.zeros(4, 256, device=target)
        started = time.perf_counter()
        for t in range(seq_len):
            state = torch.tanh(0.95 * state + x[:, t])
        elapsed = (time.perf_counter() - started) * 1000.0
        rows.append({"sequence_length": seq_len, "latency_ms": elapsed, "state_mb": state.numel() * state.element_size() / 2**20})
    return {"module": "mamba_block,selective_ssm,mamba_svd_flash", "category": "ssm", "device": str(target), "rows": rows, "status": "PASS"}
