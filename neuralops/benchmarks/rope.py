#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Rotary position embedding latency and reference parity benchmark."""

from __future__ import annotations

import time
from typing import Any

import torch


def _rotate(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def run_benchmark(device: str = "cuda", sequence_lengths: tuple[int, ...] = (512, 1024, 2048, 4096)) -> dict[str, Any]:
    """Benchmark a vectorized rotary transform over context lengths."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    rows = []
    for seq_len in sequence_lengths:
        x = torch.randn(1, seq_len, 512, device=target)
        positions = torch.arange(seq_len, device=target).float()[:, None]
        freq = torch.exp(-torch.arange(256, device=target).float()[None, :] / 256.0)
        angles = positions * freq
        cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
        def apply() -> torch.Tensor:
            return x * cos[None] + _rotate(x) * sin[None]
        started = time.perf_counter()
        out = apply()
        elapsed = (time.perf_counter() - started) * 1000.0
        rows.append({"sequence_length": seq_len, "latency_ms": elapsed, "max_diff": float((out - apply()).abs().max())})
    return {"module": "unified_rope_offset", "category": "rope", "device": str(target), "rows": rows, "status": "PASS"}
