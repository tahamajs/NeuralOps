#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Ragged versus padded attention throughput benchmark."""

from __future__ import annotations

import time
from typing import Any

import torch


def run_benchmark(device: str = "cuda", lengths: tuple[int, ...] = (128, 256, 512, 384)) -> dict[str, Any]:
    """Compare valid-token work in a ragged list with padded batching."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    hidden = 128
    ragged = [torch.randn(1, n, hidden, device=target) for n in lengths]
    padded = torch.nn.utils.rnn.pad_sequence([x.squeeze(0) for x in ragged], batch_first=True)
    started = time.perf_counter()
    ragged_out = [x @ x.transpose(-1, -2) for x in ragged]
    ragged_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    padded_out = padded @ padded.transpose(-1, -2)
    padded_ms = (time.perf_counter() - started) * 1000.0
    return {"module": "varlen_attention", "category": "varlen", "device": str(target), "ragged_ms": ragged_ms, "padded_ms": padded_ms, "padding_overhead_pct": (padded.numel() / sum(x.numel() for x in ragged) - 1.0) * 100.0, "status": "PASS" if all(torch.isfinite(x).all() for x in ragged_out) and torch.isfinite(padded_out).all() else "FAIL"}
