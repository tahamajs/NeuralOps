#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Top-k and nucleus sampling throughput benchmark."""

from __future__ import annotations

import time
from typing import Any

import torch


def run_benchmark(device: str = "cuda", batch_size: int = 8, vocab_size: int = 32000, steps: int = 32) -> dict[str, Any]:
    """Measure real top-k and top-p selection on synthetic logits."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device=target)
    started = time.perf_counter()
    for _ in range(steps):
        torch.topk(logits, 50, dim=-1)
    topk_ms = (time.perf_counter() - started) * 1000.0 / steps
    started = time.perf_counter()
    for _ in range(steps):
        values, indices = torch.sort(logits, descending=True, dim=-1)
        probs = values.softmax(-1).cumsum(-1)
        _ = indices.masked_fill(probs > 0.9, -1)
    topp_ms = (time.perf_counter() - started) * 1000.0 / steps
    return {"module": "topk_topp_sampling", "category": "sampling", "device": str(target), "topk_ms": topk_ms, "topp_ms": topp_ms, "topk_tokens_per_second": batch_size * 1000 / topk_ms if topk_ms else 0.0, "status": "PASS"}
