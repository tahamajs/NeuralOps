#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Measured top-k routing and grouped expert execution for MoE workloads."""

from __future__ import annotations

import time
from typing import Any

import torch


def run_benchmark(device: str = "cuda", tokens: int = 256, experts: int = 8, top_k: int = 2) -> dict[str, Any]:
    """Benchmark router selection, load variance, and grouped expert GEMM."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    hidden = 256
    x = torch.randn(tokens, hidden, device=target)
    router = torch.randn(hidden, experts, device=target)
    expert_weights = torch.randn(experts, hidden, hidden, device=target) / hidden**0.5
    weights, indices = (x @ router).softmax(-1).topk(top_k, dim=-1)
    counts = torch.bincount(indices.flatten(), minlength=experts).float()
    output = torch.zeros_like(x)

    def grouped() -> torch.Tensor:
        result = torch.zeros_like(x)
        for expert in range(experts):
            mask = indices == expert
            rows = mask.any(dim=-1)
            coeff = torch.where(mask, weights, torch.zeros_like(weights)).sum(-1)
            result[rows] += (x[rows] @ expert_weights[expert]) * coeff[rows, None]
        return result

    for _ in range(3):
        output = grouped()
    started = time.perf_counter()
    output = grouped()
    elapsed = (time.perf_counter() - started) * 1000.0
    return {"module": "moe_routing", "category": "moe", "device": str(target), "latency_ms": elapsed, "tokens_per_second": tokens / (elapsed / 1000.0) if elapsed else 0.0, "load_variance": float(counts.var(unbiased=False)), "experts": experts, "top_k": top_k, "status": "PASS" if torch.isfinite(output).all() else "FAIL"}
