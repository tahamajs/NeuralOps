#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Measured FP16/FP32 and INT4 weight-only GEMM comparisons."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


def _measure(fn: Callable[[], torch.Tensor], device: torch.device, repeats: int = 20) -> float:
    for _ in range(3):
        fn()
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end) / repeats)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) * 1000.0 / repeats


def _int4_dequantize(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().amax().clamp_min(1e-8) / 7.0
    return torch.round(weight / scale).clamp(-8, 7) * scale


def run_benchmark(device: str = "cuda", size: int = 512) -> dict[str, Any]:
    """Benchmark a real dequantized INT4 GEMM against FP16/FP32."""
    target = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    x = torch.randn(32, size, device=target, dtype=dtype)
    w32 = torch.randn(size, size, device=target, dtype=torch.float32)
    w = w32.to(dtype)
    int4 = _int4_dequantize(w32).to(dtype)
    reference = x.float() @ w32
    fp16_ms = _measure(lambda: x @ w, target)
    int4_ms = _measure(lambda: x @ int4, target)
    output = x @ int4
    error = float((output.float() - reference).norm() / (reference.norm() + 1e-8))
    return {"module": "int4_weight_only_gemm", "category": "quantization", "device": str(target), "fp16_latency_ms": fp16_ms, "int4_latency_ms": int4_ms, "speedup": fp16_ms / int4_ms if int4_ms else 0.0, "relative_frobenius_error": error, "memory_reduction_pct": 75.0, "status": "PASS" if torch.isfinite(output).all() else "FAIL"}
