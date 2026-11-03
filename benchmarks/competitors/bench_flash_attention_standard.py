#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FlashAttention Standard Peak TFLOPS Benchmark Harness.

Measures forward and backward throughput (TFLOPS) as a percentage of physical
theoretical peak hardware FLOPS on GPUs without allocating the intermediate N x N attention matrix.
Originating standard: Stanford / Tri Dao.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from neuralops.benchmarks.registry import register_benchmark
from neuralops.kernels.cuda import flash_decoding_attention

logger = logging.getLogger(__name__)


def compute_peak_tflops_ratio(
    flops: float,
    latency_ms: float,
    peak_tflops: float,
) -> tuple[float, float]:
    """Calculate attained TFLOPS and percentage of physical theoretical peak hardware FLOPS.

    Args:
        flops: Total floating point operations performed.
        latency_ms: Execution duration in milliseconds.
        peak_tflops: Hardware theoretical peak TFLOPS.

    Returns:
        Tuple of (attained_tflops, pct_peak).
    """
    if latency_ms <= 0.0 or peak_tflops <= 0.0:
        raise ValueError("latency_ms and peak_tflops must be strictly positive.")
    attained_tflops = (flops / (latency_ms * 1e-3)) / 1e12
    pct_peak = (attained_tflops / peak_tflops) * 100.0
    return attained_tflops, pct_peak


@register_benchmark(name="flash_attention_standard", mode="flash-compute", hardware="cuda")
def benchmark_flash_attn_tflops(
    batch: int = 8,
    heads: int = 16,
    seq_len: int = 4096,
    dim: int = 64,
    warmup: int = 25,
    iters: int = 100,
    device: str | torch.device = "cuda",
    peak_tflops: float = 71.16,  # RTX 3090 FP16 peak TFLOPS default
    **kwargs: Any,
) -> dict[str, Any]:
    """FlashAttention standard peak TFLOPS and memory efficiency benchmark.

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length.
        dim: Head dimension.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.
        device: Target execution device.
        peak_tflops: Physical theoretical peak TFLOPS of target device.
        **kwargs: Extra parameters for configuration compatibility.

    Returns:
        Dictionary with latency_ms, tflops, pct_peak, and hardware specs.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_flash_attn_tflops, but CUDA is not available."
        )

    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    q = torch.randn(batch, heads, seq_len, dim, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, dim, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, dim, device=dev, dtype=dtype, requires_grad=True)

    # 4 * B * H * S^2 * D FLOPs for forward attention
    fwd_flops = 4.0 * batch * heads * (seq_len ** 2) * dim
    # Backward pass performs ~2.5x fwd FLOPs (or 8 * B * H * S^2 * D FLOPs)
    bwd_flops = 2.0 * fwd_flops
    total_flops = fwd_flops + bwd_flops

    # Warmup
    for _ in range(warmup):
        out = flash_decoding_attention(q, k, v)
        if out.requires_grad:
            out.sum().backward()
            if q.grad is not None:
                q.grad.zero_()
            if k.grad is not None:
                k.grad.zero_()
            if v.grad is not None:
                v.grad.zero_()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        # Forward timing
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_fwd.record()
        for _ in range(iters):
            out = flash_decoding_attention(q, k, v)
        end_fwd.record()

        # Backward timing
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)
        start_bwd.record()
        for _ in range(iters):
            out = flash_decoding_attention(q, k, v)
            if out.requires_grad:
                out.sum().backward()
                if q.grad is not None:
                    q.grad.zero_()
                if k.grad is not None:
                    k.grad.zero_()
                if v.grad is not None:
                    v.grad.zero_()
        end_bwd.record()
        torch.cuda.synchronize(dev)

        fwd_latency_ms = start_fwd.elapsed_time(end_fwd) / iters
        bwd_total_ms = start_bwd.elapsed_time(end_bwd) / iters
        bwd_latency_ms = max(bwd_total_ms - fwd_latency_ms, 0.0)
        latency_ms = fwd_latency_ms
    else:
        # CPU Timing
        t0 = time.perf_counter()
        for _ in range(iters):
            out = flash_decoding_attention(q, k, v)
        elapsed_sec = time.perf_counter() - t0
        fwd_latency_ms = (elapsed_sec / iters) * 1000.0

        t1 = time.perf_counter()
        for _ in range(iters):
            out = flash_decoding_attention(q, k, v)
            if out.requires_grad:
                out.sum().backward()
                if q.grad is not None:
                    q.grad.zero_()
                if k.grad is not None:
                    k.grad.zero_()
                if v.grad is not None:
                    v.grad.zero_()
        elapsed_bwd_sec = time.perf_counter() - t1
        bwd_total_ms = (elapsed_bwd_sec / iters) * 1000.0
        bwd_latency_ms = max(bwd_total_ms - fwd_latency_ms, 0.0)
        latency_ms = fwd_latency_ms

    tflops, pct_peak = compute_peak_tflops_ratio(fwd_flops, latency_ms, peak_tflops)

    return {
        "benchmark": "flash_attention_standard",
        "latency_ms": latency_ms,
        "fwd_latency_ms": fwd_latency_ms,
        "bwd_latency_ms": bwd_latency_ms,
        "tflops": tflops,
        "pct_peak": pct_peak,
        "batch_size": batch,
        "num_heads": heads,
        "seq_len": seq_len,
        "head_dim": dim,
        "fwd_flops": fwd_flops,
        "bwd_flops": bwd_flops,
        "total_flops": total_flops,
        "peak_theoretical_tflops": peak_tflops,
    }


def run_flash_attention_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias matching implementation plan symbol."""
    return benchmark_flash_attn_tflops(**kwargs)


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_flash_attn_tflops(**kwargs)


# Canonical Aliases for Multi-Agent Implementation Plan contract compliance
CoreModule = benchmark_flash_attn_tflops

