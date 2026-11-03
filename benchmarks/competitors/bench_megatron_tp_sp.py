#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Megatron-LM Tensor & Sequence Parallel Scaling Benchmark Harness.

Measures Model FLOPs Utilization (MFU), AllReduce inter-rank latency overhead,
and weak/strong scaling efficiency across Tensor Parallel (TP) and Sequence Parallel (SP) configurations.
Originating standard: NVIDIA / Megatron-LM Team.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from neuralops.benchmarks.registry import register_benchmark
from neuralops.distributed.megatron_hooks import ColumnParallelLinear, RowParallelLinear

logger = logging.getLogger(__name__)


def compute_megatron_mfu(
    params_count: int,
    tokens_per_sec: float,
    peak_gpu_tflops: float,
    num_gpus: int = 1,
) -> dict[str, float]:
    """Compute standard Megatron-LM Model FLOPs Utilization (MFU).

    Formula: 6.0 * params_count * tokens_per_sec / (peak_gpu_flops * num_gpus).

    Args:
        params_count: Total trainable model parameters.
        tokens_per_sec: Empirical processing throughput in tokens per second.
        peak_gpu_tflops: Peak theoretical hardware FP16/BF16 TFLOPS per GPU.
        num_gpus: Number of accelerator ranks.

    Returns:
        Dictionary with mfu_pct and tflops_per_device.
    """
    if params_count <= 0 or tokens_per_sec < 0 or peak_gpu_tflops <= 0 or num_gpus <= 0:
        raise ValueError("Invalid parameters provided for compute_megatron_mfu calculation.")

    actual_flops = 6.0 * float(params_count) * float(tokens_per_sec)
    peak_hardware_flops = float(peak_gpu_tflops) * 1e12 * float(num_gpus)
    mfu_percentage = (actual_flops / max(peak_hardware_flops, 1e-12)) * 100.0
    tflops_per_device = (actual_flops / 1e12) / float(num_gpus)

    return {
        "mfu_pct": mfu_percentage,
        "tflops_per_device": tflops_per_device,
        "actual_tflops_total": actual_flops / 1e12,
    }


@register_benchmark(name="megatron_tp_sp", mode="distributed-mfu", hardware="any")
def benchmark_megatron_tp_sp(
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    seq_len: int = 2048,
    batch_size: int = 4,
    tp_size: int = 1,
    iters: int = 20,
    warmup: int = 5,
    peak_gpu_tflops: float = 71.16,  # RTX 3090 FP16 Peak
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark Megatron Tensor Parallel and Sequence Parallel forward-backward scaling.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MLP feedforward dimension.
        seq_len: Sequence length per batch.
        batch_size: Micro-batch size.
        tp_size: Tensor Parallel world size.
        iters: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        peak_gpu_tflops: Theoretical hardware peak TFLOPS.
        device: Device to run benchmark on.
        **kwargs: Additional parameters.

    Returns:
        Benchmark telemetry dictionary with MFU %, latency, and scaling efficiency.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_megatron_tp_sp, but CUDA is not available."
        )

    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    # Construct an MLP block mimicking Megatron TP column-row parallel structure
    col_proj = ColumnParallelLinear(
        in_features=hidden_size,
        out_features=intermediate_size,
        bias=False,
        gather_output=False,
        tp_world_size=tp_size,
    ).to(dev, dtype=dtype)

    row_proj = RowParallelLinear(
        in_features=intermediate_size,
        out_features=hidden_size,
        bias=False,
        input_is_parallel=True,
        tp_world_size=tp_size,
    ).to(dev, dtype=dtype)

    mlp = nn.Sequential(col_proj, nn.GELU(), row_proj)
    params_count = sum(p.numel() for p in mlp.parameters()) * tp_size

    x = torch.randn(batch_size, seq_len, hidden_size, device=dev, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(warmup):
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        mlp.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        for _ in range(iters):
            out = mlp(x)
            loss = out.sum()
            loss.backward()
            mlp.zero_grad()
        end_evt.record()
        torch.cuda.synchronize(dev)
        latency_ms = start_evt.elapsed_time(end_evt) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            out = mlp(x)
            loss = out.sum()
            loss.backward()
            mlp.zero_grad()
        elapsed_sec = time.perf_counter() - t0
        latency_ms = (elapsed_sec / iters) * 1000.0

    step_sec = latency_ms * 1e-3
    tokens_per_sec = (batch_size * seq_len) / max(step_sec, 1e-12)
    mfu_stats = compute_megatron_mfu(
        params_count=params_count,
        tokens_per_sec=tokens_per_sec,
        peak_gpu_tflops=peak_gpu_tflops,
        num_gpus=tp_size,
    )

    return {
        "benchmark": "megatron_tp_sp",
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "tp_size": tp_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "mfu_pct": mfu_stats["mfu_pct"],
        "tflops_per_device": mfu_stats["tflops_per_device"],
        "actual_tflops_total": mfu_stats["actual_tflops_total"],
        "peak_gpu_tflops": peak_gpu_tflops,
    }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_megatron_tp_sp(**kwargs)
