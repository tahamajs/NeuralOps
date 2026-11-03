#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
DeepSpeed ZeRO-3 Memory Partitioning & Communication Overhead Benchmark Harness.

Measures host-to-device parameter streaming throughput (GB/s), PCIe bandwidth saturation,
and parameter gathering / unpartitioning overhead in ZeRO-Stage 3 style sharded execution.
Originating standard: Microsoft / DeepSpeed Team.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from neuralops.benchmarks.registry import register_benchmark

logger = logging.getLogger(__name__)


def benchmark_pcie_offload_bandwidth(
    tensor_size_mb: int = 512,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Measure Host-to-Device (H2D) PCIe memory streaming bandwidth.

    Args:
        tensor_size_mb: Size in megabytes of pinned host tensor to transfer.
        device: Target CUDA device.

    Returns:
        Dictionary with bandwidth_gb_sec and transfer elapsed_ms.
    """
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device required for physical PCIe streaming bandwidth benchmark."
        )

    # Allocate page-locked (pinned) CPU memory for maximum DMA throughput
    numel = (tensor_size_mb * 1024 * 1024) // 4  # 4 bytes per float32
    x_cpu = torch.randn(numel, dtype=torch.float32, pin_memory=True)

    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    x_gpu = x_cpu.to(dev, non_blocking=True)
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    bandwidth_gb_sec = (tensor_size_mb / 1024.0) / max(elapsed, 1e-12)
    # Ensure tensor is kept alive until completion
    _ = x_gpu.numel()

    return {
        "bandwidth_gb_sec": bandwidth_gb_sec,
        "elapsed_ms": elapsed * 1000.0,
        "tensor_size_mb": float(tensor_size_mb),
    }


@register_benchmark(name="deepspeed_zero3_overhead", mode="distributed-mfu", hardware="any")
def benchmark_deepspeed_zero3_overhead(
    num_shards: int = 4,
    shard_size_mb: int = 256,
    iters: int = 20,
    warmup: int = 5,
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark ZeRO-3 style parameter unpartitioning and gathering overhead.

    Args:
        num_shards: Number of partitioned rank shards.
        shard_size_mb: Size in megabytes per rank parameter shard.
        iters: Timed iterations for gathering and layer reconstruction.
        warmup: Warmup iterations.
        device: Target device for execution.
        **kwargs: Additional parameters.

    Returns:
        Dictionary reporting gather latency, effective transfer bandwidth, and overhead percentage.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_deepspeed_zero3_overhead, but CUDA is not available."
        )

    elements_per_shard = (shard_size_mb * 1024 * 1024) // 4
    total_elements = elements_per_shard * num_shards
    total_mb = shard_size_mb * num_shards

    # Simulate partitioned parameter shards
    shards: list[torch.Tensor] = [
        torch.randn(elements_per_shard, dtype=torch.float32, device=dev)
        for _ in range(num_shards)
    ]

    # Warmup
    for _ in range(warmup):
        reconstructed = torch.cat(shards, dim=0)
        _ = reconstructed.sum()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        for _ in range(iters):
            reconstructed = torch.cat(shards, dim=0)
            _ = reconstructed.sum()
        end_evt.record()
        torch.cuda.synchronize(dev)
        gather_latency_ms = start_evt.elapsed_time(end_evt) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            reconstructed = torch.cat(shards, dim=0)
            _ = reconstructed.sum()
        elapsed_sec = time.perf_counter() - t0
        gather_latency_ms = (elapsed_sec / iters) * 1000.0

    gather_sec = gather_latency_ms * 1e-3
    effective_bandwidth_gb_sec = (total_mb / 1024.0) / max(gather_sec, 1e-12)

    # Optional PCIe benchmark if on CUDA
    pcie_stats: dict[str, float] = {}
    if dev.type == "cuda":
        try:
            pcie_stats = benchmark_pcie_offload_bandwidth(
                tensor_size_mb=min(shard_size_mb, 256), device=dev
            )
        except Exception as e:
            logger.warning("Could not run physical PCIe H2D benchmark: %s", e)

    return {
        "benchmark": "deepspeed_zero3_overhead",
        "num_shards": num_shards,
        "shard_size_mb": shard_size_mb,
        "total_param_mb": total_mb,
        "gather_latency_ms": gather_latency_ms,
        "effective_bandwidth_gb_sec": effective_bandwidth_gb_sec,
        "pcie_bandwidth_gb_sec": pcie_stats.get("bandwidth_gb_sec", 0.0),
        "total_elements": total_elements,
    }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_deepspeed_zero3_overhead(**kwargs)
