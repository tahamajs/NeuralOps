#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM Poisson Cloud Serving & PagedAttention Benchmark Harness.

Simulates multi-client Poisson request arrival traffic to evaluate Time-To-First-Token (TTFT),
Time-Per-Output-Token (TPOT / ITL), request throughput, and PagedKVCache block allocation overhead.
Originating standard: UC Berkeley / vLLM Team.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from neuralops.benchmarks.registry import register_benchmark
from neuralops.inference.paged_cache import PagedKVCache

logger = logging.getLogger(__name__)


def compute_serving_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Compute standard serving percentiles (p50, p90, p95, p99) monotonically sorted.

    Args:
        latencies_ms: List of measured latency samples in milliseconds.

    Returns:
        Dictionary mapping percentile keys to float values.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


class PagedKVCacheSimulator:
    """Simulation harness wrapping PagedKVCache allocation and preemption dynamics."""

    def __init__(
        self,
        num_blocks: int = 2048,
        block_size: int = 16,
        num_heads: int = 12,
        head_dim: int = 64,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        self.cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.preemptions_count = 0

    def allocate_request(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate sequence blocks in cache for initial prompt tokens."""
        self.cache.allocate_sequence(seq_id)
        # Ingest prompt tokens
        for _ in range(num_tokens):
            dummy_k = torch.zeros(self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
            dummy_v = torch.zeros(self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
            self.cache.append_token(seq_id, dummy_k, dummy_v)
        return list(self.cache.block_tables.get(seq_id, []))

    def append_decode_token(self, seq_id: int) -> None:
        """Append a single generated decode token to the sequence cache."""
        dummy_k = torch.zeros(self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        dummy_v = torch.zeros(self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        self.cache.append_token(seq_id, dummy_k, dummy_v)

    def free_request(self, seq_id: int) -> None:
        """Free memory blocks associated with sequence."""
        if seq_id in self.cache.block_tables:
            self.cache.free_sequence(seq_id)

    def get_fragmentation_stats(self) -> dict[str, Any]:
        """Compute internal and external memory fragmentation metrics."""
        return self.cache.get_fragmentation_stats()


@register_benchmark(name="vllm_poisson_serving", mode="serving-poisson", hardware="any")
@register_benchmark(name="vllm_poisson", mode="serving-poisson", hardware="any")
def benchmark_vllm_poisson_serving(
    arrival_rate_lambda: float = 16.0,
    num_requests: int = 100,
    max_seq: int = 1024,
    min_seq: int = 64,
    decode_steps: int = 16,
    num_blocks: int = 2048,
    block_size: int = 16,
    num_heads: int = 12,
    head_dim: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """vLLM Poisson continuous serving benchmark.

    Args:
        arrival_rate_lambda: Poisson arrival rate (requests per second).
        num_requests: Total client requests to simulate.
        max_seq: Maximum prompt sequence length.
        min_seq: Minimum prompt sequence length.
        decode_steps: Autoregressive decode token steps per request.
        num_blocks: Number of physical cache blocks.
        block_size: Block size in tokens.
        num_heads: Number of attention heads.
        head_dim: Attention head dimension.
        device: Device to allocate cache and evaluate on.
        seed: Random seed for Poisson inter-arrival intervals.
        **kwargs: Additional parameters.

    Returns:
        Dictionary containing TTFT percentiles (p50, p90, p95, p99), TPOT, throughput, and fragmentation metrics.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_vllm_poisson_serving, but CUDA is not available."
        )

    rng = np.random.default_rng(seed)
    deltas = rng.exponential(1.0 / arrival_rate_lambda, size=num_requests)

    sim = PagedKVCacheSimulator(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=dev,
        dtype=torch.float16 if dev.type == "cuda" else torch.float32,
    )

    ttfts: list[float] = []
    tpots: list[float] = []
    allocated_seq_ids: list[int] = []

    t_start = time.perf_counter()

    for idx, d in enumerate(deltas):
        time.sleep(min(float(d), 0.05))

        prompt_len = int(rng.integers(min_seq, max_seq + 1))
        t0 = time.perf_counter()

        seq_id = idx
        allocated_seq_ids.append(seq_id)
        sim.allocate_request(seq_id=seq_id, num_tokens=prompt_len)

        ttft_ms = (time.perf_counter() - t0) * 1000.0
        ttfts.append(ttft_ms)

        t_dec = time.perf_counter()
        for _ in range(decode_steps):
            sim.append_decode_token(seq_id=seq_id)
        tpot_ms = ((time.perf_counter() - t_dec) / max(decode_steps, 1)) * 1000.0
        tpots.append(tpot_ms)

    total_time_sec = time.perf_counter() - t_start
    total_tokens_generated = num_requests * decode_steps
    frag_stats = sim.get_fragmentation_stats()
    preemptions = sim.preemptions_count

    for s_id in allocated_seq_ids:
        sim.free_request(s_id)

    ttft_pct = compute_serving_percentiles(ttfts)
    tpot_pct = compute_serving_percentiles(tpots)

    req_throughput = num_requests / max(total_time_sec, 1e-6)
    token_throughput = total_tokens_generated / max(total_time_sec, 1e-6)

    return {
        "benchmark": "vllm_poisson_serving",
        "arrival_rate_lambda": arrival_rate_lambda,
        "num_requests": num_requests,
        "total_time_sec": total_time_sec,
        "ttft_p50_ms": ttft_pct["p50"],
        "ttft_p90_ms": ttft_pct["p90"],
        "ttft_p95_ms": ttft_pct["p95"],
        "ttft_p99_ms": ttft_pct["p99"],
        "tpot_p50_ms": tpot_pct["p50"],
        "tpot_p90_ms": tpot_pct["p90"],
        "tpot_p95_ms": tpot_pct["p95"],
        "tpot_p99_ms": tpot_pct["p99"],
        "throughput_req_sec": req_throughput,
        "throughput_tokens_sec": token_throughput,
        "num_blocks_total": num_blocks,
        "block_size": block_size,
        "preemptions_count": preemptions,
        "fragmentation_stats": frag_stats,
    }


def run_vllm_serving_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias matching implementation plan symbol."""
    return benchmark_vllm_poisson_serving(**kwargs)


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_vllm_poisson_serving(**kwargs)
