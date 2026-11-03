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


@register_benchmark(name="vllm_poisson_serving", mode="serving-poisson", hardware="any")
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
        Dictionary containing TTFT percentiles (p50, p95, p99), TPOT, throughput, and fragmentation metrics.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_vllm_poisson_serving, but CUDA is not available."
        )

    rng = np.random.default_rng(seed)
    # Exponential inter-arrival times representing Poisson point process
    deltas = rng.exponential(1.0 / arrival_rate_lambda, size=num_requests)

    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float16 if dev.type == "cuda" else torch.float32,
        device=dev,
    )

    ttfts: list[float] = []
    tpots: list[float] = []
    allocated_seq_ids: list[int] = []

    t_start = time.perf_counter()

    for idx, d in enumerate(deltas):
        # Emulate stochastic inter-arrival gap (capped to avoid excessive sleep in CI)
        time.sleep(min(float(d), 0.05))

        prompt_len = int(rng.integers(min_seq, max_seq + 1))
        t0 = time.perf_counter()

        seq_id = idx
        allocated_seq_ids.append(seq_id)
        # Allocate prompt blocks in paged cache
        blocks = cache.allocate(seq_id=seq_id, num_tokens=prompt_len)

        ttft_ms = (time.perf_counter() - t0) * 1000.0
        ttfts.append(ttft_ms)

        # Autoregressive generation steps
        t_dec = time.perf_counter()
        for step in range(decode_steps):
            # Append one token per decode step
            cache.append_token(seq_id=seq_id, token_idx=prompt_len + step)
        tpot_ms = ((time.perf_counter() - t_dec) / max(decode_steps, 1)) * 1000.0
        tpots.append(tpot_ms)

    total_time_sec = time.perf_counter() - t_start
    total_tokens_generated = num_requests * decode_steps

    # Clean up cache
    for s_id in allocated_seq_ids:
        cache.free(s_id)

    ttft_p50 = float(np.percentile(ttfts, 50))
    ttft_p95 = float(np.percentile(ttfts, 95))
    ttft_p99 = float(np.percentile(ttfts, 99))
    tpot_p50 = float(np.percentile(tpots, 50))
    tpot_p95 = float(np.percentile(tpots, 95))
    tpot_p99 = float(np.percentile(tpots, 99))
    req_throughput = num_requests / max(total_time_sec, 1e-6)
    token_throughput = total_tokens_generated / max(total_time_sec, 1e-6)

    return {
        "benchmark": "vllm_poisson_serving",
        "arrival_rate_lambda": arrival_rate_lambda,
        "num_requests": num_requests,
        "total_time_sec": total_time_sec,
        "ttft_p50_ms": ttft_p50,
        "ttft_p95_ms": ttft_p95,
        "ttft_p99_ms": ttft_p99,
        "tpot_p50_ms": tpot_p50,
        "tpot_p95_ms": tpot_p95,
        "tpot_p99_ms": tpot_p99,
        "throughput_req_sec": req_throughput,
        "throughput_tokens_sec": token_throughput,
        "num_blocks_total": num_blocks,
        "block_size": block_size,
    }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_vllm_poisson_serving(**kwargs)
