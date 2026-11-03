#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang RadixAttention & Prefix Cache Reuse Benchmark Harness.

Evaluates prefix cache hit rate, Radix Trie matching latency, and effective token reuse speedup
when multiple inference requests share common system prompts, few-shot demonstrations, or multi-turn history.
Originating standard: LMSYS / SGLang Team.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from neuralops.benchmarks.registry import register_benchmark
from neuralops.integrations.radix_prefix_cache import SGLangRadixAdapter

logger = logging.getLogger(__name__)


@register_benchmark(name="sglang_radix_cache", mode="serving-poisson", hardware="any")
def benchmark_radix_prefix_cache(
    common_prefix_len: int = 1024,
    query_suffix_len: int = 128,
    num_queries: int = 100,
    prefix_diversity: int = 4,
    device: str | torch.device = "cpu",
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """SGLang RadixAttention prefix caching and lookup benchmark.

    Args:
        common_prefix_len: Number of tokens in common shared prefix (e.g. system prompt).
        query_suffix_len: Unique suffix token count per query.
        num_queries: Total number of requests in the benchmark stream.
        prefix_diversity: Number of distinct prefix templates shared across queries.
        device: Device for any simulated cache state allocations.
        seed: Random seed for query prompt generation.
        **kwargs: Additional parameters.

    Returns:
        Dictionary with prefix hit rate, mean lookup latency, p95 lookup latency, and reuse statistics.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_radix_prefix_cache, but CUDA is not available."
        )

    rng = np.random.default_rng(seed)

    adapter = SGLangRadixAdapter(
        model=None,
        tokenizer=None,
        max_cached_tokens=max(65536, common_prefix_len * prefix_diversity * 2),
        max_cached_prefixes=1024,
        device=dev,
    )

    # Pre-populate Radix Tree with several shared system prompts / context templates
    distinct_prefixes: list[list[int]] = []
    for p_idx in range(prefix_diversity):
        base_val = (p_idx + 1) * 1000
        prefix_tokens = [int(base_val + i) for i in range(common_prefix_len)]
        distinct_prefixes.append(prefix_tokens)
        dummy_state = torch.zeros(1, 1, 16, device=dev)
        adapter.insert_prefix(prefix_tokens, kv_cache=dummy_state)

    lookup_times_us: list[float] = []
    matched_lengths: list[int] = []
    hits = 0

    t_start = time.perf_counter()

    for q_idx in range(num_queries):
        # Pick one of the shared prefixes with high probability (80%), or an unseen prefix (20%)
        if rng.random() < 0.8:
            prefix_idx = int(rng.integers(0, prefix_diversity))
            prefix = distinct_prefixes[prefix_idx]
        else:
            prefix = [int(90000 + i) for i in range(common_prefix_len // 2)]

        # Unique suffix for this query
        suffix = [int(500000 + q_idx * 1000 + i) for i in range(query_suffix_len)]
        full_query = prefix + suffix

        t0 = time.perf_counter()
        matched_len, cached_kv = adapter.match_prefix(full_query)
        t_elapsed = (time.perf_counter() - t0) * 1e6  # microseconds

        lookup_times_us.append(t_elapsed)
        matched_lengths.append(matched_len)

        if matched_len >= common_prefix_len:
            hits += 1

        # Periodically insert newly completed queries to test dynamic trie growth
        if q_idx % 5 == 0:
            adapter.insert_prefix(full_query, kv_cache=torch.zeros(1, 1, 16, device=dev))

    total_time_sec = time.perf_counter() - t_start
    hit_rate = (hits / max(num_queries, 1)) * 100.0
    mean_lookup_us = float(np.mean(lookup_times_us))
    p95_lookup_us = float(np.percentile(lookup_times_us, 95))
    mean_matched_tokens = float(np.mean(matched_lengths))

    return {
        "benchmark": "sglang_radix_cache",
        "common_prefix_len": common_prefix_len,
        "query_suffix_len": query_suffix_len,
        "num_queries": num_queries,
        "prefix_diversity": prefix_diversity,
        "prefix_hit_rate_pct": hit_rate,
        "mean_lookup_us": mean_lookup_us,
        "p95_lookup_us": p95_lookup_us,
        "mean_matched_tokens": mean_matched_tokens,
        "total_time_sec": total_time_sec,
    }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_radix_prefix_cache(**kwargs)
