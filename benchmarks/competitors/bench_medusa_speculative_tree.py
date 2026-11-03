#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Medusa & Self-Speculative Multi-Head Tree Decoding Benchmark Harness.

Benchmarks multi-head speculative draft-and-verify tree decoding, evaluating candidate tree
acceptance rates, average accepted tokens per step (TPS), and theoretical wall-clock speedup.
Originating standard: Together AI / Princeton University / Medusa Team.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from neuralops.benchmarks.registry import register_benchmark

logger = logging.getLogger(__name__)


def evaluate_medusa_tree_acceptance(
    draft_tree_candidates: list[int],
    target_verification_logits: torch.Tensor,
) -> dict[str, Any]:
    """Verify speculative tree candidates sequentially against target verification logits.

    Args:
        draft_tree_candidates: List of draft token predictions (length K).
        target_verification_logits: Target model logits tensor of shape (K, vocab_size).

    Returns:
        Dictionary with accepted_tokens count and efficiency_ratio.
    """
    accepted_tokens = 0
    num_candidates = len(draft_tree_candidates)

    for depth, candidate in enumerate(draft_tree_candidates):
        target_token = int(target_verification_logits[depth].argmax().item())
        if candidate == target_token:
            accepted_tokens += 1
        else:
            break

    # Final accepted tokens includes the bonus token from the target verification distribution
    total_emitted = accepted_tokens + 1
    efficiency_ratio = total_emitted / max(num_candidates + 1, 1)

    return {
        "accepted_tokens": total_emitted,
        "efficiency_ratio": efficiency_ratio,
        "perfect_match": accepted_tokens == num_candidates,
    }


@register_benchmark(name="medusa_speculative_tree", mode="serving-poisson", hardware="any")
@register_benchmark(name="medusa_speculative", mode="serving-poisson", hardware="any")
def benchmark_medusa_speculative_tree(
    num_heads: int = 4,
    tree_depth: int = 5,
    num_steps: int = 50,
    vocab_size: int = 32000,
    acceptance_bias: float = 0.75,
    device: str | torch.device = "cpu",
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark Medusa-style multi-head speculative tree decoding.

    Args:
        num_heads: Number of speculative prediction heads.
        tree_depth: Speculative candidate draft depth.
        num_steps: Verification steps to simulate.
        vocab_size: Vocabulary dimension.
        acceptance_bias: Probability of head agreement with target model.
        device: Device for verification tensors.
        seed: Random seed.
        **kwargs: Additional parameters.

    Returns:
        Dictionary containing average accepted tokens per step (TPS), speedup multiplier, and acceptance rates.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for benchmark_medusa_speculative_tree, but CUDA is not available."
        )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    accepted_per_step: list[int] = []
    step_latencies_ms: list[float] = []

    t_start = time.perf_counter()

    for _ in range(num_steps):
        t0 = time.perf_counter()

        # Generate draft candidates
        draft_candidates = [int(rng.integers(10, vocab_size)) for _ in range(tree_depth)]

        # Simulate target model logits with controlled agreement bias
        target_logits = torch.randn(tree_depth, vocab_size, device=dev)
        for d in range(tree_depth):
            if rng.random() < acceptance_bias:
                target_logits[d, draft_candidates[d]] = 100.0  # Force target agreement

        stats = evaluate_medusa_tree_acceptance(draft_candidates, target_logits)
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        accepted_per_step.append(stats["accepted_tokens"])
        step_latencies_ms.append(t_elapsed)

    total_time_sec = time.perf_counter() - t_start
    total_tokens_emitted = sum(accepted_per_step)
    mean_tps = float(np.mean(accepted_per_step))
    p95_tps = float(np.percentile(accepted_per_step, 95))

    # Baseline standard autoregressive emits 1 token per verification step
    baseline_tokens = num_steps
    speculative_speedup = total_tokens_emitted / max(baseline_tokens, 1)

    return {
        "benchmark": "medusa_speculative_tree",
        "num_heads": num_heads,
        "tree_depth": tree_depth,
        "num_steps": num_steps,
        "total_tokens_emitted": total_tokens_emitted,
        "mean_accepted_tokens_per_step": mean_tps,
        "p95_accepted_tokens_per_step": p95_tps,
        "speculative_speedup_factor": speculative_speedup,
        "total_time_sec": total_time_sec,
        "mean_step_latency_ms": float(np.mean(step_latencies_ms)),
    }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return benchmark_medusa_speculative_tree(**kwargs)


# Canonical Aliases for Multi-Agent Implementation Plan contract compliance
run_medusa_speculative_benchmark = benchmark_medusa_speculative_tree
CoreModule = benchmark_medusa_speculative_tree

