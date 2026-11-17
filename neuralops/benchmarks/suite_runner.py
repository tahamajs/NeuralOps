#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive benchmarking suite for speed, memory, accuracy, and sequence-length scaling

Implements suite runner functionality for NeuralOps.
"""

from __future__ import annotations

import os
import subprocess
import uuid
import datetime
from typing import Any, Callable

import torch
import torch.nn.functional as F

from neuralops.benchmarks.utils.timer import benchmark_callable
from neuralops.benchmarks.results_v2.ledger import BenchmarkLedger
from neuralops.benchmarks.results_v2.record import CanonicalRecord, BenchmarkStatus

__all__ = [
    "benchmark_attention_speed",
    "benchmark_attention_memory",
    "benchmark_attention_accuracy",
    "benchmark_scaling_sweep",
    "run_and_save_benchmark_suite",
]

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        return commit, len(status) > 0
    except:
        return "unknown", False

def benchmark_attention_speed(
    fn: Callable[[], Any],
    device: str | torch.device = "cpu",
    warmup: int = 10,
    repeat: int = 25,
) -> float:
    """Measures median execution latency in milliseconds."""
    stats = benchmark_callable(fn, device=device, warmup_iters=warmup, timed_iters=repeat)
    return stats.median_ms


def benchmark_attention_memory(
    fn: Callable[[], Any],
    device: str | torch.device = "cpu",
) -> float:
    """Measures peak memory in MB if CUDA is available, else returns 0.0."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return 0.0

    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    fn()
    torch.cuda.synchronize(dev)
    return float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024))


def benchmark_attention_accuracy(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    """Computes max absolute error and relative Frobenius error."""
    diff = (output - reference).abs()
    max_err = float(diff.max().item())
    rel_err = float((torch.norm(diff) / torch.norm(reference).clamp(min=1e-8)).item())
    return {"max_abs_err": max_err, "rel_frob_err": rel_err}


def benchmark_scaling_sweep(
    seq_lens: list[int] | None = None,
    batch_size: int = 2,
    num_heads: int = 4,
    head_dim: int = 32,
) -> list[dict[str, Any]]:
    """Runs sequence length scaling sweep on standard SDPA attention."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_lens = [128, 256, 512, 1024] if seq_lens is None else seq_lens
    results = []
    scale = 1.0 / (head_dim**0.5)

    for N in seq_lens:
        q = torch.randn(batch_size, num_heads, N, head_dim).to(device)
        k = torch.randn(batch_size, num_heads, N, head_dim).to(device)
        v = torch.randn(batch_size, num_heads, N, head_dim).to(device)

        def run_sdpa():
            s = torch.matmul(q, k.transpose(-2, -1)) * scale
            w = F.softmax(s, dim=-1)
            return torch.matmul(w, v)

        ms = benchmark_attention_speed(run_sdpa, device=device)
        tokens_per_sec = (batch_size * N) / (ms / 1000.0) if ms > 0 else 0.0
        results.append(
            {
                "seq_len": N,
                "latency_ms": round(ms, 4),
                "tokens_per_sec": round(tokens_per_sec, 2),
            }
        )

    return results


def run_and_save_benchmark_suite(
    output_dir: str = "results",
) -> None:
    """Runs the suite and records results through BenchmarkLedger."""
    os.makedirs(output_dir, exist_ok=True)
    ledger = BenchmarkLedger(storage_dir=output_dir)
    sweep_data = benchmark_scaling_sweep()
    
    commit, dirty = get_git_info()
    run_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for row in sweep_data:
        record = CanonicalRecord(
            run_id=run_id,
            timestamp=timestamp,
            git_commit=commit,
            git_dirty=dirty,
            suite="attention",
            module="sdpa",
            model="sdpa",
            dataset="synthetic",
            device=device,
            status=BenchmarkStatus.SUCCESS,
            metrics={"latency_ms": row["latency_ms"], "tokens_per_sec": row["tokens_per_sec"]},
            config={"seq_len": row["seq_len"]},
            telemetry={"device": device},
            checksum=str(uuid.uuid4()), # Placeholder
        )
        ledger.write(record)
