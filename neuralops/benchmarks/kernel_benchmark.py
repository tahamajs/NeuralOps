"""Triton & PyTorch attention kernel benchmark harness."""
from __future__ import annotations
from typing import Callable, Any
import time
import torch

__all__ = ["benchmark_kernel", "run_attention_kernel_benchmarks"]


def benchmark_kernel(
    fn: Callable[[], Any],
    warmup: int = 2,
    repeat: int = 5,
) -> float:
    """Measures average execution time of a callable in milliseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / repeat * 1000.0


def run_attention_kernel_benchmarks(
    batch_size: int = 2,
    seq_len: int = 64,
    head_dim: int = 32,
    num_heads: int = 4,
) -> dict[str, float]:
    """Runs standard benchmark suite over baseline and registered kernel functions."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    scale = 1.0 / (head_dim ** 0.5)

    def naive_sdpa():
        s = torch.matmul(q, k.transpose(-2, -1)) * scale
        w = torch.softmax(s, dim=-1)
        return torch.matmul(w, v)

    sdpa_ms = benchmark_kernel(naive_sdpa)
    return {
        "sdpa_ms": sdpa_ms,
        "tokens_per_sec": (batch_size * seq_len) / (sdpa_ms / 1000.0) if sdpa_ms > 0 else 0.0,
    }
