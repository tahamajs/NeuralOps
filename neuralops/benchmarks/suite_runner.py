"""Comprehensive benchmarking suite for speed, memory, accuracy, and sequence-length scaling."""
from __future__ import annotations
from typing import Callable, Any
import csv
import os
import time
import torch
import torch.nn.functional as F

__all__ = [
    "benchmark_attention_speed",
    "benchmark_attention_memory",
    "benchmark_attention_accuracy",
    "benchmark_scaling_sweep",
    "run_and_save_benchmark_suite",
]


def benchmark_attention_speed(
    fn: Callable[[], Any],
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """Measures mean execution latency in milliseconds."""
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
    seq_lens: list[int] = [128, 256, 512, 1024],
    batch_size: int = 2,
    num_heads: int = 4,
    head_dim: int = 32,
) -> list[dict[str, Any]]:
    """Runs sequence length scaling sweep on standard SDPA attention."""
    results = []
    scale = 1.0 / (head_dim ** 0.5)

    for N in seq_lens:
        q = torch.randn(batch_size, num_heads, N, head_dim)
        k = torch.randn(batch_size, num_heads, N, head_dim)
        v = torch.randn(batch_size, num_heads, N, head_dim)

        def run_sdpa():
            s = torch.matmul(q, k.transpose(-2, -1)) * scale
            w = F.softmax(s, dim=-1)
            return torch.matmul(w, v)

        ms = benchmark_attention_speed(run_sdpa, warmup=2, repeat=5)
        tokens_per_sec = (batch_size * N) / (ms / 1000.0) if ms > 0 else 0.0
        results.append({
            "seq_len": N,
            "latency_ms": round(ms, 4),
            "tokens_per_sec": round(tokens_per_sec, 2),
        })

    return results


def run_and_save_benchmark_suite(
    output_dir: str = "benchmarks",
) -> tuple[str, str]:
    """Runs the suite and outputs CSV and Markdown summary tables."""
    os.makedirs(output_dir, exist_ok=True)
    sweep_data = benchmark_scaling_sweep()

    csv_path = os.path.join(output_dir, "scaling_results.csv")
    md_path = os.path.join(output_dir, "scaling_results.md")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_len", "latency_ms", "tokens_per_sec"])
        writer.writeheader()
        writer.writerows(sweep_data)

    with open(md_path, "w") as f:
        f.write("# NeuralOps Attention Scaling Benchmark Results\n\n")
        f.write("| Sequence Length | Latency (ms) | Throughput (Tokens/sec) |\n")
        f.write("|:---:|:---:|:---:|\n")
        for row in sweep_data:
            f.write(f"| {row['seq_len']} | {row['latency_ms']:.4f} | {row['tokens_per_sec']:.2f} |\n")

    return csv_path, md_path
