# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""Continuous Serving Multi-Concurrency, TTFT, ITL, and PagedKVCache Benchmark.

Measures:
1. Time-to-First-Token (TTFT) across request concurrency N_req in {1, 4, 8, 16, 32}.
2. Inter-Token Latency (ITL) under continuous token generation.
3. PagedKVCache vs Contiguous KV Cache memory fragmentation, overhead, and maximum concurrent capacity limit.
Outputs structured JSON to results/benchmarks_serving_ttft_itl.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.benchmarks.harness import BenchmarkRunner, save_benchmark_json
from neuralops.inference.paged_cache import PagedKVCache

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ContiguousKVCache:
    """Standard preallocated contiguous tensor KV cache (baseline comparison)."""

    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, dtype=torch.float16, device="cuda"):
        self.device = torch.device(device)
        self.k_cache = torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), dtype=dtype, device=self.device)
        self.v_cache = torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), dtype=dtype, device=self.device)
        self.current_lens = [0] * max_batch_size

    def append_token(self, batch_idx: int, k_vec: torch.Tensor, v_vec: torch.Tensor):
        pos = self.current_lens[batch_idx]
        self.k_cache[batch_idx, pos] = k_vec
        self.v_cache[batch_idx, pos] = v_vec
        self.current_lens[batch_idx] += 1


class MockDecoderLayer(nn.Module):
    """Transformer decoder layer for testing TTFT and ITL under serving load."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + FFN
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Scaled dot-product
        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = self.out_proj(torch.matmul(attn, v)) + x
        ffn = self.fc2(torch.relu(self.fc1(out)))
        return out + ffn


def benchmark_serving(
    concurrencies: list[int] | None = None,
    prefill_len: int = 128,
    decode_steps: int = 16,
    hidden_dim: int = 512,
    num_heads: int = 8,
    device: str = "cuda:0",
) -> dict:
    concurrencies = [1, 4, 8, 16, 32] if concurrencies is None else concurrencies
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    runner = BenchmarkRunner(device=dev, warmup_iters=2, benchmark_iters=5)
    model = MockDecoderLayer(hidden_dim=hidden_dim, num_heads=num_heads).to(device=dev, dtype=torch.float16).eval()

    ttft_itl_results = []

    print(f"=== Serving Multi-Concurrency Benchmark (Hidden={hidden_dim}, Heads={num_heads}) ===")
    for N_req in concurrencies:
        runner.reset_vram()
        # Measure TTFT (prefill pass)
        x_prefill = torch.randn(N_req, prefill_len, hidden_dim, device=dev, dtype=torch.float16)

        def prefill_step():
            with torch.no_grad():
                return model(x_prefill)

        prefill_metric = runner.measure_callable(
            prefill_step,
            batch_size=N_req,
            seq_len=prefill_len,
        )

        ttft_ms = prefill_metric.latency_ms_trimmed_mean
        ttft_p99 = prefill_metric.latency_ms_max

        # Measure ITL (inter-token decoding latency per step)
        x_decode = torch.randn(N_req, 1, hidden_dim, device=dev, dtype=torch.float16)

        def decode_step():
            with torch.no_grad():
                return model(x_decode)

        decode_metric = runner.measure_callable(
            decode_step,
            batch_size=N_req,
            seq_len=1,
            warmup=3,
            iters=10,
        )

        itl_ms = decode_metric.latency_ms_trimmed_mean
        gen_tokens_sec = (N_req * 1000.0) / itl_ms if itl_ms > 0 else 0.0

        rec = {
            "concurrency": N_req,
            "ttft_mean_ms": round(ttft_ms, 3),
            "ttft_p99_ms": round(ttft_p99, 3),
            "itl_mean_ms": round(itl_ms, 3),
            "generation_tokens_per_sec": round(gen_tokens_sec, 2),
            "peak_vram_mb": decode_metric.peak_vram_mb,
            "status": "OK",
        }
        ttft_itl_results.append(rec)
        print(f"  N_req={N_req:2d} | TTFT: {ttft_ms:6.2f} ms | ITL: {itl_ms:5.2f} ms | Throughput: {gen_tokens_sec:8.1f} tok/s | VRAM: {decode_metric.peak_vram_mb:5.1f} MB")

    # KV Cache Fragmentation & Capacity Benchmark
    print("\n=== PagedKVCache vs Contiguous Cache Memory Benchmark ===")
    cache_results = []
    block_sizes = [16, 32]
    total_tokens_per_seq = 2048
    head_dim = hidden_dim // num_heads

    for bs in block_sizes:
        num_blocks = 2048
        runner.reset_vram()
        # Initialize PagedKVCache
        paged_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=bs,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float16,
            device=dev,
        )
        paged_vram = runner.get_allocated_vram_mb()

        # Allocate 32 sequences of varied dynamic lengths
        t0 = time.perf_counter()
        active_seqs = 32
        for s in range(active_seqs):
            paged_cache.allocate_sequence(s)
            tokens_to_add = 128 + (s * 32)
            k_tok = torch.randn(num_heads, head_dim, dtype=torch.float16, device=dev)
            v_tok = torch.randn(num_heads, head_dim, dtype=torch.float16, device=dev)
            for _ in range(tokens_to_add):
                paged_cache.append_token(s, k_tok, v_tok)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        paged_alloc_time = (time.perf_counter() - t0) * 1000.0

        # Contiguous baseline
        runner.reset_vram()
        _contig_cache = ContiguousKVCache(
            max_batch_size=active_seqs,
            max_seq_len=total_tokens_per_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float16,
            device=dev,
        )
        contig_vram = runner.get_allocated_vram_mb()

        cache_rec = {
            "block_size": bs,
            "paged_static_vram_mb": round(paged_vram, 2),
            "contiguous_vram_mb": round(contig_vram, 2),
            "memory_savings_ratio": round(contig_vram / max(paged_vram, 0.01), 2),
            "dynamic_insert_ms": round(paged_alloc_time, 2),
            "max_supported_concurrent_seqs": (num_blocks * bs) // 512,
        }
        cache_results.append(cache_rec)
        print(f"  BlockSize={bs:2d} | Paged VRAM: {paged_vram:6.1f} MB | Contig VRAM: {contig_vram:6.1f} MB | Ratio: {cache_rec['memory_savings_ratio']:.2f}x | Dynamic append: {paged_alloc_time:6.2f} ms")

    return {
        "serving_ttft_itl": ttft_itl_results,
        "kv_cache_comparison": cache_results,
    }


def main():
    parser = argparse.ArgumentParser(description="NeuralOps Serving Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", "--out-file", dest="out_file", type=str, default="results/benchmarks_serving_ttft_itl.json", help="Path to save JSON output")
    args = parser.parse_args()

    results = benchmark_serving(device=args.device)
    out_path = save_benchmark_json(results, args.out_file)
    print(f"\nSaved serving benchmark results to {out_path.resolve()}")

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Serving Multi-Concurrency TTFT & ITL Performance")
        table.add_column("Concurrency (N_req)", justify="center", style="cyan")
        table.add_column("TTFT Mean (ms)", justify="right")
        table.add_column("TTFT P99 (ms)", justify="right")
        table.add_column("ITL Mean (ms)", justify="right")
        table.add_column("Gen Throughput (tok/s)", justify="right", style="green")
        table.add_column("VRAM (MB)", justify="right", style="magenta")

        for r in results["serving_ttft_itl"]:
            table.add_row(
                str(r["concurrency"]),
                f"{r['ttft_mean_ms']:.2f}",
                f"{r['ttft_p99_ms']:.2f}",
                f"{r['itl_mean_ms']:.2f}",
                f"{r['generation_tokens_per_sec']:,.1f}",
                f"{r['peak_vram_mb']:.1f}",
            )
        console.print(table)


if __name__ == "__main__":
    main()
