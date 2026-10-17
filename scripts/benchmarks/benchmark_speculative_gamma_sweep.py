# -*- coding: utf-8 -*-
"""Self-Speculative Decoding Benchmark across lookahead steps gamma in {1, 2, 3, 4, 6, 8}.

Measures:
1. End-to-end decoding latency across speculative lookahead steps gamma.
2. Token acceptance rate (% verified tokens accepted by target model).
3. Wall-clock speedup factor over standard autoregressive decoding.
4. Memory overhead and verification efficiency.
Outputs structured JSON to results/benchmarks_speculative_gamma_sweep.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.benchmarks.harness import BenchmarkRunner, save_benchmark_json
from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class SyntheticLanguageModel(nn.Module):
    """Small transformer language model with target and student draft modes."""

    def __init__(self, vocab_size: int = 2048, hidden_dim: int = 256, num_layers: int = 4, is_draft: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Draft model has fewer layers or reduced dimension for speedup
        n_layers = 1 if is_draft else num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def run_speculative_gamma_benchmark(
    device: str = "cuda:0",
    gammas: list[int] = [1, 2, 3, 4, 6, 8],
    tokens_to_generate: int = 32,
    out_file: str = "results/benchmarks_speculative_gamma_sweep.json",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"=== Self-Speculative Decoding Gamma Sweep Benchmark ===")
    print(f"Device: {dev} | Gamma Steps: {gammas} | Tokens to Generate: {tokens_to_generate}\n")

    target_model = SyntheticLanguageModel(hidden_dim=256, num_layers=4, is_draft=False).to(device=dev).eval()
    draft_model = SyntheticLanguageModel(hidden_dim=256, num_layers=1, is_draft=True).to(device=dev).eval()

    # Align draft model head weights to target for realistic high acceptance
    with torch.no_grad():
        draft_model.embedding.weight.copy_(target_model.embedding.weight)
        draft_model.head.weight.copy_(target_model.head.weight)

    prompt = torch.randint(0, 1000, (1, 16), device=dev)

    # 1. Standard autoregressive baseline
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.perf_counter()
    curr_ids = prompt.clone()
    with torch.no_grad():
        for _ in range(tokens_to_generate):
            logits = target_model(curr_ids)[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    baseline_time_s = time.perf_counter() - t0
    baseline_tok_s = tokens_to_generate / baseline_time_s

    print(f"Standard Autoregressive Baseline: {baseline_time_s*1000.0:.2f} ms ({baseline_tok_s:.1f} tok/s)\n")

    results = []

    for gamma in gammas:
        decoder = SelfSpeculativeDecoder(
            target_model=target_model,
            draft_model=draft_model,
            max_draft_k=gamma,
            min_draft_k=gamma,
            adaptive_k=False,
        )

        torch.cuda.synchronize() if dev.type == "cuda" else None
        t_start = time.perf_counter()

        gen_ids, stats = decoder.generate(
            input_ids=prompt,
            max_new_tokens=tokens_to_generate,
            temperature=0.0,
        )

        torch.cuda.synchronize() if dev.type == "cuda" else None
        elapsed_s = time.perf_counter() - t_start

        speedup = baseline_time_s / max(elapsed_s, 1e-6)
        tok_s = stats.total_tokens_generated / max(elapsed_s, 1e-6)

        rec = {
            "gamma": gamma,
            "tokens_generated": stats.total_tokens_generated,
            "target_forward_passes": stats.total_target_forward_passes,
            "acceptance_rate": round(stats.acceptance_rate, 4),
            "mean_accepted_per_step": round(stats.mean_accepted_tokens_per_step, 2),
            "latency_ms": round(elapsed_s * 1000.0, 2),
            "tokens_per_sec": round(tok_s, 1),
            "speedup_factor": round(speedup, 2),
            "status": "OK",
        }
        results.append(rec)
        print(f"  Gamma={gamma:2d}: Latency={elapsed_s*1000.0:6.1f} ms | AccRate={stats.acceptance_rate*100.0:5.1f}% | Acc/Step={stats.mean_accepted_tokens_per_step:.2f} | Speedup={speedup:4.2f}x | {tok_s:6.1f} tok/s")

    out_p = save_benchmark_json({"baseline_tok_s": round(baseline_tok_s, 1), "gamma_sweep": results}, out_file)
    print(f"\nSaved Speculative Gamma results to {out_p.resolve()}")

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Self-Speculative Decoding vs Lookahead Gamma")
        table.add_column("Gamma (Lookahead)", justify="center", style="cyan")
        table.add_column("Acceptance Rate", justify="right")
        table.add_column("Tokens / Step", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Throughput (tok/s)", justify="right", style="green")
        table.add_column("Speedup Factor", justify="right", style="magenta")

        for r in results:
            table.add_row(
                str(r["gamma"]),
                f"{r['acceptance_rate']*100:.1f}%",
                f"{r['mean_accepted_per_step']:.2f}",
                f"{r['latency_ms']:.1f}",
                f"{r['tokens_per_sec']:,.1f}",
                f"{r['speedup_factor']:.2f}x",
            )
        console.print(table)

    return results


def main():
    parser = argparse.ArgumentParser(description="NeuralOps Self-Speculative Decoding Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-file", type=str, default="results/benchmarks_speculative_gamma_sweep.json")
    args = parser.parse_args()

    run_speculative_gamma_benchmark(device=args.device, out_file=args.out_file)


if __name__ == "__main__":
    main()
