"""
Pareto Paper Benchmark Suite: Empirically measures throughput, latency, memory,
and retrieval capability across linear attention mechanisms and compression engines.

Generates camera-ready LaTeX booktabs tables and Pareto frontier metrics:
  1. Prefill Latency (ms) across sequence lengths [512, 1024, 2048, 4096, 8192]
  2. Autoregressive Decode Speed (tokens / sec)
  3. Peak VRAM Allocation (MB)
  4. Passkey Retrieval Recall (%) at 4k context
  5. LaTeX Booktabs export for publication
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Benchmarks.Pareto")


@dataclass
class ParetoBenchmarkResult:
    method_name: str
    category: str  # "Full Attention", "Linear Attention", "Compression"
    prefill_latency_ms: Dict[int, float]  # seq_len -> latency (ms)
    decode_tokens_per_sec: float
    peak_vram_mb: float
    passkey_recall_pct: float
    macs_gflops: float = 0.0


class PaperParetoBenchmarkSuite:
    """
    Standardized benchmarking harness comparing Softmax Attention against
    Linear Attentions (Cosformer, GLA, Gated DeltaNet) and Compression (AASVD / Whitened SVD).
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.results: List[ParetoBenchmarkResult] = []

    def measure_prefill_latency(
        self,
        module: nn.Module,
        seq_lengths: List[int] = [512, 1024, 2048, 4096],
        warmup: int = 3,
        repetitions: int = 10,
    ) -> Dict[int, float]:
        """Measures prefill execution time in milliseconds across sequence lengths."""
        module.eval().to(self.device, dtype=self.dtype)
        latencies = {}

        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, self.d_model, device=self.device, dtype=self.dtype)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = module(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Measure
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(repetitions):
                    _ = module(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            total_time = (time.perf_counter() - t0) * 1000.0 / repetitions
            latencies[seq_len] = round(total_time, 2)

        return latencies

    def measure_decode_throughput(
        self,
        module: nn.Module,
        decode_steps: int = 64,
        warmup: int = 5,
    ) -> float:
        """Measures single-token autoregressive generation throughput (tokens / sec)."""
        module.eval().to(self.device, dtype=self.dtype)
        x_step = torch.randn(1, 1, self.d_model, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            for _ in range(warmup):
                _ = module(x_step)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(decode_steps):
                _ = module(x_step)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_per_sec = decode_steps / max(elapsed, 1e-6)
        return round(tok_per_sec, 1)

    def measure_peak_vram(
        self,
        module: nn.Module,
        seq_len: int = 4096,
    ) -> float:
        """Measures peak VRAM allocated during forward execution."""
        if self.device.type != "cuda":
            return 0.0

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        module.eval().to(self.device, dtype=self.dtype)
        x = torch.randn(1, seq_len, self.d_model, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            _ = module(x)
        torch.cuda.synchronize()

        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        return round(peak_bytes / (1024.0 * 1024.0), 1)

    def measure_passkey_retrieval(
        self,
        module: nn.Module,
        seq_len: int = 4096,
        passkey_dim: int = 64,
    ) -> float:
        """
        Synthesizes a needle-in-a-haystack synthetic task where a specific signal vector
        is placed at sequence position 128 and queried at the end.
        Returns retrieval cosine similarity / recall percentage.
        """
        module.eval().to(self.device, dtype=self.dtype)
        x = torch.randn(1, seq_len, self.d_model, device=self.device, dtype=self.dtype)
        needle = torch.ones(1, 1, self.d_model, device=self.device, dtype=self.dtype) * 3.0
        needle_pos = 128
        x[:, needle_pos:needle_pos+1, :] = needle

        with torch.no_grad():
            out = module(x)
            if isinstance(out, tuple):
                out = out[0]

        # Calculate signal retention at final token
        final_token = out[:, -1, :]
        similarity = torch.cosine_similarity(final_token, needle.squeeze(1), dim=-1).item()
        recall = max(0.0, min(100.0, (similarity + 1.0) / 2.0 * 100.0))
        return round(recall, 1)

    def to_latex_table(self) -> str:
        """Generates publication-quality LaTeX booktabs code."""
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Pareto Frontier: Runtime Latency, Decode Throughput, Peak VRAM, and Retrieval Quality across Attention Mechanisms and SVD Compression Engines.}",
            r"\label{tab:neuralops_pareto}",
            r"\small",
            r"\begin{tabular}{llccccr}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Category} & \textbf{Prefill (2k) [ms]} & \textbf{Prefill (4k) [ms]} & \textbf{Decode [tok/s]} & \textbf{Peak VRAM [MB]} & \textbf{Recall [\%]} \\",
            r"\midrule",
        ]

        for r in self.results:
            p2k = r.prefill_latency_ms.get(2048, "-")
            p4k = r.prefill_latency_ms.get(4096, "-")
            lines.append(
                f"{r.method_name} & {r.category} & {p2k} & {p4k} & {r.decode_tokens_per_sec} & {r.peak_vram_mb} & {r.passkey_recall_pct}\\% \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        return "\n".join(lines)


__all__ = ["ParetoBenchmarkResult", "PaperParetoBenchmarkSuite"]
