# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""End-to-End GPT-2 Physical Hardware Comparison: Triton vs PyTorch Fallback."""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

import neuralops
import neuralops.core.triton_kernels as tk_types


def run_benchmark():
    """Execute run benchmark routine."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running End-to-End GPT-2 Benchmark on: {device} ({torch.cuda.get_device_name(0)})")

    config = GPT2Config(
        n_embd=768,
        n_layer=12,
        n_head=12,
        vocab_size=50257,
    )
    base_model = GPT2LMHeadModel(config).to(device=device, dtype=torch.float16).eval()

    print("Transplanting GPT-2 causal attention with Triton PositiveKernelLinearAttention...")
    model = neuralops.replace_attention(base_model, attention="positive_kernel_linear_attention")
    model.eval()

    eval_configs = [
        {"batch_size": 2, "seq_len": 512},
        {"batch_size": 4, "seq_len": 512},
        {"batch_size": 2, "seq_len": 1024},
    ]

    benchmark_records = []

    for cfg in eval_configs:
        B = cfg["batch_size"]
        T = cfg["seq_len"]
        print(f"\nEvaluating configuration: Batch={B}, SeqLen={T}...")
        x = torch.randint(0, 50257, (B, T), device=device)

        with torch.no_grad():
            tk_types.FORCE_PYTORCH_FALLBACK = False
            os.environ["FORCE_PYTORCH_FALLBACK"] = "0"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            for _ in range(5):
                _ = model(x)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            iters = 25
            for _ in range(iters):
                out_tri = model(x)
            torch.cuda.synchronize()
            t_tri_ms = (time.perf_counter() - t0) / iters * 1000.0
            peak_vram_tri_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            tokens_per_sec_tri = (B * T) / (t_tri_ms / 1000.0)

            tk_types.FORCE_PYTORCH_FALLBACK = True
            os.environ["FORCE_PYTORCH_FALLBACK"] = "1"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            for _ in range(5):
                _ = model(x)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iters):
                out_pt = model(x)
            torch.cuda.synchronize()
            t_pt_ms = (time.perf_counter() - t0) / iters * 1000.0
            peak_vram_pt_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            tokens_per_sec_pt = (B * T) / (t_pt_ms / 1000.0)

            tk_types.FORCE_PYTORCH_FALLBACK = False
            os.environ["FORCE_PYTORCH_FALLBACK"] = "0"

            logits_tri = out_tri.logits.float().view(-1, 50257)
            logits_pt = out_pt.logits.float().view(-1, 50257)
            cos_sim = F.cosine_similarity(logits_tri, logits_pt, dim=-1).mean().item()
            max_abs_err = (logits_tri - logits_pt).abs().max().item()
            speedup = t_pt_ms / t_tri_ms

            record = {
                "batch_size": B,
                "seq_len": T,
                "triton": {
                    "latency_ms": round(t_tri_ms, 3),
                    "peak_vram_mb": round(peak_vram_tri_mb, 2),
                    "tokens_per_sec": round(tokens_per_sec_tri, 1),
                },
                "pytorch_fallback": {
                    "latency_ms": round(t_pt_ms, 3),
                    "peak_vram_mb": round(peak_vram_pt_mb, 2),
                    "tokens_per_sec": round(tokens_per_sec_pt, 1),
                },
                "speedup_factor": round(speedup, 2),
                "output_cosine_similarity": round(cos_sim, 6),
                "max_absolute_error": round(max_abs_err, 6),
                "status": "PASS" if (speedup >= 1.30 and cos_sim >= 0.999) else "WARN",
            }
            benchmark_records.append(record)

            print(
                f"  Triton       : Latency={t_tri_ms:6.2f} ms | VRAM={peak_vram_tri_mb:6.1f} MB | Throughput={tokens_per_sec_tri:8.1f} tok/s"
            )
            print(
                f"  PyTorch      : Latency={t_pt_ms:6.2f} ms | VRAM={peak_vram_pt_mb:6.1f} MB | Throughput={tokens_per_sec_pt:8.1f} tok/s"
            )
            print(f"  Speedup      : {speedup:5.2f}x (Target: >=1.30x)")
            print(f"  Cosine Parity: {cos_sim:7.5f} (Target: >=0.99900)")

    out_file = Path("results/triton_e2e_gpt2_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"\nMaster benchmark results written to {out_file}")


if __name__ == "__main__":
    run_benchmark()
