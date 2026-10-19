#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Multi-Precision (FP32, FP16, BF16) Roofline & TFLOPs Benchmark for Attention Mechanisms.

Measures latency, effective TFLOPs/s, memory bandwidth utilization against
RTX 4070 Laptop (264 GB/s theoretical max) across FP32, FP16, and BF16 precisions.
Outputs structured JSON to results/benchmarks_multi_precision_roofline.json.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


def fmt_opt(val, fmt=".2f", suffix="", default="-"):
    """Execute fmt opt routine."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f"{f:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return default


import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.benchmarks.harness import BenchmarkRunner, HardwareProfile, save_benchmark_json
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


TARGET_MECHANISMS = [
    ("flash_attention3", "Quadratic FlashAttention (O(N^2))"),
    ("gated_delta_net", "Gated Delta Net (GDN)"),
    ("gated_linear_attention", "Gated Linear Attention (GLA)"),
    ("cosformer", "Cosformer Linear"),
    ("svd_linear_attention", "SVD Linear Attention"),
    ("selective_ssm", "Selective SSM (Mamba)"),
]


def resolve_attention_key(key: str) -> str:
    """Execute resolve attention key routine."""
    if key in ATTENTION_REGISTRY:
        return key
    for k in ATTENTION_REGISTRY:
        if key.split("_")[0] in k:
            return k
    return next(iter(ATTENTION_REGISTRY.keys()))


def main():
    """Execute main routine."""
    parser = argparse.ArgumentParser(description="Multi-Precision Roofline Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--out-file", type=str, default="results/benchmarks_multi_precision_roofline.json")
    args = parser.parse_args()

    dev = torch.device(args.device)
    hw = HardwareProfile()
    runner = BenchmarkRunner(device=dev, warmup_iters=3, benchmark_iters=15, hardware_profile=hw)

    B, T, D, H = args.batch_size, args.seq_len, args.embed_dim, args.num_heads
    precisions = [
        ("FP32", torch.float32, 4),
        ("FP16", torch.float16, 2),
        ("BF16", torch.bfloat16, 2),
    ]

    print("=== Multi-Precision Roofline & Arithmetic Intensity Benchmark ===")
    print(f"Hardware: {hw.name} (Peak Bandwidth: {hw.peak_bandwidth_gb_s} GB/s)")
    print(f"Shapes: Batch={B}, SeqLen={T}, Dim={D}, Heads={H}\n")

    results = []

    for name_key, display_name in TARGET_MECHANISMS:
        actual_key = resolve_attention_key(name_key)
        is_linear = "linear" in actual_key or "mamba" in actual_key or "gdn" in actual_key

        entry = {
            "name": name_key,
            "actual_key": actual_key,
            "display_name": display_name,
            "precisions": {},
        }
        print(f"--> Mechanism: {display_name} ({actual_key})")

        for prec_name, dtype, byte_per_elem in precisions:
            runner.reset_vram()
            if is_linear:
                flops = 4.0 * B * T * (D**2)
            else:
                flops = 4.0 * B * (T**2) * D + 4.0 * B * T * (D**2)
            bytes_moved = 2.0 * (B * T * D + B * T * D) * byte_per_elem

            try:
                mod = create_attention(actual_key, embed_dim=D, num_heads=H).to(device=dev, dtype=dtype).eval()
                x = torch.randn(B, T, D, device=dev, dtype=dtype)

                def _make_forward(target_mod, target_x):
                    """Execute make forward routine."""

                    def run_forward():
                        """Execute run forward routine."""
                        with torch.no_grad():
                            out = target_mod(target_x)
                            if isinstance(out, tuple):
                                out = out[0]
                            return out

                    return run_forward

                metric = runner.measure_callable(
                    _make_forward(mod, x),
                    total_flops=flops,
                    total_bytes=bytes_moved,
                    batch_size=B,
                    seq_len=T,
                )

                mod = None
                x = None
                runner.reset_vram()

                entry["precisions"][prec_name] = {
                    "latency_ms": metric.latency_ms_trimmed_mean,
                    "tokens_per_sec": metric.throughput_tokens_sec,
                    "tflops_per_sec": metric.tflops_per_sec,
                    "bandwidth_gb_s": metric.bandwidth_gb_s,
                    "bandwidth_util_pct": metric.bandwidth_utilization_pct,
                    "peak_vram_mb": metric.peak_vram_mb,
                    "status": metric.status,
                    "error": metric.error,
                }
                print(
                    f"    {prec_name:4s}: Latency={metric.latency_ms_trimmed_mean:6.2f} ms | TFLOPs={metric.tflops_per_sec:5.2f} | BW={metric.bandwidth_gb_s:6.1f} GB/s ({metric.bandwidth_utilization_pct:4.1f}% util) | {metric.throughput_tokens_sec:8.1f} tok/s"
                )

            except Exception as e:
                err = str(e)
                entry["precisions"][prec_name] = {
                    "latency_ms": 0.0,
                    "tflops_per_sec": 0.0,
                    "bandwidth_gb_s": 0.0,
                    "bandwidth_util_pct": 0.0,
                    "status": "FAILED",
                    "error": err[:128],
                }
                print(f"    {prec_name:4s}: FAILED ({err[:60]})")

        results.append(entry)
        print()

    out_p = save_benchmark_json(results, args.out_file)
    print(f"Saved Multi-Precision Roofline results to {out_p.resolve()}")

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Multi-Precision Latency & Arithmetic Intensity (T=2048)")
        table.add_column("Mechanism", style="cyan")
        table.add_column("FP32 (ms)", justify="right")
        table.add_column("FP16 (ms)", justify="right")
        table.add_column("BF16 (ms)", justify="right")
        table.add_column("FP16 TFLOPs", justify="right", style="green")
        table.add_column("FP16 BW Util", justify="right", style="magenta")

        for r in results:
            p = r["precisions"]
            fp32_ms = fmt_opt(p.get("FP32", {}).get("latency_ms"), fmt=".2f", default="-")
            fp16_ms = fmt_opt(p.get("FP16", {}).get("latency_ms"), fmt=".2f", default="-")
            bf16_ms = fmt_opt(p.get("BF16", {}).get("latency_ms"), fmt=".2f", default="-")
            fp16_tf = fmt_opt(p.get("FP16", {}).get("tflops_per_sec"), fmt=".2f", default="-")
            fp16_bw = fmt_opt(p.get("FP16", {}).get("bandwidth_util_pct"), fmt=".1f", suffix="%", default="-")
            table.add_row(r["display_name"], fp32_ms, fp16_ms, bf16_ms, fp16_tf, fp16_bw)

        console.print(table)


if __name__ == "__main__":
    main()
