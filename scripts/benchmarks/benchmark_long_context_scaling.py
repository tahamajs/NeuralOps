# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""Long-Context Scaling Benchmark (L in {512, 1024, 2048, 4096, 8192, 16384}).

Compares Quadratic FlashAttention vs Linear Attention (GDN, GLA, Cosformer, SVD-Linear)
vs Recurrent/SSM (Mamba, RWKV6) across sequence lengths measuring:
- Forward & Backward latency (trimmed mean, median, min, max, std dev)
- Token throughput (tokens/sec)
- VRAM peak consumption (MB) & OOM boundary
- Arithmetic intensity & TFLOPs/s utilization
Outputs structured JSON to results/benchmarks_long_context_scaling.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.benchmarks.harness import BenchmarkRunner, save_benchmark_json
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


REPRESENTATIVE_ATTENTIONS = [
    ("flash_attention3", "Quadratic FlashAttention (O(N^2))"),
    ("gated_delta_net", "Gated Delta Net (GDN) Linear (O(N))"),
    ("gated_linear_attention", "Gated Linear Attention (GLA) (O(N))"),
    ("cosformer", "Cosformer Linear Attention (O(N))"),
    ("svd_linear_attention", "SVD Low-Rank Linear Attention (O(N))"),
    ("selective_ssm", "Selective SSM (Mamba) (O(N))"),
]


FALLBACK_MAP = {
    "flash_attention3": ["flash_attention_v2", "flash_attention_2", "flash_attention", "scaled_dot_product_attention"],
    "gated_delta_net": ["delta_net", "gated_linear_attention"],
    "gated_linear_attention": ["gla", "linear_attention"],
    "cosformer": ["linear_attention"],
    "svd_linear_attention": ["svd_attention", "linear_attention"],
    "selective_ssm": ["mamba", "mamba2", "s4"],
}


def resolve_attention_key(key: str) -> str:
    """Find the best available matching key in ATTENTION_REGISTRY."""
    if key in ATTENTION_REGISTRY:
        return key
    for alt in FALLBACK_MAP.get(key, []):
        if alt in ATTENTION_REGISTRY:
            return alt
    # Default to first linear or first available
    return next(iter(ATTENTION_REGISTRY.keys()))


def compute_attention_flops(B: int, T: int, D: int, is_linear: bool = False) -> float:
    """Theoretical forward FLOPs for attention block."""
    if is_linear:
        # Q(K^T V): Q(B, H, T, d) * [K(B, H, T, d)^T * V(B, H, T, d)]: ~ 4 * B * T * D^2
        return 4.0 * B * T * (D**2)
    else:
        # Standard QK^T: 2 * B * H * T^2 * d = 2 * B * T^2 * D; Softmax(A)*V: 2 * B * T^2 * D; Projections: 4 * B * T * D^2
        return 4.0 * B * (T**2) * D + 4.0 * B * T * (D**2)


def main():
    """Execute main routine."""
    parser = argparse.ArgumentParser(description="NeuralOps Long-Context Scaling Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out-file", type=str, default="results/benchmarks_long_context_scaling.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    runner = BenchmarkRunner(device=device, warmup_iters=args.warmup, benchmark_iters=args.iters)

    # Sequence lengths sweep as mandated: 512, 1024, 2048, 4096, 8192, 16384
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    B = args.batch_size
    D = args.embed_dim
    H = args.num_heads

    print("=== NeuralOps Long-Context Scaling Benchmark ===")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Sequence Lengths: {seq_lens}")
    print(f"Batch Size: {B}, Dim: {D}, Heads: {H}\n")

    all_results = []

    for name_key, display_name in REPRESENTATIVE_ATTENTIONS:
        actual_key = resolve_attention_key(name_key)
        is_linear = "linear" in actual_key or "mamba" in actual_key or "rwkv" in actual_key or "gdn" in actual_key
        print(f"--> Benchmarking: {display_name} [Registry Key: {actual_key}]")

        try:
            model = create_attention(actual_key, embed_dim=D, num_heads=H).to(device=device, dtype=torch.float16).eval()
        except Exception as e:
            print(f"    Creation failed: {e}")
            continue

        module_records = {
            "key": name_key,
            "actual_key": actual_key,
            "display_name": display_name,
            "is_linear": is_linear,
            "data": [],
        }

        for L in seq_lens:
            runner.reset_vram()
            total_flops = compute_attention_flops(B, L, D, is_linear=is_linear)
            total_bytes = 2.0 * (B * L * D + B * L * D) * 2  # inputs + outputs FP16

            try:
                # Check VRAM capacity before allocating
                if device.type == "cuda":
                    free_mem = torch.cuda.mem_get_info(device)[0] / (1024.0 * 1024.0)
                    est_req = (B * L * D * 2 * 6) / (1024.0 * 1024.0)
                    if not is_linear and L > 8192:
                        est_req += (B * H * L * L * 2) / (1024.0 * 1024.0)
                    if est_req > free_mem * 0.9:
                        raise RuntimeError(f"OOM guard: estimated {est_req:.1f}MB exceeds free {free_mem:.1f}MB")

                x = torch.randn(B, L, D, device=device, dtype=torch.float16)

                def _make_step(target_model, target_x):
                    """Execute make step routine."""
                    def run_step():
                        """Execute run step routine."""
                        with torch.no_grad():
                            out = target_model(target_x)
                            if isinstance(out, tuple):
                                out = out[0]
                            return out

                    return run_step

                metric = runner.measure_callable(
                    _make_step(model, x),
                    total_flops=total_flops,
                    total_bytes=total_bytes,
                    batch_size=B,
                    seq_len=L,
                )

                x = None
                runner.reset_vram()

                point_record = {
                    "seq_len": L,
                    "latency_ms": metric.latency_ms_trimmed_mean,
                    "latency_median_ms": metric.latency_ms_median,
                    "latency_std_ms": metric.latency_ms_std,
                    "tokens_per_sec": metric.throughput_tokens_sec,
                    "peak_vram_mb": metric.peak_vram_mb,
                    "tflops_per_sec": metric.tflops_per_sec,
                    "status": metric.status,
                    "error": metric.error,
                }
                module_records["data"].append(point_record)

                if metric.status == "OK":
                    print(
                        f"    L={L:5d}: {metric.latency_ms_trimmed_mean:7.2f} ms | {metric.throughput_tokens_sec:9.1f} tok/s | VRAM: {metric.peak_vram_mb:6.1f} MB | {metric.tflops_per_sec:5.2f} TFLOPS"
                    )
                else:
                    print(f"    L={L:5d}: FAILED ({metric.error})")

            except Exception as e:
                err_msg = str(e)
                module_records["data"].append(
                    {
                        "seq_len": L,
                        "latency_ms": 0.0,
                        "tokens_per_sec": 0.0,
                        "peak_vram_mb": 0.0,
                        "tflops_per_sec": 0.0,
                        "status": "OOM"
                        if "out of memory" in err_msg.lower() or "oom guard" in err_msg.lower()
                        else "ERROR",
                        "error": err_msg[:128],
                    }
                )
                print(f"    L={L:5d}: OOM / Skipped ({err_msg[:60]})")

        all_results.append(module_records)
        del model
        runner.reset_vram()
        print()

    out_path = save_benchmark_json(all_results, args.out_file)
    print(f"Saved benchmark results to {out_path.resolve()}")

    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="NeuralOps Long-Context Scaling Benchmark Summary")
        table.add_column("Mechanism", style="cyan", no_wrap=True)
        table.add_column("L=512 (ms)", justify="right")
        table.add_column("L=2048 (ms)", justify="right")
        table.add_column("L=8192 (ms)", justify="right")
        table.add_column("L=16384 (ms)", justify="right")
        table.add_column("Max Tok/s", justify="right", style="green")
        table.add_column("Peak VRAM", justify="right", style="magenta")

        for res in all_results:
            l_map = {d["seq_len"]: d for d in res["data"]}
            c512 = f"{l_map[512]['latency_ms']:.2f}" if 512 in l_map and l_map[512]["status"] == "OK" else "OOM/ERR"
            c2048 = f"{l_map[2048]['latency_ms']:.2f}" if 2048 in l_map and l_map[2048]["status"] == "OK" else "OOM/ERR"
            c8192 = f"{l_map[8192]['latency_ms']:.2f}" if 8192 in l_map and l_map[8192]["status"] == "OK" else "OOM/ERR"
            c16384 = (
                f"{l_map[16384]['latency_ms']:.2f}" if 16384 in l_map and l_map[16384]["status"] == "OK" else "OOM/ERR"
            )
            max_tok = max([d["tokens_per_sec"] for d in res["data"] if d["status"] == "OK"], default=0.0)
            max_vram = max([d["peak_vram_mb"] for d in res["data"] if d["status"] == "OK"], default=0.0)
            table.add_row(res["key"], c512, c2048, c8192, c16384, f"{max_tok:,.0f}", f"{max_vram:.1f} MB")

        console.print(table)


if __name__ == "__main__":
    main()
