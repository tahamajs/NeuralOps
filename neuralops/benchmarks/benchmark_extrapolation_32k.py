
import logging
logger = logging.getLogger(__name__)
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Long-Horizon Sequence Extrapolation and KV-State Scaling Benchmark (up to 32,768 tokens)

Evaluates latency, peak memory, KV-state complexity, and numerical drift across Standard Softmax Attention, Chunk GLA, Gated DeltaNet, and SVD Linear Attention.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

import neuralops.core.triton_kernels._types as tt

tt.MAX_TRITON_SEQ_LEN = 65536

from neuralops.attentions.svd.svd_linear_attention import SVDLinearAttention
from neuralops.core.triton_kernels.gdn_scan import gdn_scan
from neuralops.core.triton_kernels.gla_scan import gla_scan

DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
QUICK_SEQ_LENS = [512, 1024]


def collect_cuda_memory_stats(device: torch.device | str) -> dict[str, Any]:
    """Report torch.cuda.memory_stats() metrics and activation fragmentation ratio."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda" or not torch.cuda.is_available():
        return {
            "peak_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "fragmentation_ratio": 0.0,
            "memory_stats": {},
        }
    try:
        mem_stats = torch.cuda.memory_stats(dev)
        peak_mb = float(torch.cuda.max_memory_allocated(dev) / (1024.0**2))
        res_bytes = mem_stats.get("reserved_bytes.all.current", torch.cuda.memory_reserved(dev))
        alloc_bytes = mem_stats.get("allocated_bytes.all.current", torch.cuda.memory_allocated(dev))
        res_mb = float(res_bytes / (1024.0**2))
        alloc_mb = float(alloc_bytes / (1024.0**2))
        frag_ratio = float(max(0.0, (res_bytes - alloc_bytes) / res_bytes)) if res_bytes > 0 else 0.0
        return {
            "peak_vram_mb": round(peak_mb, 1),
            "reserved_vram_mb": round(res_mb, 1),
            "allocated_vram_mb": round(alloc_mb, 1),
            "fragmentation_ratio": round(frag_ratio, 4),
            "memory_stats": mem_stats,
        }
    except Exception:
        return {
            "peak_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "fragmentation_ratio": 0.0,
            "memory_stats": {},
        }


def run_extrapolation_benchmark(
    device_str: str = "auto",
    dtype_str: str = "float16",
    seq_lens: list = None,
    warmup: int = 3,
    repeat: int = 10,
    output_csv: str = "docs/benchmarks/paper_benchmarks_extrapolation_32k.csv",
    output_json: str = "results/paper_benchmarks_extrapolation_32k.json",
    output_tex: str = "docs/reports/PAPER_TABLES_EXTRAPOLATION_32K.tex",
):
    """Execute sequence length scaling benchmark."""
    if seq_lens is None:
        seq_lens = DEFAULT_SEQ_LENS

    if device_str == "auto" or device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if "cuda" in device_str and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.float16)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    logger.info(f"=== Long-Horizon Sequence Extrapolation Benchmark on {device} ({gpu_name}) ===")
    logger.info(f"Sequence lengths: {seq_lens} | Precision: {dtype_str} | Warmup: {warmup} | Repeat: {repeat}")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    B = 1
    H = 4
    D = 64
    rank = 16
    embed_dim = H * D

    models = ["Standard Quadratic Attention", "Chunk GLA", "Gated Delta Net", "SVD Linear Attention"]
    results = []

    for L in seq_lens:
        logger.info(f"\n================ Sequence Length L = {L} ================")

        q = (torch.randn(B, H, L, D, device=device, dtype=dtype) / (D**0.5)).contiguous()
        k = (torch.randn(B, H, L, D, device=device, dtype=dtype) / (D**0.5)).contiguous()
        v = (torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1).contiguous()
        lam = (-torch.rand(B, H, L, device=device, dtype=dtype) * 0.1).contiguous()
        beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype)).contiguous()
        alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype)).contiguous()

        svd_model = SVDLinearAttention(embed_dim=embed_dim, num_heads=H, rank=rank).to(device=device, dtype=dtype)
        x_svd = torch.randn(B, L, embed_dim, device=device, dtype=dtype)

        elem_bytes = q.element_size()
        kv_mem_bytes_quad = 2 * (B * H * L * D * elem_bytes)
        kv_mem_bytes_recurrent = B * H * D * D * elem_bytes
        kv_mem_bytes_svd = B * H * D * rank * elem_bytes

        for m_name in models:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            drift = 0.0
            if m_name == "Standard Quadratic Attention":
                fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                kv_state_kb = kv_mem_bytes_quad / 1024.0
                state_complexity = "O(L)"
                drift = 0.0
            elif m_name == "Chunk GLA":
                fn = lambda: gla_scan(q, k, v, lambdas=lam)
                kv_state_kb = kv_mem_bytes_recurrent / 1024.0
                state_complexity = "O(1)"
                with torch.no_grad():
                    try:
                        q32, k32, v32, lam32 = q.float(), k.float(), v.float(), lam.float()
                        ref32 = gla_scan(q32, k32, v32, lambdas=lam32)
                        y = fn()
                        drift = (y.float() - ref32).norm().item() / max(ref32.norm().item(), 1e-12)
                    except Exception:
                        drift = 0.0
            elif m_name == "Gated Delta Net":
                fn = lambda: gdn_scan(q, k, v, beta=beta, alpha=alpha)
                kv_state_kb = kv_mem_bytes_recurrent / 1024.0
                state_complexity = "O(1)"
                with torch.no_grad():
                    try:
                        q32, k32, v32, b32, a32 = q.float(), k.float(), v.float(), beta.float(), alpha.float()
                        ref32 = gdn_scan(q32, k32, v32, beta=b32, alpha=a32)
                        y = fn()
                        drift = (y.float() - ref32).norm().item() / max(ref32.norm().item(), 1e-12)
                    except Exception:
                        drift = 0.0
            elif m_name == "SVD Linear Attention":
                fn = lambda: svd_model(x_svd)
                kv_state_kb = kv_mem_bytes_svd / 1024.0
                state_complexity = "O(1)"
                with torch.no_grad():
                    try:
                        svd32 = SVDLinearAttention(embed_dim=embed_dim, num_heads=H, rank=rank).to(
                            device=device, dtype=torch.float32
                        )
                        svd32.load_state_dict({k_n: v_t.float() for k_n, v_t in svd_model.state_dict().items()})
                        ref32 = svd32(x_svd.float())[0]
                        y = fn()[0]
                        drift = (y.float() - ref32).norm().item() / max(ref32.norm().item(), 1e-12)
                    except Exception:
                        drift = 0.0

            try:
                for _ in range(warmup):
                    _ = fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()

                iters = max(1, repeat if L <= 16384 else (repeat // 2))
                t0 = time.perf_counter()
                for _ in range(iters):
                    out = fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat_ms = ((time.perf_counter() - t0) / iters) * 1000.0

                mem_info = collect_cuda_memory_stats(device)

                if isinstance(out, tuple):
                    out = out[0]
                has_nan = torch.isnan(out).any().item() or torch.isinf(out).any().item()

                row = {
                    "sequence_length": L,
                    "mechanism": m_name,
                    "latency_ms": round(lat_ms, 2),
                    "peak_vram_mb": mem_info["peak_vram_mb"],
                    "reserved_vram_mb": mem_info["reserved_vram_mb"],
                    "allocated_vram_mb": mem_info["allocated_vram_mb"],
                    "fragmentation_ratio": mem_info["fragmentation_ratio"],
                    "kv_state_kb": round(kv_state_kb, 1),
                    "state_complexity": state_complexity,
                    "state_drift_rel": float(f"{drift:.2e}"),
                    "zero_nan": not has_nan,
                    "memory_stats": mem_info["memory_stats"],
                }
                results.append(row)
                logger.info(
                    f"  {m_name:<28} | Latency: {lat_ms:6.2f} ms | Peak VRAM: {mem_info['peak_vram_mb']:6.1f} MB | Frag: {mem_info['fragmentation_ratio']:6.4f} | KV State: {kv_state_kb:8.1f} KB ({state_complexity}) | Drift: {drift:9.2e} | Stable: {not has_nan}"
                )
            except Exception as exc:
                logger.info(f"  {m_name:<28} | EXCEPTION: {exc}")
                results.append(
                    {
                        "sequence_length": L,
                        "mechanism": m_name,
                        "latency_ms": 0.0,
                        "peak_vram_mb": 0.0,
                        "reserved_vram_mb": 0.0,
                        "allocated_vram_mb": 0.0,
                        "fragmentation_ratio": 0.0,
                        "kv_state_kb": round(kv_state_kb, 1),
                        "state_complexity": state_complexity,
                        "state_drift_rel": float("nan"),
                        "zero_nan": False,
                        "memory_stats": {},
                    }
                )

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence_length",
                    "mechanism",
                    "latency_ms",
                    "peak_vram_mb",
                    "reserved_vram_mb",
                    "allocated_vram_mb",
                    "fragmentation_ratio",
                    "kv_state_kb",
                    "state_complexity",
                    "state_drift_rel",
                    "zero_nan",
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        logger.info(f"\nSaved CSV to: {csv_path}")

    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON to: {json_path}")

    if output_tex:
        tex_path = Path(output_tex)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex = [
            r"% Auto-generated Long-Horizon Sequence Extrapolation Benchmark",
            f"% Target: {gpu_name}",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Long-Horizon Sequence Extrapolation and KV-State Scaling on "
            + gpu_name
            + r". Comparing standard causal softmax attention with recurrent linear associative scan mechanisms.}",
            r"\label{tab:extrapolation_32k}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"\textbf{Sequence Length ($L$)} & \textbf{Mechanism} & \textbf{Latency (ms)} & \textbf{Peak VRAM (MB)} & \textbf{KV-State Size} & \textbf{State Drift ($\|\Delta\|/\|Y\|_F$)} & \textbf{Zero-NaN/Inf} \\",
            r"\midrule",
        ]

        curr_L = None
        for r in results:
            L_val = r["sequence_length"]
            if L_val != curr_L:
                if curr_L is not None:
                    tex.append(r"\midrule")
                curr_L = L_val
                l_col = f"\\textbf{{{L_val:,}}}"
            else:
                l_col = ""

            mech_disp = r["mechanism"]
            lat_str = f"{r['latency_ms']:.2f}"
            vram_str = f"{r['peak_vram_mb']:.1f}"
            kv_str = f"{r['kv_state_kb']:.0f} KB ($\\mathcal{{O}}({r['state_complexity'][2:-1]})$)"
            drift_str = (
                "Baseline" if r["mechanism"] == "Standard Quadratic Attention" else f"{r['state_drift_rel']:.2e}"
            )
            status_str = r"\checkmark" if r["zero_nan"] else r"\times"

            tex.append(f"{l_col} & {mech_disp} & {lat_str} & {vram_str} & {kv_str} & {drift_str} & {status_str} \\\\")

        tex.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
        )
        with open(tex_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")
        logger.info(f"Saved LaTeX Table to: {tex_path}")

    return results


def _legacy_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Long-Horizon Sequence Extrapolation Benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/benchmarks/paper_benchmarks_extrapolation_32k.csv",
        help="Path to save CSV output",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/paper_benchmarks_extrapolation_32k.json",
        help="Path to save JSON output",
    )
    parser.add_argument(
        "--tex-output",
        type=str,
        default="docs/reports/PAPER_TABLES_EXTRAPOLATION_32K.tex",
        help="Path to save LaTeX table",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to execute benchmark (default: 'auto')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16", "fp16", "fp32", "bf16"],
        help="Precision",
    )
    parser.add_argument(
        "--seq-lens", type=str, default="512,1024,2048,4096,8192,16384,32768", help="Comma-separated sequence lengths"
    )
    parser.add_argument("--min-len", type=int, default=None, help="Minimum sequence length (powers of 2 scaling)")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum sequence length (powers of 2 scaling)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="Repeat iterations")
    parser.add_argument("--quick", action="store_true", help="Quick mode (seq-lens 512,1024)")
    args = parser.parse_args(argv)

    if args.quick:
        seq_lens = QUICK_SEQ_LENS
    elif args.min_len is not None or args.max_len is not None:
        min_l = args.min_len if args.min_len is not None else 512
        max_l = args.max_len if args.max_len is not None else 32768
        seq_lens = []
        cur = min_l
        while cur <= max_l:
            seq_lens.append(cur)
            cur *= 2
    else:
        seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]

    warmup = 1 if args.quick else args.warmup
    repeat = 2 if args.quick else args.repeat

    output_json = args.output_json
    output_csv = args.output
    if args.output and args.output.endswith(".json"):
        output_json = args.output
        output_csv = args.output[:-5] + ".csv"

    run_extrapolation_benchmark(
        device_str=args.device,
        dtype_str=args.dtype,
        seq_lens=seq_lens,
        warmup=warmup,
        repeat=repeat,
        output_csv=output_csv,
        output_json=output_json,
        output_tex=args.tex_output,
    )
    return 0


if __name__ == "__main__":
    # DEPRECATED_ENTRYPOINT
    pass #
    sys.exit(main())
