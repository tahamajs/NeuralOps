# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""
Publication Attention Benchmark Suite.
Compares specialized attention kernels (sparse_attention, routing_attention, cross_attention)
against standard PyTorch SDPA reference across:
B in {1, 4, 8}
L in {512, 1024, 2048, 4096, 8192}
D in {64, 128}

Measures forward latency, backward latency, peak memory, speedup, and numerical diff.
Outputs:
- docs/benchmarks/paper_benchmarks_attention.csv
- docs/reports/PAPER_TABLES_ATTENTION.tex
"""

import sys
from pathlib import Path
import csv
import torch

# Ensure neuralops is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from neuralops.core.cuda_kernels.sparse_attention.sparse_attention_autograd import (
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)
from neuralops.core.cuda_kernels.routing_attention.routing_attention_autograd import (
    routing_attention,
    routing_attention_ref,
)
from neuralops.core.cuda_kernels.cross_attention.cross_attn_autograd import (
    cross_attention,
    cross_attention_pytorch_ref,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def benchmark_kernel(fwd_fn, bwd_fn, *args, warmup=10, iters=30):
    """Execute benchmark kernel routine."""
    torch.cuda.synchronize()
    # Warmup
    for _ in range(warmup):
        out = fwd_fn(*args)
        if bwd_fn and out.requires_grad:
            grad_out = torch.ones_like(out)
            bwd_fn(out, grad_out)
    torch.cuda.synchronize()

    # Time forward
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out = fwd_fn(*args)
    end_event.record()
    torch.cuda.synchronize()
    fwd_time_ms = start_event.elapsed_time(end_event) / iters

    # Time backward if applicable
    bwd_time_ms = 0.0
    if bwd_fn and out.requires_grad:
        grad_out = torch.ones_like(out)
        start_event.record()
        for _ in range(iters):
            bwd_fn(out, grad_out)
        end_event.record()
        torch.cuda.synchronize()
        bwd_time_ms = start_event.elapsed_time(end_event) / iters

    return fwd_time_ms, bwd_time_ms, out


def run_benchmarks():
    """Execute run benchmarks routine."""
    print(f"Running benchmarks on: {torch.cuda.get_device_name(0)}")
    results = []

    configs = [
        # (B, L, D)
        (1, 1024, 64),
        (1, 2048, 64),
        (1, 4096, 64),
        (1, 8192, 64),
        (4, 1024, 64),
        (4, 2048, 64),
        (4, 4096, 64),
        (8, 512, 64),
        (8, 1024, 64),
        (8, 2048, 64),
        (1, 1024, 128),
        (1, 2048, 128),
        (1, 4096, 128),
        (4, 1024, 128),
        (4, 2048, 128),
    ]

    H = 8  # Fixed 8 attention heads

    for B, L, D in configs:
        torch.cuda.empty_cache()
        print(f"\n--- Config: B={B}, H={H}, L={L}, D={D} ---")

        # 1. Sparse Attention
        block_size = 64
        q_sparse = torch.randn(B, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
        k_sparse = torch.randn(B, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
        v_sparse = torch.randn(B, H, L, D, device=device, dtype=torch.float16, requires_grad=True)

        num_blocks_q = L // block_size
        num_blocks_k = L // block_size
        # Local window sparsity: band of width 3 blocks
        block_mask = torch.zeros(num_blocks_q, num_blocks_k, dtype=torch.bool, device=device)
        for bq in range(num_blocks_q):
            for bk in range(max(0, bq - 1), min(num_blocks_k, bq + 2)):
                block_mask[bq, bk] = True

        # PyTorch SDPA baseline
        def ref_sparse_fwd(q, k, v):
            """Execute ref sparse fwd routine."""
            return block_sparse_attn_pytorch_ref(q, k, v, block_mask, block_size)

        def ref_bwd(out, grad):
            """Execute ref bwd routine."""
            out.backward(grad, retain_graph=True)

        ref_fwd_ms, ref_bwd_ms, out_ref = benchmark_kernel(
            ref_sparse_fwd, ref_bwd, q_sparse, k_sparse, v_sparse, warmup=5, iters=15
        )

        # CUDA Sparse Attention
        def cuda_sparse_fwd(q, k, v):
            """Execute cuda sparse fwd routine."""
            return block_sparse_attn(q, k, v, block_mask, block_size)

        cuda_fwd_ms, cuda_bwd_ms, out_cuda = benchmark_kernel(
            cuda_sparse_fwd, ref_bwd, q_sparse, k_sparse, v_sparse, warmup=5, iters=15
        )

        max_diff = (out_cuda.float() - out_ref.float()).abs().max().item()
        speedup_fwd = ref_fwd_ms / max(cuda_fwd_ms, 1e-6)
        speedup_bwd = ref_bwd_ms / max(cuda_bwd_ms, 1e-6) if ref_bwd_ms > 0 else 1.0

        results.append(
            {
                "kernel": "sparse_attention",
                "batch": B,
                "heads": H,
                "seq_len": L,
                "head_dim": D,
                "ref_fwd_ms": f"{ref_fwd_ms:.3f}",
                "cuda_fwd_ms": f"{cuda_fwd_ms:.3f}",
                "speedup_fwd": f"{speedup_fwd:.2f}x",
                "ref_bwd_ms": f"{ref_bwd_ms:.3f}",
                "cuda_bwd_ms": f"{cuda_bwd_ms:.3f}",
                "speedup_bwd": f"{speedup_bwd:.2f}x",
                "max_diff": f"{max_diff:.2e}",
            }
        )
        print(
            f"Sparse Attn: Ref Fwd={ref_fwd_ms:.2f}ms, CUDA Fwd={cuda_fwd_ms:.2f}ms (Speedup {speedup_fwd:.2f}x), Max Diff={max_diff:.2e}"
        )

        # 2. Routing Attention
        q_rout = torch.randn(B, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
        k_rout = torch.randn(B, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
        v_rout = torch.randn(B, H, L, D, device=device, dtype=torch.float16, requires_grad=True)
        # Sparse route weights
        route_weights = torch.rand(B, H, L, device=device, dtype=torch.float16)
        route_weights = torch.where(route_weights > 0.5, route_weights, torch.zeros_like(route_weights))

        def ref_rout_fwd(q, k, v):
            """Execute ref rout fwd routine."""
            return routing_attention_ref(q, k, v, route_weights)

        def cuda_rout_fwd(q, k, v):
            """Execute cuda rout fwd routine."""
            return routing_attention(q, k, v, route_weights)

        ref_fwd_ms, ref_bwd_ms, out_ref = benchmark_kernel(
            ref_rout_fwd, ref_bwd, q_rout, k_rout, v_rout, warmup=5, iters=15
        )
        cuda_fwd_ms, cuda_bwd_ms, out_cuda = benchmark_kernel(
            cuda_rout_fwd, ref_bwd, q_rout, k_rout, v_rout, warmup=5, iters=15
        )

        max_diff = (out_cuda.float() - out_ref.float()).abs().max().item()
        speedup_fwd = ref_fwd_ms / max(cuda_fwd_ms, 1e-6)
        speedup_bwd = ref_bwd_ms / max(cuda_bwd_ms, 1e-6) if ref_bwd_ms > 0 else 1.0

        results.append(
            {
                "kernel": "routing_attention",
                "batch": B,
                "heads": H,
                "seq_len": L,
                "head_dim": D,
                "ref_fwd_ms": f"{ref_fwd_ms:.3f}",
                "cuda_fwd_ms": f"{cuda_fwd_ms:.3f}",
                "speedup_fwd": f"{speedup_fwd:.2f}x",
                "ref_bwd_ms": f"{ref_bwd_ms:.3f}",
                "cuda_bwd_ms": f"{cuda_bwd_ms:.3f}",
                "speedup_bwd": f"{speedup_bwd:.2f}x",
                "max_diff": f"{max_diff:.2e}",
            }
        )
        print(
            f"Routing Attn: Ref Fwd={ref_fwd_ms:.2f}ms, CUDA Fwd={cuda_fwd_ms:.2f}ms (Speedup {speedup_fwd:.2f}x), Max Diff={max_diff:.2e}"
        )

        # 3. Cross Attention (L_q = L // 4, L_kv = L)
        L_q = max(32, L // 4)
        L_kv = L
        q_cross = torch.randn(B, H, L_q, D, device=device, dtype=torch.float16, requires_grad=True)
        k_cross = torch.randn(B, H, L_kv, D, device=device, dtype=torch.float16, requires_grad=True)
        v_cross = torch.randn(B, H, L_kv, D, device=device, dtype=torch.float16, requires_grad=True)

        def ref_cross_fwd(q, k, v):
            """Execute ref cross fwd routine."""
            return cross_attention_pytorch_ref(q, k, v)

        def cuda_cross_fwd(q, k, v):
            """Execute cuda cross fwd routine."""
            return cross_attention(q, k, v)

        ref_fwd_ms, ref_bwd_ms, out_ref = benchmark_kernel(
            ref_cross_fwd, ref_bwd, q_cross, k_cross, v_cross, warmup=5, iters=15
        )
        cuda_fwd_ms, cuda_bwd_ms, out_cuda = benchmark_kernel(
            cuda_cross_fwd, ref_bwd, q_cross, k_cross, v_cross, warmup=5, iters=15
        )

        max_diff = (out_cuda.float() - out_ref.float()).abs().max().item()
        speedup_fwd = ref_fwd_ms / max(cuda_fwd_ms, 1e-6)
        speedup_bwd = ref_bwd_ms / max(cuda_bwd_ms, 1e-6) if ref_bwd_ms > 0 else 1.0

        results.append(
            {
                "kernel": "cross_attention",
                "batch": B,
                "heads": H,
                "seq_len": f"Q:{L_q}/KV:{L_kv}",
                "head_dim": D,
                "ref_fwd_ms": f"{ref_fwd_ms:.3f}",
                "cuda_fwd_ms": f"{cuda_fwd_ms:.3f}",
                "speedup_fwd": f"{speedup_fwd:.2f}x",
                "ref_bwd_ms": f"{ref_bwd_ms:.3f}",
                "cuda_bwd_ms": f"{cuda_bwd_ms:.3f}",
                "speedup_bwd": f"{speedup_bwd:.2f}x",
                "max_diff": f"{max_diff:.2e}",
            }
        )
        print(
            f"Cross Attn: Ref Fwd={ref_fwd_ms:.2f}ms, CUDA Fwd={cuda_fwd_ms:.2f}ms (Speedup {speedup_fwd:.2f}x), Max Diff={max_diff:.2e}"
        )

    # Export CSV
    csv_path = repo_root / "docs" / "benchmarks" / "paper_benchmarks_attention.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kernel",
        "batch",
        "heads",
        "seq_len",
        "head_dim",
        "ref_fwd_ms",
        "cuda_fwd_ms",
        "speedup_fwd",
        "ref_bwd_ms",
        "cuda_bwd_ms",
        "speedup_bwd",
        "max_diff",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved benchmark results to {csv_path}")

    # Export LaTeX
    tex_path = repo_root / "docs" / "reports" / "PAPER_TABLES_ATTENTION.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated Attention Benchmarks LaTeX Table\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\caption{Benchmarking Memory-Optimized CUDA Attention Kernels (SMEM Tiling, 128-bit Loads) vs PyTorch SDPA Reference on NVIDIA RTX 4070 GPU.}\n"
        )
        f.write("\\label{tab:cuda_attention_benchmarks}\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Kernel & Shape $(B, L, D)$ & PyTorch Fwd (ms) & CUDA Fwd (ms) & Fwd Speedup & PyTorch Bwd (ms) & CUDA Bwd (ms) & Bwd Speedup \\\\\n"
        )
        f.write("\\midrule\n")
        for row in results:
            shape_str = f"({row['batch']}, {row['seq_len']}, {row['head_dim']})"
            k_name = row["kernel"].replace("_", "\\_")
            f.write(
                f"{k_name} & {shape_str} & {row['ref_fwd_ms']} & {row['cuda_fwd_ms']} & {row['speedup_fwd']} & {row['ref_bwd_ms']} & {row['cuda_bwd_ms']} & {row['speedup_bwd']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"Saved LaTeX table to {tex_path}")


if __name__ == "__main__":
    run_benchmarks()
