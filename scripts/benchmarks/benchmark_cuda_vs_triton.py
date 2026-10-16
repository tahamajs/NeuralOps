"""
Comprehensive benchmark: CUDA C++ vs Triton vs PyTorch for 8 hot-path modules.

Usage:
    ssh gpu "cd /mnt/d/SVD_linear_Attention && \
        PYTHONPATH=. /mnt/d/venv/bin/python3 \
        scripts/benchmarks/benchmark_cuda_vs_triton.py \
        --output results/cuda_vs_triton_benchmark.json"
"""

import argparse
import json
import time
from pathlib import Path
import torch

from neuralops.core.cuda_kernels.paged_attention import paged_attention, paged_attention_pytorch_reference
from neuralops.core.cuda_kernels.selective_scan import selective_scan, selective_scan_pytorch_ref
from neuralops.core.cuda_kernels.quantized_gemm import (
    w4a16_gemm, w4a16_gemm_pytorch_ref, fp8_gemm, fp8_gemm_pytorch_ref
)
from neuralops.core.cuda_kernels.flash_gqa import flash_gqa, flash_gqa_pytorch_ref
from neuralops.core.cuda_kernels.moe_routing import moe_dispatch_gather
from neuralops.core.cuda_kernels.sparse_attention import block_sparse_attn, block_sparse_attn_pytorch_ref
from neuralops.core.cuda_kernels.sliding_window import sliding_window_attn, sliding_window_pytorch_ref
from neuralops.core.cuda_kernels.paged_kv_cache import copy_blocks_cow, copy_blocks_cow_pytorch_ref


def _benchmark_fn(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def benchmark_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print("Benchmarking 1: PagedAttention...")
    q = torch.randn(8, 8, 64, device=device)
    k_cache = torch.randn(64, 2, 64, 16, device=device)
    v_cache = torch.randn(64, 2, 64, 16, device=device)
    context_lens = torch.tensor([64] * 8, device=device, dtype=torch.int32)
    block_tables = torch.arange(8 * 4, device=device, dtype=torch.int32).view(8, 4)
    args_paged = (q, k_cache, v_cache, block_tables, context_lens, 16, 64, 1.0 / 8.0)

    t_cuda = _benchmark_fn(paged_attention, (q, k_cache, v_cache, block_tables, context_lens, 16))
    t_ref = _benchmark_fn(paged_attention_pytorch_reference, args_paged)
    results["paged_attention"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    print("Benchmarking 2: Selective Scan...")
    u = torch.randn(2, 64, 128, device=device)
    delta = torch.randn(2, 64, 128, device=device).sigmoid()
    A = -torch.rand(64, 16, device=device) - 0.5
    B_sel = torch.randn(2, 16, 128, device=device)
    C_sel = torch.randn(2, 16, 128, device=device)
    D_val = torch.randn(64, device=device)
    args_scan = (u, delta, A, B_sel, C_sel, D_val)

    t_cuda = _benchmark_fn(selective_scan, args_scan)
    t_ref = _benchmark_fn(selective_scan_pytorch_ref, args_scan)
    results["selective_scan"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    print("Benchmarking 3: Quantized GEMM (W4A16 & FP8)...")
    A_mat = torch.randn(64, 128, device=device)
    W_q = torch.randint(0, 255, (128, 64), device=device, dtype=torch.uint8)
    scales = torch.rand(4, 128, device=device) * 0.1
    zeros = torch.randint(0, 8, (4, 128), device=device, dtype=torch.float32)
    args_w4 = (A_mat, W_q, scales, zeros, 32)

    t_cuda_w4 = _benchmark_fn(w4a16_gemm, args_w4)
    t_ref_w4 = _benchmark_fn(w4a16_gemm_pytorch_ref, args_w4)
    results["w4a16_gemm"] = {
        "cuda_ms": round(t_cuda_w4, 3),
        "pytorch_ms": round(t_ref_w4, 3),
        "speedup_vs_pytorch": round(t_ref_w4 / max(1e-5, t_cuda_w4), 2),
    }

    B_mat = torch.randn(128, 128, device=device)
    scale_a = torch.tensor(0.5, device=device)
    scale_b = torch.tensor(0.25, device=device)
    args_fp8 = (A_mat, B_mat, scale_a, scale_b)
    t_cuda_fp8 = _benchmark_fn(fp8_gemm, args_fp8)
    t_ref_fp8 = _benchmark_fn(fp8_gemm_pytorch_ref, args_fp8)
    results["fp8_gemm"] = {
        "cuda_ms": round(t_cuda_fp8, 3),
        "pytorch_ms": round(t_ref_fp8, 3),
        "speedup_vs_pytorch": round(t_ref_fp8 / max(1e-5, t_cuda_fp8), 2),
    }

    print("Benchmarking 4: Flash GQA...")
    q_gqa = torch.randn(2, 8, 128, 64, device=device)
    k_gqa = torch.randn(2, 2, 128, 64, device=device)
    v_gqa = torch.randn(2, 2, 128, 64, device=device)
    args_gqa = (q_gqa, k_gqa, v_gqa, False)

    t_cuda = _benchmark_fn(flash_gqa, args_gqa)
    t_ref = _benchmark_fn(flash_gqa_pytorch_ref, args_gqa)
    results["flash_gqa"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    print("Benchmarking 5: MoE Routing...")
    tokens = torch.randn(256, 128, device=device)
    gate_logits = torch.randn(256, 8, device=device)
    def dummy_expert(x): return x * 1.5

    t_cuda = _benchmark_fn(moe_dispatch_gather, (tokens, gate_logits, dummy_expert, 2))
    def py_moe(t, g):
        topk_vals, topk_idx = torch.topk(g, k=2, dim=-1)
        w = torch.softmax(topk_vals, dim=-1)
        disp = t.repeat_interleave(2, dim=0)
        exp_out = dummy_expert(disp)
        return (exp_out.view(256, 2, 128) * w.unsqueeze(-1)).sum(dim=1)
    t_ref = _benchmark_fn(py_moe, (tokens, gate_logits))
    results["moe_routing"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    print("Benchmarking 6: Block Sparse Attention...")
    q_sp = torch.randn(2, 4, 64, 64, device=device)
    k_sp = torch.randn(2, 4, 64, 64, device=device)
    v_sp = torch.randn(2, 4, 64, 64, device=device)
    block_mask = torch.eye(4, device=device, dtype=torch.int32)
    block_mask[0, 1] = 1
    args_sp = (q_sp, k_sp, v_sp, block_mask, 16)

    t_cuda = _benchmark_fn(block_sparse_attn, args_sp)
    t_ref = _benchmark_fn(block_sparse_attn_pytorch_ref, args_sp)
    results["sparse_attention"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    print("Benchmarking 7: Sliding Window...")
    q_sw = torch.randn(2, 4, 128, 64, device=device)
    k_sw = torch.randn(2, 4, 128, 64, device=device)
    v_sw = torch.randn(2, 4, 128, 64, device=device)
    args_sw = (q_sw, k_sw, v_sw, 32, 4)

    t_cuda = _benchmark_fn(sliding_window_attn, args_sw)
    t_ref = _benchmark_fn(sliding_window_pytorch_ref, args_sw)
    results["sliding_window"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    print("Benchmarking 8: Paged KV Cache (CoW)...")
    src_k = torch.randn(32, 8, 64, 16, device=device)
    src_v = torch.randn(32, 8, 64, 16, device=device)
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)
    block_map = torch.tensor([[0, 2], [1, 5], [3, 7], [4, 8]], device=device, dtype=torch.int32)
    args_cow = (dst_k, dst_v, src_k, src_v, block_map, 16, 8, 64)

    t_cuda = _benchmark_fn(copy_blocks_cow, args_cow)
    t_ref = _benchmark_fn(copy_blocks_cow_pytorch_ref, args_cow)
    results["paged_kv_cache"] = {
        "cuda_ms": round(t_cuda, 3),
        "pytorch_ms": round(t_ref, 3),
        "speedup_vs_pytorch": round(t_ref / max(1e-5, t_cuda), 2),
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/cuda_vs_triton_benchmark.json")
    args = parser.parse_args()

    res = benchmark_all()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    md_path = args.output.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CUDA C++ vs PyTorch Reference — Measured Hot-Path Benchmarks\n\n")
        f.write("| Module | CUDA (ms) | PyTorch (ms) | Speedup vs PyTorch |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for k, v in res.items():
            f.write(f"| **{k}** | {v['cuda_ms']:.3f} | {v['pytorch_ms']:.3f} | **{v['speedup_vs_pytorch']:.2f}×** |\n")

    print(f"\nSaved benchmark results to {args.output} and {md_path}")
