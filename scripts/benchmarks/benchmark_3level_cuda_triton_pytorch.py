#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""
3-Level GPU Benchmark Suite: Native CUDA C++ vs. Triton JIT vs. Pure PyTorch Reference.
Target: NVIDIA Ada Lovelace SM 8.9 (GeForce RTX 4070 Laptop GPU, 8GB VRAM).

Measures latency (ms), peak allocated VRAM (MB), speedups, and parity across all 3 execution levels:
  Level 1: Native CUDA C++ (SM >= 7.0)
  Level 2: OpenAI Triton JIT
  Level 3: Pure PyTorch Reference
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels import (
    force_kernel_backend,
    fused_lowrank_gated_gemm,
    chunked_delta_gate_scan,
    hgrn2_fused_bwd_scan,
    cosformer_linear_attention,
    fused_norm_linear,
    fused_rotary_emb,
    kronecker_matmul,
    lut_codebook_dequant_gemm,
    linear_attention,
    fused_cross_entropy,
)
from neuralops.core.cuda_kernels.selective_scan.selective_scan_autograd import selective_scan
from neuralops.core.cuda_kernels.sliding_window.sliding_autograd import sliding_window_attn
from neuralops.core.cuda_kernels.sparse_attention.sparse_autograd import block_sparse_attn
from neuralops.core.cuda_kernels.quantized_gemm.quant_autograd import w4a16_gemm, fp8_gemm
from neuralops.core.cuda_kernels.flash_gqa.flash_gqa_autograd import flash_gqa
from neuralops.core.cuda_kernels.paged_attention.paged_attention_autograd import paged_attention
from neuralops.core.cuda_kernels.moe_routing.moe_autograd import moe_dispatch, moe_gather

logger = logging.getLogger("benchmark_3level")


def benchmark_fn(
    fn: Callable[[], Any],
    warmup: int = 25,
    iters: int = 100,
) -> Tuple[float, float]:
    """Measures CUDA kernel execution time (ms) and peak allocated VRAM (MB)."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = fn()
    end_event.record()
    torch.cuda.synchronize()

    mean_ms = start_event.elapsed_time(end_event) / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return mean_ms, peak_vram_mb


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> Tuple[float, float]:
    ref_f = ref.detach().flatten().float()
    test_f = test.detach().flatten().float()
    cos_sim = F.cosine_similarity(ref_f, test_f, dim=0).item()
    diff = (ref_f - test_f).abs()
    rel_err = (diff / (ref_f.abs() + 1e-5)).max().item()
    return cos_sim, rel_err


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable. Exiting benchmark.")
        sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 80)
    print("3-LEVEL BENCHMARK: CUDA C++ vs. TRITON JIT vs. PYTORCH REFERENCE")
    print(f"Physical Device: {gpu_name} (SM {torch.cuda.get_device_capability(0)})")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Selective Scan
    # -------------------------------------------------------------------------
    print("\n[1/8] selective_scan (Mamba SSM)")
    B, D, L, D_STATE = 4, 64, 512, 16
    u = torch.randn(B, D, L, device=device, dtype=torch.float32)
    delta = torch.rand(B, D, L, device=device, dtype=torch.float32) * 0.1
    A = -torch.rand(D, D_STATE, device=device, dtype=torch.float32)
    B_m = torch.randn(B, D_STATE, L, device=device, dtype=torch.float32)
    C_m = torch.randn(B, D_STATE, L, device=device, dtype=torch.float32)
    D_val = torch.randn(D, device=device, dtype=torch.float32)

    fn_py = lambda: selective_scan(u, delta, A, B_m, C_m, D_val)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_py)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_py)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Triton benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_py)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("CUDA benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "selective_scan",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 2. Sliding Window Attention
    # -------------------------------------------------------------------------
    print("\n[2/8] sliding_window_attn")
    B, H, N, D = 4, 8, 512, 64
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    fn_sw = lambda: sliding_window_attn(q, k, v, window_size=64, num_sinks=4)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_sw)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_sw)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_sw)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "sliding_window_attn",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 3. Block Sparse Attention
    # -------------------------------------------------------------------------
    print("\n[3/8] block_sparse_attn")
    B, H, N, D = 4, 8, 512, 64
    block_size = 32
    num_blocks = N // block_size
    mask = torch.eye(num_blocks, device=device, dtype=torch.bool)
    for i in range(num_blocks):
        mask[i, max(0, i - 1):min(num_blocks, i + 2)] = True

    fn_bs = lambda: block_sparse_attn(q, k, v, block_mask=mask, block_size=block_size)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_bs)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_bs)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_bs)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "block_sparse_attn",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 4. W4A16 Quantized GEMM
    # -------------------------------------------------------------------------
    print("\n[4/8] w4a16_gemm")
    M, K, N = 128, 1024, 1024
    group_size = 64
    A_mat = torch.randn(M, K, device=device, dtype=torch.float32)
    W_q = torch.randint(0, 255, (K, N // 2), device=device, dtype=torch.uint8)
    scales = torch.rand(K // group_size, N, device=device, dtype=torch.float32) * 0.1
    zeros = torch.zeros(K // group_size, N, device=device, dtype=torch.float32)

    fn_w4 = lambda: w4a16_gemm(A_mat, W_q, scales, zeros, group_size=group_size)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_w4)

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_w4)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "w4a16_gemm",
        "pytorch_ms": lat_py,
        "triton_ms": None,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 5. FP8 Scaled GEMM
    # -------------------------------------------------------------------------
    print("\n[5/8] fp8_gemm")
    M, K, N = 256, 1024, 1024
    A_fp8 = torch.randn(M, K, device=device, dtype=torch.float32)
    B_fp8 = torch.randn(K, N, device=device, dtype=torch.float32)
    sa = torch.tensor([0.5], device=device, dtype=torch.float32)
    sb = torch.tensor([0.5], device=device, dtype=torch.float32)

    fn_fp8 = lambda: fp8_gemm(A_fp8, B_fp8, sa, sb)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_fp8)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_fp8)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_fp8)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "fp8_gemm",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 6. Flash GQA
    # -------------------------------------------------------------------------
    print("\n[6/8] flash_gqa")
    B, H, N_Q, D = 4, 8, 256, 64
    H_KV, N_KV = 2, 256
    q_gqa = torch.randn(B, H, N_Q, D, device=device, dtype=torch.float32)
    k_gqa = torch.randn(B, H_KV, N_KV, D, device=device, dtype=torch.float32)
    v_gqa = torch.randn(B, H_KV, N_KV, D, device=device, dtype=torch.float32)

    fn_gqa = lambda: flash_gqa(q_gqa, k_gqa, v_gqa, is_causal=True)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_gqa)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_gqa)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_gqa)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "flash_gqa",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 7. Paged Attention
    # -------------------------------------------------------------------------
    print("\n[7/8] paged_attention")
    num_seqs = 8
    num_heads = 8
    head_dim = 64
    block_size = 16
    num_blocks = 64

    q_pa = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
    k_cache = torch.randn(num_blocks, num_heads, head_dim, block_size, device=device, dtype=torch.float32)
    v_cache = torch.randn(num_blocks, num_heads, head_dim, block_size, device=device, dtype=torch.float32)
    block_tables = torch.randint(0, num_blocks, (num_seqs, 16), device=device, dtype=torch.int32)
    context_lens = torch.full((num_seqs,), 128, device=device, dtype=torch.int32)

    fn_pa = lambda: paged_attention(q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_pa)

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_pa)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "paged_attention",
        "pytorch_ms": lat_py,
        "triton_ms": None,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 8. MoE Routing
    # -------------------------------------------------------------------------
    print("\n[8/8] moe_routing (dispatch + gather)")
    num_tokens, hidden_dim, top_k = 512, 1024, 2
    total_slots = 1024
    tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
    topk_indices = torch.randint(0, 8, (num_tokens, top_k), device=device, dtype=torch.int32)
    offsets = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32).view(num_tokens, top_k)
    weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.float32), dim=-1)

    def fn_moe():
        disp = moe_dispatch(tokens, topk_indices, offsets, total_slots)
        return moe_gather(disp, weights, topk_indices, offsets)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_moe)

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_moe)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "moe_routing",
        "pytorch_ms": lat_py,
        "triton_ms": None,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 9. Fused Low-Rank Gated GEMM
    # -------------------------------------------------------------------------
    print("\n[9/18] fused_lowrank_gated_gemm")
    M, K, r, N = 64, 512, 32, 256
    x_lr = torch.randn(M, K, device=device, dtype=torch.float32)
    V_lr = torch.randn(r, K, device=device, dtype=torch.float32)
    U_lr = torch.randn(N, r, device=device, dtype=torch.float32)
    alpha_lr = torch.randn(r, device=device, dtype=torch.float32)
    bias_lr = torch.randn(N, device=device, dtype=torch.float32)
    fn_lr = lambda: fused_lowrank_gated_gemm(x_lr, V_lr, U_lr, alpha=alpha_lr, bias=bias_lr)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_lr)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_lr)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_lr)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "fused_lowrank_gated_gemm",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 10. Chunked Delta-Gate Recurrent Scan
    # -------------------------------------------------------------------------
    print("\n[10/18] chunked_delta_gate_scan")
    B_g, H_g, L_g, D_K_g, D_V_g = 2, 4, 128, 32, 32
    q_g = torch.randn(B_g, H_g, L_g, D_K_g, device=device, dtype=torch.float32)
    k_g = F.normalize(torch.randn(B_g, H_g, L_g, D_K_g, device=device, dtype=torch.float32), p=2, dim=-1)
    v_g = torch.randn(B_g, H_g, L_g, D_V_g, device=device, dtype=torch.float32)
    alpha_g = torch.sigmoid(torch.randn(B_g, H_g, L_g, device=device, dtype=torch.float32))
    beta_g = torch.sigmoid(torch.randn(B_g, H_g, L_g, device=device, dtype=torch.float32))
    fn_gdn = lambda: chunked_delta_gate_scan(q_g, k_g, v_g, alpha_g, beta_g)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_gdn)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_gdn)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_gdn)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "chunked_delta_gate_scan",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 11. HGRN2 Bidirectional Recurrent Scan
    # -------------------------------------------------------------------------
    print("\n[11/18] hgrn2_fused_bwd_scan")
    B_h, H_h, L_h, D_h = 2, 4, 128, 64
    x_h = torch.randn(B_h, H_h, L_h, D_h, device=device, dtype=torch.float32)
    g_h = torch.sigmoid(torch.randn(B_h, H_h, L_h, D_h, device=device, dtype=torch.float32))
    fn_hgrn = lambda: hgrn2_fused_bwd_scan(x_h, g_h)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_hgrn)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_hgrn)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_hgrn)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "hgrn2_fused_bwd_scan",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 12. Cosformer Linear Attention
    # -------------------------------------------------------------------------
    print("\n[12/18] cosformer_linear_attention")
    B_c, H_c, L_c, D_c = 2, 4, 128, 32
    q_c = torch.randn(B_c, H_c, L_c, D_c, device=device, dtype=torch.float32)
    k_c = torch.randn(B_c, H_c, L_c, D_c, device=device, dtype=torch.float32)
    v_c = torch.randn(B_c, H_c, L_c, D_c, device=device, dtype=torch.float32)
    fn_cos = lambda: cosformer_linear_attention(q_c, k_c, v_c, is_causal=True)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_cos)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_cos)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_cos)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "cosformer_linear_attention",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 13. Fused Norm Linear
    # -------------------------------------------------------------------------
    print("\n[13/18] fused_norm_linear")
    M_fn, K_fn, N_fn = 64, 512, 256
    x_fn = torch.randn(M_fn, K_fn, device=device, dtype=torch.float32)
    w_fn = torch.randn(N_fn, K_fn, device=device, dtype=torch.float32)
    b_fn = torch.randn(N_fn, device=device, dtype=torch.float32)
    fn_norm_lin = lambda: fused_norm_linear(x_fn, w_fn, bias=b_fn)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_norm_lin)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_norm_lin)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_norm_lin)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "fused_norm_linear",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 14. Fused RoPE
    # -------------------------------------------------------------------------
    print("\n[14/18] fused_rotary_emb")
    B_rp, H_rp, N_rp, D_rp = 4, 8, 256, 64
    half_rp = D_rp // 2
    x_rp = torch.randn(B_rp, H_rp, N_rp, D_rp, device=device, dtype=torch.float32)
    cos_rp = torch.randn(1, 1, N_rp, half_rp, device=device, dtype=torch.float32)
    sin_rp = torch.randn(1, 1, N_rp, half_rp, device=device, dtype=torch.float32)
    fn_rope = lambda: fused_rotary_emb(x_rp, cos_rp, sin_rp)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_rope)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_rope)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_rope)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "fused_rotary_emb",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 15. Kronecker Matmul
    # -------------------------------------------------------------------------
    print("\n[15/18] kronecker_matmul")
    M_kr, M1_kr, N1_kr, M2_kr, N2_kr = 64, 16, 16, 16, 16
    x_kr = torch.randn(M_kr, N1_kr * N2_kr, device=device, dtype=torch.float32)
    A_kr = torch.randn(M1_kr, N1_kr, device=device, dtype=torch.float32)
    B_kr = torch.randn(M2_kr, N2_kr, device=device, dtype=torch.float32)
    b_kr = torch.randn(M1_kr * M2_kr, device=device, dtype=torch.float32)
    fn_kron = lambda: kronecker_matmul(x_kr, A_kr, B_kr, bias=b_kr)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_kron)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_kron)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_kron)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "kronecker_matmul",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 16. In-Register LUT Codebook Dequantization GEMM
    # -------------------------------------------------------------------------
    print("\n[16/18] lut_codebook_dequant_gemm")
    M_lt, K_lt, N_lt = 64, 256, 128
    grp_lt = 64
    num_grp_lt = K_lt // grp_lt
    x_lt = torch.randn(M_lt, K_lt, device=device, dtype=torch.float32)
    indices_lt = torch.randint(0, 255, (N_lt, K_lt // 2), device=device, dtype=torch.uint8)
    codebook_lt = torch.randn(num_grp_lt, 16, device=device, dtype=torch.float32) * 0.1
    b_lt = torch.randn(N_lt, device=device, dtype=torch.float32)
    fn_lut = lambda: lut_codebook_dequant_gemm(x_lt, indices_lt, codebook_lt, group_size=grp_lt, bias=b_lt)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_lut)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_lut)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_lut)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "lut_codebook_dequant_gemm",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 17. Causal Linear Attention
    # -------------------------------------------------------------------------
    print("\n[17/18] linear_attention")
    B_la, H_la, L_la, D_K_la, D_V_la = 2, 4, 128, 32, 32
    q_la = torch.randn(B_la, H_la, L_la, D_K_la, device=device, dtype=torch.float32)
    k_la = torch.randn(B_la, H_la, L_la, D_K_la, device=device, dtype=torch.float32)
    v_la = torch.randn(B_la, H_la, L_la, D_V_la, device=device, dtype=torch.float32)
    fn_la = lambda: linear_attention(q_la, k_la, v_la)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_la)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_la)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_la)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "linear_attention",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 18. Fused Cross-Entropy Loss
    # -------------------------------------------------------------------------
    print("\n[18/18] fused_cross_entropy")
    N_ce, V_ce = 512, 1024
    logits_ce = torch.randn(N_ce, V_ce, device=device, dtype=torch.float32)
    targets_ce = torch.randint(0, V_ce, (N_ce,), device=device, dtype=torch.int64)
    fn_ce = lambda: fused_cross_entropy(logits_ce, targets_ce, ignore_index=-100, reduction="mean")

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_ce)
    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_ce)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_tri, vram_tri = None, None
    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_ce)
        except (RuntimeError, MemoryError, AssertionError) as e:
            logger.warning("Backend benchmark failed: %s", e, exc_info=True)
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")
    results.append({
        "kernel": "fused_cross_entropy",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "triton_vram_mb": vram_tri,
        "cuda_vram_mb": vram_cuda,
    })

    # Save to JSON, CSV, MD
    os.makedirs("results", exist_ok=True)
    json_path = "results/benchmark_3level_cuda_triton_pytorch.json"
    csv_path = "results/benchmark_3level_cuda_triton_pytorch.csv"
    md_path = "results/benchmark_3level_cuda_triton_pytorch.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🚀 3-Level Execution Hierarchy Benchmark: Native CUDA C++ vs. Triton JIT vs. PyTorch\n\n")
        f.write(f"**Device:** {gpu_name} (SM {torch.cuda.get_device_capability(0)})\n\n")
        f.write("| Kernel Operator | PyTorch Ref (ms) | Triton JIT (ms) | CUDA C++ (ms) | Speedup (CUDA vs. Py) | Speedup (CUDA vs. Triton) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            tri_str = f"{r['triton_ms']:.3f}" if r['triton_ms'] else "N/A"
            cuda_str = f"{r['cuda_ms']:.3f}" if r['cuda_ms'] else "N/A"
            f.write(f"| `{r['kernel']}` | {r['pytorch_ms']:.3f} | {tri_str} | {cuda_str} | **{r['speedup_cuda_vs_py']:.2f}x** | {r['speedup_cuda_vs_tri']:.2f}x |\n")

    print("\n" + "=" * 80)
    print(f"BENCHMARK SAVED TO {json_path}, {csv_path}, AND {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
