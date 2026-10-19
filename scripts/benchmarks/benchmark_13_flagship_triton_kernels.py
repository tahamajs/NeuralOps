# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Rigorous Micro-Benchmarking Suite for the 13 Flagship Triton GPU Kernels.
Executed natively on NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace SM 8.9, 8GB VRAM).

Measures PyTorch vs Triton:
  1. Mean latency (ms) across 100 timed iterations after 25 warmups via torch.cuda.Event.
  2. Peak allocated VRAM (MB) via torch.cuda.max_memory_allocated().
  3. Speedup factor (x).
  4. VRAM savings (%).
  5. Maximum relative difference and cosine similarity.

Outputs:
  - results/triton_13_flagship_kernels_benchmark.json
  - results/triton_13_flagship_kernels_benchmark.csv
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels import (
    triton_fused_lowrank_gated_gemm,
    triton_fused_lowrank_gated_gemm_pytorch_ref,
    triton_fused_lowrank_sparse_gemm,
    triton_fused_lowrank_sparse_gemm_pytorch_ref,
    triton_strided_tt_linear,
    triton_strided_tt_contraction_pytorch_ref,
    triton_fused_hadamard_w4a16_gemm,
    triton_fused_hadamard_w4a16_gemm_pytorch_ref,
    triton_lut_codebook_dequant_gemm,
    triton_lut_codebook_dequant_gemm_pytorch_ref,
    triton_fused_kronecker_matmul,
    triton_fused_kronecker_matmul_pytorch_ref,
    triton_chunked_delta_gate_scan,
    triton_chunked_delta_gate_scan_pytorch_ref,
    triton_cosformer_linear_attention,
    triton_cosformer_linear_attention_pytorch_ref,
    triton_block_sparse_streaming_attn,
    triton_block_sparse_streaming_attn_pytorch_ref,
    triton_hgrn2_fused_bwd_scan,
    triton_hgrn2_fused_bwd_scan_pytorch_ref,
    triton_spectral_entropy_svd_gate,
    triton_spectral_entropy_svd_gate_pytorch_ref,
    triton_fused_act_scale_quant_gemm,
    triton_fused_act_scale_quant_gemm_pytorch_ref,
    triton_fused_latent_kv_cache_attention,
    triton_fused_latent_kv_cache_attention_pytorch_ref,
)


def benchmark_fn(
    fn: Callable[[], torch.Tensor],
    warmup: int = 25,
    iters: int = 100,
) -> Tuple[float, float]:
    """Measures CUDA kernel execution time (ms) and peak allocated VRAM (MB)."""
    assert callable(fn), "fn must be callable"
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

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return elapsed_ms, peak_vram_mb


def compute_metrics(out_ref: torch.Tensor, out_tri: torch.Tensor) -> Tuple[float, float]:
    """Execute compute metrics routine."""
    ref_f = out_ref.float().view(-1)
    tri_f = out_tri.float().view(-1)

    cos_sim = F.cosine_similarity(ref_f.unsqueeze(0), tri_f.unsqueeze(0)).item()
    denom = torch.norm(ref_f) + 1e-7
    rel_diff = (torch.norm(ref_f - tri_f) / denom).item()
    return cos_sim, rel_diff


def main():
    """Execute main routine."""
    if not torch.cuda.is_available():
        print("CUDA is required for Triton GPU benchmarks. Exiting.")
        sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 80)
    print(f"BENCHMARKING 13 FLAGSHIP TRITON KERNELS ON {gpu_name}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    print("\n[Kernel 1/13] triton_fused_lowrank_gated_gemm")
    M, K, r, N = 1024, 2048, 32, 2048
    x1 = torch.randn(M, K, device=device, dtype=torch.float16)
    V1 = torch.randn(r, K, device=device, dtype=torch.float16)
    U1 = torch.randn(N, r, device=device, dtype=torch.float16)
    a1 = torch.randn(r, device=device, dtype=torch.float16)
    b1 = torch.randn(N, device=device, dtype=torch.float16)

    fn_ref1 = lambda: triton_fused_lowrank_gated_gemm_pytorch_ref(x1, V1, U1, alpha=a1, b_drift=b1)
    fn_tri1 = lambda: triton_fused_lowrank_gated_gemm(x1, V1, U1, alpha=a1, b_drift=b1)

    lat_ref, vram_ref = benchmark_fn(fn_ref1)
    lat_tri, vram_tri = benchmark_fn(fn_tri1)
    cos_sim, rel_err = compute_metrics(fn_ref1(), fn_tri1())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 1,
            "kernel": "triton_fused_lowrank_gated_gemm",
            "target_module": "hnc_svd, dobi_svd, direct_svd",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 2/13] triton_fused_lowrank_sparse_gemm")
    M, K, r, N = 1024, 2048, 32, 2048
    x2 = torch.randn(M, K, device=device, dtype=torch.float16)
    V2 = torch.randn(r, K, device=device, dtype=torch.float16)
    U2 = torch.randn(N, r, device=device, dtype=torch.float16)
    s2 = torch.randn(N, K, device=device, dtype=torch.float16) * (torch.rand(N, K, device=device) < 0.05).to(
        torch.float16
    )
    s2_csr = s2.to_sparse_csr()
    b2 = torch.randn(N, device=device, dtype=torch.float16)

    fn_ref2 = lambda: triton_fused_lowrank_sparse_gemm_pytorch_ref(x2, V2, U2, s=s2_csr, bias=b2)
    fn_tri2 = lambda: triton_fused_lowrank_sparse_gemm(x2, V2, U2, s=s2_csr, bias=b2)

    lat_ref, vram_ref = benchmark_fn(fn_ref2)
    lat_tri, vram_tri = benchmark_fn(fn_tri2)
    cos_sim, rel_err = compute_metrics(fn_ref2(), fn_tri2())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 2,
            "kernel": "triton_fused_lowrank_sparse_gemm",
            "target_module": "rpca_compressor",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 3/13] triton_strided_tt_linear")
    p_dims = [32, 64]
    q_dims = [32, 64]
    r = 16
    M = 1024
    c1 = torch.randn(1, p_dims[0], q_dims[0], r, device=device, dtype=torch.float16)
    c2 = torch.randn(r, p_dims[1], q_dims[1], 1, device=device, dtype=torch.float16)
    x3 = torch.randn(M, p_dims[0] * p_dims[1], device=device, dtype=torch.float16)
    b3 = torch.randn(q_dims[0] * q_dims[1], device=device, dtype=torch.float16)

    fn_ref3 = lambda: F.linear(x3, triton_strided_tt_contraction_pytorch_ref(c1, c2, p_dims, q_dims), b3)
    fn_tri3 = lambda: triton_strided_tt_linear(x3, c1, c2, p_dims, q_dims, bias=b3)

    lat_ref, vram_ref = benchmark_fn(fn_ref3)
    lat_tri, vram_tri = benchmark_fn(fn_tri3)
    cos_sim, rel_err = compute_metrics(fn_ref3(), fn_tri3())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 3,
            "kernel": "triton_strided_tt_linear",
            "target_module": "tt_compressor",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 4/13] triton_fused_hadamard_w4a16_gemm")
    M, K, N = 128, 512, 512
    group_size = 32
    x4 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_p4 = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8)
    s4 = torch.rand(N, K // group_size, device=device, dtype=torch.float16) * 0.1
    z4 = torch.zeros(N, K // group_size, device=device, dtype=torch.float16)
    b4 = torch.randn(N, device=device, dtype=torch.float16)

    fn_ref4 = lambda: triton_fused_hadamard_w4a16_gemm_pytorch_ref(x4, w_p4, s4, z4, group_size=group_size, bias=b4)
    fn_tri4 = lambda: triton_fused_hadamard_w4a16_gemm(x4, w_p4, s4, z4, group_size=group_size, bias=b4)

    lat_ref, vram_ref = benchmark_fn(fn_ref4)
    lat_tri, vram_tri = benchmark_fn(fn_tri4)
    cos_sim, rel_err = compute_metrics(fn_ref4(), fn_tri4())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 4,
            "kernel": "triton_fused_hadamard_w4a16_gemm",
            "target_module": "rotational_slicegpt_int4, quarot",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 5/13] triton_lut_codebook_dequant_gemm")
    M, K, N = 128, 512, 512
    group_size = 64
    x5 = torch.randn(M, K, device=device, dtype=torch.float16)
    idx5 = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8)
    cb5 = torch.randn(K // group_size, 16, device=device, dtype=torch.float16) * 0.2
    b5 = torch.randn(N, device=device, dtype=torch.float16)

    fn_ref5 = lambda: triton_lut_codebook_dequant_gemm_pytorch_ref(x5, idx5, cb5, group_size=group_size, bias=b5)
    fn_tri5 = lambda: triton_lut_codebook_dequant_gemm(x5, idx5, cb5, group_size=group_size, bias=b5)

    lat_ref, vram_ref = benchmark_fn(fn_ref5)
    lat_tri, vram_tri = benchmark_fn(fn_tri5)
    cos_sim, rel_err = compute_metrics(fn_ref5(), fn_tri5())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 5,
            "kernel": "triton_lut_codebook_dequant_gemm",
            "target_module": "coopq_quantizer, kmeans_quantize",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 6/13] triton_fused_kronecker_matmul")
    M1, N1 = 32, 32
    M2, N2 = 64, 64
    M = 1024
    x6 = torch.randn(M, N1 * N2, device=device, dtype=torch.float16)
    A6 = torch.randn(M1, N1, device=device, dtype=torch.float16)
    B6 = torch.randn(M2, N2, device=device, dtype=torch.float16)
    b6 = torch.randn(M1 * M2, device=device, dtype=torch.float16)

    fn_ref6 = lambda: triton_fused_kronecker_matmul_pytorch_ref(x6, A6, B6, bias=b6)
    fn_tri6 = lambda: triton_fused_kronecker_matmul(x6, A6, B6, bias=b6)

    lat_ref, vram_ref = benchmark_fn(fn_ref6)
    lat_tri, vram_tri = benchmark_fn(fn_tri6)
    cos_sim, rel_err = compute_metrics(fn_ref6(), fn_tri6())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 6,
            "kernel": "triton_fused_kronecker_matmul",
            "target_module": "kronecker_compressor",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 7/13] triton_chunked_delta_gate_scan")
    B, H, L, D_K, D_V = 2, 8, 128, 32, 32
    q7 = torch.randn(B, H, L, D_K, device=device, dtype=torch.float32)
    k7 = F.normalize(torch.randn(B, H, L, D_K, device=device, dtype=torch.float32), p=2, dim=-1)
    v7 = torch.randn(B, H, L, D_V, device=device, dtype=torch.float32)
    a7 = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
    be7 = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))

    fn_ref7 = lambda: triton_chunked_delta_gate_scan_pytorch_ref(q7, k7, v7, a7, be7)
    fn_tri7 = lambda: triton_chunked_delta_gate_scan(q7, k7, v7, a7, be7)

    lat_ref, vram_ref = benchmark_fn(fn_ref7, warmup=5, iters=20)
    lat_tri, vram_tri = benchmark_fn(fn_tri7, warmup=10, iters=50)
    cos_sim, rel_err = compute_metrics(fn_ref7(), fn_tri7())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 7,
            "kernel": "triton_chunked_delta_gate_scan",
            "target_module": "gated_delta_net, gla_attention",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 8/13] triton_cosformer_linear_attention")
    B, H, L, D = 2, 8, 256, 64
    q8 = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    k8 = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    v8 = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

    fn_ref8 = lambda: triton_cosformer_linear_attention_pytorch_ref(q8, k8, v8, is_causal=True)
    fn_tri8 = lambda: triton_cosformer_linear_attention(q8, k8, v8, is_causal=True)

    lat_ref, vram_ref = benchmark_fn(fn_ref8)
    lat_tri, vram_tri = benchmark_fn(fn_tri8)
    cos_sim, rel_err = compute_metrics(fn_ref8(), fn_tri8())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 8,
            "kernel": "triton_cosformer_linear_attention",
            "target_module": "cosformer, performer",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 9/13] triton_block_sparse_streaming_attn")
    B, H, L, D = 2, 4, 256, 64
    block_size = 32
    num_q_blocks = L // block_size
    K_active = 4
    q9 = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    k9 = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    v9 = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
    ab9 = torch.randint(0, num_q_blocks, (B, H, num_q_blocks, K_active), device=device, dtype=torch.int32)

    fn_ref9 = lambda: triton_block_sparse_streaming_attn_pytorch_ref(q9, k9, v9, ab9, block_size=block_size)
    fn_tri9 = lambda: triton_block_sparse_streaming_attn(q9, k9, v9, ab9, block_size=block_size)

    lat_ref, vram_ref = benchmark_fn(fn_ref9)
    lat_tri, vram_tri = benchmark_fn(fn_tri9)
    cos_sim, rel_err = compute_metrics(fn_ref9(), fn_tri9())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 9,
            "kernel": "triton_block_sparse_streaming_attn",
            "target_module": "nsa_attention, moba_attention",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 10/13] triton_hgrn2_fused_bwd_scan")
    B, H, L, D = 2, 8, 128, 64
    x10 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    g10 = torch.sigmoid(torch.randn(B, H, L, D, device=device, dtype=torch.float32))

    fn_ref10 = lambda: triton_hgrn2_fused_bwd_scan_pytorch_ref(x10, g10)
    fn_tri10 = lambda: triton_hgrn2_fused_bwd_scan(x10, g10)

    lat_ref, vram_ref = benchmark_fn(fn_ref10)
    lat_tri, vram_tri = benchmark_fn(fn_tri10)
    cos_sim, rel_err = compute_metrics(fn_ref10(), fn_tri10())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 10,
            "kernel": "triton_hgrn2_fused_bwd_scan",
            "target_module": "hgrn2_attention, mamba_block",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 11/13] triton_spectral_entropy_svd_gate")
    M, K, R_MAX, N = 1024, 2048, 32, 2048
    eff_r = 16
    x11 = torch.randn(M, K, device=device, dtype=torch.float16)
    V11 = torch.randn(K, R_MAX, device=device, dtype=torch.float16)
    U11 = torch.randn(R_MAX, N, device=device, dtype=torch.float16)
    w11 = torch.rand(R_MAX, device=device, dtype=torch.float16)
    b11 = torch.randn(N, device=device, dtype=torch.float16)

    fn_ref11 = lambda: triton_spectral_entropy_svd_gate_pytorch_ref(
        x11, V11, U11, effective_rank=eff_r, s_weights=w11, bias=b11
    )
    fn_tri11 = lambda: triton_spectral_entropy_svd_gate(x11, V11, U11, effective_rank=eff_r, s_weights=w11, bias=b11)

    lat_ref, vram_ref = benchmark_fn(fn_ref11)
    lat_tri, vram_tri = benchmark_fn(fn_tri11)
    cos_sim, rel_err = compute_metrics(fn_ref11(), fn_tri11())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 11,
            "kernel": "triton_spectral_entropy_svd_gate",
            "target_module": "entropy_budget_allocation",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 12/13] triton_fused_act_scale_quant_gemm")
    M, K, N = 128, 768, 768
    x12 = torch.randn(M, K, device=device, dtype=torch.float16)
    s12 = torch.rand(K, device=device, dtype=torch.float16) + 0.5
    w12 = torch.randn(N, K, device=device, dtype=torch.float16)
    b12 = torch.randn(N, device=device, dtype=torch.float16)

    fn_ref12 = lambda: triton_fused_act_scale_quant_gemm_pytorch_ref(x12, s12, w12, bias=b12)
    fn_tri12 = lambda: triton_fused_act_scale_quant_gemm(x12, s12, w12, bias=b12)

    lat_ref, vram_ref = benchmark_fn(fn_ref12)
    lat_tri, vram_tri = benchmark_fn(fn_tri12)
    cos_sim, rel_err = compute_metrics(fn_ref12(), fn_tri12())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 12,
            "kernel": "triton_fused_act_scale_quant_gemm",
            "target_module": "smoothquant, faq_quantizer",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    print("\n[Kernel 13/13] triton_fused_latent_kv_cache_attention")
    B, H, L_Q, L_KV, R, D_K, D_V = 2, 8, 64, 256, 32, 64, 64
    q13 = torch.randn(B, H, L_Q, D_K, device=device, dtype=torch.float16)
    c13 = torch.randn(B, L_KV, R, device=device, dtype=torch.float16)
    uk13 = torch.randn(D_K, R, device=device, dtype=torch.float16) * 0.1
    uv13 = torch.randn(R, D_V, device=device, dtype=torch.float16) * 0.1

    fn_ref13 = lambda: triton_fused_latent_kv_cache_attention_pytorch_ref(q13, c13, uk13, uv13)
    fn_tri13 = lambda: triton_fused_latent_kv_cache_attention(q13, c13, uk13, uv13)

    lat_ref, vram_ref = benchmark_fn(fn_ref13)
    lat_tri, vram_tri = benchmark_fn(fn_tri13)
    cos_sim, rel_err = compute_metrics(fn_ref13(), fn_tri13())
    speedup = lat_ref / max(lat_tri, 1e-6)
    print(
        f"  PyTorch: {lat_ref:.3f} ms, {vram_ref:.2f} MB | Triton: {lat_tri:.3f} ms, {vram_tri:.2f} MB | Speedup: {speedup:.2f}x"
    )

    results.append(
        {
            "id": 13,
            "kernel": "triton_fused_latent_kv_cache_attention",
            "target_module": "svd_linear_attention, mla_attention",
            "pytorch_ms": lat_ref,
            "triton_ms": lat_tri,
            "speedup": speedup,
            "pytorch_vram_mb": vram_ref,
            "triton_vram_mb": vram_tri,
            "cosine_sim": cos_sim,
            "rel_error": rel_err,
        }
    )

    os.makedirs("results", exist_ok=True)
    json_path = "results/triton_13_flagship_kernels_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = "results/triton_13_flagship_kernels_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"SAVED BENCHMARK RESULTS TO {json_path} AND {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
