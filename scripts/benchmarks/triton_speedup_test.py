"""
Standalone Benchmark & Profiling Script for High-ROI Triton Kernels (Phase 4).

Validates:
1. Numerical parity between Triton and vanilla PyTorch references (atol=1e-3, rtol=1e-3).
2. Latency measurement using torch.cuda.Event across 100 iterations (10 warmup iterations).
3. Maximum VRAM allocation (SRAM vs HBM savings).
4. Speedup factor on NVIDIA RTX 4070 (SM 8.9).

Shapes simulated: Standard LLM layer (B=8, seq_len=1024, in_features=768, out_features=768, rank=32).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from neuralops.kernels.triton.fused_lowrank_gemm import triton_fused_lowrank_gated_gemm
from neuralops.kernels.triton.rpca_sparse_gemm import triton_fused_lowrank_sparse_gemm
from neuralops.kernels.triton.strided_tt import (
    triton_strided_tt_contraction,
    triton_strided_tt_linear,
)


def measure_latency_and_memory(
    fn: Callable[[], torch.Tensor],
    warmup: int = 10,
    iters: int = 100,
) -> Tuple[float, float]:
    """
    Measures mean execution time (ms) using CUDA events and peak allocated VRAM (MB).
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Reset peak memory before benchmark run
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return elapsed_ms, peak_vram_mb


def benchmark_kernel1_lowrank_gated_gemm(
    B: int = 8,
    seq_len: int = 1024,
    in_features: int = 768,
    out_features: int = 768,
    rank: int = 32,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    print(f"\n{'='*70}\n[Benchmark 1] triton_fused_lowrank_gated_gemm (Target: Low-Rank SVD)\n{'='*70}")
    print(f"Config: B={B}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}, rank={rank}, dtype={dtype}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    X = torch.randn(B, seq_len, in_features, device=device, dtype=dtype)
    V_r = torch.randn(rank, in_features, device=device, dtype=dtype)
    U_r = torch.randn(out_features, rank, device=device, dtype=dtype)
    alpha = torch.randn(rank, device=device, dtype=dtype)
    b_drift = torch.randn(out_features, device=device, dtype=dtype)

    # 1. Vanilla PyTorch reference
    def pytorch_ref():
        # Intermediate Z materialized in HBM
        Z = torch.matmul(X, V_r.t())
        Z_gated = Z * torch.sigmoid(alpha).unsqueeze(0).unsqueeze(0)
        Y = torch.matmul(Z_gated, U_r.t()) + b_drift
        return Y

    # 2. Triton fused implementation
    def triton_fn():
        return triton_fused_lowrank_gated_gemm(X, V_r, U_r, alpha=alpha, b_drift=b_drift)

    out_ref = pytorch_ref()
    out_triton = triton_fn()

    # Parity verification
    max_diff = (out_ref - out_triton).abs().max().item()
    is_close = torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3)
    print(f"Numerical Parity: {'PASSED' if is_close else 'FAILED'} (Max Diff: {max_diff:.6f}, atol=1e-3, rtol=1e-3)")
    assert is_close, f"Kernel 1 output mismatch! Max diff: {max_diff}"

    t_pytorch, mem_pytorch = measure_latency_and_memory(pytorch_ref)
    t_triton, mem_triton = measure_latency_and_memory(triton_fn)
    speedup = t_pytorch / max(1e-5, t_triton)
    mem_saving_pct = ((mem_pytorch - mem_triton) / max(1e-5, mem_pytorch)) * 100

    print(f"PyTorch Latency : {t_pytorch:.4f} ms | Peak VRAM : {mem_pytorch:.2f} MB")
    print(f"Triton Latency  : {t_triton:.4f} ms | Peak VRAM : {mem_triton:.2f} MB")
    print(f"Speedup         : {speedup:.2f}x | VRAM Saved : {mem_saving_pct:.1f}%")

    return {
        "kernel": "triton_fused_lowrank_gated_gemm",
        "pytorch_ms": round(t_pytorch, 4),
        "triton_ms": round(t_triton, 4),
        "speedup": round(speedup, 2),
        "pytorch_vram_mb": round(mem_pytorch, 2),
        "triton_vram_mb": round(mem_triton, 2),
        "vram_saving_pct": round(mem_saving_pct, 1),
        "parity_passed": is_close,
    }


def benchmark_kernel2_rpca_sparse_gemm(
    B: int = 8,
    seq_len: int = 1024,
    in_features: int = 768,
    out_features: int = 768,
    rank: int = 32,
    sparsity: float = 0.05,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    print(f"\n{'='*70}\n[Benchmark 2] triton_fused_lowrank_sparse_gemm (Target: RPCA)\n{'='*70}")
    print(f"Config: B={B}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}, rank={rank}, sparsity={sparsity*100:.1f}%, dtype={dtype}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    X = torch.randn(B, seq_len, in_features, device=device, dtype=dtype)
    V_r = torch.randn(rank, in_features, device=device, dtype=dtype)
    U_r = torch.randn(out_features, rank, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)

    # Generate sparse matrix S
    raw_s = torch.randn(out_features, in_features, device=device, dtype=dtype)
    mask = torch.rand(out_features, in_features, device=device) < sparsity
    S_sparse = raw_s * mask
    S_csr = S_sparse.to_sparse_csr()

    def pytorch_ref():
        Y_lr = torch.matmul(torch.matmul(X, V_r.t()), U_r.t())
        Y_sp = torch.matmul(X, S_sparse.t())
        return Y_lr + Y_sp + bias

    def triton_fn():
        return triton_fused_lowrank_sparse_gemm(X, V_r, U_r, s=S_csr, bias=bias)

    out_ref = pytorch_ref()
    out_triton = triton_fn()

    max_diff = (out_ref - out_triton).abs().max().item()
    is_close = torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3)
    print(f"Numerical Parity: {'PASSED' if is_close else 'FAILED'} (Max Diff: {max_diff:.6f}, atol=1e-3, rtol=1e-3)")
    assert is_close, f"Kernel 2 output mismatch! Max diff: {max_diff}"

    t_pytorch, mem_pytorch = measure_latency_and_memory(pytorch_ref)
    t_triton, mem_triton = measure_latency_and_memory(triton_fn)
    speedup = t_pytorch / max(1e-5, t_triton)
    mem_saving_pct = ((mem_pytorch - mem_triton) / max(1e-5, mem_pytorch)) * 100

    print(f"PyTorch Latency : {t_pytorch:.4f} ms | Peak VRAM : {mem_pytorch:.2f} MB")
    print(f"Triton Latency  : {t_triton:.4f} ms | Peak VRAM : {mem_triton:.2f} MB")
    print(f"Speedup         : {speedup:.2f}x | VRAM Saved : {mem_saving_pct:.1f}%")

    return {
        "kernel": "triton_fused_lowrank_sparse_gemm",
        "pytorch_ms": round(t_pytorch, 4),
        "triton_ms": round(t_triton, 4),
        "speedup": round(speedup, 2),
        "pytorch_vram_mb": round(mem_pytorch, 2),
        "triton_vram_mb": round(mem_triton, 2),
        "vram_saving_pct": round(mem_saving_pct, 1),
        "parity_passed": is_close,
    }


def benchmark_kernel3_strided_tt(
    B: int = 8,
    seq_len: int = 1024,
    in_features: int = 768,
    out_features: int = 768,
    tt_rank: int = 16,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    print(f"\n{'='*70}\n[Benchmark 3] triton_strided_tt_contraction (Target: Tensor-Train)\n{'='*70}")
    print(f"Config: B={B}, seq_len={seq_len}, in_features={in_features}, out_features={out_features}, tt_rank={tt_rank}, dtype={dtype}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Balanced 2-factor split: 768 = 24 * 32
    p_dims = (24, 32)
    q_dims = (24, 32)

    core0 = torch.randn(p_dims[0], q_dims[0], tt_rank, device=device, dtype=dtype)
    core1 = torch.randn(tt_rank, p_dims[1], q_dims[1], device=device, dtype=dtype)
    X = torch.randn(B, seq_len, in_features, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)

    # 1. Vanilla PyTorch contraction (forcing .permute().contiguous().view())
    def pytorch_reconstruct():
        res = torch.tensordot(core0, core1, dims=([-1], [0]))  # (p0, q0, p1, q1)
        res = res.permute(1, 3, 0, 2).contiguous()              # forces VRAM alloc and copy
        W = res.view(out_features, in_features)
        return W

    def pytorch_linear():
        W = pytorch_reconstruct()
        return F.linear(X, W, bias)

    # 2. Triton zero-copy contraction
    def triton_reconstruct():
        return triton_strided_tt_contraction(core0, core1, p_dims, q_dims)

    def triton_linear():
        return triton_strided_tt_linear(X, core0, core1, p_dims, q_dims, bias=bias)

    # Parity check on reconstructed weight matrix
    W_ref = pytorch_reconstruct()
    W_triton = triton_reconstruct()
    w_diff = (W_ref - W_triton).abs().max().item()
    is_w_close = torch.allclose(W_triton, W_ref, atol=1e-3, rtol=1e-3)
    print(f"Weight Reconstruct Parity: {'PASSED' if is_w_close else 'FAILED'} (Max Diff: {w_diff:.6f})")
    assert is_w_close, f"TT Reconstructed weight mismatch! Max diff: {w_diff}"

    # Parity check on linear layer forward
    out_ref = pytorch_linear()
    out_triton = triton_linear()
    out_diff = (out_ref - out_triton).abs().max().item()
    is_out_close = torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3)
    print(f"Linear Forward Parity    : {'PASSED' if is_out_close else 'FAILED'} (Max Diff: {out_diff:.6f})")
    assert is_out_close, f"TT Linear forward mismatch! Max diff: {out_diff}"

    t_pytorch_rec, mem_pytorch_rec = measure_latency_and_memory(pytorch_reconstruct)
    t_triton_rec, mem_triton_rec = measure_latency_and_memory(triton_reconstruct)
    speedup_rec = t_pytorch_rec / max(1e-5, t_triton_rec)
    mem_saving_pct = ((mem_pytorch_rec - mem_triton_rec) / max(1e-5, mem_pytorch_rec)) * 100

    print(f"PyTorch TT Contraction : {t_pytorch_rec:.4f} ms | Peak VRAM : {mem_pytorch_rec:.2f} MB")
    print(f"Triton Strided TT      : {t_triton_rec:.4f} ms | Peak VRAM : {mem_triton_rec:.2f} MB")
    print(f"Speedup (Contraction)  : {speedup_rec:.2f}x | VRAM Saved : {mem_saving_pct:.1f}%")

    t_pytorch_lin, mem_pytorch_lin = measure_latency_and_memory(pytorch_linear)
    t_triton_lin, mem_triton_lin = measure_latency_and_memory(triton_linear)
    speedup_lin = t_pytorch_lin / max(1e-5, t_triton_lin)

    print(f"End-to-End Linear Layer: PyTorch = {t_pytorch_lin:.4f} ms vs Triton = {t_triton_lin:.4f} ms (Speedup: {speedup_lin:.2f}x)")

    return {
        "kernel": "triton_strided_tt_contraction",
        "pytorch_ms": round(t_pytorch_rec, 4),
        "triton_ms": round(t_triton_rec, 4),
        "speedup": round(speedup_rec, 2),
        "linear_speedup": round(speedup_lin, 2),
        "pytorch_vram_mb": round(mem_pytorch_rec, 2),
        "triton_vram_mb": round(mem_triton_rec, 2),
        "vram_saving_pct": round(mem_saving_pct, 1),
        "parity_passed": is_w_close and is_out_close,
    }


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA device is required to run Triton benchmarks.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"\nRunning High-ROI Triton Kernel Suite on {device_name}")
    print(f"CUDA Version: {torch.version.cuda} | PyTorch: {torch.__version__}")

    results = []
    res1 = benchmark_kernel1_lowrank_gated_gemm()
    results.append(res1)

    res2 = benchmark_kernel2_rpca_sparse_gemm()
    results.append(res2)

    res3 = benchmark_kernel3_strided_tt()
    results.append(res3)

    output_path = Path("results/triton_speedup_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY BENCHMARK REPORT\n{'='*70}")
    print(f"{'Kernel':<35} | {'PyTorch (ms)':<12} | {'Triton (ms)':<11} | {'Speedup':<8} | {'VRAM Saved'}")
    print("-" * 80)
    for r in results:
        print(f"{r['kernel']:<35} | {r['pytorch_ms']:<12.4f} | {r['triton_ms']:<11.4f} | {r['speedup']:<7.2f}x | {r['vram_saving_pct']:.1f}%")
    print("=" * 80)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
