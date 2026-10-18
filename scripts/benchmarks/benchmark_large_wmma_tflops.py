"""Benchmark large wmma tflops module.

Author: Taha Majlesi
"""

from __future__ import annotations
import sys
import time
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from neuralops.core.cuda_kernels import force_kernel_backend
from neuralops.core.cuda_kernels.quantized_gemm import quant_autograd as m_quant
from neuralops.core.cuda_kernels.ternary_gemm import ternary_autograd as m_ternary
from neuralops.core.cuda_kernels.lut_gemm import lut_autograd as m_lut
from neuralops.core.cuda_kernels.low_rank_svd import lowrank_autograd as m_lowrank


def bench_fn(fn, warmup=10, iters=30):
    """Execute bench fn routine."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def run_large_wmma_benchmarks():
    """Execute run large wmma benchmarks routine."""
    shapes = [
        (512, 1024, 1024),
        (1024, 2048, 2048),
        (1024, 2048, 4096),
        (2048, 2048, 2048),
    ]

    results = []
    print("=" * 110)
    header = (
        f"{chr(37)}-18s | {chr(37)}-18s | {chr(37)}-6s | {chr(37)}-12s | {chr(37)}-10s | {chr(37)}-9s | {chr(37)}-8s"
        % ("Kernel", "Shape (M,K,N)", "Dtype", "Latency (ms)", "TFLOPS", "Zero NaNs", "Diff")
    )
    print(header)
    print("=" * 110)

    # 1. Quantized FP8 GEMM (SM 8.9 Tensor Core Accelerated)
    for M, K, N in shapes:
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(M, K, device="cuda", dtype=dtype)
            w = torch.randn(K, N, device="cuda", dtype=dtype)
            sx = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            sw = torch.tensor(1.0, device="cuda", dtype=torch.float32)

            with force_kernel_backend("cuda"):
                out_cuda = m_quant.fp8_gemm(x, w, sx, sw)
            out_ref = m_quant.fp8_gemm_pytorch_ref(x, w, sx, sw)
            diff = (out_cuda.float() - out_ref.float()).abs().max().item()
            zero_nans = not (torch.isnan(out_cuda).any().item() or torch.isinf(out_cuda).any().item())

            with force_kernel_backend("cuda"):
                lat_ms = bench_fn(lambda: m_quant.fp8_gemm(x, w, sx, sw))

            flops = 2.0 * M * K * N
            tflops = (flops / (lat_ms * 1e-3)) / 1e12

            dt_str = "FP16" if dtype == torch.float16 else "BF16"
            shape_str = f"({M},{K},{N})"
            row_str = (
                f"{chr(37)}-18s | {chr(37)}-18s | {chr(37)}-6s | {lat_ms:<12.4f} | {tflops:<10.2f} | {str(zero_nans):<9} | {diff:<8.5f}"
                % ("quantized_fp8", shape_str, dt_str)
            )
            print(row_str)
            results.append(
                {
                    "kernel": "quantized_fp8",
                    "shape": shape_str,
                    "dtype": dt_str,
                    "latency_ms": round(lat_ms, 4),
                    "tflops": round(tflops, 2),
                    "zero_nans": zero_nans,
                    "diff": float(diff),
                }
            )

    # 2. Ternary GEMM (1.58-bit packed weights)
    for M, K, N in shapes:
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
            w_packed = torch.randint(0, 256, (N, K // 4), device="cuda", dtype=torch.uint8)
            scales = torch.randn(N, device="cuda", dtype=dtype) * 0.1

            with force_kernel_backend("cuda"):
                out_cuda = m_ternary.ternary_gemm(x, w_packed, scales)
            out_ref = m_ternary.ternary_gemm_pytorch_ref(x, w_packed, scales)
            diff = (out_cuda.float() - out_ref.float()).abs().max().item()
            zero_nans = not (torch.isnan(out_cuda).any().item() or torch.isinf(out_cuda).any().item())

            with force_kernel_backend("cuda"):
                lat_ms = bench_fn(lambda: m_ternary.ternary_gemm(x, w_packed, scales))

            flops = 2.0 * M * K * N
            effective_tflops = (flops / (lat_ms * 1e-3)) / 1e12

            dt_str = "FP16" if dtype == torch.float16 else "BF16"
            shape_str = f"({M},{K},{N})"
            row_str = (
                f"{chr(37)}-18s | {chr(37)}-18s | {chr(37)}-6s | {lat_ms:<12.4f} | {effective_tflops:<10.2f} | {str(zero_nans):<9} | {diff:<8.5f}"
                % ("ternary_gemm", shape_str, dt_str)
            )
            print(row_str)
            results.append(
                {
                    "kernel": "ternary_gemm",
                    "shape": shape_str,
                    "dtype": dt_str,
                    "latency_ms": round(lat_ms, 4),
                    "tflops": round(effective_tflops, 2),
                    "zero_nans": zero_nans,
                    "diff": float(diff),
                }
            )

    # 3. LUT Codebook Dequantization GEMM
    for M, K, N in shapes:
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
            indices_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
            codebook = torch.randn(max(1, K // 64), 16, device="cuda", dtype=dtype) * 0.1

            with force_kernel_backend("cuda"):
                out_cuda = m_lut.lut_codebook_dequant_gemm(x, indices_packed, codebook, group_size=64)
            out_ref = m_lut.lut_codebook_dequant_gemm_pytorch_ref(x, indices_packed, codebook, group_size=64)
            diff = (out_cuda.float() - out_ref.float()).abs().max().item()
            zero_nans = not (torch.isnan(out_cuda).any().item() or torch.isinf(out_cuda).any().item())

            with force_kernel_backend("cuda"):
                lat_ms = bench_fn(lambda: m_lut.lut_codebook_dequant_gemm(x, indices_packed, codebook, group_size=64))

            flops = 2.0 * M * K * N
            effective_tflops = (flops / (lat_ms * 1e-3)) / 1e12

            dt_str = "FP16" if dtype == torch.float16 else "BF16"
            shape_str = f"({M},{K},{N})"
            row_str = (
                f"{chr(37)}-18s | {chr(37)}-18s | {chr(37)}-6s | {lat_ms:<12.4f} | {effective_tflops:<10.2f} | {str(zero_nans):<9} | {diff:<8.5f}"
                % ("lut_gemm", shape_str, dt_str)
            )
            print(row_str)
            results.append(
                {
                    "kernel": "lut_gemm",
                    "shape": shape_str,
                    "dtype": dt_str,
                    "latency_ms": round(lat_ms, 4),
                    "tflops": round(effective_tflops, 2),
                    "zero_nans": zero_nans,
                    "diff": float(diff),
                }
            )

    # 4. Low-Rank SVD Gated GEMM (x @ V_r.t() @ U_r.t())
    # x: [M, K], V_r: [R, K] -> z: [M, R], U_r: [N, R] -> y: [M, N]
    for M, K, N in shapes:
        R = 64
        for dtype in [torch.float16, torch.bfloat16]:
            x = torch.randn(M, K, device="cuda", dtype=dtype)
            v_r = torch.randn(R, K, device="cuda", dtype=dtype)
            u_r = torch.randn(N, R, device="cuda", dtype=dtype)

            with force_kernel_backend("cuda"):
                out_cuda = m_lowrank.fused_lowrank_gated_gemm(x, v_r, u_r)
            out_ref = m_lowrank.fused_lowrank_gated_gemm_pytorch_ref(x, v_r, u_r)
            diff = (out_cuda.float() - out_ref.float()).abs().max().item()
            zero_nans = not (torch.isnan(out_cuda).any().item() or torch.isinf(out_cuda).any().item())

            with force_kernel_backend("cuda"):
                lat_ms = bench_fn(lambda: m_lowrank.fused_lowrank_gated_gemm(x, v_r, u_r))

            # Low-rank FLOPs: 2*M*K*R + 2*M*R*N
            flops = 2.0 * M * K * R + 2.0 * M * R * N
            tflops = (flops / (lat_ms * 1e-3)) / 1e12

            dt_str = "FP16" if dtype == torch.float16 else "BF16"
            shape_str = f"({M},{K},{N},r={R})"
            row_str = (
                f"{chr(37)}-18s | {chr(37)}-18s | {chr(37)}-6s | {lat_ms:<12.4f} | {tflops:<10.2f} | {str(zero_nans):<9} | {diff:<8.5f}"
                % ("low_rank_svd", shape_str, dt_str)
            )
            print(row_str)
            results.append(
                {
                    "kernel": "low_rank_svd",
                    "shape": shape_str,
                    "dtype": dt_str,
                    "latency_ms": round(lat_ms, 4),
                    "tflops": round(tflops, 2),
                    "zero_nans": zero_nans,
                    "diff": float(diff),
                }
            )

    print("=" * 110)

    with open("/mnt/d/SVD_linear_Attention/results/benchmark_large_wmma_tflops.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/benchmark_large_wmma_tflops.json")


if __name__ == "__main__":
    run_large_wmma_benchmarks()
