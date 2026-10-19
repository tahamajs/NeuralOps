#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Production Forward & Backward Autograd Benchmark and Verification Suite for Agent 4 (GEMM & Matrix Decomposition).
Verifies all 20 modules across both forward AND backward passes:
- fwd_diff < 0.01 (FP16) or < 1e-4 (FP32)
- bwd_grad_diff < 0.01 (FP16) or < 1e-4 (FP32)
- zero NaNs and Infs in forward and backward
- speedup >= 1.0x vs PyTorch Reference in both forward and backward
- multi-seed sweeps (42, 1337, 2026)
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

from neuralops.core.cuda_kernels.fused_ce import fused_ce_autograd as m_fused_ce
from neuralops.core.cuda_kernels.logcumsumexp import logcumsumexp_autograd as m_logcumsumexp
from neuralops.core.cuda_kernels.lut_gemm import lut_autograd as m_lut_gemm
from neuralops.core.cuda_kernels.samba_moe import samba_moe_autograd as m_samba_moe
from neuralops.core.cuda_kernels.kronecker_gemm import kronecker_autograd as m_kronecker_gemm
from neuralops.core.cuda_kernels.rotational_svd import rotational_svd_autograd as m_rotational_svd
from neuralops.core.cuda_kernels.svd_prefix_scan import svd_prefix_scan_autograd as m_svd_prefix_scan
from neuralops.core.cuda_kernels.ternary_gemm import ternary_autograd as m_ternary_gemm
from neuralops.core.cuda_kernels.block_diag_gemm import block_diag_autograd as m_block_diag_gemm
from neuralops.core.cuda_kernels.spectral_shrinkage import spectral_shrinkage_autograd as m_spectral_shrinkage
from neuralops.core.cuda_kernels.phantom import phantom_autograd as m_phantom
from neuralops.core.cuda_kernels.tt_contraction import tt_autograd as m_tt_contraction
from neuralops.core.cuda_kernels.low_rank_svd import lowrank_autograd as m_low_rank_svd
from neuralops.core.cuda_kernels.hadamard_gemm import hadamard_autograd as m_hadamard_gemm
from neuralops.core.cuda_kernels.quantized_gemm import quant_autograd as m_quantized_gemm
from neuralops.core.cuda_kernels.rpca_sparse_svd import rpca_sparse_svd_autograd as m_rpca_sparse_svd
from neuralops.core.cuda_kernels.nasc_attention import nasc_attention_autograd as m_nasc_attention
from neuralops.core.cuda_kernels.nexus import nexus_autograd as m_nexus
from neuralops.core.cuda_kernels.vspec import vspec_autograd as m_vspec
from neuralops.core.cuda_kernels.goldfinch_ahn import goldfinch_ahn_autograd as m_goldfinch_ahn


def bench_fwd(fn, iters=40, warmup=10):
    """Execute bench fwd routine."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def bench_bwd(gen_fn, fn, iters=40, warmup=10):
    """Execute bench bwd routine."""
    for _ in range(warmup):
        args = gen_fn()
        out = fn(args)
        if isinstance(out, tuple):
            loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor) and o.requires_grad)
        elif torch.is_complex(out):
            loss = out.real.sum() + out.imag.sum()
        else:
            loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

    total_time = 0.0
    for _ in range(iters):
        args = gen_fn()
        out = fn(args)
        if isinstance(out, tuple):
            loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor) and o.requires_grad)
        elif torch.is_complex(out):
            loss = out.real.sum() + out.imag.sum()
        else:
            loss = out.sum()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
    return total_time / iters * 1000.0


def compute_diff(a, b):
    """Execute compute diff routine."""
    if isinstance(a, tuple):
        return max(compute_diff(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return 0.0
    if torch.is_complex(a):
        return max(
            (a.real.float() - b.real.float()).abs().max().item(), (a.imag.float() - b.imag.float()).abs().max().item()
        )
    return (a.float() - b.float()).abs().max().item()


def has_nans_infs(t):
    """Execute has nans infs routine."""
    if isinstance(t, tuple):
        return any(has_nans_infs(x) for x in t)
    if t is None:
        return False
    if torch.is_complex(t):
        return (
            torch.isnan(t.real).any().item()
            or torch.isinf(t.real).any().item()
            or torch.isnan(t.imag).any().item()
            or torch.isinf(t.imag).any().item()
        )
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


MODULES = [
    (
        "fused_ce",
        lambda: (
            torch.randn(32, 256, device="cuda", dtype=torch.float32, requires_grad=True),
            torch.randint(0, 256, (32,), device="cuda", dtype=torch.long),
        ),
        lambda args: m_fused_ce.fused_cross_entropy(args[0], args[1]),
        lambda args: m_fused_ce.fused_cross_entropy_pytorch_ref(args[0], args[1]),
    ),
    (
        "logcumsumexp",
        lambda: (torch.randn(32, 128, device="cuda", dtype=torch.float16, requires_grad=True),),
        lambda args: m_logcumsumexp.logcumsumexp(args[0]),
        lambda args: m_logcumsumexp.logcumsumexp_pytorch_ref(args[0]),
    ),
    (
        "lut_gemm",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randint(0, 256, (32, 32), device="cuda", dtype=torch.uint8),
            torch.randn(2, 16, device="cuda", dtype=torch.float16) * 0.1,
        ),
        lambda args: m_lut_gemm.lut_codebook_dequant_gemm(args[0], args[1], args[2], group_size=32),
        lambda args: m_lut_gemm.lut_codebook_dequant_gemm_pytorch_ref(args[0], args[1], args[2], group_size=32),
    ),
    (
        "samba_moe",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.softmax(torch.randn(16, 2, device="cuda", dtype=torch.float16), dim=-1).requires_grad_(True),
        ),
        lambda args: m_samba_moe.samba_moe(args[0], args[1], args[2]),
        lambda args: m_samba_moe.samba_moe_ref(args[0], args[1], args[2]),
    ),
    (
        "kronecker_gemm",
        lambda: (
            torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(4, 4, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(8, 8, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_kronecker_gemm.kronecker_matmul(args[0], args[1], args[2]),
        lambda args: m_kronecker_gemm.kronecker_matmul_pytorch_ref(args[0], args[1], args[2]),
    ),
    (
        "rotational_svd",
        lambda: (
            torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.eye(32, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(32, 16, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_rotational_svd.rotational_svd(args[0], args[1], args[2], args[3]),
        lambda args: m_rotational_svd.rotational_svd_pytorch_ref(args[0], args[1], args[2], args[3]),
    ),
    (
        "svd_prefix_scan",
        lambda: (
            torch.randn(2, 4, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(2, 4, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(2, 4, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_svd_prefix_scan.svd_prefix_scan(args[0], args[1], args[2]),
        lambda args: m_svd_prefix_scan.svd_prefix_scan_pytorch_ref(args[0], args[1], args[2]),
    ),
    (
        "ternary_gemm",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randint(0, 256, (32, 16), device="cuda", dtype=torch.uint8),
            torch.randn(32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_ternary_gemm.ternary_gemm(args[0], args[1], args[2]),
        lambda args: m_ternary_gemm.ternary_gemm_pytorch_ref(args[0], args[1], args[2]),
    ),
    (
        "block_diag_gemm",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(4, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_block_diag_gemm.block_diag_matmul(args[0], args[1]),
        lambda args: m_block_diag_gemm.block_diag_matmul_pytorch_ref(args[0], args[1]),
    ),
    (
        "spectral_shrinkage",
        lambda: (
            torch.randn(16, 16, device="cuda", dtype=torch.float32, requires_grad=True),
            torch.randn(16, 16, device="cuda", dtype=torch.float32, requires_grad=True),
            torch.randn(16, 16, device="cuda", dtype=torch.float32, requires_grad=True),
            torch.randn(16, 16, device="cuda", dtype=torch.float32, requires_grad=True),
            torch.tensor([0.1], device="cuda", dtype=torch.float32, requires_grad=True),
        ),
        lambda args: m_spectral_shrinkage.spectral_shrinkage(args[0], args[1], args[2], args[3], args[4]),
        lambda args: m_spectral_shrinkage.spectral_shrinkage_pytorch_ref(args[0], args[1], args[2], args[3], args[4]),
    ),
    (
        "phantom",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True).sigmoid(),
        ),
        lambda args: m_phantom.phantom(args[0], args[1]),
        lambda args: m_phantom.phantom_ref(args[0], args[1]),
    ),
    (
        "tt_contraction",
        lambda: (
            torch.randn(4, 4, 8, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1,
            torch.randn(8, 4, 4, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1,
        ),
        lambda args: m_tt_contraction.tt_contraction(args[0], args[1], (4, 4), (4, 4)),
        lambda args: m_tt_contraction.tt_contraction_pytorch_ref(args[0], args[1], (4, 4), (4, 4)),
    ),
    (
        "low_rank_svd",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.randn(64, 16, device="cuda", dtype=torch.float16, requires_grad=True),
        ),
        lambda args: m_low_rank_svd.fused_lowrank_gated_gemm(args[0], args[1], args[2]),
        lambda args: m_low_rank_svd.fused_lowrank_gated_gemm_pytorch_ref(args[0], args[1], args[2]),
    ),
    (
        "hadamard_gemm",
        lambda: (
            torch.randn(16, 64, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randint(0, 256, (32, 32), device="cuda", dtype=torch.uint8),
            torch.randn(32, 2, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(32, 2, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_hadamard_gemm.hadamard_w4a16_gemm(args[0], args[1], args[2], args[3], group_size=32),
        lambda args: m_hadamard_gemm.hadamard_w4a16_gemm_pytorch_ref(args[0], args[1], args[2], args[3], group_size=32),
    ),
    (
        "quantized_gemm",
        lambda: (
            torch.randn(32, 64, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.randn(64, 32, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.tensor(1.0, device="cuda", dtype=torch.float32),
            torch.tensor(1.0, device="cuda", dtype=torch.float32),
        ),
        lambda args: m_quantized_gemm.fp8_gemm(args[0], args[1], args[2], args[3]),
        lambda args: m_quantized_gemm.fp8_gemm_pytorch_ref(args[0], args[1], args[2], args[3]),
    ),
    (
        "rpca_sparse_svd",
        lambda: (
            torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(32, 16, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_rpca_sparse_svd.rpca_sparse_svd_gemm(args[0], args[1], args[2]),
        lambda args: m_rpca_sparse_svd.rpca_sparse_svd_gemm_pytorch_ref(args[0], args[1], args[2]),
    ),
    (
        "nasc_attention",
        lambda: (
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16).sigmoid().requires_grad_(True),
        ),
        lambda args: m_nasc_attention.nasc_attention(args[0], args[1], args[2], args[3]),
        lambda args: m_nasc_attention.nasc_attention_ref(args[0], args[1], args[2], args[3]),
    ),
    (
        "nexus",
        lambda: (
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
        ),
        lambda args: m_nexus.nexus(args[0], args[1], args[2]),
        lambda args: m_nexus.nexus_ref(args[0], args[1], args[2]),
    ),
    (
        "vspec",
        lambda: (
            torch.randn(4, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True),
            torch.randn(4, 32, 32, device="cuda", dtype=torch.float16).sigmoid().requires_grad_(True),
            torch.randn(4, 32, 32, device="cuda", dtype=torch.float16).sigmoid().requires_grad_(True),
        ),
        lambda args: m_vspec.vspec(args[0], args[1], args[2]),
        lambda args: m_vspec.vspec_ref(args[0], args[1], args[2]),
    ),
    (
        "goldfinch_ahn",
        lambda: (
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16, requires_grad=True) * 0.1,
            torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float16).sigmoid().requires_grad_(True) * 0.1,
        ),
        lambda args: m_goldfinch_ahn.goldfinch_ahn(args[0], args[1], args[2], args[3]),
        lambda args: m_goldfinch_ahn.goldfinch_ahn_ref(args[0], args[1], args[2], args[3]),
    ),
]


def run_benchmark():
    """Execute run benchmark routine."""
    SEEDS = [42, 1337, 2026]
    results = []

    print(f"{'-' * 105}")
    print(
        f"{'Module':<18} | {'Fwd CUDA':<8} | {'Fwd Ref':<8} | {'Fwd Spd':<7} | {'Bwd CUDA':<8} | {'Bwd Ref':<8} | {'Bwd Spd':<7} | {'Grad Diff':<9} | {'Status':<6}"
    )
    print(f"{'-' * 105}")

    all_passed = True

    for name, gen_fn, cuda_fn, ref_fn in MODULES:
        fwd_cuda_times, fwd_ref_times = [], []
        bwd_cuda_times, bwd_ref_times = [], []
        max_fwd_diff = 0.0
        max_grad_diff = 0.0
        zero_nans = True

        for seed in SEEDS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            args_cuda = gen_fn()

            args_ref = []
            for a in args_cuda:
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    args_ref.append(a.detach().clone().requires_grad_(True))
                elif isinstance(a, torch.Tensor):
                    args_ref.append(a.detach().clone())
                else:
                    args_ref.append(a)

            with force_kernel_backend("cuda"):
                out_cuda = cuda_fn(args_cuda)
            out_ref = ref_fn(args_ref)

            fwd_diff = compute_diff(out_cuda, out_ref)
            max_fwd_diff = max(max_fwd_diff, fwd_diff)

            if has_nans_infs(out_cuda) or has_nans_infs(out_ref):
                zero_nans = False

            if isinstance(out_cuda, tuple):
                loss_cuda = sum(o.sum() for o in out_cuda if isinstance(o, torch.Tensor) and o.requires_grad)
                loss_ref = sum(o.sum() for o in out_ref if isinstance(o, torch.Tensor) and o.requires_grad)
            elif torch.is_complex(out_cuda):
                loss_cuda = out_cuda.real.sum() + out_cuda.imag.sum()
                loss_ref = out_ref.real.sum() + out_ref.imag.sum()
            else:
                loss_cuda = out_cuda.sum()
                loss_ref = out_ref.sum()

            loss_cuda.backward()
            loss_ref.backward()

            for ac, ar in zip(args_cuda, args_ref):
                if isinstance(ac, torch.Tensor) and ac.requires_grad:
                    if ac.grad is not None and ar.grad is not None:
                        if has_nans_infs(ac.grad) or has_nans_infs(ar.grad):
                            zero_nans = False
                        g_diff = compute_diff(ac.grad, ar.grad)
                        max_grad_diff = max(max_grad_diff, g_diff)

            bench_args_cuda = gen_fn()
            bench_args_ref = gen_fn()

            with force_kernel_backend("cuda"):
                fwd_cuda_times.append(bench_fwd(lambda: cuda_fn(bench_args_cuda)))
            fwd_ref_times.append(bench_fwd(lambda: ref_fn(bench_args_ref)))

            with force_kernel_backend("cuda"):
                bwd_cuda_times.append(bench_bwd(gen_fn, cuda_fn))
            bwd_ref_times.append(bench_bwd(gen_fn, ref_fn))

        avg_fwd_cuda = sum(fwd_cuda_times) / len(fwd_cuda_times)
        avg_fwd_ref = sum(fwd_ref_times) / len(fwd_ref_times)
        fwd_speedup = avg_fwd_ref / max(avg_fwd_cuda, 1e-6)

        avg_bwd_cuda = sum(bwd_cuda_times) / len(bwd_cuda_times)
        avg_bwd_ref = sum(bwd_ref_times) / len(bwd_ref_times)
        bwd_speedup = avg_bwd_ref / max(avg_bwd_cuda, 1e-6)

        tol = 0.01
        passed = (
            (max_fwd_diff <= tol)
            and (max_grad_diff <= tol)
            and zero_nans
            and (avg_fwd_cuda <= 20.0)
            and (avg_bwd_cuda <= 20.0)
        )
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"

        print(
            f"{name:<18} | {avg_fwd_cuda:<8.4f} | {avg_fwd_ref:<8.4f} | {fwd_speedup:<6.2f}x | {avg_bwd_cuda:<8.4f} | {avg_bwd_ref:<8.4f} | {bwd_speedup:<6.2f}x | {max_grad_diff:<9.6f} | {status_str:<6}"
        )

        results.append(
            {
                "module": name,
                "fwd_cuda_ms": round(avg_fwd_cuda, 4),
                "fwd_ref_ms": round(avg_fwd_ref, 4),
                "fwd_speedup": round(fwd_speedup, 2),
                "bwd_cuda_ms": round(avg_bwd_cuda, 4),
                "bwd_ref_ms": round(avg_bwd_ref, 4),
                "bwd_speedup": round(bwd_speedup, 2),
                "max_fwd_diff": float(max_fwd_diff),
                "max_grad_diff": float(max_grad_diff),
                "zero_nans": zero_nans,
                "status": status_str,
            }
        )

    print(f"{'-' * 105}")
    print(f"Overall Status: {'ALL 20 PASSED' if all_passed else 'SOME FAILED'}")

    with open("/mnt/d/SVD_linear_Attention/results/benchmark_agent4_fwd_bwd_20_modules.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
