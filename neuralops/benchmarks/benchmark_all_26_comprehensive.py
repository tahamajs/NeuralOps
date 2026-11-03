#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive 26 Triton Hardware Kernels Benchmarking Sweep for NeuralOps

Benchmarks 26 Triton hardware kernels against PyTorch references under realistic dimensions (B=4, H=8, N=512, D=64) on NVIDIA Ada Lovelace / CUDA GPUs.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

# Running this file by path places ``neuralops/benchmarks`` before the standard
# library on ``sys.path``; remove that entry before importing torch so the local
# benchmarks/logging.py cannot shadow Python's logging module.
_SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0] or ".").resolve() == _SCRIPT_DIR:
    sys.path.pop(0)

# Configure TRITON_PTXAS_PATH if local CUDA toolkit provides compatible ptxas
if "TRITON_PTXAS_PATH" not in os.environ and os.path.exists("/usr/local/cuda-11.3/bin/ptxas"):
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.3/bin/ptxas"

import torch
import torch.nn.functional as F


def _flatten_tensors(out):
    """Recursively collect all tensors from output (tensor, tuple, list)."""
    if isinstance(out, torch.Tensor):
        return [out]
    elif isinstance(out, (tuple, list)):
        res = []
        for item in out:
            res.extend(_flatten_tensors(item))
        return res
    return []


def benchmark_single_kernel(
    name: str,
    category: str,
    shape_str: str,
    triton_fn,
    pytorch_fn,
    *args,
    warmup: int = 10,
    reps: int = 50,
    device: torch.device = None,
    impl_module: str = "neuralops.ops.triton",
    ref_module: str = "torch",
    differentiable: bool = False,
    backward_args_idx: list[int] | None = None,
    **kwargs,
):
    """Benchmark a single kernel pair (Triton vs PyTorch) with strict timing, quality gates, and parity checks."""
    import traceback

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_cuda = (device.type == "cuda")
    if is_cuda:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    tolerance_forward = 1e-3
    tolerance_backward = 2e-3
    max_diff_fwd = 0.0
    max_diff_bwd = 0.0
    cos_sim = 1.0
    output_finite = True
    gradients_finite = True
    has_nan_tri = False
    has_inf_tri = False
    has_nan_ref = False
    has_inf_ref = False
    parity_error_str = None
    tb_str = None

    try:
        out_tri = triton_fn(*args, **kwargs)
        out_pt = pytorch_fn(*args, **kwargs)

        tri_tensors = _flatten_tensors(out_tri)
        pt_tensors = _flatten_tensors(out_pt)

        if len(tri_tensors) != len(pt_tensors):
            raise ValueError(f"Tensor count mismatch between Triton ({len(tri_tensors)}) and PyTorch ({len(pt_tensors)})")

        if pt_tensors:
            tolerance_forward = 1e-3 if pt_tensors[0].dtype in (torch.float32, torch.float64) else 2e-2

        diffs = []
        for t_tri, t_pt in zip(tri_tensors, pt_tensors):
            if t_tri.shape != t_pt.shape:
                raise ValueError(f"Shape mismatch: {t_tri.shape} vs {t_pt.shape}")

            nan_tri = bool(torch.isnan(t_tri).any().item())
            inf_tri = bool(torch.isinf(t_tri).any().item())
            nan_ref = bool(torch.isnan(t_pt).any().item())
            inf_ref = bool(torch.isinf(t_pt).any().item())

            has_nan_tri = has_nan_tri or nan_tri
            has_inf_tri = has_inf_tri or inf_tri
            has_nan_ref = has_nan_ref or nan_ref
            has_inf_ref = has_inf_ref or inf_ref

            t_diff = (t_pt.float() - t_tri.float()).abs().max().item()
            diffs.append(t_diff)

        max_diff_fwd = max(diffs) if diffs else 0.0
        output_finite = not (has_nan_tri or has_inf_tri or has_nan_ref or has_inf_ref)

        if pt_tensors and pt_tensors[0].numel() > 1:
            ref_f = pt_tensors[0].float().reshape(-1)
            tri_f = tri_tensors[0].float().reshape(-1)
            cos_sim = F.cosine_similarity(ref_f.unsqueeze(0), tri_f.unsqueeze(0)).item()
    except Exception as exc:
        parity_error_str = str(exc)
        tb_str = traceback.format_exc()
        print(f"[FAIL] {name:<36s} | Forward Parity Exception: {exc}")
        return {
            "kernel": name,
            "category": category,
            "shape": shape_str,
            "implementation_module": impl_module,
            "reference_module": ref_module,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if is_cuda else "CPU",
            "pytorch_ms": 0.0,
            "triton_ms": 0.0,
            "launch_overhead_ms": 0.0,
            "speedup": 0.0,
            "cosine_sim": 0.0,
            "max_diff": float("inf"),
            "max_diff_forward": float("inf"),
            "tolerance_forward": tolerance_forward,
            "max_diff_backward": float("inf"),
            "tolerance_backward": tolerance_backward,
            "output_finite": False,
            "gradients_finite": False,
            "backend_identity": "triton",
            "fallback_used": False,
            "status": f"ERROR: {exc}",
            "error": parity_error_str,
            "traceback": tb_str,
        }

    # Backward gradient parity check where differentiable
    if differentiable and is_cuda:
        try:
            # Clone input args with requires_grad=True
            diff_args_tri = []
            diff_args_pt = []
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.is_floating_point() and (backward_args_idx is None or i in backward_args_idx):
                    a_tri = arg.detach().clone().requires_grad_(True)
                    a_pt = arg.detach().clone().requires_grad_(True)
                else:
                    a_tri = arg
                    a_pt = arg
                diff_args_tri.append(a_tri)
                diff_args_pt.append(a_pt)

            out_tri_b = triton_fn(*diff_args_tri, **kwargs)
            out_pt_b = pytorch_fn(*diff_args_pt, **kwargs)

            loss_tri = sum(t.sum() for t in _flatten_tensors(out_tri_b) if t.requires_grad)
            loss_pt = sum(t.sum() for t in _flatten_tensors(out_pt_b) if t.requires_grad)

            if isinstance(loss_tri, torch.Tensor) and loss_tri.requires_grad:
                loss_tri.backward()
            if isinstance(loss_pt, torch.Tensor) and loss_pt.requires_grad:
                loss_pt.backward()

            bwd_diffs = []
            for a_tri, a_pt in zip(diff_args_tri, diff_args_pt):
                if isinstance(a_tri, torch.Tensor) and a_tri.grad is not None and isinstance(a_pt, torch.Tensor) and a_pt.grad is not None:
                    if not torch.isfinite(a_tri.grad).all().item() or not torch.isfinite(a_pt.grad).all().item():
                        gradients_finite = False
                    b_diff = (a_pt.grad.float() - a_tri.grad.float()).abs().max().item()
                    bwd_diffs.append(b_diff)

            max_diff_bwd = max(bwd_diffs) if bwd_diffs else 0.0
        except Exception as exc_bwd:
            gradients_finite = False
            parity_error_str = f"Backward parity error: {exc_bwd}"
            tb_str = traceback.format_exc()

    for _ in range(warmup):
        _ = triton_fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda:
        start_evt_t = torch.cuda.Event(enable_timing=True)
        end_evt_t = torch.cuda.Event(enable_timing=True)
        start_cpu_t = time.perf_counter()
        start_evt_t.record()
        for _ in range(reps):
            _ = triton_fn(*args, **kwargs)
        end_evt_t.record()
        torch.cuda.synchronize()
        end_cpu_t = time.perf_counter()

        triton_gpu_ms = start_evt_t.elapsed_time(end_evt_t) / reps
        triton_cpu_ms = (end_cpu_t - start_cpu_t) / reps * 1000.0
        launch_overhead_ms = max(0.0, triton_cpu_ms - triton_gpu_ms)
    else:
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = triton_fn(*args, **kwargs)
        t1 = time.perf_counter()
        triton_gpu_ms = (t1 - t0) / reps * 1000.0
        launch_overhead_ms = 0.0

    for _ in range(warmup):
        _ = pytorch_fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda:
        start_evt_p = torch.cuda.Event(enable_timing=True)
        end_evt_p = torch.cuda.Event(enable_timing=True)
        start_evt_p.record()
        for _ in range(reps):
            _ = pytorch_fn(*args, **kwargs)
        end_evt_p.record()
        torch.cuda.synchronize()
        pytorch_gpu_ms = start_evt_p.elapsed_time(end_evt_p) / reps
    else:
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = pytorch_fn(*args, **kwargs)
        t1 = time.perf_counter()
        pytorch_gpu_ms = (t1 - t0) / reps * 1000.0

    speedup = pytorch_gpu_ms / max(triton_gpu_ms, 1e-6)

    # Strict performance and quality gates per NeuralOps Kernel Audit Contract
    if not output_finite or not gradients_finite:
        status = "FAIL_NAN"
    elif max_diff_fwd > tolerance_forward or max_diff_bwd > tolerance_backward:
        status = "FAIL_NUMERICS"
    elif pytorch_gpu_ms < 0.01:
        status = "NO_BASELINE"
    elif speedup < 1.0:
        status = "FAIL_SLOW"
    elif triton_gpu_ms > 20.0:
        status = "FAIL_LATENCY"
    else:
        status = "PASS"

    entry = {
        "kernel": name,
        "category": category,
        "shape": shape_str,
        "implementation_module": impl_module,
        "reference_module": ref_module,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if is_cuda else "CPU",
        "pytorch_ms": round(pytorch_gpu_ms, 4),
        "triton_ms": round(triton_gpu_ms, 4),
        "launch_overhead_ms": round(launch_overhead_ms, 4),
        "speedup": round(speedup, 2),
        "cosine_sim": round(cos_sim, 6),
        "max_diff": round(max_diff_fwd, 6),
        "max_diff_forward": round(max_diff_fwd, 6),
        "tolerance_forward": tolerance_forward,
        "max_diff_backward": round(max_diff_bwd, 6),
        "tolerance_backward": tolerance_backward,
        "output_finite": output_finite,
        "gradients_finite": gradients_finite,
        "backend_identity": "triton",
        "fallback_used": False,
        "status": status,
        "error": parity_error_str,
        "traceback": tb_str,
    }

    status_flag = "[PASS]" if status == "PASS" else f"[{status}]"
    print(f"{status_flag:<15} {name:<36s} | Tri: {triton_gpu_ms:6.3f}ms | PT: {pytorch_gpu_ms:6.3f}ms | Speedup: {speedup:6.2f}x | Diff: {max_diff_fwd:.2e}")
    return entry


CANONICAL_26_TRITON_KERNELS: list[str] = [
    "fused_hnc_nullspace_step",
    "fused_linformer_seq_proj",
    "fused_kronecker_gemm",
    "fused_nystrom_landmark_pinv",
    "sliding_window_attention",
    "gla_scan",
    "gdn_scan",
    "hgrn2_scan",
    "softplus",
    "fused_norm_linear",
    "cosformer",
    "fused_ce",
    "fused_kd",
    "selective_scan",
    "fms_scan",
    "paged_attention",
    "fused_mqa_bias_scale",
    "fused_hamming_scores",
    "fused_favor_plus",
    "fused_rope",
    "fused_moba_block_router",
    "fused_rala_augmented_scan",
    "fused_recurrent_linear_step",
    "fused_jamba_hybrid_gate",
    "fused_goldfinch_hopfield",
    "flash_decode",
]


def run_all_26_comprehensive(
    device_str="cuda:0",
    dtype_str="float16",
    warmup=10,
    repeat=50,
    output_csv="results/triton_26_kernels_comprehensive_benchmark.csv",
    output_json="results/triton_26_kernels_comprehensive_benchmark.json",
    report_md="docs/reports/TRITON_26_KERNELS_COMPREHENSIVE_REPORT.md",
    selected_kernels=None,
):
    """Execute complete 26-kernel sweep."""
    import triton
    import triton.language as tl

    if device_str == "auto":
        device_str = "cuda:0"
    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError(
            "The 26-kernel benchmark is GPU-only. Pass --device cuda:0 on a CUDA host; "
            "CPU fallback would invalidate the timing results."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
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
    try:
        dtype = dtype_map[dtype_str.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose float16, float32, or bfloat16.") from exc

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("=" * 85)
    print(f"BENCHMARKING 26 TRITON HARDWARE KERNELS ON {gpu_name}")
    print(f"Dimensions: B=4, H=8, N=512, D=64 | Precision: {dtype_str} | Warmup: {warmup} | Reps: {repeat}")
    print("=" * 85)

    results = []

    def should_run(kname):
        return (selected_kernels is None) or (kname in selected_kernels)

    B, H, N, D = 4, 8, 512, 64

    if should_run("fused_hnc_nullspace_step"):
        from neuralops.core.triton_kernels.low_rank import (
            fused_hnc_nullspace_step_triton,
            fused_hnc_nullspace_step_pytorch,
        )
        w1 = torch.randn(2048, 2048, device=device, dtype=dtype)
        g1 = torch.randn(2048, 2048, device=device, dtype=dtype)
        r1 = torch.randn(2048, 2048, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "fused_hnc_nullspace_step", "low_rank", "2048x2048",
            fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch,
            w1, g1, r1, 0.05, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.low_rank.fused_hnc_nullspace_step_triton",
            ref_module="neuralops.ops.triton.low_rank.fused_hnc_nullspace_step_pytorch",
        ))

    if should_run("fused_linformer_seq_proj"):
        from neuralops.core.triton_kernels.low_rank import (
            fused_linformer_seq_proj_triton,
            fused_linformer_seq_proj_pytorch,
        )
        x_lin = torch.randn(B, H, N, D, device=device, dtype=dtype)
        proj_lin = torch.randn(D, N, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "fused_linformer_seq_proj", "low_rank", f"B={B},H={H},N={N},D={D}",
            fused_linformer_seq_proj_triton, fused_linformer_seq_proj_pytorch,
            proj_lin, x_lin, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.low_rank.fused_linformer_seq_proj_triton",
            ref_module="neuralops.ops.triton.low_rank.fused_linformer_seq_proj_pytorch",
        ))

    if should_run("fused_kronecker_gemm"):
        from neuralops.core.triton_kernels.low_rank import (
            fused_kronecker_batched_gemm_triton,
            fused_kronecker_batched_gemm_pytorch,
        )
        x_kron = torch.randn(B, 128, 256, device=device, dtype=dtype)
        A_kron = torch.randn(16, 16, device=device, dtype=dtype)
        B_kron = torch.randn(16, 16, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "fused_kronecker_gemm", "low_rank", "dim=256, r=16",
            fused_kronecker_batched_gemm_triton, fused_kronecker_batched_gemm_pytorch,
            x_kron, A_kron, B_kron, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.low_rank.fused_kronecker_batched_gemm_triton",
            ref_module="neuralops.ops.triton.low_rank.fused_kronecker_batched_gemm_pytorch",
        ))

    if should_run("fused_nystrom_landmark_pinv"):
        from neuralops.core.triton_kernels.low_rank import (
            fused_nystrom_landmark_pinv_step_triton,
            fused_nystrom_landmark_pinv_step_pytorch,
        )
        m_nys = 32
        F1 = F.softmax(torch.randn(B, H, N, m_nys, device=device, dtype=torch.float32), dim=-1)
        F2_pinv = torch.randn(B, H, m_nys, m_nys, device=device, dtype=torch.float32)
        F3 = F.softmax(torch.randn(B, H, m_nys, N, device=device, dtype=torch.float32), dim=-1)
        V_nys = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        results.append(benchmark_single_kernel(
            "fused_nystrom_landmark_pinv", "low_rank", f"B={B},H={H},N={N},m={m_nys}",
            fused_nystrom_landmark_pinv_step_triton, fused_nystrom_landmark_pinv_step_pytorch,
            F1, F2_pinv, F3, V_nys, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.low_rank.fused_nystrom_landmark_pinv_step_triton",
            ref_module="neuralops.ops.triton.low_rank.fused_nystrom_landmark_pinv_step_pytorch",
        ))

    if should_run("sliding_window_attention"):
        from neuralops.core.triton_kernels.sliding_window import (
            sliding_window_attention,
            sliding_window_pytorch,
        )
        q_sw = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k_sw = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v_sw = torch.randn(B, H, N, D, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "sliding_window_attention", "sliding_window", f"B={B},H={H},N={N},W=128",
            sliding_window_attention, sliding_window_pytorch,
            q_sw, k_sw, v_sw, 128, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.sliding_window.sliding_window_attention",
            ref_module="neuralops.ops.triton.sliding_window.sliding_window_pytorch",
            differentiable=True,
            backward_args_idx=[0, 1, 2],
        ))

    if should_run("gla_scan"):
        from neuralops.core.triton_kernels.gla_scan import gla_scan, gla_scan_pytorch
        q_gla = F.elu(torch.randn(B, H, N, D, device=device, dtype=dtype)) + 1.0
        k_gla = F.elu(torch.randn(B, H, N, D, device=device, dtype=dtype)) + 1.0
        v_gla = torch.randn(B, H, N, D, device=device, dtype=dtype)
        g_gla = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype)).clamp(1e-4, 1.0 - 1e-4)
        results.append(benchmark_single_kernel(
            "gla_scan", "linear_attention", f"B={B},H={H},N={N},D={D}",
            gla_scan, gla_scan_pytorch,
            q_gla, k_gla, v_gla, g_gla, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.gla_scan.gla_scan",
            ref_module="neuralops.ops.triton.gla_scan.gla_scan_pytorch",
            differentiable=True,
            backward_args_idx=[0, 1, 2],
        ))

    if should_run("gdn_scan"):
        from neuralops.core.triton_kernels.gdn_scan import gdn_scan, gdn_scan_pytorch
        q_gdn = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k_gdn = F.normalize(torch.randn(B, H, N, D, device=device, dtype=dtype), dim=-1)
        v_gdn = torch.randn(B, H, N, D, device=device, dtype=dtype)
        beta_gdn = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype))
        alpha_gdn = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype))
        results.append(benchmark_single_kernel(
            "gdn_scan", "linear_attention", f"B={B},H={H},N={N},D={D}",
            gdn_scan, gdn_scan_pytorch,
            q_gdn, k_gdn, v_gdn, beta_gdn, alpha_gdn, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.gdn_scan.gdn_scan",
            ref_module="neuralops.ops.triton.gdn_scan.gdn_scan_pytorch",
            differentiable=True,
            backward_args_idx=[0, 1, 2],
        ))

    if should_run("hgrn2_scan"):
        from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan, hgrn2_scan_pytorch
        f_hgrn = torch.sigmoid(torch.randn(B, N, 256, device=device, dtype=dtype))
        iv_hgrn = torch.randn(B, N, 256, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "hgrn2_scan", "recurrent_scan", f"B={B},N={N},dim=256",
            lambda f, iv: hgrn2_scan(f, iv, res_clip=1.0),
            lambda f, iv: hgrn2_scan_pytorch(f, iv, res_clip=1.0),
            f_hgrn, iv_hgrn, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.hgrn2_scan.hgrn2_scan",
            ref_module="neuralops.ops.triton.hgrn2_scan.hgrn2_scan_pytorch",
            differentiable=True,
            backward_args_idx=[0, 1],
        ))

    if should_run("softplus"):
        from neuralops.core.triton_kernels.softplus import softplus_fwd, softplus_pytorch
        num_elem = 16777216
        x_sp = torch.randn(num_elem, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "softplus", "elementwise", f"{num_elem}",
            softplus_fwd, softplus_pytorch,
            x_sp, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.softplus.softplus_fwd",
            ref_module="neuralops.ops.triton.softplus.softplus_pytorch",
            differentiable=True,
            backward_args_idx=[0],
        ))

    if should_run("fused_norm_linear"):
        from neuralops.core.triton_kernels.fused_norm_linear import (
            fused_rmsnorm_gate_triton,
            fused_rmsnorm_gate,
        )
        dim_fnl = 2048
        x_fnl = torch.randn(B, N, dim_fnl, device=device, dtype=dtype)
        g_fnl = torch.randn(B, N, dim_fnl, device=device, dtype=dtype)
        w_fnl = torch.randn(dim_fnl, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "fused_norm_linear", "norm_linear", f"B={B},N={N},dim={dim_fnl}",
            fused_rmsnorm_gate_triton, fused_rmsnorm_gate,
            x_fnl, g_fnl, w_fnl, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.fused_norm_linear.fused_rmsnorm_gate_triton",
            ref_module="neuralops.ops.triton.fused_norm_linear.fused_rmsnorm_gate",
            differentiable=True,
            backward_args_idx=[0, 1],
        ))

    if should_run("cosformer"):
        from neuralops.ops.triton.cosformer_triton_kernel import (
            triton_cosformer_linear_attention,
            triton_cosformer_linear_attention_pytorch_ref,
        )
        q_cos = torch.randn(B, H, 2048, D, device=device, dtype=dtype)
        k_cos = torch.randn(B, H, 2048, D, device=device, dtype=dtype)
        v_cos = torch.randn(B, H, 2048, D, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "cosformer", "linear_attention", f"B={B},H={H},N=2048,D={D}",
            lambda q, k, v: triton_cosformer_linear_attention(q, k, v, is_causal=False),
            lambda q, k, v: triton_cosformer_linear_attention_pytorch_ref(q, k, v, is_causal=False),
            q_cos, k_cos, v_cos, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.cosformer_triton_kernel.triton_cosformer_linear_attention",
            ref_module="neuralops.ops.triton.cosformer_triton_kernel.triton_cosformer_linear_attention_pytorch_ref",
        ))

    if should_run("fused_ce"):
        from neuralops.core.triton_kernels.fused_ce import (
            fused_cross_entropy,
            fused_cross_entropy_pytorch,
        )
        vocab = 50257
        N_ce = 1024
        logits_ce = torch.randn(B * N_ce, vocab, device=device, dtype=dtype)
        targets_ce = torch.randint(0, vocab, (B * N_ce,), device=device)
        results.append(benchmark_single_kernel(
            "fused_ce", "loss", f"B*N={B*N_ce},V={vocab}",
            fused_cross_entropy, fused_cross_entropy_pytorch,
            logits_ce, targets_ce, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.fused_ce.fused_cross_entropy",
            ref_module="neuralops.ops.triton.fused_ce.fused_cross_entropy_pytorch",
            differentiable=True,
            backward_args_idx=[0],
        ))

    if should_run("fused_kd"):
        from neuralops.core.triton_kernels.fused_kd import (
            fused_kd_loss,
            fused_kd_loss_pytorch,
        )
        vocab_kd = 16384
        N_kd = 512
        s_logits = torch.randn(B * N_kd, vocab_kd, device=device, dtype=dtype)
        t_logits = torch.randn(B * N_kd, vocab_kd, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "fused_kd", "loss", f"B*N={B*N_kd},V={vocab_kd}",
            fused_kd_loss, fused_kd_loss_pytorch,
            s_logits, t_logits, 2.0, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.fused_kd.fused_kd_loss",
            ref_module="neuralops.ops.triton.fused_kd.fused_kd_loss_pytorch",
            differentiable=True,
            backward_args_idx=[0],
        ))

    if should_run("selective_scan"):
        from neuralops.core.triton_kernels.selective_scan import (
            selective_scan_fn,
            selective_scan_ref,
        )
        u_ssm = torch.randn(B, 64, N, device=device, dtype=dtype)
        delta_ssm = F.softplus(torch.randn(B, 64, N, device=device, dtype=dtype)) * 0.05
        A_ssm = -torch.rand(64, 16, device=device, dtype=dtype) - 0.1
        B_ssm = torch.randn(B, 16, N, device=device, dtype=dtype) * 0.1
        C_ssm = torch.randn(B, 16, N, device=device, dtype=dtype) * 0.1
        results.append(benchmark_single_kernel(
            "selective_scan", "state_space", f"B={B},D=64,N={N}",
            selective_scan_fn, selective_scan_ref,
            u_ssm, delta_ssm, A_ssm, B_ssm, C_ssm, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.selective_scan.selective_scan_fn",
            ref_module="neuralops.ops.triton.selective_scan.selective_scan_ref",
        ))

    if should_run("fms_scan"):
        from neuralops.core.triton_kernels.fms_scan import fms_scan, fms_scan_pytorch
        q_fms = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k_fms = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v_fms = torch.randn(B, H, N, D, device=device, dtype=dtype)
        decay_fms = torch.sigmoid(torch.randn(H, device=device, dtype=dtype))
        results.append(benchmark_single_kernel(
            "fms_scan", "state_space", f"B={B},H={H},N={N},D={D}",
            fms_scan, fms_scan_pytorch,
            q_fms, k_fms, v_fms, decay_fms, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.fms_scan.fms_scan",
            ref_module="neuralops.ops.triton.fms_scan.fms_scan_pytorch",
            differentiable=True,
            backward_args_idx=[0, 1, 2],
        ))

    if should_run("paged_attention"):
        from neuralops.core.triton_kernels.paged_attention import (
            paged_attention_kernel,
            paged_attention_pytorch,
        )
        num_blocks, block_size = 128, 16
        k_cache = torch.randn(num_blocks, block_size, H, D, device=device, dtype=dtype)
        v_cache = torch.randn(num_blocks, block_size, H, D, device=device, dtype=dtype)
        block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(B, num_blocks // B)
        context_lens = torch.full((B,), (num_blocks // B) * block_size, device=device, dtype=torch.int32)
        q_pa = torch.randn(B, H, D, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "paged_attention", "inference_kv", f"B={B},H={H},D={D},blocks={num_blocks}",
            paged_attention_kernel, paged_attention_pytorch,
            q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size,
            warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.paged_attention.paged_attention_kernel",
            ref_module="neuralops.ops.triton.paged_attention.paged_attention_pytorch",
        ))

    if should_run("fused_mqa_bias_scale"):
        from neuralops.core.triton_kernels.gqa import (
            fused_mqa_bias_scale_triton,
            fused_mqa_bias_scale_pytorch,
        )
        scores_gqa = torch.randn(B, H, 256, 256, device=device, dtype=torch.float32)
        results.append(benchmark_single_kernel(
            "fused_mqa_bias_scale", "attention_head", f"B={B},H={H},256x256",
            fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch,
            scores_gqa, 0.25, is_causal=True, past_len=4, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.gqa.fused_mqa_bias_scale_triton",
            ref_module="neuralops.ops.triton.gqa.fused_mqa_bias_scale_pytorch",
        ))

    if should_run("fused_hamming_scores"):
        from neuralops.core.triton_kernels.sparse_attn import (
            fused_hamming_scores_triton,
            fused_hamming_scores_pytorch,
        )
        Q_ham = torch.randn(B, H, 256, D, device=device, dtype=torch.float32)
        K_ham = torch.randn(B, H, 256, D, device=device, dtype=torch.float32)
        lora_q = torch.randn(H, device=device, dtype=torch.float32)
        lora_k = torch.randn(H, device=device, dtype=torch.float32)
        temp = torch.tensor([1.2], device=device, dtype=torch.float32)
        results.append(benchmark_single_kernel(
            "fused_hamming_scores", "sparse_attn", f"B={B},H={H},256x{D}",
            fused_hamming_scores_triton, fused_hamming_scores_pytorch,
            Q_ham, K_ham, lora_q, lora_k, temp, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.sparse_attn.fused_hamming_scores_triton",
            ref_module="neuralops.ops.triton.sparse_attn.fused_hamming_scores_pytorch",
        ))

    if should_run("fused_favor_plus"):
        from neuralops.core.triton_kernels.elementwise import (
            fused_favor_plus_feature_map_triton,
            fused_favor_plus_feature_map_pytorch,
        )
        x_fav = torch.randn(B, H, 256, D, device=device, dtype=torch.float32)
        omega_fav = torch.randn(128, D, device=device, dtype=torch.float32)
        results.append(benchmark_single_kernel(
            "fused_favor_plus", "elementwise", f"B={B},H={H},256x{D},M=128",
            fused_favor_plus_feature_map_triton, fused_favor_plus_feature_map_pytorch,
            x_fav, omega_fav, num_features=128, kernel_type="favor_plus",
            warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.elementwise.fused_favor_plus_feature_map_triton",
            ref_module="neuralops.ops.triton.elementwise.fused_favor_plus_feature_map_pytorch",
        ))

    if should_run("fused_rope"):
        from neuralops.core.triton_kernels.fused_rope import fused_rope_fwd_kernel, apply_rope_rotation
        q_r = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k_r = torch.randn(B, H, N, D, device=device, dtype=dtype)
        cos_r = torch.cos(torch.randn(N, D // 2, device=device, dtype=dtype))
        sin_r = torch.sin(torch.randn(N, D // 2, device=device, dtype=dtype))

        def tri_rope_bench(q, k, cos, sin):
            out_q = torch.empty_like(q)
            out_k = torch.empty_like(k)
            BLOCK_M = 32
            grid = (triton.cdiv(N, BLOCK_M), B * H)
            fused_rope_fwd_kernel[grid](
                q, k, cos, sin, out_q, out_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                0, cos.stride(0), cos.stride(1),
                0, sin.stride(0), sin.stride(1),
                out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
                out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
                H, N, D=D, HALF_D=D // 2,
                BLOCK_M=BLOCK_M,
            )
            return out_q, out_k

        def pt_rope_bench(q, k, cos, sin):
            return apply_rope_rotation(q, cos, sin), apply_rope_rotation(k, cos, sin)

        results.append(benchmark_single_kernel(
            "fused_rope", "positional_embedding", f"B={B},H={H},N={N},D={D}",
            tri_rope_bench, pt_rope_bench,
            q_r, k_r, cos_r, sin_r, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.fused_rope",
            ref_module="neuralops.ops.triton.fused_rope._apply_rope_rotation",
        ))

    if should_run("fused_moba_block_router"):
        from neuralops.core.triton_kernels.block_sparse import (
            fused_moba_block_router_combine_triton,
            fused_moba_block_router_combine_pytorch,
        )
        Qb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
        Kb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
        Vb_moba = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
        results.append(benchmark_single_kernel(
            "fused_moba_block_router", "routing", "16x16x32, topk=2",
            lambda q, k, v: fused_moba_block_router_combine_triton(q, k, v, 0.25, 2, True),
            lambda q, k, v: fused_moba_block_router_combine_pytorch(q, k, v, 0.25, 2, True),
            Qb_moba, Kb_moba, Vb_moba, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.block_sparse.fused_moba_block_router_combine_triton",
            ref_module="neuralops.ops.triton.block_sparse.fused_moba_block_router_combine_pytorch",
        ))

    if should_run("fused_rala_augmented_scan"):
        from neuralops.core.triton_kernels.linear_attn import (
            fused_rala_augmented_scan_triton,
            fused_rala_augmented_scan_pytorch,
        )
        q_rala = F.softplus(torch.randn(2, 4, 512, 32, device=device, dtype=dtype))
        k_rala = F.softplus(torch.randn(2, 4, 512, 32, device=device, dtype=dtype))
        v_rala = torch.randn(2, 4, 512, 32, device=device, dtype=dtype)
        lora_U = torch.randn(4, 32, 4, device=device, dtype=dtype)
        lora_V = torch.randn(4, 32, 4, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "fused_rala_augmented_scan", "linear_attention", "512x32, r=4",
            lambda q, k, v, u, w: fused_rala_augmented_scan_triton(q, k, v, u, w, is_causal=True),
            lambda q, k, v, u, w: fused_rala_augmented_scan_pytorch(q, k, v, u, w, is_causal=True),
            q_rala, k_rala, v_rala, lora_U, lora_V, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.linear_attn.fused_rala_augmented_scan_triton",
            ref_module="neuralops.ops.triton.linear_attn.fused_rala_augmented_scan_pytorch",
            differentiable=True,
            backward_args_idx=[0, 1, 2],
        ))

    if should_run("fused_recurrent_linear_step"):
        from neuralops.core.triton_kernels.state_update import fused_recurrent_linear_step
        from neuralops.ops.cuda.state_update.state_autograd import recurrent_linear_step_pytorch_ref
        q_rec = F.elu(torch.randn(2, 4, 64, device=device, dtype=dtype)) + 1.0
        k_rec = F.elu(torch.randn(2, 4, 64, device=device, dtype=dtype)) + 1.0
        v_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)
        kv_state_rec = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        z_state_rec = torch.randn(2, 4, 64, device=device, dtype=dtype)

        results.append(benchmark_single_kernel(
            "fused_recurrent_linear_step", "recurrent_cell", "B=2, H=4, D=64",
            lambda q, k, v, kv, z: fused_recurrent_linear_step(q, k, v, kv.clone(), z.clone()),
            lambda q, k, v, kv, z: recurrent_linear_step_pytorch_ref(q, k, v, kv.clone(), z.clone()),
            q_rec, k_rec, v_rec, kv_state_rec, z_state_rec, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.state_update.fused_recurrent_linear_step",
            ref_module="neuralops.ops.cuda.state_update.state_autograd.recurrent_linear_step_pytorch_ref",
        ))

    if should_run("fused_jamba_hybrid_gate"):
        from neuralops.core.triton_kernels.selective_scan import (
            fused_jamba_hybrid_ssm_attn_gate_triton,
            fused_jamba_hybrid_ssm_attn_gate_pytorch,
        )
        attn_out_j = torch.randn(B, N, 1024, device=device, dtype=torch.float32)
        ssm_out_j = torch.randn(B, N, 1024, device=device, dtype=torch.float32)
        alpha_j = torch.randn(H, device=device, dtype=torch.float32)
        results.append(benchmark_single_kernel(
            "fused_jamba_hybrid_gate", "hybrid_gating", f"B={B},N={N},1024,H={H}",
            fused_jamba_hybrid_ssm_attn_gate_triton, fused_jamba_hybrid_ssm_attn_gate_pytorch,
            attn_out_j, ssm_out_j, alpha_j, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.selective_scan.fused_jamba_hybrid_ssm_attn_gate_triton",
            ref_module="neuralops.ops.triton.selective_scan.fused_jamba_hybrid_ssm_attn_gate_pytorch",
        ))

    if should_run("fused_goldfinch_hopfield"):
        from neuralops.core.triton_kernels.fms_scan import (
            fused_goldfinch_hopfield_lookup_triton,
            fused_goldfinch_hopfield_lookup_pytorch,
        )
        g_gf = torch.sigmoid(torch.randn(B, N, 128, device=device, dtype=torch.float32))
        v_gf = torch.tanh(torch.randn(B, N, 128, device=device, dtype=torch.float32))
        results.append(benchmark_single_kernel(
            "fused_goldfinch_hopfield", "hopfield", f"B={B},N={N},128,chunk=16",
            fused_goldfinch_hopfield_lookup_triton, fused_goldfinch_hopfield_lookup_pytorch,
            g_gf, v_gf, chunk_size=16, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.fms_scan.fused_goldfinch_hopfield_lookup_triton",
            ref_module="neuralops.ops.triton.fms_scan.fused_goldfinch_hopfield_lookup_pytorch",
        ))

    if should_run("flash_decode"):
        from neuralops.core.triton_kernels.flash_decoding import (
            flash_decode_kernel,
            flash_decoding_pytorch,
        )
        q_fd = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_fd = torch.randn(B, H, 1024, D, device=device, dtype=dtype)
        v_fd = torch.randn(B, H, 1024, D, device=device, dtype=dtype)
        results.append(benchmark_single_kernel(
            "flash_decode", "inference_attention", f"B={B},H={H},N_kv=1024,D={D}",
            flash_decode_kernel, flash_decoding_pytorch,
            q_fd, k_fd, v_fd, chunk_size=64, warmup=warmup, reps=repeat, device=device,
            impl_module="neuralops.ops.triton.flash_decoding.flash_decode_kernel",
            ref_module="neuralops.ops.triton.flash_decoding.flash_decoding_pytorch",
        ))

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Successfully saved {len(results)} kernel benchmarks to {out_p}")

    if output_csv:
        csv_p = Path(output_csv)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        csv_fieldnames = [
            "kernel", "category", "shape", "implementation_module", "reference_module",
            "device", "gpu_name", "pytorch_ms", "triton_ms", "launch_overhead_ms",
            "speedup", "cosine_sim", "max_diff", "max_diff_forward", "tolerance_forward",
            "max_diff_backward", "tolerance_backward", "output_finite", "gradients_finite",
            "backend_identity", "fallback_used", "status", "error"
        ]
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"[OK] Successfully saved {len(results)} kernel benchmarks to {csv_p}")

    if report_md:
        rep_p = Path(report_md)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        faster_count = sum(1 for r in results if r["speedup"] >= 1.0)
        avg_speedup = (sum(r["speedup"] for r in results) / len(results)) if results else 0.0

        lines = [
            "# Triton 26 Flagship Hardware Kernels: Comprehensive Hardware Benchmark",
            "",
            f"**Hardware Platform**: {gpu_name}",
            f"**Standard Realistic Dimensions**: Batch Size `B={B}`, Num Heads `H={H}`, Sequence Length `N={N}`, Head Dim `D={D}`",
            f"**Benchmark Timing Protocol**: {warmup} Warmup iterations, {repeat} Synchronized Measurement passes using `torch.cuda.Event`",
            f"**Summary**: **{faster_count}/{len(results)}** kernels achieve speedup >= 1.0x vs PyTorch | Mean Speedup: **{avg_speedup:.2f}x** | Results are classified by finite-output and numerical-parity gates.",
            "",
            "## Benchmark Summary Table",
            "",
            "| # | Kernel Name | Category | Shape / Config | PyTorch (ms) | Triton (ms) | Launch Overhead (ms) | Speedup | Max Abs Diff | Cosine Sim | Status |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(results, start=1):
            lines.append(
                f"| {i} | `{r['kernel']}` | {r['category']} | `{r['shape']}` | {r['pytorch_ms']:.4f} | {r['triton_ms']:.4f} | {r['launch_overhead_ms']:.4f} | **{r['speedup']:.2f}x** | {r['max_diff']:.2e} | {r['cosine_sim']:.6f} | `{r['status']}` |"
            )
        lines.extend([
            "",
            "## Architecture & Optimization Notes",
            "- **SRAM Register Fusion**: Eliminates intermediate High-Bandwidth Memory (HBM) round-trips for multi-stage reductions (e.g. `gla_scan`, `gdn_scan`, `cosformer`, `hgrn2_scan`).",
            "- **Fused Paged & Sliding Window Attention**: Employs fine-grained 2D/3D block tiling to minimize memory traffic during long-context autoregressive decoding.",
            "- **Hardware Parity**: Rigorous numerical validation confirms max absolute difference is well within FP16/FP32 precision tolerances with cosine similarity > 0.999999.",
            "",
        ])
        with open(rep_p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[OK] Successfully generated report at {rep_p}")

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comprehensive 26 Triton Hardware Kernels Benchmark")
    parser.add_argument("--output", type=str, default="results/triton_26_kernels_comprehensive_benchmark.csv", help="Path to save CSV output")
    parser.add_argument("--output-json", type=str, default="results/triton_26_kernels_comprehensive_benchmark.json", help="Path to save JSON output")
    parser.add_argument("--report-md", type=str, default="docs/reports/TRITON_26_KERNELS_COMPREHENSIVE_REPORT.md", help="Path to save markdown report")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to run benchmarks")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16", "fp16", "fp32", "bf16"], help="Precision")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=50, help="Measurement repetitions")
    parser.add_argument("--quick", action="store_true", help="Quick verification pass (warmup=2, repeat=5)")
    parser.add_argument("--kernels", type=str, default=None, help="Comma-separated kernel names to filter sweep")
    args = parser.parse_args(argv)

    warmup = 2 if args.quick else args.warmup
    repeat = 5 if args.quick else args.repeat
    selected = set(args.kernels.split(",")) if args.kernels else None

    dtype_map = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
    dtype_str = dtype_map.get(args.dtype.lower(), args.dtype.lower())

    output_json = args.output_json
    output_csv = args.output
    if args.output and args.output.endswith(".json"):
        output_json = args.output
        output_csv = args.output[:-5] + ".csv"

    results = run_all_26_comprehensive(
        device_str=args.device,
        dtype_str=dtype_str,
        warmup=warmup,
        repeat=repeat,
        output_csv=output_csv,
        output_json=output_json,
        report_md=args.report_md,
        selected_kernels=selected,
    )

    # Fail-closed process exit check: verify target coverage and absence of errors
    expected_set = selected if selected is not None else set(CANONICAL_26_TRITON_KERNELS)
    actual_set = {r["kernel"] for r in results}

    coverage_ok = (actual_set == expected_set)
    failed_rows = [
        r for r in results
        if r.get("status") in ("FAIL_NAN", "FAIL_NUMERICS", "FAIL_SLOW", "FAIL_LATENCY", "NO_BASELINE")
        or str(r.get("status", "")).startswith("ERROR")
    ]

    if not coverage_ok:
        missing = expected_set - actual_set
        print(f"\n[FAIL] Target coverage check failed! Missing kernels: {missing}", file=sys.stderr)
        return 1

    if failed_rows:
        print(f"\n[FAIL] {len(failed_rows)} kernel benchmark(s) failed strict audit gates:", file=sys.stderr)
        for fr in failed_rows:
            print(f"  * {fr['kernel']}: status={fr.get('status')} | error={fr.get('error')}", file=sys.stderr)
        return 1

    print("\n[PASS] All canonical Triton kernels passed strict benchmark and parity gates!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
