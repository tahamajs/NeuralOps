"""
Multi-Precision Numerical Cross-Validation and Throughput Benchmark

Evaluates FP32, FP16, BF16, FP8 (E4M3, E5M2) trade-offs across flagship kernels: GLA Scan, Differential Attention, Block-Sparse Attention, Rotational SVD, and FP8 GEMM.

Author: Taha Majlesi
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_PRECISIONS = [
    ("FP32", torch.float32, False),
    ("FP16", torch.float16, False),
    ("BF16", torch.bfloat16, False),
    ("FP8_E4M3", torch.float8_e4m3fn, True),
    ("FP8_E5M2", torch.float8_e5m2, True),
]

QUICK_PRECISIONS = [
    ("FP32", torch.float32, False),
    ("FP16", torch.float16, False),
]


def measure_benchmark(
    device_str: str = "cuda:0",
    precisions_list: list = None,
    warmup: int = 5,
    repeat: int = 20,
    output_csv: str = "docs/benchmarks/paper_benchmarks_multiprecision_crossval.csv",
    output_json: str = "results/paper_benchmarks_multiprecision_crossval.json",
    output_tex: str = "docs/reports/PAPER_TABLES_MULTIPRECISION.tex",
    selected_kernels: list = None,
):
    """Execute multi-precision numerical verification and throughput benchmark."""
    if precisions_list is None:
        precisions_list = DEFAULT_PRECISIONS

    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    from neuralops.core.triton_kernels.gla_scan import gla_scan
    from neuralops.core.triton_kernels.differential_attention import triton_differential_attention
    from neuralops.core.cuda_kernels.sparse_attention import block_sparse_attn
    from neuralops.core.cuda_kernels.rotational_svd.rotational_svd_autograd import rotational_svd
    from neuralops.core.cuda_kernels.fp8_gemm.fp8_autograd import _get_module as get_fp8_gemm_module

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Executing Multi-Precision Cross-Validation Benchmark on {device} ({gpu_name})")
    print(f"Precisions: {[p[0] for p in precisions_list]} | Warmup: {warmup} | Repeat: {repeat}")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Dimensions
    B, H, N, D = 4, 8, 512, 64
    r_dim = 16
    M_gemm, K_gemm, N_gemm = 512, 512, 512

    results_table = []

    # 1. Reference FP32 tensors
    gla_q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    gla_k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    gla_v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 0.1
    gla_l_ref = -torch.rand(B, H, N, device=device, dtype=torch.float32) * 0.5

    diff_q1_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    diff_k1_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    diff_q2_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    diff_k2_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    diff_v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 0.1

    sparse_q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    sparse_k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) / (D ** 0.5)
    sparse_v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32) * 0.1
    num_blocks = N // 16
    sparse_mask = torch.eye(num_blocks, device=device, dtype=torch.int32)

    rot_x_ref = torch.randn(B, N, D, device=device, dtype=torch.float32) * 0.1
    rot_rin_ref = torch.eye(D, device=device, dtype=torch.float32)
    rot_vr_ref = torch.randn(D, r_dim, device=device, dtype=torch.float32) / (r_dim ** 0.5)
    rot_ur_ref = torch.randn(r_dim, D, device=device, dtype=torch.float32) / (r_dim ** 0.5)

    gemm_mod = get_fp8_gemm_module()
    gemm_a_ref = torch.randn(M_gemm, K_gemm, device=device, dtype=torch.float32)
    gemm_b_ref = torch.randn(K_gemm, N_gemm, device=device, dtype=torch.float32)
    gemm_sa = torch.tensor([1.0], device=device, dtype=torch.float32)
    gemm_sb = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Reference Outputs in FP32
    with torch.no_grad():
        y_gla_ref = gla_scan(gla_q_ref, gla_k_ref, gla_v_ref, lambdas=gla_l_ref)
        if isinstance(y_gla_ref, tuple):
            y_gla_ref = y_gla_ref[0]

        y_diff_ref = triton_differential_attention(diff_q1_ref, diff_k1_ref, diff_q2_ref, diff_k2_ref, diff_v_ref)
        y_sparse_ref = block_sparse_attn(sparse_q_ref, sparse_k_ref, sparse_v_ref, sparse_mask, block_size=16)
        y_rot_ref = rotational_svd(rot_x_ref, rot_rin_ref, rot_vr_ref, rot_ur_ref)
        y_gemm_ref = torch.matmul(gemm_a_ref, gemm_b_ref)

    # Theoretical FLOPs
    flops_gla = 4 * B * H * N * D * D
    flops_diff = 4 * B * H * N * N * D
    flops_sparse = 2 * (1.0 / num_blocks) * 2 * B * H * N * N * D
    flops_rot = 2 * B * N * D * D + 4 * B * N * D * r_dim
    flops_gemm = 2 * M_gemm * N_gemm * K_gemm

    kernels = [
        ("gla_scan", flops_gla, y_gla_ref),
        ("differential_attention", flops_diff, y_diff_ref),
        ("sparse_attention", flops_sparse, y_sparse_ref),
        ("rotational_svd", flops_rot, y_rot_ref),
        ("fp8_gemm", flops_gemm, y_gemm_ref),
    ]

    for k_name, k_flops, y_ref in kernels:
        if selected_kernels and (k_name not in selected_kernels):
            continue

        norm_ref = torch.linalg.norm(y_ref.float()).item()
        print(f"\nBenchmarking Kernel: {k_name}")

        for prec_name, dtype, is_fp8 in precisions_list:
            try:
                if k_name == "gla_scan":
                    if is_fp8:
                        q = gla_q_ref.to(dtype).float()
                        k = gla_k_ref.to(dtype).float()
                        v = gla_v_ref.to(dtype).float()
                        l = gla_l_ref.to(dtype).float()
                    else:
                        q = gla_q_ref.to(dtype)
                        k = gla_k_ref.to(dtype)
                        v = gla_v_ref.to(dtype)
                        l = gla_l_ref.to(dtype)
                    fn = lambda: gla_scan(q, k, v, lambdas=l)
                elif k_name == "differential_attention":
                    if is_fp8:
                        q1 = diff_q1_ref.to(dtype).float()
                        k1 = diff_k1_ref.to(dtype).float()
                        q2 = diff_q2_ref.to(dtype).float()
                        k2 = diff_k2_ref.to(dtype).float()
                        v = diff_v_ref.to(dtype).float()
                    else:
                        q1 = diff_q1_ref.to(dtype)
                        k1 = diff_k1_ref.to(dtype)
                        q2 = diff_q2_ref.to(dtype)
                        k2 = diff_k2_ref.to(dtype)
                        v = diff_v_ref.to(dtype)
                    fn = lambda: triton_differential_attention(q1, k1, q2, k2, v)
                elif k_name == "sparse_attention":
                    if is_fp8:
                        q = sparse_q_ref.to(dtype).float()
                        k = sparse_k_ref.to(dtype).float()
                        v = sparse_v_ref.to(dtype).float()
                    else:
                        q = sparse_q_ref.to(dtype)
                        k = sparse_k_ref.to(dtype)
                        v = sparse_v_ref.to(dtype)
                    fn = lambda: block_sparse_attn(q, k, v, sparse_mask, block_size=16)
                elif k_name == "rotational_svd":
                    if is_fp8:
                        x = rot_x_ref.to(dtype).float()
                        rin = rot_rin_ref.to(dtype).float()
                        vr = rot_vr_ref.to(dtype).float()
                        ur = rot_ur_ref.to(dtype).float()
                    else:
                        x = rot_x_ref.to(dtype)
                        rin = rot_rin_ref.to(dtype)
                        vr = rot_vr_ref.to(dtype)
                        ur = rot_ur_ref.to(dtype)
                    fn = lambda: rotational_svd(x, rin, vr, ur)
                elif k_name == "fp8_gemm":
                    if prec_name == "FP32":
                        a = gemm_a_ref
                        b = gemm_b_ref
                        fn = lambda: torch.matmul(a, b)
                    elif prec_name == "FP16":
                        a = gemm_a_ref.half()
                        b = gemm_b_ref.half()
                        fn = lambda: torch.matmul(a, b)
                    elif prec_name == "BF16":
                        a = gemm_a_ref.bfloat16()
                        b = gemm_b_ref.bfloat16()
                        fn = lambda: torch.matmul(a, b)
                    elif prec_name == "FP8_E4M3":
                        a = gemm_a_ref.to(torch.float8_e4m3fn)
                        b = gemm_b_ref.to(torch.float8_e5m2)
                        fn = lambda: gemm_mod.fp8_gemm_fwd(a, b, gemm_sa, gemm_sb, None, torch.float16)
                    elif prec_name == "FP8_E5M2":
                        a = gemm_a_ref.to(torch.float8_e4m3fn)
                        b = gemm_b_ref.to(torch.float8_e5m2)
                        fn = lambda: gemm_mod.fp8_gemm_fwd(a, b, gemm_sa, gemm_sb, None, torch.float16)

                # Warmup
                for _ in range(warmup):
                    _ = fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # Timing
                t0 = time.perf_counter()
                for _ in range(repeat):
                    out = fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency_s = (time.perf_counter() - t0) / repeat
                latency_ms = latency_s * 1000.0

                if isinstance(out, tuple):
                    out = out[0]

                out_f32 = out.float()
                has_nan = torch.isnan(out_f32).any().item()
                has_inf = torch.isinf(out_f32).any().item()
                zero_nan = (not has_nan) and (not has_inf)

                diff_norm = torch.linalg.norm(out_f32 - y_ref.float()).item()
                rel_error = diff_norm / max(norm_ref, 1e-12)
                tflops = (k_flops / (latency_s * 1e12)) if latency_s > 0 else 0.0

                row = {
                    "kernel": k_name,
                    "precision": prec_name,
                    "latency_ms": round(latency_ms, 3),
                    "tflops": round(tflops, 2),
                    "frob_rel_error": float(f"{rel_error:.4e}"),
                    "zero_nan": zero_nan,
                }
                results_table.append(row)
                print(f"  {prec_name:<10} | Latency: {latency_ms:6.3f} ms | Throughput: {tflops:6.2f} TFLOPS | Rel Error: {rel_error:10.4e} | Zero-NaN: {zero_nan}")

            except Exception as exc:
                print(f"  {prec_name:<10} | EXCEPTION: {exc}")
                results_table.append({
                    "kernel": k_name,
                    "precision": prec_name,
                    "latency_ms": 0.0,
                    "tflops": 0.0,
                    "frob_rel_error": float("nan"),
                    "zero_nan": False,
                })

    # Save CSV
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["kernel", "precision", "latency_ms", "tflops", "frob_rel_error", "zero_nan"])
            writer.writeheader()
            for r in results_table:
                writer.writerow(r)
        print(f"\nSaved CSV to: {csv_path}")

    # Save JSON
    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(results_table, f, indent=2)
        print(f"Saved JSON to: {json_path}")

    # Generate LaTeX Table
    if output_tex:
        tex_path = Path(output_tex)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_content = [
            r"% Auto-generated Multi-Precision Cross-Validation & Regression Audit",
            f"% Target: {gpu_name}",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Multi-Precision Numerical Cross-Validation and Throughput Comparison on " + gpu_name + r". Relative error is Frobenius norm relative to FP32 baseline.}",
            r"\label{tab:multiprecision_crossval}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"\textbf{Kernel} & \textbf{Precision} & \textbf{Latency (ms)} & \textbf{Throughput (TFLOPS)} & \textbf{Rel. Error ($\|\Delta\|_F/\|Y\|_F$)} & \textbf{Zero-NaN/Inf} \\",
            r"\midrule",
        ]

        curr_kernel = None
        for r in results_table:
            k_disp = r['kernel'].replace('_', r'\_')
            if k_disp != curr_kernel:
                if curr_kernel is not None:
                    tex_content.append(r"\midrule")
                curr_kernel = k_disp
                k_col = f"\\textbf{{{k_disp}}}"
            else:
                k_col = ""

            prec_disp = r['precision'].replace('_', r'\_')
            lat = f"{r['latency_ms']:.3f}"
            tflops = f"{r['tflops']:.2f}"
            err_str = "Baseline ($0.0$)" if r['precision'] == "FP32" else f"{r['frob_rel_error']:.2e}"
            status_str = r"\checkmark" if r['zero_nan'] else r"\times"

            tex_content.append(f"{k_col} & {prec_disp} & {lat} & {tflops} & {err_str} & {status_str} \\\\")

        tex_content.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        with open(tex_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(tex_content) + "\n")
        print(f"Saved LaTeX Table to: {tex_path}")

    return results_table


def main():
    parser = argparse.ArgumentParser(description="Multi-Precision Cross-Validation Benchmark")
    parser.add_argument("--output", type=str, default="docs/benchmarks/paper_benchmarks_multiprecision_crossval.csv", help="Path to save CSV output")
    parser.add_argument("--output-json", type=str, default="results/paper_benchmarks_multiprecision_crossval.json", help="Path to save JSON output")
    parser.add_argument("--tex-output", type=str, default="docs/reports/PAPER_TABLES_MULTIPRECISION.tex", help="Path to save LaTeX table")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to execute benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=20, help="Repeat iterations")
    parser.add_argument("--quick", action="store_true", help="Quick mode (FP32 & FP16 only, warmup=1, repeat=3)")
    parser.add_argument("--precisions", type=str, default=None, help="Comma-separated precisions: FP32,FP16,BF16,FP8_E4M3,FP8_E5M2")
    parser.add_argument("--kernels", type=str, default=None, help="Comma-separated kernel names to test")
    args = parser.parse_args()

    warmup = 1 if args.quick else args.warmup
    repeat = 3 if args.quick else args.repeat

    if args.precisions:
        req_p = set(p.strip().upper() for p in args.precisions.split(","))
        prec_list = [p for p in DEFAULT_PRECISIONS if p[0] in req_p]
    elif args.quick:
        prec_list = QUICK_PRECISIONS
    else:
        prec_list = DEFAULT_PRECISIONS

    selected_kernels = [k.strip() for k in args.kernels.split(",")] if args.kernels else None

    output_json = args.output_json
    output_csv = args.output
    if args.output and args.output.endswith(".json"):
        output_json = args.output
        output_csv = args.output[:-5] + ".csv"

    measure_benchmark(
        device_str=args.device,
        precisions_list=prec_list,
        warmup=warmup,
        repeat=repeat,
        output_csv=output_csv,
        output_json=output_json,
        output_tex=args.tex_output,
        selected_kernels=selected_kernels,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
