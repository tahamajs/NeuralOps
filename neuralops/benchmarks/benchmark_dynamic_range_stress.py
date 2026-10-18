"""
Dynamic Range Stress Test across extreme numerical scales (1e-4 to 1e4)

Evaluates numerical stability, min/max values, frobenius norms, and NaN/Inf robustness under extreme dynamic ranges across flagship kernels (GLA Scan, Block Sparse, Rotational SVD, Samba MoE).

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

DEFAULT_DYNAMIC_SCALES = [1e-4, 1e-2, 1.0, 1e2, 1e4]
QUICK_DYNAMIC_SCALES = [1e-2, 1.0, 1e2]
DEFAULT_KERNEL_NAMES = ["gla_scan", "sparse_attention", "rotational_svd", "samba_moe"]


def run_dynamic_range_benchmark(
    device_str: str = "cuda:0",
    scales: list = None,
    kernel_names: list = None,
    warmup: int = 5,
    repeat: int = 30,
    output_csv: str = "docs/benchmarks/paper_benchmarks_dynamic_range_stress.csv",
    output_json: str = "results/paper_benchmarks_dynamic_range_stress.json",
    output_tex: str = "docs/reports/PAPER_TABLES_DYNAMIC_RANGE.tex",
):
    """Execute dynamic range stress test sweep."""
    if scales is None:
        scales = DEFAULT_DYNAMIC_SCALES
    if kernel_names is None:
        kernel_names = DEFAULT_KERNEL_NAMES

    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    from neuralops.core.triton_kernels.gla_scan import gla_scan
    from neuralops.core.cuda_kernels.sparse_attention import block_sparse_attn
    from neuralops.core.cuda_kernels.rotational_svd.rotational_svd_autograd import rotational_svd
    from neuralops.core.cuda_kernels.samba_moe.samba_moe_autograd import samba_moe

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Dynamic Range Stress Test on {device} ({gpu_name}) ===")
    print(f"Scales: {scales} | Kernels: {kernel_names} | Warmup: {warmup} | Repeat: {repeat}")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    B, H, N, D = 2, 4, 256, 64
    r_dim = 16
    num_blocks = N // 16
    sparse_mask = torch.eye(num_blocks, device=device, dtype=torch.int32)

    results = []

    for k_name in kernel_names:
        print(f"\n--- Testing Kernel: {k_name} ---")
        for scale in scales:
            scale_str = f"1e{int(math.log10(scale)):+03d}" if scale != 1.0 else "1e+00"

            try:
                if k_name == "gla_scan":
                    q = (2.0 * torch.rand(B, H, N, D, device=device, dtype=torch.float32) - 1.0) * scale / (D ** 0.5)
                    k = (2.0 * torch.rand(B, H, N, D, device=device, dtype=torch.float32) - 1.0) * scale / (D ** 0.5)
                    v = (2.0 * torch.rand(B, H, N, D, device=device, dtype=torch.float32) - 1.0) * scale
                    lam = -torch.rand(B, H, N, device=device, dtype=torch.float32) * 0.5
                    fn = lambda: gla_scan(q, k, v, lambdas=lam)
                elif k_name == "sparse_attention":
                    q_sp = (2.0 * torch.rand(B, H, N, D, device=device, dtype=torch.float32) - 1.0) * scale / (D ** 0.5)
                    k_sp = (2.0 * torch.rand(B, H, N, D, device=device, dtype=torch.float32) - 1.0) * scale / (D ** 0.5)
                    v_sp = (2.0 * torch.rand(B, H, N, D, device=device, dtype=torch.float32) - 1.0) * scale
                    fn = lambda: block_sparse_attn(q_sp, k_sp, v_sp, sparse_mask, block_size=16)
                elif k_name == "rotational_svd":
                    x_rot = (2.0 * torch.rand(B, N, D, device=device, dtype=torch.float32) - 1.0) * scale
                    rin = torch.eye(D, device=device, dtype=torch.float32)
                    vr = torch.randn(D, r_dim, device=device, dtype=torch.float32) / (r_dim ** 0.5)
                    ur = torch.randn(r_dim, D, device=device, dtype=torch.float32) / (r_dim ** 0.5)
                    fn = lambda: rotational_svd(x_rot, rin, vr, ur)
                elif k_name == "samba_moe":
                    e1 = (2.0 * torch.rand(B, N, D, device=device, dtype=torch.float32) - 1.0) * scale
                    e2 = (2.0 * torch.rand(B, N, D, device=device, dtype=torch.float32) - 1.0) * scale
                    rw = torch.softmax(torch.randn(B, N, 2, device=device, dtype=torch.float32), dim=-1)
                    fn = lambda: samba_moe(e1, e2, rw)
                else:
                    raise ValueError(f"Unknown kernel name: {k_name}")

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
                lat_ms = ((time.perf_counter() - t0) / repeat) * 1000.0

                if isinstance(out, tuple):
                    out = out[0]

                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                min_val = out.min().item()
                max_val = out.max().item()
                norm_val = torch.linalg.norm(out).item()

                row = {
                    "kernel": k_name,
                    "scale": scale,
                    "scale_label": scale_str,
                    "latency_ms": round(lat_ms, 3),
                    "min_val": min_val,
                    "max_val": max_val,
                    "frobenius_norm": norm_val,
                    "zero_nan": not has_nan,
                    "zero_inf": not has_inf,
                    "status": "PASS" if (not has_nan and not has_inf) else "FAIL",
                }
                results.append(row)
                print(f"  Scale {scale_str:<6} | Latency: {lat_ms:6.3f} ms | Min: {min_val:10.2e} | Max: {max_val:10.2e} | Norm: {norm_val:10.2e} | Stable: {row['status']}")

            except Exception as exc:
                print(f"  Scale {scale_str:<6} | EXCEPTION: {exc}")
                results.append({
                    "kernel": k_name,
                    "scale": scale,
                    "scale_label": scale_str,
                    "latency_ms": 0.0,
                    "min_val": 0.0,
                    "max_val": 0.0,
                    "frobenius_norm": 0.0,
                    "zero_nan": False,
                    "zero_inf": False,
                    "status": f"FAIL ({exc})",
                })

    # Save CSV
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "kernel", "scale", "scale_label", "latency_ms", "min_val", "max_val", "frobenius_norm", "zero_nan", "zero_inf", "status"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nSaved CSV to: {csv_path}")

    # Save JSON
    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON to: {json_path}")

    # Save LaTeX Table
    if output_tex:
        tex_path = Path(output_tex)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex = [
            r"% Auto-generated Dynamic Range Stress Test & Numerical Stability Audit",
            f"% Target: {gpu_name}",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Dynamic Range Numerical Stress Audit across extreme input scales ($10^{-4}$ to $10^{4}$) on " + gpu_name + r". Testing numerical stability, norm preservation, and floating-point overflow resistance.}",
            r"\label{tab:dynamic_range_stress}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"\textbf{Kernel} & \textbf{Dynamic Scale} & \textbf{Latency (ms)} & \textbf{Min Output} & \textbf{Max Output} & \textbf{Frobenius Norm} & \textbf{Zero-NaN/Inf} \\",
            r"\midrule",
        ]

        curr_kernel = None
        for r in results:
            k_disp = r['kernel'].replace('_', r'\_')
            if k_disp != curr_kernel:
                if curr_kernel is not None:
                    tex.append(r"\midrule")
                curr_kernel = k_disp
                k_col = f"\\textbf{{{k_disp}}}"
            else:
                k_col = ""

            scale_disp = f"$10^{{{int(math.log10(r['scale']))}}}$" if r['scale'] != 1.0 else "$10^{0}$"
            lat = f"{r['latency_ms']:.3f}"
            min_str = f"{r['min_val']:.2e}"
            max_str = f"{r['max_val']:.2e}"
            norm_str = f"{r['frobenius_norm']:.2e}"
            status_str = r"\checkmark" if (r['zero_nan'] and r['zero_inf']) else r"\times"

            tex.append(f"{k_col} & {scale_disp} & {lat} & {min_str} & {max_str} & {norm_str} & {status_str} \\\\")

        tex.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        with open(tex_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")
        print(f"Saved LaTeX Table to: {tex_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Dynamic Range Stress Test Benchmark")
    parser.add_argument("--output", type=str, default="docs/benchmarks/paper_benchmarks_dynamic_range_stress.csv", help="Path to save CSV output")
    parser.add_argument("--output-json", type=str, default="results/paper_benchmarks_dynamic_range_stress.json", help="Path to save JSON output")
    parser.add_argument("--tex-output", type=str, default="docs/reports/PAPER_TABLES_DYNAMIC_RANGE.tex", help="Path to save LaTeX table")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to execute benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=30, help="Repeat iterations")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3 scales, warmup=1, repeat=3)")
    parser.add_argument("--kernels", type=str, default=None, help="Comma-separated kernel names to test")
    args = parser.parse_args()

    scales = QUICK_DYNAMIC_SCALES if args.quick else DEFAULT_DYNAMIC_SCALES
    warmup = 1 if args.quick else args.warmup
    repeat = 3 if args.quick else args.repeat
    kernel_names = [k.strip() for k in args.kernels.split(",")] if args.kernels else DEFAULT_KERNEL_NAMES

    run_dynamic_range_benchmark(
        device_str=args.device,
        scales=scales,
        kernel_names=kernel_names,
        warmup=warmup,
        repeat=repeat,
        output_csv=args.output,
        output_json=args.output_json,
        output_tex=args.tex_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
