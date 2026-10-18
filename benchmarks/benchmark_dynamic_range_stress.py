import os
import sys
import time
import csv
import math
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels.gla_scan import gla_scan
from neuralops.core.cuda_kernels.sparse_attention import block_sparse_attn
from neuralops.core.cuda_kernels.rotational_svd.rotational_svd_autograd import rotational_svd
from neuralops.core.cuda_kernels.samba_moe.samba_moe_autograd import samba_moe

DYNAMIC_SCALES = [1e-4, 1e-2, 1.0, 1e2, 1e4]
KERNEL_NAMES = ["gla_scan", "sparse_attention", "rotational_svd", "samba_moe"]

def run_dynamic_range_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Dynamic Range Stress Test on {device} ({torch.cuda.get_device_name(0)}) ===")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    B, H, N, D = 2, 4, 256, 64
    r_dim = 16
    num_blocks = N // 16
    sparse_mask = torch.eye(num_blocks, device=device, dtype=torch.int32)

    results = []

    for k_name in KERNEL_NAMES:
        print(f"\n--- Testing Kernel: {k_name} ---")
        for scale in DYNAMIC_SCALES:
            scale_str = f"1e{int(math.log10(scale)):+03d}" if scale != 1.0 else "1e+00"

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

            # Warmup
            for _ in range(5):
                _ = fn()
            torch.cuda.synchronize()

            # Timing
            iters = 30
            t0 = time.perf_counter()
            for _ in range(iters):
                out = fn()
            torch.cuda.synchronize()
            lat_ms = ((time.perf_counter() - t0) / iters) * 1000

            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            min_val = out.min().item()
            max_val = out.max().item()
            norm_val = torch.linalg.norm(out).item()

            row = {
                "kernel": k_name,
                "scale": scale,
                "scale_label": scale_str,
                "latency_ms": lat_ms,
                "min_val": min_val,
                "max_val": max_val,
                "frobenius_norm": norm_val,
                "zero_nan": not has_nan,
                "zero_inf": not has_inf,
                "status": "PASS" if (not has_nan and not has_inf) else "FAIL"
            }
            results.append(row)
            print(f"  Scale {scale_str:<6} | Latency: {lat_ms:6.3f} ms | Min: {min_val:10.2e} | Max: {max_val:10.2e} | Norm: {norm_val:10.2e} | Stable: {row['status']}")

    # Paths
    base_dir = "/mnt/d/SVD_linear_Attention"
    csv_path = os.path.join(base_dir, "docs/benchmarks/paper_benchmarks_dynamic_range_stress.csv")
    tex_path = os.path.join(base_dir, "docs/reports/PAPER_TABLES_DYNAMIC_RANGE.tex")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)

    # 1. Export CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "scale", "scale_label", "latency_ms", "min_val", "max_val", "frobenius_norm", "zero_nan", "zero_inf", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved CSV to: {csv_path}")

    # 2. Export LaTeX Table
    tex = []
    tex.append(r"% Auto-generated Global Floating-Point Dynamic Range Stress Test")
    tex.append(r"% Target: NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9 Ada Lovelace)")
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\caption{Global Floating-Point Dynamic Range Stress Test across Flagship NeuralOps Kernels. Evaluated on NVIDIA RTX 4070 Laptop GPU (Ada Lovelace, SM 8.9) across activation magnitudes $\|X\|_\infty \in [10^{-4}, 10^{4}]$.}")
    tex.append(r"\label{tab:dynamic_range_stress}")
    tex.append(r"\begin{tabular}{llcccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Kernel} & \textbf{Dynamic Scale ($\|X\|_\infty$)} & \textbf{Latency (ms)} & \textbf{Output Range $[\min, \max]$} & \textbf{Frobenius Norm $\|Y\|_F$} & \textbf{Zero-NaN/Inf Status} \\")
    tex.append(r"\midrule")

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

        scale_disp = f"$10^{{{int(math.log10(r['scale']))}}}$"
        lat_str = f"{r['latency_ms']:.3f}"
        range_str = f"$[{r['min_val']:.1e}, {r['max_val']:.1e}]$"
        norm_str = f"{r['frobenius_norm']:.2e}"
        status_str = r"\checkmark \text{ PASS}" if r['status'] == "PASS" else r"\times \text{ FAIL}"

        tex.append(f"{k_col} & {scale_disp} & {lat_str} & {range_str} & {norm_str} & {status_str} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")

    with open(tex_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved LaTeX Table to: {tex_path}")

if __name__ == "__main__":
    run_dynamic_range_benchmark()
