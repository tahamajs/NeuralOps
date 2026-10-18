import os
import sys
import time
import csv
import math
import torch
import torch.nn.functional as F

import neuralops.core.triton_kernels._types as tt
tt.MAX_TRITON_SEQ_LEN = 65536

from neuralops.core.triton_kernels.gla_scan import gla_scan
from neuralops.core.triton_kernels.gdn_scan import gdn_scan
from neuralops.attentions.svd.svd_linear_attention import SVDLinearAttention

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

def run_extrapolation_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Long-Horizon Sequence Extrapolation Benchmark on {device} ({torch.cuda.get_device_name(0)}) ===")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    B = 1
    H = 4
    D = 64
    rank = 16
    embed_dim = H * D

    models = ["Standard Quadratic Attention", "Chunk GLA", "Gated Delta Net", "SVD Linear Attention"]
    results = []

    for L in SEQ_LENS:
        print(f"\n================ Sequence Length L = {L} ================")

        # Prepare base inputs
        q = (torch.randn(B, H, L, D, device=device, dtype=torch.float16) / (D ** 0.5)).contiguous()
        k = (torch.randn(B, H, L, D, device=device, dtype=torch.float16) / (D ** 0.5)).contiguous()
        v = (torch.randn(B, H, L, D, device=device, dtype=torch.float16) * 0.1).contiguous()
        lam = (-torch.rand(B, H, L, device=device, dtype=torch.float16) * 0.1).contiguous()
        beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float16)).contiguous()
        alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float16)).contiguous()

        svd_model = SVDLinearAttention(embed_dim=embed_dim, num_heads=H, rank=rank).to(device=device, dtype=torch.float16)
        x_svd = torch.randn(B, L, embed_dim, device=device, dtype=torch.float16)

        # Theoretical recurrent KV-state memory in bytes
        # Quadratic SDPA requires storing full causal key-value sequences over context window: B * H * L * D * 2 bytes * 2 (K, V)
        kv_mem_bytes_quad = 2 * (B * H * L * D * 2) # O(L)
        # Recurrent scan state: S = B * H * D * D * 2 bytes (FP16 state matrix)
        kv_mem_bytes_recurrent = B * H * D * D * 2 # O(1) wrt L: 4 * 64 * 64 * 2 = 32 KB
        # SVD Low-rank state: S = B * H * D * rank * 2 bytes
        kv_mem_bytes_svd = B * H * D * rank * 2 # 4 * 64 * 16 * 2 = 8 KB

        for m_name in models:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if m_name == "Standard Quadratic Attention":
                fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                kv_state_kb = kv_mem_bytes_quad / 1024.0
                state_complexity = "O(L)"
                drift = 0.0  # Baseline
            elif m_name == "Chunk GLA":
                fn = lambda: gla_scan(q, k, v, lambdas=lam)
                kv_state_kb = kv_mem_bytes_recurrent / 1024.0
                state_complexity = "O(1)"
                # Drift vs FP32
                with torch.no_grad():
                    q32, k32, v32, lam32 = q.float(), k.float(), v.float(), lam.float()
                    ref32 = gla_scan(q32, k32, v32, lambdas=lam32)
                    y16 = fn()
                    drift = (y16.float() - ref32).norm().item() / max(ref32.norm().item(), 1e-12)
            elif m_name == "Gated Delta Net":
                fn = lambda: gdn_scan(q, k, v, beta=beta, alpha=alpha)
                kv_state_kb = kv_mem_bytes_recurrent / 1024.0
                state_complexity = "O(1)"
                with torch.no_grad():
                    q32, k32, v32, b32, a32 = q.float(), k.float(), v.float(), beta.float(), alpha.float()
                    ref32 = gdn_scan(q32, k32, v32, beta=b32, alpha=a32)
                    y16 = fn()
                    drift = (y16.float() - ref32).norm().item() / max(ref32.norm().item(), 1e-12)
            elif m_name == "SVD Linear Attention":
                fn = lambda: svd_model(x_svd)
                kv_state_kb = kv_mem_bytes_svd / 1024.0
                state_complexity = "O(1)"
                with torch.no_grad():
                    svd32 = SVDLinearAttention(embed_dim=embed_dim, num_heads=H, rank=rank).to(device=device, dtype=torch.float32)
                    svd32.load_state_dict({k: v.float() for k, v in svd_model.state_dict().items()})
                    ref32 = svd32(x_svd.float())[0]
                    y16 = fn()[0]
                    drift = (y16.float() - ref32).norm().item() / max(ref32.norm().item(), 1e-12)

            # Warmup
            for _ in range(3):
                _ = fn()
            torch.cuda.synchronize()

            # Measure timing
            iters = 10 if L <= 16384 else 5
            t0 = time.perf_counter()
            for _ in range(iters):
                out = fn()
            torch.cuda.synchronize()
            lat_ms = ((time.perf_counter() - t0) / iters) * 1000

            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

            if isinstance(out, tuple):
                out = out[0]
            has_nan = torch.isnan(out).any().item() or torch.isinf(out).any().item()

            row = {
                "sequence_length": L,
                "mechanism": m_name,
                "latency_ms": lat_ms,
                "peak_vram_mb": peak_mem_mb,
                "kv_state_kb": kv_state_kb,
                "state_complexity": state_complexity,
                "state_drift_rel": drift,
                "zero_nan": not has_nan
            }
            results.append(row)
            print(f"  {m_name:<28} | Latency: {lat_ms:6.2f} ms | Peak VRAM: {peak_mem_mb:6.1f} MB | KV State: {kv_state_kb:8.1f} KB ({state_complexity}) | Drift: {drift:9.2e} | Stable: {not has_nan}")

    # File exports
    base_dir = "/mnt/d/SVD_linear_Attention"
    csv_path = os.path.join(base_dir, "docs/benchmarks/paper_benchmarks_extrapolation_32k.csv")
    tex_path = os.path.join(base_dir, "docs/reports/PAPER_TABLES_EXTRAPOLATION_32K.tex")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)

    # 1. Export CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence_length", "mechanism", "latency_ms", "peak_vram_mb", "kv_state_kb", "state_complexity", "state_drift_rel", "zero_nan"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved CSV to: {csv_path}")

    # 2. Export LaTeX Table
    tex = []
    tex.append(r"% Auto-generated Long-Horizon Sequence Extrapolation Benchmark")
    tex.append(r"% Target: NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9 Ada Lovelace)")
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\caption{Long-Horizon Sequence Extrapolation and KV-State Scaling across $L \in \{4096, 8192, 16384, 32768\}$ on NVIDIA RTX 4070 Laptop GPU. Comparing standard causal softmax attention with recurrent linear associative scan mechanisms.}")
    tex.append(r"\label{tab:extrapolation_32k}")
    tex.append(r"\begin{tabular}{llccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Sequence Length ($L$)} & \textbf{Mechanism} & \textbf{Latency (ms)} & \textbf{Peak VRAM (MB)} & \textbf{KV-State Size} & \textbf{State Drift ($\|\Delta\|/\|Y\|_F$)} & \textbf{Zero-NaN/Inf} \\")
    tex.append(r"\midrule")

    curr_L = None
    for r in results:
        L_val = r['sequence_length']
        if L_val != curr_L:
            if curr_L is not None:
                tex.append(r"\midrule")
            curr_L = L_val
            l_col = f"\\textbf{{{L_val:,}}}"
        else:
            l_col = ""

        mech_disp = r['mechanism']
        lat_str = f"{r['latency_ms']:.2f}"
        vram_str = f"{r['peak_vram_mb']:.1f}"
        
        if r['state_complexity'] == "O(1)":
            kv_str = f"{r['kv_state_kb']:.0f} KB ($\\mathcal{{O}}(1)$)"
        else:
            kv_str = f"{r['kv_state_kb']:.0f} KB ($\\mathcal{{O}}(L)$)"

        drift_str = "Baseline" if r['mechanism'] == "Standard Quadratic Attention" else f"{r['state_drift_rel']:.2e}"
        status_str = r"\checkmark" if r['zero_nan'] else r"\times"

        tex.append(f"{l_col} & {mech_disp} & {lat_str} & {vram_str} & {kv_str} & {drift_str} & {status_str} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")

    with open(tex_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved LaTeX Table to: {tex_path}")

if __name__ == "__main__":
    run_extrapolation_benchmark()
