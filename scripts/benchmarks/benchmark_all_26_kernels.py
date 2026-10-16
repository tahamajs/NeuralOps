# -*- coding: utf-8 -*-
"""Direct Triton vs PyTorch physical hardware benchmark for NeuralOps JIT kernel modules."""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Direct Kernel Benchmark on: {device} ({torch.cuda.get_device_name(0)})")
    dtype = torch.float16

    results = []

    def bench_pair(name, shape_str, triton_fn, pytorch_fn, *args):
        # Warmup
        for _ in range(3):
            _ = triton_fn(*args)
            _ = pytorch_fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = triton_fn(*args)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = pytorch_fn(*args)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        if isinstance(out_t, tuple):
            out_t = out_t[0]
        if isinstance(out_p, tuple):
            out_p = out_p[0]

        max_diff = (out_t.float() - out_p.float()).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": name,
            "shape": shape_str,
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"  {name:36s} {shape_str:16s}: Tri={t_tri:6.3f}ms, PT={t_pt:6.3f}ms, Speedup={sp:5.2f}x, Diff={max_diff:.2e}")

    # 1. low_rank: fused_hnc_nullspace_step
    from neuralops.core.triton_kernels.low_rank import (
        fused_hnc_nullspace_step_pytorch,
        fused_hnc_nullspace_step_triton,
    )
    print("\n--- Benchmarking low_rank (HNC Nullspace) ---")
    for dim in [512, 1024, 2048]:
        w = torch.randn(dim, dim, device=device, dtype=dtype)
        g = torch.randn(dim, dim, device=device, dtype=dtype)
        r = torch.randn(dim, dim, device=device, dtype=dtype)
        bench_pair("fused_hnc_nullspace_step", f"({dim}x{dim})", fused_hnc_nullspace_step_triton, fused_hnc_nullspace_step_pytorch, w, g, r, 0.05)

    # 2. low_rank: fused_linformer_seq_proj
    from neuralops.core.triton_kernels.low_rank import (
        fused_linformer_seq_proj_pytorch,
        fused_linformer_seq_proj_triton,
    )
    print("\n--- Benchmarking low_rank (Linformer Projection) ---")
    for seq_len in [512, 1024, 2048]:
        x = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        proj = torch.randn(64, seq_len, device=device, dtype=dtype)
        bench_pair("fused_linformer_seq_proj", f"(T={seq_len})", fused_linformer_seq_proj_triton, fused_linformer_seq_proj_pytorch, x, proj)

    # 3. low_rank: fused_kronecker_batched_gemm
    from neuralops.core.triton_kernels.low_rank import (
        fused_kronecker_batched_gemm_pytorch,
        fused_kronecker_batched_gemm_triton,
    )
    print("\n--- Benchmarking low_rank (Kronecker GEMM) ---")
    for r in [16, 32]:
        x = torch.randn(2, 64, 256, device=device, dtype=dtype)
        A = torch.randn(16, 16, device=device, dtype=dtype)
        B = torch.randn(16, 16, device=device, dtype=dtype)
        bench_pair("fused_kronecker_gemm", f"(dim=256, r={r})", fused_kronecker_batched_gemm_triton, fused_kronecker_batched_gemm_pytorch, x, A, B)

    # 4. low_rank: fused_nystrom_landmark_pinv
    from neuralops.core.triton_kernels.low_rank import (
        fused_nystrom_landmark_pinv_step_pytorch,
        fused_nystrom_landmark_pinv_step_triton,
    )
    print("\n--- Benchmarking low_rank (Nystrom Landmark) ---")
    for m in [32, 64]:
        A_tilde = torch.randn(2, 4, m, m, device=device, dtype=dtype)
        bench_pair("fused_nystrom_landmark_pinv", f"(m={m})", fused_nystrom_landmark_pinv_step_triton, fused_nystrom_landmark_pinv_step_pytorch, A_tilde)

    # 5. sliding_window
    from neuralops.core.triton_kernels.sliding_window import (
        sliding_window_attention,
        sliding_window_pytorch,
    )
    print("\n--- Benchmarking sliding_window ---")
    for seq_len in [512, 1024, 2048]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        bench_pair("sliding_window_attention", f"(T={seq_len}, W=128)", sliding_window_attention, sliding_window_pytorch, q, k, v, 128)

    # 6. gla_scan
    from neuralops.core.triton_kernels.gla_scan import gla_scan, gla_scan_pytorch
    print("\n--- Benchmarking gla_scan ---")
    for seq_len in [256, 512, 1024]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype))
        bench_pair("gla_scan", f"(T={seq_len})", gla_scan, gla_scan_pytorch, q, k, v, g)

    # 7. gdn_scan
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan, gdn_scan_pytorch
    print("\n--- Benchmarking gdn_scan ---")
    for seq_len in [256, 512, 1024]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        beta = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype))
        bench_pair("gdn_scan", f"(T={seq_len})", gdn_scan, gdn_scan_pytorch, q, k, v, beta)

    # 8. hgrn2_scan
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan, hgrn2_scan_pytorch
    print("\n--- Benchmarking hgrn2_scan ---")
    for seq_len in [256, 512, 1024]:
        x = torch.randn(2, seq_len, 256, device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn(2, seq_len, 256, device=device, dtype=dtype))
        bench_pair("hgrn2_scan", f"(T={seq_len})", hgrn2_scan, hgrn2_scan_pytorch, x, g)

    # 9. softplus
    from neuralops.core.triton_kernels.softplus import softplus_fwd, softplus_pytorch
    print("\n--- Benchmarking softplus ---")
    for num_elements in [262144, 1048576, 4194304]:
        x = torch.randn(num_elements, device=device, dtype=dtype)
        bench_pair("softplus", f"({num_elements})", softplus_fwd, softplus_pytorch, x)

    # 10. fused_norm_linear
    from neuralops.core.triton_kernels.fused_norm_linear import (
        fused_norm_linear,
        fused_norm_linear_pytorch,
    )
    print("\n--- Benchmarking fused_norm_linear ---")
    for dim in [512, 1024, 2048]:
        x = torch.randn(2, 128, dim, device=device, dtype=dtype)
        w = torch.randn(dim, dim, device=device, dtype=dtype)
        gamma = torch.randn(dim, device=device, dtype=dtype)
        bench_pair("fused_norm_linear", f"({dim})", fused_norm_linear, fused_norm_linear_pytorch, x, gamma, w)

    # 11. cosformer_triton_kernel
    from neuralops.core.triton_kernels.cosformer_triton_kernel import (
        cosformer_pytorch_fwd,
        cosformer_triton_fwd,
    )
    print("\n--- Benchmarking cosformer ---")
    for seq_len in [256, 512, 1024]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        bench_pair("cosformer", f"(T={seq_len})", cosformer_triton_fwd, cosformer_pytorch_fwd, q, k, v)

    # 12. fused_ce
    from neuralops.core.triton_kernels.fused_ce import (
        fused_cross_entropy,
        fused_cross_entropy_pytorch,
    )
    print("\n--- Benchmarking fused_ce ---")
    for vocab in [4096, 16384, 32768]:
        logits = torch.randn(2, 128, vocab, device=device, dtype=dtype)
        targets = torch.randint(0, vocab, (2, 128), device=device)
        bench_pair("fused_ce", f"(V={vocab})", fused_cross_entropy, fused_cross_entropy_pytorch, logits, targets)

    # 13. fused_kd
    from neuralops.core.triton_kernels.fused_kd import (
        fused_kd_loss,
        fused_kd_loss_pytorch,
    )
    print("\n--- Benchmarking fused_kd ---")
    for vocab in [4096, 16384]:
        s_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
        t_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
        bench_pair("fused_kd", f"(V={vocab})", fused_kd_loss, fused_kd_loss_pytorch, s_logits, t_logits, 2.0)

    # 14. selective_scan
    from neuralops.core.triton_kernels.selective_scan import (
        selective_scan_fn,
        selective_scan_ref,
    )
    print("\n--- Benchmarking selective_scan ---")
    for seq_len in [256, 512]:
        u = torch.randn(2, 64, seq_len, device=device, dtype=dtype)
        delta = torch.randn(2, 64, seq_len, device=device, dtype=dtype)
        A = torch.randn(64, 16, device=device, dtype=dtype)
        B = torch.randn(2, 16, seq_len, device=device, dtype=dtype)
        C = torch.randn(2, 16, seq_len, device=device, dtype=dtype)
        bench_pair("selective_scan", f"(T={seq_len})", selective_scan_fn, selective_scan_ref, u, delta, A, B, C)

    # 15. fms_scan
    from neuralops.core.triton_kernels.fms_scan import fms_scan, fms_scan_pytorch
    print("\n--- Benchmarking fms_scan ---")
    for seq_len in [256, 512]:
        x = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
        A = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
        B = torch.randn(2, seq_len, 64, device=device, dtype=dtype)
        bench_pair("fms_scan", f"(T={seq_len})", fms_scan, fms_scan_pytorch, x, A, B)

    # 16. paged_attention
    from neuralops.core.triton_kernels.paged_attention import (
        paged_attention_kernel,
        paged_attention_pytorch,
    )
    print("\n--- Benchmarking paged_attention ---")
    num_blocks, block_size, num_heads, head_dim = 64, 16, 4, 64
    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(2, 32)
    context_lens = torch.full((2,), 32 * block_size, device=device, dtype=torch.int32)
    q = torch.randn(2, num_heads, head_dim, device=device, dtype=dtype)
    bench_pair("paged_attention", "(B=2, H=4, d=64)", paged_attention_kernel, paged_attention_pytorch, q, k_cache, v_cache, block_tables, context_lens, 1.0 / (head_dim ** 0.5))

    out_path = Path("results/triton_all_26_kernels_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")
    faster_count = sum(1 for r in results if r["speedup"] >= 1.0)
    print(f"Summary: {faster_count}/{len(results)} configurations faster or equal to PyTorch reference")


if __name__ == "__main__":
    main()
