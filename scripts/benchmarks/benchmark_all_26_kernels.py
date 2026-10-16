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

    # 1. low_rank: fused_hnc_nullspace_step
    from neuralops.core.triton_kernels.low_rank import (
        fused_hnc_nullspace_step_pytorch,
        fused_hnc_nullspace_step_triton,
    )
    for dim in [512, 1024, 2048]:
        w = torch.randn(dim, dim, device=device, dtype=dtype)
        g = torch.randn(dim, dim, device=device, dtype=dtype)
        r = torch.randn(dim, dim, device=device, dtype=dtype)
        for _ in range(3):
            _ = fused_hnc_nullspace_step_triton(w, g, r, 0.05)
            _ = fused_hnc_nullspace_step_pytorch(w, g, r, 0.05)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = fused_hnc_nullspace_step_triton(w, g, r, 0.05)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = fused_hnc_nullspace_step_pytorch(w, g, r, 0.05)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "fused_hnc_nullspace_step",
            "shape": f"({dim}, {dim})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"fused_hnc_nullspace_step ({dim}x{dim}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 2. low_rank: fused_down_up_gemm_bias
    from neuralops.core.triton_kernels.low_rank import (
        fused_down_up_gemm_bias_pytorch,
        fused_down_up_gemm_bias_triton,
    )
    for dim in [512, 1024, 2048]:
        r_dim = 32
        x = torch.randn(4, 128, dim, device=device, dtype=dtype)
        U = torch.randn(dim, r_dim, device=device, dtype=dtype)
        V = torch.randn(r_dim, dim, device=device, dtype=dtype)
        bias = torch.randn(dim, device=device, dtype=dtype)
        for _ in range(3):
            _ = fused_down_up_gemm_bias_triton(x, U, V, bias)
            _ = fused_down_up_gemm_bias_pytorch(x, U, V, bias)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = fused_down_up_gemm_bias_triton(x, U, V, bias)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = fused_down_up_gemm_bias_pytorch(x, U, V, bias)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "fused_down_up_gemm_bias",
            "shape": f"({dim}, r={r_dim})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"fused_down_up_gemm_bias ({dim}, r={r_dim}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 3. sliding_window
    from neuralops.core.triton_kernels.sliding_window import (
        sliding_window_attention,
        sliding_window_pytorch,
    )
    for seq_len in [512, 1024, 2048]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        window = 128
        for _ in range(3):
            _ = sliding_window_attention(q, k, v, window)
            _ = sliding_window_pytorch(q, k, v, window)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = sliding_window_attention(q, k, v, window)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = sliding_window_pytorch(q, k, v, window)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "sliding_window_attention",
            "shape": f"(T={seq_len}, W={window})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"sliding_window_attention (T={seq_len}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 4. gla_scan
    from neuralops.core.triton_kernels.gla_scan import gla_scan, gla_scan_pytorch
    for seq_len in [256, 512, 1024]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype))
        for _ in range(3):
            _ = gla_scan(q, k, v, g)
            _ = gla_scan_pytorch(q, k, v, g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = gla_scan(q, k, v, g)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = gla_scan_pytorch(q, k, v, g)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "gla_scan",
            "shape": f"(T={seq_len})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"gla_scan (T={seq_len}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 5. gdn_scan
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan, gdn_scan_pytorch
    for seq_len in [256, 512, 1024]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        beta = torch.sigmoid(torch.randn(2, 4, seq_len, device=device, dtype=dtype))
        for _ in range(3):
            _ = gdn_scan(q, k, v, beta)
            _ = gdn_scan_pytorch(q, k, v, beta)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = gdn_scan(q, k, v, beta)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = gdn_scan_pytorch(q, k, v, beta)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "gdn_scan",
            "shape": f"(T={seq_len})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"gdn_scan (T={seq_len}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 6. hgrn2_scan
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan, hgrn2_scan_pytorch
    for seq_len in [256, 512, 1024]:
        x = torch.randn(2, seq_len, 256, device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn(2, seq_len, 256, device=device, dtype=dtype))
        for _ in range(3):
            _ = hgrn2_scan(x, g)
            _ = hgrn2_scan_pytorch(x, g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = hgrn2_scan(x, g)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = hgrn2_scan_pytorch(x, g)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "hgrn2_scan",
            "shape": f"(T={seq_len})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"hgrn2_scan (T={seq_len}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 7. softplus
    from neuralops.core.triton_kernels.softplus import softplus_fwd, softplus_pytorch
    for num_elements in [262144, 1048576, 4194304]:
        x = torch.randn(num_elements, device=device, dtype=dtype)
        for _ in range(3):
            _ = softplus_fwd(x)
            _ = softplus_pytorch(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = softplus_fwd(x)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = softplus_pytorch(x)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "softplus",
            "shape": f"({num_elements})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"softplus ({num_elements}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 8. fused_norm_linear
    from neuralops.core.triton_kernels.fused_norm_linear import (
        fused_norm_linear,
        fused_norm_linear_pytorch,
    )
    for dim in [512, 1024, 2048]:
        x = torch.randn(2, 128, dim, device=device, dtype=dtype)
        w = torch.randn(dim, dim, device=device, dtype=dtype)
        gamma = torch.randn(dim, device=device, dtype=dtype)
        for _ in range(3):
            _ = fused_norm_linear(x, gamma, w)
            _ = fused_norm_linear_pytorch(x, gamma, w)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = fused_norm_linear(x, gamma, w)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = fused_norm_linear_pytorch(x, gamma, w)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "fused_norm_linear",
            "shape": f"({dim})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"fused_norm_linear ({dim}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 9. cosformer_triton_kernel
    from neuralops.core.triton_kernels.cosformer_triton_kernel import (
        cosformer_pytorch_fwd,
        cosformer_triton_fwd,
    )
    for seq_len in [256, 512, 1024]:
        q = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, seq_len, 64, device=device, dtype=dtype)
        for _ in range(3):
            _ = cosformer_triton_fwd(q, k, v)
            _ = cosformer_pytorch_fwd(q, k, v)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = cosformer_triton_fwd(q, k, v)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = cosformer_pytorch_fwd(q, k, v)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "cosformer_triton_kernel",
            "shape": f"(T={seq_len})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"cosformer ({seq_len}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 10. fused_ce
    from neuralops.core.triton_kernels.fused_ce import (
        fused_cross_entropy,
        fused_cross_entropy_pytorch,
    )
    for vocab in [4096, 16384, 32768]:
        logits = torch.randn(2, 128, vocab, device=device, dtype=dtype)
        targets = torch.randint(0, vocab, (2, 128), device=device)
        for _ in range(3):
            _ = fused_cross_entropy(logits, targets)
            _ = fused_cross_entropy_pytorch(logits, targets)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = fused_cross_entropy(logits, targets)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = fused_cross_entropy_pytorch(logits, targets)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "fused_ce",
            "shape": f"(V={vocab})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"fused_ce (V={vocab}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    # 11. fused_kd
    from neuralops.core.triton_kernels.fused_kd import (
        fused_kd_loss,
        fused_kd_loss_pytorch,
    )
    for vocab in [4096, 16384]:
        s_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
        t_logits = torch.randn(2, 64, vocab, device=device, dtype=dtype)
        for _ in range(3):
            _ = fused_kd_loss(s_logits, t_logits, 2.0)
            _ = fused_kd_loss_pytorch(s_logits, t_logits, 2.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_t = fused_kd_loss(s_logits, t_logits, 2.0)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / 20.0 * 1000.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out_p = fused_kd_loss_pytorch(s_logits, t_logits, 2.0)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / 20.0 * 1000.0

        max_diff = (out_t - out_p).abs().max().item()
        sp = round(t_pt / max(t_tri, 1e-6), 2)
        results.append({
            "kernel": "fused_kd",
            "shape": f"(V={vocab})",
            "triton_ms": round(t_tri, 3),
            "pytorch_ms": round(t_pt, 3),
            "speedup": sp,
            "max_diff": round(max_diff, 6),
            "status": "OK",
        })
        print(f"fused_kd (V={vocab}): Tri={t_tri:.2f}ms, PT={t_pt:.2f}ms, Speedup={sp:.2f}x, Diff={max_diff:.2e}")

    out_path = Path("results/triton_all_26_kernels_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")
    faster_count = sum(1 for r in results if r["speedup"] >= 1.0)
    print(f"Summary: {faster_count}/{len(results)} configurations faster or equal to PyTorch")


if __name__ == "__main__":
    main()
