# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""
Benchmark and Correctness Verification Suite for Part 1 (12 Modules) on NVIDIA RTX 4070 GPU.
Measures:
1. Real CUDA latency (ms)
2. Real PyTorch reference latency (ms)
3. Speedup (Ref / CUDA)
4. Max Absolute Difference (strictly diff < 1e-3)
5. Status PASS
"""

import time
import torch

from neuralops.core.cuda_kernels.alibi_attention.alibi_autograd import alibi_attention, alibi_attention_pytorch_ref
from neuralops.core.cuda_kernels.cluster_attention.cluster_attention_autograd import (
    cluster_attention,
    cluster_attention_ref,
)
from neuralops.core.cuda_kernels.cross_attention.cross_attn_autograd import cross_attention, cross_attention_pytorch_ref
from neuralops.core.cuda_kernels.differential_attention.differential_attention_autograd import (
    differential_attention,
    differential_attention_ref,
)
from neuralops.core.cuda_kernels.dynamic_conv1d.dynamic_conv1d_autograd import dynamic_conv1d, dynamic_conv1d_ref
from neuralops.core.cuda_kernels.entropy_gate.entropy_gate_autograd import (
    spectral_entropy_svd_gate,
    spectral_entropy_svd_gate_pytorch_ref,
)
from neuralops.core.cuda_kernels.flash_decoding.flash_dec_autograd import (
    flash_decoding_attention,
    flash_decoding_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fms_scan.fms_autograd import fms_scan, fms_scan_pytorch_ref
from neuralops.core.cuda_kernels.gla_scan.gla_autograd import gla_scan, gla_scan_pytorch_ref
from neuralops.core.cuda_kernels.latent_kv_attn.latent_kv_autograd import (
    latent_kv_cache_attention,
    latent_kv_cache_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.linformer_attention.linformer_attention_autograd import (
    linformer_attention,
    linformer_attention_ref,
)
from neuralops.core.cuda_kernels.local_attention.local_attention_autograd import local_attention, local_attention_ref


def time_fn(fn, warmup=5, iters=20):
    """Measures execution latency using high-resolution CUDA Events."""
    assert callable(fn), "fn must be callable"
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000.0


def benchmark_module(name, cuda_fn, ref_fn):
    """Execute benchmark module routine."""
    assert callable(cuda_fn), f"cuda_fn for {name} must be callable"
    assert callable(ref_fn), f"ref_fn for {name} must be callable"
    out_cuda = cuda_fn()
    out_ref = ref_fn()
    diff = (out_cuda - out_ref).abs().max().item()

    cuda_ms = time_fn(cuda_fn)
    ref_ms = time_fn(ref_fn)
    speedup = ref_ms / cuda_ms if cuda_ms > 0 else 1.0
    status = "PASS" if diff < 1e-3 else "FAIL_ACCURACY"

    return {
        "name": name,
        "cuda_ms": cuda_ms,
        "ref_ms": ref_ms,
        "speedup": speedup,
        "diff": diff,
        "status": status,
    }


def main():
    """Execute main routine."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    results = []

    # 1. alibi_attention
    B, H, L, D = 2, 8, 256, 64
    q1 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k1 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v1 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    slopes1 = torch.randn(H, device=device, dtype=torch.float32)
    results.append(
        benchmark_module(
            "alibi_attention",
            lambda: alibi_attention(q1, k1, v1, slopes1, is_causal=True),
            lambda: alibi_attention_pytorch_ref(q1, k1, v1, slopes1, is_causal=True),
        )
    )

    # 2. cluster_attention
    B, H, N, D, K_clust = 2, 8, 256, 64, 32
    q2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    c2 = torch.randn(B, H, K_clust, D, device=device, dtype=dtype) * 0.1
    v2 = torch.randn(B, H, K_clust, D, device=device, dtype=dtype) * 0.1
    idx2 = torch.randint(0, K_clust, (B, H, N), device=device, dtype=torch.int32)
    results.append(
        benchmark_module(
            "cluster_attention",
            lambda: cluster_attention(q2, c2, v2, idx2),
            lambda: cluster_attention_ref(q2, c2, v2, idx2),
        )
    )

    # 3. cross_attention
    B, H, L_q, L_kv, D = 2, 8, 256, 256, 64
    q3 = torch.randn(B, H, L_q, D, device=device, dtype=dtype) * 0.1
    k3 = torch.randn(B, H, L_kv, D, device=device, dtype=dtype) * 0.1
    v3 = torch.randn(B, H, L_kv, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "cross_attention",
            lambda: cross_attention(q3, k3, v3),
            lambda: cross_attention_pytorch_ref(q3, k3, v3),
        )
    )

    # 4. differential_attention
    B, H, N, D = 2, 8, 256, 64
    q4_1 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k4_1 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    q4_2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k4_2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    v4 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "differential_attention",
            lambda: differential_attention(q4_1, k4_1, q4_2, k4_2, v4, lambda_val=0.8),
            lambda: differential_attention_ref(q4_1, k4_1, q4_2, k4_2, v4, lambda_val=0.8),
        )
    )

    # 5. dynamic_conv1d
    B, D_conv, L_conv, K_conv = 4, 64, 512, 8
    x5 = torch.randn(B, D_conv, L_conv, device=device, dtype=dtype) * 0.1
    w5 = torch.randn(D_conv, K_conv, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "dynamic_conv1d",
            lambda: dynamic_conv1d(x5, w5),
            lambda: dynamic_conv1d_ref(x5, w5),
        )
    )

    # 6. entropy_gate
    B, M, K_dim, N_dim, R = 4, 128, 128, 128, 32
    x6 = torch.randn(B, M, K_dim, device=device, dtype=dtype) * 0.1
    V6 = torch.randn(K_dim, R, device=device, dtype=dtype) * 0.1
    U6 = torch.randn(R, N_dim, device=device, dtype=dtype) * 0.1
    eff_r = 16
    results.append(
        benchmark_module(
            "entropy_gate",
            lambda: spectral_entropy_svd_gate(x6, V6, U6, eff_r),
            lambda: spectral_entropy_svd_gate_pytorch_ref(x6, V6, U6, eff_r),
        )
    )

    # 7. flash_decoding
    B, H, N_q, N_kv, D = 4, 8, 1, 1024, 128
    q7 = torch.randn(B, H, N_q, D, device=device, dtype=dtype) * 0.1
    k7 = torch.randn(B, H, N_kv, D, device=device, dtype=dtype) * 0.1
    v7 = torch.randn(B, H, N_kv, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "flash_decoding",
            lambda: flash_decoding_attention(q7, k7, v7),
            lambda: flash_decoding_attention_pytorch_ref(q7, k7, v7),
        )
    )

    # 8. fms_scan
    B, H, L, D = 2, 8, 256, 64
    decay8 = torch.full((H,), 0.95, device=device, dtype=torch.float32)
    q8 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k8 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v8 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "fms_scan",
            lambda: fms_scan(q8, k8, v8, decay8),
            lambda: fms_scan_pytorch_ref(q8, k8, v8, decay8),
        )
    )

    # 9. gla_scan
    B, H, L, D = 2, 8, 256, 64
    q9 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k9 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v9 = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "gla_scan",
            lambda: gla_scan(q9, k9, v9),
            lambda: gla_scan_pytorch_ref(q9, k9, v9),
        )
    )

    # 10. latent_kv_attn
    B, H, L_Q, L_KV, D_K, D_V, R = 2, 8, 128, 256, 64, 64, 32
    q10 = torch.randn(B, H, L_Q, D_K, device=device, dtype=dtype) * 0.1
    c10 = torch.randn(B, L_KV, R, device=device, dtype=dtype) * 0.1
    uk10 = torch.randn(D_K, R, device=device, dtype=dtype) * 0.1
    uv10 = torch.randn(R, D_V, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "latent_kv_attn",
            lambda: latent_kv_cache_attention(q10, c10, uk10, uv10),
            lambda: latent_kv_cache_attention_pytorch_ref(q10, c10, uk10, uv10),
        )
    )

    # 11. linformer_attention
    B, H, N, D, K_proj_len = 2, 8, 256, 64, 32
    q11 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    kp11 = torch.randn(B, H, K_proj_len, D, device=device, dtype=dtype) * 0.1
    vp11 = torch.randn(B, H, K_proj_len, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "linformer_attention",
            lambda: linformer_attention(q11, kp11, vp11),
            lambda: linformer_attention_ref(q11, kp11, vp11),
        )
    )

    # 12. local_attention
    B, H, N, D = 2, 8, 256, 64
    q12 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k12 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    v12 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    results.append(
        benchmark_module(
            "local_attention",
            lambda: local_attention(q12, k12, v12, window_size=64),
            lambda: local_attention_ref(q12, k12, v12, window_size=64),
        )
    )

    print("=" * 105)
    print(
        f"{'Idx':<4} | {'Module Name':<25} | {'CUDA (ms)':<10} | {'Ref (ms)':<10} | {'Speedup':<9} | {'Max Diff':<12} | {'Status'}"
    )
    print("-" * 105)

    all_pass = True
    for idx, r in enumerate(results, 1):
        print(
            f"{idx:<4} | {r['name']:<25} | {r['cuda_ms']:<10.4f} | {r['ref_ms']:<10.4f} | {r['speedup']:<9.2f}x | {r['diff']:<12.6e} | {r['status']}"
        )
        if r["status"] != "PASS":
            all_pass = False

    print("=" * 105)
    assert all_pass, "Some modules failed accuracy checks!"
    print("ALL 12 MODULES VERIFIED & BENCHMARKED SUCCESSFULLY ON RTX 4070!")


if __name__ == "__main__":
    main()
