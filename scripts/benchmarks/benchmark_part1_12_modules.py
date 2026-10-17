"""
Benchmark and Correctness Verification Suite for Part 1 (12 Modules) on NVIDIA RTX 4070 GPU.
Measures:
1. Real CUDA latency (ms)
2. Real PyTorch reference latency (ms)
3. Speedup (Ref / CUDA)
4. Max Absolute Difference (strictly diff < 1e-3)
5. Status PASS
"""

import math
import time
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels.alibi_attention.alibi_autograd import (
    alibi_attention, alibi_attention_pytorch_ref
)
from neuralops.core.cuda_kernels.cluster_attention.cluster_attention_autograd import (
    cluster_attention, cluster_attention_ref
)
from neuralops.core.cuda_kernels.cross_attention.cross_attn_autograd import (
    cross_attention, cross_attention_pytorch_ref
)
from neuralops.core.cuda_kernels.differential_attention.differential_attention_autograd import (
    differential_attention, differential_attention_ref
)
from neuralops.core.cuda_kernels.dynamic_conv1d.dynamic_conv1d_autograd import (
    dynamic_conv1d, dynamic_conv1d_ref
)
from neuralops.core.cuda_kernels.entropy_gate.entropy_gate_autograd import (
    spectral_entropy_svd_gate, spectral_entropy_svd_gate_pytorch_ref
)
from neuralops.core.cuda_kernels.flash_decoding.flash_dec_autograd import (
    flash_decoding_attention, flash_decoding_attention_pytorch_ref
)
from neuralops.core.cuda_kernels.fms_scan.fms_autograd import (
    fms_scan, fms_scan_pytorch_ref
)
from neuralops.core.cuda_kernels.gla_scan.gla_autograd import (
    gla_scan, gla_scan_pytorch_ref
)
from neuralops.core.cuda_kernels.latent_kv_attn.latent_kv_autograd import (
    latent_kv_cache_attention, latent_kv_cache_attention_pytorch_ref
)
from neuralops.core.cuda_kernels.linformer_attention.linformer_attention_autograd import (
    linformer_attention, linformer_attention_ref
)
from neuralops.core.cuda_kernels.local_attention.local_attention_autograd import (
    local_attention, local_attention_ref
)


def benchmark_fn(fn, args, kwargs=None, warmup=5, iters=20):
    kwargs = kwargs or {}
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def run_all_benchmarks():
    device = torch.device("cuda:0")
    dtype = torch.float16

    modules = {}

    # 1. alibi_attention
    B, H, L, D = 2, 8, 256, 64
    q = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    slopes = torch.randn(H, device=device, dtype=torch.float32)
    modules["alibi_attention"] = {
        "cuda_fn": lambda: alibi_attention(q, k, v, slopes, is_causal=True),
        "ref_fn": lambda: alibi_attention_pytorch_ref(q, k, v, slopes, is_causal=True),
    }

    # 2. cluster_attention
    B, H, N, D, K_clust = 2, 8, 256, 64, 32
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    c_cen = torch.randn(B, H, K_clust, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, K_clust, D, device=device, dtype=dtype) * 0.1
    c_idx = torch.randint(0, K_clust, (B, H, N), device=device, dtype=torch.int32)
    modules["cluster_attention"] = {
        "cuda_fn": lambda: cluster_attention(q, c_cen, v, c_idx),
        "ref_fn": lambda: cluster_attention_ref(q, c_cen, v, c_idx),
    }

    # 3. cross_attention
    B, H, L_q, L_kv, D = 2, 8, 256, 256, 64
    q = torch.randn(B, H, L_q, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, L_kv, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, L_kv, D, device=device, dtype=dtype) * 0.1
    modules["cross_attention"] = {
        "cuda_fn": lambda: cross_attention(q, k, v),
        "ref_fn": lambda: cross_attention_pytorch_ref(q, k, v),
    }

    # 4. differential_attention
    B, H, N, D = 2, 8, 256, 64
    q1 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k1 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    q2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k2 = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    modules["differential_attention"] = {
        "cuda_fn": lambda: differential_attention(q1, k1, q2, k2, v, lambda_val=0.8),
        "ref_fn": lambda: differential_attention_ref(q1, k1, q2, k2, v, lambda_val=0.8),
    }

    # 5. dynamic_conv1d
    B, D_conv, L_conv, K_conv = 4, 64, 512, 8
    x_3d = torch.randn(B, D_conv, L_conv, device=device, dtype=dtype) * 0.1
    w_3d = torch.randn(D_conv, K_conv, device=device, dtype=dtype) * 0.1
    modules["dynamic_conv1d"] = {
        "cuda_fn": lambda: dynamic_conv1d(x_3d, w_3d),
        "ref_fn": lambda: dynamic_conv1d_ref(x_3d, w_3d),
    }

    # 6. entropy_gate
    B, M, K_dim, N_dim, R = 4, 128, 128, 128, 32
    x = torch.randn(B, M, K_dim, device=device, dtype=dtype) * 0.1
    V_r = torch.randn(K_dim, R, device=device, dtype=dtype) * 0.1
    U_r = torch.randn(R, N_dim, device=device, dtype=dtype) * 0.1
    eff_r = 16
    modules["entropy_gate"] = {
        "cuda_fn": lambda: spectral_entropy_svd_gate(x, V_r, U_r, eff_r),
        "ref_fn": lambda: spectral_entropy_svd_gate_pytorch_ref(x, V_r, U_r, eff_r),
    }

    # 7. flash_decoding
    B, H, N_q, N_kv, D = 4, 8, 1, 1024, 128
    q = torch.randn(B, H, N_q, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, N_kv, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, N_kv, D, device=device, dtype=dtype) * 0.1
    modules["flash_decoding"] = {
        "cuda_fn": lambda: flash_decoding_attention(q, k, v),
        "ref_fn": lambda: flash_decoding_attention_pytorch_ref(q, k, v),
    }

    # 8. fms_scan
    B, H, L, D = 2, 8, 256, 64
    decay = torch.full((H,), 0.95, device=device, dtype=torch.float32)
    q = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    modules["fms_scan"] = {
        "cuda_fn": lambda: fms_scan(q, k, v, decay),
        "ref_fn": lambda: fms_scan_pytorch_ref(q, k, v, decay),
    }

    # 9. gla_scan
    B, H, L, D = 2, 8, 256, 64
    q = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    modules["gla_scan"] = {
        "cuda_fn": lambda: gla_scan(q, k, v),
        "ref_fn": lambda: gla_scan_pytorch_ref(q, k, v),
    }

    # 10. latent_kv_attn
    B, H, L_Q, L_KV, D_K, D_V, R = 2, 8, 128, 256, 64, 64, 32
    q = torch.randn(B, H, L_Q, D_K, device=device, dtype=dtype) * 0.1
    c_kv = torch.randn(B, L_KV, R, device=device, dtype=dtype) * 0.1
    u_k = torch.randn(D_K, R, device=device, dtype=dtype) * 0.1
    u_v = torch.randn(R, D_V, device=device, dtype=dtype) * 0.1
    modules["latent_kv_attn"] = {
        "cuda_fn": lambda: latent_kv_cache_attention(q, c_kv, u_k, u_v),
        "ref_fn": lambda: latent_kv_cache_attention_pytorch_ref(q, c_kv, u_k, u_v),
    }

    # 11. linformer_attention
    B, H, N, D, K_proj_len = 2, 8, 256, 64, 32
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k_proj = torch.randn(B, H, K_proj_len, D, device=device, dtype=dtype) * 0.1
    v_proj = torch.randn(B, H, K_proj_len, D, device=device, dtype=dtype) * 0.1
    modules["linformer_attention"] = {
        "cuda_fn": lambda: linformer_attention(q, k_proj, v_proj),
        "ref_fn": lambda: linformer_attention_ref(q, k_proj, v_proj),
    }

    # 12. local_attention
    B, H, N, D = 2, 8, 256, 64
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    modules["local_attention"] = {
        "cuda_fn": lambda: local_attention(q, k, v, window_size=64),
        "ref_fn": lambda: local_attention_ref(q, k, v, window_size=64),
    }

    print("=" * 105)
    print(f"{'Idx':<4} | {'Module Name':<25} | {'CUDA (ms)':<10} | {'Ref (ms)':<10} | {'Speedup':<9} | {'Max Diff':<12} | {'Status'}")
    print("-" * 105)

    all_pass = True
    for idx, (name, mod) in enumerate(modules.items(), 1):
        out_cuda = mod["cuda_fn"]()
        out_ref = mod["ref_fn"]()
        max_diff = (out_cuda - out_ref).abs().max().item()

        cuda_ms = benchmark_fn(mod["cuda_fn"], [])
        ref_ms = benchmark_fn(mod["ref_fn"], [])
        speedup = ref_ms / cuda_ms if cuda_ms > 0 else 1.0

        status = "PASS" if max_diff < 1e-3 else "FAIL_ACCURACY"
        if status != "PASS":
            all_pass = False

        print(f"{idx:<4} | {name:<25} | {cuda_ms:<10.4f} | {ref_ms:<10.4f} | {speedup:<9.2f}x | {max_diff:<12.6e} | {status}")

    print("=" * 105)
    assert all_pass, "Some modules failed accuracy checks!"
    print("ALL 12 MODULES VERIFIED & BENCHMARKED SUCCESSFULLY ON RTX 4070!")


if __name__ == "__main__":
    run_all_benchmarks()
