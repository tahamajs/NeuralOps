#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Benchmark for All Available Triton Modules in NeuralOps.
Measures Triton latency, PyTorch baseline latency, Speedup, Max Diff, and Status.
"""

import os
import sys
import time
import math
import csv
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def time_fn(fn, *args, warmups=10, iters=30, **kwargs):
    if device == "cuda":
        for _ in range(warmups):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
    else:
        for _ in range(warmups):
            fn(*args, **kwargs)
        start = time.perf_counter()
        for _ in range(iters):
            fn(*args, **kwargs)
        end = time.perf_counter()
    return (end - start) / iters * 1000.0

def run_benchmarks():
    print(f"Benchmarking All Triton Modules on device: {device}, dtype: {dtype}")
    results = []

    # 1. Softplus
    try:
        from neuralops.ops.triton.softplus import softplus_triton, softplus_pytorch
        x = torch.randn(1024, 2048, device=device, dtype=dtype)
        t_tri = time_fn(softplus_triton, x)
        t_pt = time_fn(softplus_pytorch, x)
        diff = (softplus_triton(x) - softplus_pytorch(x)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "softplus", "category": "elementwise", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'softplus':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"softplus setup error: {e}")

    # 2. Fused Cross Entropy
    try:
        from neuralops.ops.triton.fused_ce import fused_cross_entropy_triton, fused_cross_entropy_pytorch
        logits = torch.randn(256, 8192, device=device, dtype=dtype)
        targets = torch.randint(0, 8192, (256,), device=device, dtype=torch.long)
        t_tri = time_fn(fused_cross_entropy_triton, logits, targets)
        t_pt = time_fn(fused_cross_entropy_pytorch, logits, targets)
        diff = (fused_cross_entropy_triton(logits, targets) - fused_cross_entropy_pytorch(logits, targets)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "fused_ce", "category": "loss", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_ce':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_ce setup error: {e}")

    # 3. Cosformer
    try:
        from neuralops.ops.triton.cosformer_triton_kernel import cosformer_linear_attention_triton, cosformer_linear_attention_pytorch
        B, H, T, D = 2, 8, 512, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        t_tri = time_fn(cosformer_linear_attention_triton, q, k, v)
        t_pt = time_fn(cosformer_linear_attention_pytorch, q, k, v)
        diff = (cosformer_linear_attention_triton(q, k, v) - cosformer_linear_attention_pytorch(q, k, v)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "cosformer", "category": "linear_attention", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'cosformer':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"cosformer setup error: {e}")

    # 4. HGRN2 Scan
    try:
        from neuralops.ops.triton.hgrn2_scan import hgrn2_scan_triton, hgrn2_scan_pytorch
        B, H, T, D = 2, 4, 512, 64
        x = torch.randn(B, H, T, D, device=device, dtype=dtype)
        g = torch.randn(B, H, T, D, device=device, dtype=dtype).sigmoid()
        t_tri = time_fn(hgrn2_scan_triton, x, g)
        t_pt = time_fn(hgrn2_scan_pytorch, x, g)
        diff = (hgrn2_scan_triton(x, g) - hgrn2_scan_pytorch(x, g)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "hgrn2_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'hgrn2_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"hgrn2_scan setup error: {e}")

    # 5. GLA Scan
    try:
        from neuralops.ops.triton.gla_scan import gla_scan_triton, gla_scan_pytorch
        B, H, T, D = 2, 4, 256, 32
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        g = -torch.rand(B, H, T, D, device=device, dtype=dtype)
        t_tri = time_fn(gla_scan_triton, q, k, v, g)
        t_pt = time_fn(gla_scan_pytorch, q, k, v, g)
        diff = (gla_scan_triton(q, k, v, g) - gla_scan_pytorch(q, k, v, g)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "gla_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'gla_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"gla_scan setup error: {e}")

    # 6. Selective Scan
    try:
        from neuralops.ops.triton.selective_scan import selective_scan_triton, selective_scan_pytorch
        B, D, L = 2, 64, 512
        N = 16
        u = torch.randn(B, D, L, device=device, dtype=dtype)
        delta = torch.randn(B, D, L, device=device, dtype=dtype).abs() + 0.1
        A = -torch.rand(D, N, device=device, dtype=dtype) - 0.1
        B_mat = torch.randn(B, N, L, device=device, dtype=dtype)
        C_mat = torch.randn(B, N, L, device=device, dtype=dtype)
        D_vec = torch.randn(D, device=device, dtype=dtype)
        t_tri = time_fn(selective_scan_triton, u, delta, A, B_mat, C_mat, D_vec)
        t_pt = time_fn(selective_scan_pytorch, u, delta, A, B_mat, C_mat, D_vec)
        out_tri = selective_scan_triton(u, delta, A, B_mat, C_mat, D_vec)
        out_pt = selective_scan_pytorch(u, delta, A, B_mat, C_mat, D_vec)
        diff = (out_tri - out_pt).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 5e-2 else ("SLOW" if diff < 5e-2 else "FAIL")
        results.append({"module": "selective_scan", "category": "ssm", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'selective_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"selective_scan setup error: {e}")

    # 7. Flash Decoding
    try:
        from neuralops.ops.triton.flash_decoding import flash_decoding_attention, flash_decoding_pytorch
        B, H, D = 2, 8, 64
        S = 1024
        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        t_tri = time_fn(flash_decoding_attention, q, k, v)
        t_pt = time_fn(flash_decoding_pytorch, q, k, v)
        diff = (flash_decoding_attention(q, k, v) - flash_decoding_pytorch(q, k, v)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "flash_decoding", "category": "attention", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'flash_decoding':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"flash_decoding setup error: {e}")

    # 8. Fused Goldfinch Hopfield
    try:
        from neuralops.ops.triton.fms_scan import fused_goldfinch_hopfield_lookup_triton, fused_goldfinch_hopfield_lookup_pytorch
        B, H, T, D = 2, 4, 128, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        mem = torch.randn(B, H, 256, D, device=device, dtype=dtype)
        val = torch.randn(B, H, 256, D, device=device, dtype=dtype)
        t_tri = time_fn(fused_goldfinch_hopfield_lookup_triton, q, mem, val)
        t_pt = time_fn(fused_goldfinch_hopfield_lookup_pytorch, q, mem, val)
        diff = (fused_goldfinch_hopfield_lookup_triton(q, mem, val) - fused_goldfinch_hopfield_lookup_pytorch(q, mem, val)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "fused_goldfinch_hopfield", "category": "hopfield", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_goldfinch_hopfield':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_goldfinch_hopfield setup error: {e}")

    # 9. Fused Unified RoPE
    try:
        from neuralops.ops.triton.fused_rope import fused_unified_rope_offset_triton, fused_unified_rope_offset_pytorch
        B, H, T, D = 2, 8, 512, 64
        x = torch.randn(B, H, T, D, device=device, dtype=dtype)
        cos = torch.randn(1, 1, T, D, device=device, dtype=dtype)
        sin = torch.randn(1, 1, T, D, device=device, dtype=dtype)
        t_tri = time_fn(fused_unified_rope_offset_triton, x, cos, sin)
        t_pt = time_fn(fused_unified_rope_offset_pytorch, x, cos, sin)
        diff = (fused_unified_rope_offset_triton(x, cos, sin) - fused_unified_rope_offset_pytorch(x, cos, sin)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "fused_rope", "category": "rotary", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_rope':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_rope setup error: {e}")

    # 10. Fused Norm Linear
    try:
        from neuralops.ops.triton.fused_norm_linear import fused_rms_norm_linear_triton, fused_rms_norm_linear_pytorch
        x = torch.randn(128, 512, 1024, device=device, dtype=dtype)
        weight_norm = torch.randn(1024, device=device, dtype=dtype)
        weight_linear = torch.randn(2048, 1024, device=device, dtype=dtype)
        t_tri = time_fn(fused_rms_norm_linear_triton, x, weight_norm, weight_linear)
        t_pt = time_fn(fused_rms_norm_linear_pytorch, x, weight_norm, weight_linear)
        diff = (fused_rms_norm_linear_triton(x, weight_norm, weight_linear) - fused_rms_norm_linear_pytorch(x, weight_norm, weight_linear)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "fused_norm_linear", "category": "normalization", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_norm_linear':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_norm_linear setup error: {e}")

    # 11. Fused MQA Bias Scale
    try:
        from neuralops.ops.triton.gqa import fused_mqa_bias_scale_triton, fused_mqa_bias_scale_pytorch
        B, H, T, D = 2, 8, 256, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        bias = torch.randn(1, H, 1, 1, device=device, dtype=dtype)
        scale = 0.125
        t_tri = time_fn(fused_mqa_bias_scale_triton, q, bias, scale)
        t_pt = time_fn(fused_mqa_bias_scale_pytorch, q, bias, scale)
        diff = (fused_mqa_bias_scale_triton(q, bias, scale) - fused_mqa_bias_scale_pytorch(q, bias, scale)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "fused_mqa_bias_scale", "category": "gqa", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_mqa_bias_scale':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_mqa_bias_scale setup error: {e}")

    # 12. Fused Hamming Scores
    try:
        from neuralops.ops.triton.sparse_attn import fused_hamming_scores_triton, fused_hamming_scores_pytorch
        codes_q = torch.randint(0, 256, (2, 8, 128), device=device, dtype=torch.uint8)
        codes_k = torch.randint(0, 256, (2, 8, 128), device=device, dtype=torch.uint8)
        t_tri = time_fn(fused_hamming_scores_triton, codes_q, codes_k)
        t_pt = time_fn(fused_hamming_scores_pytorch, codes_q, codes_k)
        diff = (fused_hamming_scores_triton(codes_q, codes_k) - fused_hamming_scores_pytorch(codes_q, codes_k)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "fused_hamming_scores", "category": "sparse", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_hamming_scores':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_hamming_scores setup error: {e}")

    # 13. RetNet Scan
    try:
        from neuralops.ops.triton.retnet_scan import retnet_scan_triton, retnet_scan_pytorch
        B, H, T, D = 2, 4, 256, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        gamma = 0.95
        t_tri = time_fn(retnet_scan_triton, q, k, v, gamma)
        t_pt = time_fn(retnet_scan_pytorch, q, k, v, gamma)
        diff = (retnet_scan_triton(q, k, v, gamma) - retnet_scan_pytorch(q, k, v, gamma)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "retnet_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'retnet_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"retnet_scan setup error: {e}")

    # 14. RWKV6 Scan
    try:
        from neuralops.ops.triton.rwkv6_scan import rwkv6_scan, rwkv6_scan_pytorch
        B, H, T, D = 2, 4, 256, 64
        r = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        w = -torch.rand(B, H, T, D, device=device, dtype=dtype)
        u = torch.randn(H, D, device=device, dtype=dtype)
        t_tri = time_fn(rwkv6_scan, r, k, v, w, u)
        t_pt = time_fn(rwkv6_scan_pytorch, r, k, v, w, u)
        diff = (rwkv6_scan(r, k, v, w, u) - rwkv6_scan_pytorch(r, k, v, w, u)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "rwkv6_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'rwkv6_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"rwkv6_scan setup error: {e}")

    # 15. Based Scan
    try:
        from neuralops.ops.triton.based_scan import based_taylor_scan, based_taylor_scan_pytorch
        B, H, T, D = 2, 4, 256, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        t_tri = time_fn(based_taylor_scan, q, k, v)
        t_pt = time_fn(based_taylor_scan_pytorch, q, k, v)
        diff = (based_taylor_scan(q, k, v) - based_taylor_scan_pytorch(q, k, v)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 1e-2 else ("SLOW" if diff < 1e-2 else "FAIL")
        results.append({"module": "based_scan", "category": "linear_attention", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'based_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"based_scan setup error: {e}")

    # 16. Delta Net Scan
    try:
        from neuralops.ops.triton.delta_net_scan import delta_net_scan_triton, delta_net_scan_pytorch
        B, H, T, D = 2, 4, 128, 32
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
        t_tri = time_fn(delta_net_scan_triton, q, k, v, beta)
        t_pt = time_fn(delta_net_scan_pytorch, q, k, v, beta)
        diff = (delta_net_scan_triton(q, k, v, beta) - delta_net_scan_pytorch(q, k, v, beta)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "delta_net_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'delta_net_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"delta_net_scan setup error: {e}")

    # 17. GDN Scan
    try:
        from neuralops.ops.triton.gdn_scan import gdn_scan_triton, gdn_scan_pytorch
        B, H, T, D = 2, 4, 128, 32
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
        alpha = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
        t_tri = time_fn(gdn_scan_triton, q, k, v, beta, alpha)
        t_pt = time_fn(gdn_scan_pytorch, q, k, v, beta, alpha)
        diff = (gdn_scan_triton(q, k, v, beta, alpha) - gdn_scan_pytorch(q, k, v, beta, alpha)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "gdn_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'gdn_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"gdn_scan setup error: {e}")

    # 18. FMS Scan
    try:
        from neuralops.ops.triton.fms_scan import fms_scan_triton, fms_scan_pytorch
        B, H, T, D = 2, 4, 256, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        decay = torch.sigmoid(torch.randn(H, device=device, dtype=dtype))
        t_tri = time_fn(fms_scan_triton, q, k, v, decay)
        t_pt = time_fn(fms_scan_pytorch, q, k, v, decay)
        diff = (fms_scan_triton(q, k, v, decay) - fms_scan_pytorch(q, k, v, decay)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "fms_scan", "category": "recurrent", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fms_scan':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fms_scan setup error: {e}")

    # 19. Differential Attention
    try:
        from neuralops.ops.triton.differential_attention import triton_differential_attention, triton_differential_attention_pytorch_ref
        B, H, T, D = 2, 4, 256, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        lambda_init = 0.8
        t_tri = time_fn(triton_differential_attention, q, k, v, lambda_init)
        t_pt = time_fn(triton_differential_attention_pytorch_ref, q, k, v, lambda_init)
        diff = (triton_differential_attention(q, k, v, lambda_init) - triton_differential_attention_pytorch_ref(q, k, v, lambda_init)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "differential_attention", "category": "attention", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'differential_attention':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"differential_attention setup error: {e}")

    # 20. CUR Linear
    try:
        from neuralops.ops.triton.cur_linear import triton_cur_linear, triton_cur_linear_pytorch_ref
        M, N, K = 512, 1024, 512
        rank = 64
        x = torch.randn(M, K, device=device, dtype=dtype)
        c = torch.randn(M, rank, device=device, dtype=dtype)
        u = torch.randn(rank, rank, device=device, dtype=dtype)
        r = torch.randn(rank, N, device=device, dtype=dtype)
        t_tri = time_fn(triton_cur_linear, x, c, u, r)
        t_pt = time_fn(triton_cur_linear_pytorch_ref, x, c, u, r)
        diff = (triton_cur_linear(x, c, u, r) - triton_cur_linear_pytorch_ref(x, c, u, r)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "cur_linear", "category": "low_rank", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'cur_linear':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"cur_linear setup error: {e}")

    # 21. Sliding Window Attention
    try:
        from neuralops.ops.triton.sliding_window import sliding_window_attn_triton, sliding_window_attn_pytorch
        B, H, T, D = 2, 4, 512, 64
        window_size = 128
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        t_tri = time_fn(sliding_window_attn_triton, q, k, v, window_size)
        t_pt = time_fn(sliding_window_attn_pytorch, q, k, v, window_size)
        diff = (sliding_window_attn_triton(q, k, v, window_size) - sliding_window_attn_pytorch(q, k, v, window_size)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "sliding_window_attention", "category": "attention", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'sliding_window_attention':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"sliding_window_attention setup error: {e}")

    # 22. Fused Lowrank Gated GEMM
    try:
        from neuralops.ops.triton.low_rank import triton_fused_lowrank_gated_gemm, triton_fused_lowrank_gated_gemm_pytorch_ref
        M, N, K = 256, 1024, 512
        rank = 64
        x = torch.randn(M, K, device=device, dtype=dtype)
        w_down = torch.randn(rank, K, device=device, dtype=dtype)
        w_up = torch.randn(N, rank, device=device, dtype=dtype)
        w_gate = torch.randn(N, K, device=device, dtype=dtype)
        t_tri = time_fn(triton_fused_lowrank_gated_gemm, x, w_down, w_up, w_gate)
        t_pt = time_fn(triton_fused_lowrank_gated_gemm_pytorch_ref, x, w_down, w_up, w_gate)
        diff = (triton_fused_lowrank_gated_gemm(x, w_down, w_up, w_gate) - triton_fused_lowrank_gated_gemm_pytorch_ref(x, w_down, w_up, w_gate)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "fused_lowrank_gated_gemm", "category": "low_rank", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_lowrank_gated_gemm':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_lowrank_gated_gemm setup error: {e}")

    # 23. Fused Lowrank Sparse GEMM (RPCA)
    try:
        from neuralops.ops.triton.low_rank import triton_fused_lowrank_sparse_gemm, triton_fused_lowrank_sparse_gemm_pytorch_ref
        M, N, K = 256, 1024, 512
        rank = 64
        x = torch.randn(M, K, device=device, dtype=dtype)
        w_low = torch.randn(rank, K, device=device, dtype=dtype)
        w_high = torch.randn(N, rank, device=device, dtype=dtype)
        w_sparse = torch.randn(N, K, device=device, dtype=dtype) * (torch.rand(N, K, device=device) > 0.95).float()
        t_tri = time_fn(triton_fused_lowrank_sparse_gemm, x, w_low, w_high, w_sparse)
        t_pt = time_fn(triton_fused_lowrank_sparse_gemm_pytorch_ref, x, w_low, w_high, w_sparse)
        diff = (triton_fused_lowrank_sparse_gemm(x, w_low, w_high, w_sparse) - triton_fused_lowrank_sparse_gemm_pytorch_ref(x, w_low, w_high, w_sparse)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "fused_lowrank_sparse_gemm", "category": "low_rank", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_lowrank_sparse_gemm':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_lowrank_sparse_gemm setup error: {e}")

    # 24. Strided TT Contraction
    try:
        from neuralops.ops.triton.low_rank import triton_strided_tt_contraction, triton_strided_tt_contraction_pytorch_ref
        B, Din = 256, 512
        rank = 16
        x = torch.randn(B, Din, device=device, dtype=dtype)
        g1 = torch.randn(1, 16, 16, rank, device=device, dtype=dtype)
        g2 = torch.randn(rank, 32, 32, 1, device=device, dtype=dtype)
        t_tri = time_fn(triton_strided_tt_contraction, x, g1, g2)
        t_pt = time_fn(triton_strided_tt_contraction_pytorch_ref, x, g1, g2)
        diff = (triton_strided_tt_contraction(x, g1, g2) - triton_strided_tt_contraction_pytorch_ref(x, g1, g2)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "strided_tt_contraction", "category": "low_rank", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'strided_tt_contraction':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"strided_tt_contraction setup error: {e}")

    # 25. Fused Kronecker Matmul
    try:
        from neuralops.ops.triton.low_rank import triton_fused_kronecker_matmul, triton_fused_kronecker_matmul_pytorch_ref
        B = 256
        A = torch.randn(32, 32, device=device, dtype=dtype)
        B_mat = torch.randn(32, 32, device=device, dtype=dtype)
        X = torch.randn(B, 1024, device=device, dtype=dtype)
        t_tri = time_fn(triton_fused_kronecker_matmul, X, A, B_mat)
        t_pt = time_fn(triton_fused_kronecker_matmul_pytorch_ref, X, A, B_mat)
        diff = (triton_fused_kronecker_matmul(X, A, B_mat) - triton_fused_kronecker_matmul_pytorch_ref(X, A, B_mat)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "fused_kronecker_matmul", "category": "low_rank", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'fused_kronecker_matmul':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"fused_kronecker_matmul setup error: {e}")

    # 26. Spectral Entropy SVD Gate
    try:
        from neuralops.ops.triton.low_rank import triton_spectral_entropy_svd_gate, triton_spectral_entropy_svd_gate_pytorch_ref
        B, D = 256, 1024
        rank = 64
        x = torch.randn(B, D, device=device, dtype=dtype)
        v = torch.randn(D, rank, device=device, dtype=dtype)
        s = torch.randn(rank, device=device, dtype=dtype).abs()
        t_tri = time_fn(triton_spectral_entropy_svd_gate, x, v, s)
        t_pt = time_fn(triton_spectral_entropy_svd_gate_pytorch_ref, x, v, s)
        diff = (triton_spectral_entropy_svd_gate(x, v, s) - triton_spectral_entropy_svd_gate_pytorch_ref(x, v, s)).abs().max().item()
        speedup = t_pt / max(t_tri, 1e-6)
        status = "PASS" if speedup >= 1.0 and diff < 2e-2 else ("SLOW" if diff < 2e-2 else "FAIL")
        results.append({"module": "spectral_entropy_svd_gate", "category": "low_rank", "triton_ms": t_tri, "pytorch_ms": t_pt, "speedup": speedup, "max_diff": diff, "status": status})
        print(f"[{status}] {'spectral_entropy_svd_gate':<30} | Tri: {t_tri:7.3f}ms | PT: {t_pt:7.3f}ms | Speedup: {speedup:6.2f}x | Diff: {diff:.2e}")
    except Exception as e:
        print(f"spectral_entropy_svd_gate setup error: {e}")

    # Save to CSV
    os.makedirs("results/reruns_2026-11-03", exist_ok=True)
    out_csv = "results/reruns_2026-11-03/all_triton_modules_comprehensive.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "category", "triton_ms", "pytorch_ms", "speedup", "max_diff", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved all Triton module benchmark results to: {out_csv}")

if __name__ == "__main__":
    run_benchmarks()
