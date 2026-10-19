#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Bench batch1 module.
"""

import time
import torch

from neuralops.core.triton_kernels import (
    HAS_TRITON,
    linear_attn,
    low_rank,
    gqa,
    gdn_scan,
    selective_scan,
)
from neuralops.core.triton_kernels.cosformer_triton_kernel import cosformer_triton_fwd
from neuralops.core.triton_kernels.fused_norm_linear import fused_norm_linear
from neuralops.core.triton_kernels.fused_ce import fused_cross_entropy
from neuralops.core.triton_kernels.fused_kd import fused_kd_loss
from neuralops.core.triton_kernels.sliding_window import sliding_window_attention

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Benchmarking on device: {device}, HAS_TRITON: {HAS_TRITON}")
if device == "cuda":
    print(f"Device Name: {torch.cuda.get_device_name(0)}")


def bench_op(name, fn_fwd, warmup=5, iters=20):
    """Execute bench op routine."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = fn_fwd()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn_fwd()
    if device == "cuda":
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / iters * 1000.0

    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    print(f"{name:30s} | Fwd: {fwd_ms:7.3f} ms | Peak VRAM: {vram_mb:7.2f} MB")
    return {"op": name, "fwd_ms": fwd_ms, "vram_mb": vram_mb}


B, H, N, D = 4, 8, 1024, 64
q = torch.randn(B, H, N, D, device=device, dtype=torch.float16, requires_grad=True)
k = torch.randn(B, H, N, D, device=device, dtype=torch.float16, requires_grad=True)
v = torch.randn(B, H, N, D, device=device, dtype=torch.float16, requires_grad=True)
bench_op("linear_attn (triton/fwd)", lambda: linear_attn.fwd(q, k, v))

x = torch.randn(B, N, 512, device=device, dtype=torch.float16, requires_grad=True)
U = torch.randn(512, 64, device=device, dtype=torch.float16, requires_grad=True)
V = torch.randn(64, 512, device=device, dtype=torch.float16, requires_grad=True)
bench_op("low_rank (triton/fwd)", lambda: low_rank.fwd(x, U, V))

k_gqa = torch.randn(B, 2, N, D, device=device, dtype=torch.float16)
v_gqa = torch.randn(B, 2, N, D, device=device, dtype=torch.float16)
bench_op("gqa (triton/fwd)", lambda: gqa.fwd(k_gqa, v_gqa, 2, 8))

beta = torch.sigmoid(torch.randn(B, H, N, D, device=device, dtype=torch.float16))
alpha = torch.sigmoid(torch.randn(B, H, N, D, device=device, dtype=torch.float16))
if gdn_scan.gdn_scan_triton is not None:
    q_inf = q.detach()
    k_inf = k.detach()
    v_inf = v.detach()
    bench_op("gdn_scan (triton/inf)", lambda: gdn_scan.gdn_scan_triton(q_inf, k_inf, v_inf, beta, alpha))
    bench_op("gdn_scan (triton/train)", lambda: gdn_scan.gdn_scan_triton(q, k, v, beta, alpha))

u_ssm = torch.randn(B, 64, 1024, device=device, dtype=torch.float16)
delta_ssm = torch.randn(B, 64, 1024, device=device, dtype=torch.float16)
A_ssm = torch.randn(64, 16, device=device, dtype=torch.float16)
B_ssm = torch.randn(B, 16, 1024, device=device, dtype=torch.float16)
C_ssm = torch.randn(B, 16, 1024, device=device, dtype=torch.float16)
if selective_scan.selective_scan_fn is not None:
    bench_op(
        "selective_scan (triton/fwd)", lambda: selective_scan.selective_scan_fn(u_ssm, delta_ssm, A_ssm, B_ssm, C_ssm)
    )

bench_op("cosformer (triton/fwd)", lambda: cosformer_triton_fwd(q, k, v))

bench_op("sliding_window (triton/fwd)", lambda: sliding_window_attention(q, k, v, window_size=128))

x_norm = torch.randn(B, N, 512, device=device, dtype=torch.float16)
weight_lin = torch.randn(512, 512, device=device, dtype=torch.float16)
bench_op("fused_norm_linear (triton)", lambda: fused_norm_linear(x_norm, weight_lin))

logits_ce = torch.randn(4096, 32000, device=device, dtype=torch.float16)
targets_ce = torch.randint(0, 32000, (4096,), device=device, dtype=torch.long)
bench_op("fused_ce (triton)", lambda: fused_cross_entropy(logits_ce, targets_ce))

teacher_ce = torch.randn(4096, 32000, device=device, dtype=torch.float16)
bench_op("fused_kd (triton)", lambda: fused_kd_loss(logits_ce, teacher_ce, temperature=2.0))
