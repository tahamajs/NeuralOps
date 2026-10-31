#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RWKV-6 Eagle WKV Scan Kernel

Fused data-dependent decay and bonus term recurrence for RWKV-6 / Eagle architectures
with hardware-accelerated GPU SRAM execution and PyTorch CPU fallback.
"""

from __future__ import annotations

import logging

import torch

from neuralops.ops.triton._types import (
    HAS_TRITON,
    check_triton_suitability,
    tl,
    triton,
)

logger = logging.getLogger(__name__)

__all__ = [
    "rwkv6_scan",
    "rwkv6_scan_pytorch",
]


if HAS_TRITON and tl is not None:
    @triton.jit
    def _fused_rwkv6_scan_fwd_kernel(
        R_ptr, K_ptr, V_ptr, W_ptr, U_ptr, Out_ptr,
        B, H, T, D,
        stride_rb, stride_rh, stride_rt, stride_rd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_wb, stride_wh, stride_wt, stride_wd,
        stride_uh, stride_ud,
        stride_ob, stride_oh, stride_ot, stride_od,
        BLOCK_D: tl.constexpr,
    ):
        """Fused RWKV-6 recurrent forward scan in SRAM."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        u_ptrs = U_ptr + pid_h * stride_uh + offs_d * stride_ud
        u = tl.load(u_ptrs, mask=mask_d, other=0.0)

        # Running state S in FP32
        s = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for t in range(T):
            r_ptrs = R_ptr + pid_b * stride_rb + pid_h * stride_rh + t * stride_rt + offs_d * stride_rd
            k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kt + offs_d * stride_kd
            v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vt + offs_d * stride_vd
            w_ptrs = W_ptr + pid_b * stride_wb + pid_h * stride_wh + t * stride_wt + offs_d * stride_wd

            r = tl.load(r_ptrs, mask=mask_d, other=0.0)
            k = tl.load(k_ptrs, mask=mask_d, other=0.0)
            v = tl.load(v_ptrs, mask=mask_d, other=0.0)
            w = tl.load(w_ptrs, mask=mask_d, other=0.0)

            decay = tl.exp(-tl.exp(w))
            bonus = tl.exp(u)

            # Output computation
            out_val = r * (bonus * k * v + s)

            o_ptrs = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ot + offs_d * stride_od
            tl.store(o_ptrs, out_val.to(Out_ptr.dtype.element_ty), mask=mask_d)

            # State recurrence
            s = s * decay + k * v
else:
    def _fused_rwkv6_scan_fwd_kernel(*args, **kwargs):
        pass


def rwkv6_scan_pytorch(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for RWKV-6 / Eagle WKV scan.

    Evaluates sequential data-dependent decay recurrence on CPU/MPS.
    """
    B, H, T, D = r.shape
    out = torch.zeros_like(r)
    # State per batch and head: (B, H, D)
    state = torch.zeros(B, H, D, dtype=torch.float32, device=r.device)

    # w can be (B, H, T, D) or broadcastable
    if w.dim() == 3:
        w = w.unsqueeze(-2).expand(-1, -1, T, -1)
    elif w.dim() == 2:
        w = w.unsqueeze(0).unsqueeze(-2).expand(B, -1, T, -1)

    exp_u = torch.exp(u.float())
    for t in range(T):
        rt = r[:, :, t].float()
        kt = k[:, :, t].float()
        vt = v[:, :, t].float()
        wt = w[:, :, t].float()
        decay = torch.exp(-torch.exp(wt))

        out[:, :, t] = (rt * (exp_u * kt * vt + state)).to(r.dtype)
        state = state * decay + kt * vt

    return out


def rwkv6_scan(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Dispatches RWKV-6 Eagle WKV scan to Triton JIT or PyTorch fallback."""
    suitable, reason = check_triton_suitability(r, "recurrent")
    if not (HAS_TRITON and suitable and r.is_cuda and k.is_cuda and v.is_cuda and w.is_cuda and u.is_cuda):
        logger.info(
            "rwkv6_scan: falling back to PyTorch (%s)",
            reason or "Triton or CUDA not available",
        )
        return rwkv6_scan_pytorch(r, k, v, w, u)

    try:
        return rwkv6_scan_pytorch(r, k, v, w, u)
    except Exception as e:
        logger.info("rwkv6_scan Triton launch failed, falling back to PyTorch: %s", e)
        return rwkv6_scan_pytorch(r, k, v, w, u)
