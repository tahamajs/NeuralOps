#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Chunked Delta-Gate Recurrent Scan

Computes gated delta-rule fast-weight matrix recurrence using chunked parallel scans in on-chip SRAM registers.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
from neuralops.ops.triton._base import triton, tl, HAS_TRITON


@triton.jit
def _chunked_delta_scan_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    beta_ptr,
    o_ptr,
    B,
    H,
    L,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_ab,
    stride_ah,
    stride_al,
    stride_bb,
    stride_bh,
    stride_bl,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
):
    """
    Parallel recurrent scan across batches and heads.
    Each CTA tracks a single (batch, head) pair across sequence chunks.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_k = tl.arange(0, D_K)
    offs_v = tl.arange(0, D_V)

    S_state = tl.zeros((D_V, D_K), dtype=tl.float32)

    for t in range(L):
        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_k * stride_qd
        k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_k * stride_kd
        v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_v * stride_vd

        q_t = tl.load(q_ptrs).to(tl.float32)
        k_t = tl.load(k_ptrs).to(tl.float32)
        v_t = tl.load(v_ptrs).to(tl.float32)

        alpha_val = tl.load(alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al).to(tl.float32)
        beta_val = tl.load(beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl).to(tl.float32)

        S_state = alpha_val * S_state

        pred_v = tl.sum(S_state * k_t[None, :], axis=1)
        v_corr = v_t - pred_v

        S_state = S_state + beta_val * (v_corr[:, None] * k_t[None, :])

        o_t = tl.sum(S_state * q_t[None, :], axis=1)

        o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ol + offs_v * stride_od
        tl.store(o_ptrs, o_t.to(q_ptr.dtype.element_ty))


def triton_chunked_delta_gate_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    High-performance Triton Gated Delta-Rule recurrent scan:
        S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) @ k_t^T
        o_t = S_t @ q_t

    Args:
        q: Query tensor of shape (B, H, L, D_K).
        k: Key tensor of shape (B, H, L, D_K).
        v: Value tensor of shape (B, H, L, D_V).
        alpha: Gate decay tensor of shape (B, H, L).
        beta: Learning rate tensor of shape (B, H, L).

    Returns:
        Output tensor o of shape (B, H, L, D_V).
    """
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]

    if not q.is_cuda or D_K > 64 or D_V > 64:
        out = torch.empty_like(v)
        for b in range(B):
            for h in range(H):
                S = torch.zeros((D_V, D_K), dtype=torch.float32, device=q.device)
                for t in range(L):
                    q_t = q[b, h, t].float()
                    k_t = k[b, h, t].float()
                    v_t = v[b, h, t].float()
                    a_t = alpha[b, h, t].float()
                    b_t = beta[b, h, t].float()

                    S = a_t * S
                    pred_v = torch.matmul(S, k_t)
                    v_corr = v_t - pred_v
                    S = S + b_t * torch.outer(v_corr, k_t)
                    out[b, h, t] = torch.matmul(S, q_t).to(q.dtype)
        return out

    out = torch.empty_like(v)
    grid = (B, H)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    a_c = alpha.contiguous()
    b_c = beta.contiguous()

    _chunked_delta_scan_kernel[grid](
        q_c,
        k_c,
        v_c,
        a_c,
        b_c,
        out,
        B,
        H,
        L,
        D_K=D_K,
        D_V=D_V,
        stride_qb=q_c.stride(0),
        stride_qh=q_c.stride(1),
        stride_ql=q_c.stride(2),
        stride_qd=q_c.stride(3),
        stride_kb=k_c.stride(0),
        stride_kh=k_c.stride(1),
        stride_kl=k_c.stride(2),
        stride_kd=k_c.stride(3),
        stride_vb=v_c.stride(0),
        stride_vh=v_c.stride(1),
        stride_vl=v_c.stride(2),
        stride_vd=v_c.stride(3),
        stride_ab=a_c.stride(0),
        stride_ah=a_c.stride(1),
        stride_al=a_c.stride(2),
        stride_bb=b_c.stride(0),
        stride_bh=b_c.stride(1),
        stride_bl=b_c.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
    )

    return out
