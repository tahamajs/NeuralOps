#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Latent KV-Cache Attention

Decompresses compressed latent KV cache representations on-the-fly during attention computation to minimize memory bandwidth.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from neuralops.ops.triton._base import triton, tl, HAS_TRITON


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    ],
    key=["L_Q", "L_KV", "R", "D_V"],
)
@triton.jit
def _latent_kv_cache_attn_kernel(
    q_latent_ptr,
    c_kv_ptr,
    u_v_ptr,
    o_ptr,
    B,
    H,
    L_Q,
    L_KV,
    R: tl.constexpr,
    D_V: tl.constexpr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qr,
    stride_cb,
    stride_cl,
    stride_cr,
    stride_ur,
    stride_ud,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = tl.arange(0, R)
    offs_dv = tl.arange(0, D_V)

    mask_m = offs_m < L_Q

    q_ptrs = (
        q_latent_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_ql
        + offs_r[None, :] * stride_qr
    )
    q_latent_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_latent = tl.zeros([BLOCK_M, R], dtype=tl.float32)

    num_n_blocks = tl.cdiv(L_KV, BLOCK_N)
    for n_idx in range(num_n_blocks):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L_KV

        c_ptrs = c_kv_ptr + pid_b * stride_cb + offs_n[:, None] * stride_cl + offs_r[None, :] * stride_cr
        c_tile = tl.load(c_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        s_tile = tl.dot(q_latent_tile, tl.trans(c_tile), allow_tf32=True) * sm_scale
        s_tile = tl.where(mask_m[:, None] & mask_n[None, :], s_tile, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(s_tile, axis=1))
        p = tl.exp(s_tile - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc_latent = acc_latent * alpha[:, None] + tl.dot(p, c_tile, allow_tf32=True)
        m_i = m_ij

    safe_l_i = tl.maximum(l_i[:, None], 1e-5)
    acc_latent_norm = acc_latent / safe_l_i

    u_ptrs = u_v_ptr + offs_r[:, None] * stride_ur + offs_dv[None, :] * stride_ud
    u_v_tile = tl.load(u_ptrs).to(tl.float32)

    o_tile = tl.dot(acc_latent_norm.to(q_latent_ptr.dtype.element_ty), u_v_tile.to(q_latent_ptr.dtype.element_ty), allow_tf32=True)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_ol
        + offs_dv[None, :] * stride_od
    )
    tl.store(o_ptrs, o_tile.to(q_latent_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_fused_latent_kv_cache_attention(
    q: torch.Tensor,
    c_kv: torch.Tensor,
    u_k: torch.Tensor,
    u_v: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    High-performance Triton Latent-Space FlashAttention.

    Args:
        q: Query tensor of shape (B, H, L_Q, D_K).
        c_kv: Compressed latent KV-cache of shape (B, L_KV, R).
        u_k: Key up-projection factor of shape (D_K, R).
        u_v: Value up-projection factor of shape (R, D_V).
        sm_scale: Softmax scale factor (default: 1.0 / sqrt(R)).

    Returns:
        Output tensor of shape (B, H, L_Q, D_V).
    """
    B, H, L_Q, D_K = q.shape
    B_c, L_KV, R = c_kv.shape
    D_V = u_v.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(R)

    q_latent = torch.matmul(q, u_k)

    if not q.is_cuda or R > 128 or D_V > 128:
        c_expanded = c_kv.unsqueeze(1)
        scores = torch.matmul(q_latent.float(), c_expanded.float().transpose(-1, -2)) * sm_scale
        probs = F.softmax(scores, dim=-1)
        acc_latent = torch.matmul(probs, c_expanded.float())
        out = torch.matmul(acc_latent.to(q.dtype), u_v)
        return out

    out = torch.empty((B, H, L_Q, D_V), device=q.device, dtype=q.dtype)

    grid = lambda META: (
        B,
        H,
        triton.cdiv(L_Q, META["BLOCK_M"]),
    )

    q_lat_c = q_latent.contiguous()
    c_kv_c = c_kv.contiguous()
    u_v_c = u_v.contiguous()

    _latent_kv_cache_attn_kernel[grid](
        q_lat_c,
        c_kv_c,
        u_v_c,
        out,
        B,
        H,
        L_Q,
        L_KV,
        R=R,
        D_V=D_V,
        sm_scale=sm_scale,
        stride_qb=q_lat_c.stride(0),
        stride_qh=q_lat_c.stride(1),
        stride_ql=q_lat_c.stride(2),
        stride_qr=q_lat_c.stride(3),
        stride_cb=c_kv_c.stride(0),
        stride_cl=c_kv_c.stride(1),
        stride_cr=c_kv_c.stride(2),
        stride_ur=u_v_c.stride(0),
        stride_ud=u_v_c.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
    )

    return out
