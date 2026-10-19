#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Block-Sparse Streaming Attention

Iterates over active block index tables with online FlashAttention softmax to eliminate redundant masked zero tiles.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    ],
    key=["L", "D"],
)
@triton.jit
def _block_sparse_streaming_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    active_blocks_ptr,
    o_ptr,
    B,
    H,
    L,
    D: tl.constexpr,
    K_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    sm_scale,
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
    stride_ab_b,
    stride_ab_h,
    stride_ab_q,
    stride_ab_k,
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
    offs_d = tl.arange(0, D)
    mask_m = offs_m < L

    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q_block_id = pid_m * BLOCK_M // BLOCK_SIZE

    for active_idx in range(K_ACTIVE):
        active_block_ptr = (
            active_blocks_ptr
            + pid_b * stride_ab_b
            + pid_h * stride_ab_h
            + q_block_id * stride_ab_q
            + active_idx * stride_ab_k
        )
        kv_block_id = tl.load(active_block_ptr)

        if kv_block_id >= 0:
            offs_n = kv_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_N)
            mask_n = offs_n < L

            k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd
            v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd

            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            s_tile = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=True) * sm_scale
            s_tile = tl.where(mask_m[:, None] & mask_n[None, :], s_tile, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(s_tile, axis=1))
            p = tl.exp(s_tile - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p, v_tile, allow_tf32=True)
            m_i = m_ij

    safe_l_i = tl.maximum(l_i[:, None], 1e-5)
    out_tile = acc / safe_l_i

    o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_tile.to(q_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_block_sparse_streaming_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_block_indices: torch.Tensor,
    block_size: int = 32,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    High-performance Triton Block-Sparse Streaming FlashAttention.

    Args:
        q: (B, H, L, D)
        k: (B, H, L, D)
        v: (B, H, L, D)
        active_block_indices: (B, H, num_q_blocks, K_active) with active KV block IDs.
        block_size: Tile size along sequence dimension (default: 32).
        sm_scale: Softmax scaling factor.

    Returns:
        Output tensor of shape (B, H, L, D).
    """
    B, H, L, D = q.shape
    K_ACTIVE = active_block_indices.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if not q.is_cuda or D > 128:
        out = torch.empty_like(v)
        num_q_blocks = (L + block_size - 1) // block_size

        for b in range(B):
            for h in range(H):
                for q_blk in range(num_q_blocks):
                    q_st = q_blk * block_size
                    q_ed = min(L, q_st + block_size)
                    q_sub = q[b, h, q_st:q_ed]

                    active_ids = active_block_indices[b, h, q_blk]
                    valid_ids = active_ids[active_ids >= 0]

                    if len(valid_ids) == 0:
                        out[b, h, q_st:q_ed] = 0.0
                        continue

                    k_chunks = []
                    v_chunks = []
                    for blk_id in valid_ids:
                        kv_st = blk_id.item() * block_size
                        kv_ed = min(L, kv_st + block_size)
                        k_chunks.append(k[b, h, kv_st:kv_ed])
                        v_chunks.append(v[b, h, kv_st:kv_ed])

                    k_sub = torch.cat(k_chunks, dim=0)
                    v_sub = torch.cat(v_chunks, dim=0)

                    scores = torch.matmul(q_sub.float(), k_sub.float().t()) * sm_scale
                    p = F.softmax(scores, dim=-1)
                    out[b, h, q_st:q_ed] = torch.matmul(p, v_sub.float()).to(q.dtype)

        return out

    out = torch.empty_like(v)

    grid = lambda META: (
        B,
        H,
        triton.cdiv(L, META["BLOCK_M"]),
    )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    ab_c = active_block_indices.contiguous().to(torch.int32)

    _block_sparse_streaming_attn_kernel[grid](
        q_c,
        k_c,
        v_c,
        ab_c,
        out,
        B,
        H,
        L,
        D=D,
        K_ACTIVE=K_ACTIVE,
        BLOCK_SIZE=block_size,
        sm_scale=sm_scale,
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
        stride_ab_b=ab_c.stride(0),
        stride_ab_h=ab_c.stride(1),
        stride_ab_q=ab_c.stride(2),
        stride_ab_k=ab_c.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
    )

    return out
