"""
Hybrid Local Kernel Attention Triton JIT Implementation

High-throughput Triton JIT kernels for sliding-window local attention and two-branch gated blending.

Author: Taha Majlesi
"""

from __future__ import annotations

import math
import torch
from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = [
    "hybrid_local_attn_triton",
    "hybrid_two_branch_blend_triton",
    "hybrid_three_branch_blend_triton",
]

if HAS_TRITON and tl is not None:
    @triton.jit
    def _hybrid_local_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        H, N, RADIUS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q + i_b * stride_qb + i_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

        m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        start_col = tl.maximum(0, (pid_m * BLOCK_M) - RADIUS)
        if IS_CAUSAL:
            end_col = tl.minimum(N, (pid_m + 1) * BLOCK_M)
        else:
            end_col = tl.minimum(N, ((pid_m + 1) * BLOCK_M) + RADIUS)

        num_k_blocks = tl.cdiv(end_col - start_col, BLOCK_N)

        for j in range(num_k_blocks):
            offs_n = start_col + j * BLOCK_N + tl.arange(0, BLOCK_N)
            k_ptrs = K + i_b * stride_kb + i_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)

            scores = tl.dot(q, k) * (1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32)))
            dist = offs_m[:, None] - offs_n[None, :]
            if IS_CAUSAL:
                mask_window = (dist >= 0) & (dist <= RADIUS) & (offs_m[:, None] < N) & (offs_n[None, :] < N)
            else:
                mask_window = (tl.abs(dist) <= RADIUS) & (offs_m[:, None] < N) & (offs_n[None, :] < N)

            scores = tl.where(mask_window, scores, -1e30)

            m_curr = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_curr)
            p = tl.where(mask_window, tl.exp(scores - m_curr[:, None]), 0.0)

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V + i_b * stride_vb + i_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_curr

        out_ptrs = Out + i_b * stride_ob + i_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        out = acc / tl.maximum(l_i[:, None], 1e-6)
        tl.store(out_ptrs, out, mask=offs_m[:, None] < N)

    @triton.jit
    def _hybrid_two_branch_blend_kernel(
        B0, B1, Alpha, Out,
        TOTAL, D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < TOTAL

        token_h = offs // D
        a = tl.load(Alpha + token_h, mask=mask, other=0.0)
        v0 = tl.load(B0 + offs, mask=mask, other=0.0)
        v1 = tl.load(B1 + offs, mask=mask, other=0.0)

        out = a * v0 + (1.0 - a) * v1
        tl.store(Out + offs, out, mask=mask)


def hybrid_local_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    radius: int = 32,
    is_causal: bool = False,
) -> torch.Tensor:
    if not HAS_TRITON or not q.is_cuda:
        raise RuntimeError("Triton is not available or tensor not on CUDA")

    B, H, N, D = q.shape
    out = torch.empty_like(q)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(N, BLOCK_M), B * H)

    _hybrid_local_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, N, RADIUS=radius,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=2,
    )
    return out


def hybrid_two_branch_blend_triton(
    b0: torch.Tensor,
    b1: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if not HAS_TRITON or not b0.is_cuda:
        raise RuntimeError("Triton is not available or tensor not on CUDA")

    total = b0.numel()
    D = b0.size(-1)
    out = torch.empty_like(b0)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    _hybrid_two_branch_blend_kernel[grid](
        b0.contiguous(), b1.contiguous(), alpha.contiguous(), out,
        TOTAL=total, D=D, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


if HAS_TRITON and tl is not None:
    @triton.jit
    def _hybrid_three_branch_blend_kernel(
        B0, B1, B2, Alpha, Out,
        TOTAL, D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < TOTAL

        token_h = offs // D
        a0 = tl.load(Alpha + token_h * 3 + 0, mask=mask, other=0.0)
        a1 = tl.load(Alpha + token_h * 3 + 1, mask=mask, other=0.0)
        a2 = tl.load(Alpha + token_h * 3 + 2, mask=mask, other=0.0)
        v0 = tl.load(B0 + offs, mask=mask, other=0.0)
        v1 = tl.load(B1 + offs, mask=mask, other=0.0)
        v2 = tl.load(B2 + offs, mask=mask, other=0.0)

        out = a0 * v0 + a1 * v1 + a2 * v2
        tl.store(Out + offs, out, mask=mask)


def hybrid_three_branch_blend_triton(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if not HAS_TRITON or not b0.is_cuda:
        raise RuntimeError("Triton is not available or tensor not on CUDA")

    total = b0.numel()
    D = b0.size(-1)
    out = torch.empty_like(b0)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    _hybrid_three_branch_blend_kernel[grid](
        b0.contiguous(), b1.contiguous(), b2.contiguous(), alpha.contiguous(), out,
        TOTAL=total, D=D, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out

