#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dynamic Sparse Attention

Selects dynamic high-scoring key-value attention blocks on-the-fly to accelerate long-sequence processing.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    class _MockTriton:
        @staticmethod
        def jit(fn):
            return fn
        @staticmethod
        def autotune(*args, **kwargs):
            return lambda fn: fn
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
        class Config:
            def __init__(self, *args, **kwargs):
                pass
    triton = _MockTriton()
    class _MockTl:
        constexpr = int
    tl = _MockTl()


if HAS_TRITON:
    @triton.jit
    def _dynamic_sparse_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Active_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ab_b, stride_ab_h, stride_ab_q, stride_ab_k,
        stride_ob, stride_oh, stride_om, stride_od,
        sm_scale,
        B, H, L,
        D: tl.constexpr,
        K_ACTIVE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        mask_m = offs_m < L

        q_base = Q_ptr + b * stride_qb + h * stride_qh
        k_base = K_ptr + b * stride_kb + h * stride_kh
        v_base = V_ptr + b * stride_vb + h * stride_vh
        out_base = Out_ptr + b * stride_ob + h * stride_oh
        act_base = Active_ptr + b * stride_ab_b + h * stride_ab_h + pid_m * stride_ab_q

        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for k_idx in range(0, K_ACTIVE):
            target_block = tl.load(act_base + k_idx * stride_ab_k)

            if target_block >= 0:
                start_n = target_block * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < L

                k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
                v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

                s = tl.dot(q, tl.trans(k)) * sm_scale
                s = tl.where(mask_m[:, None] & mask_n[None, :], s, -float("inf"))

                m_curr = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp(m_i - m_curr)
                p = tl.exp(s - m_curr[:, None])

                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_curr

        out = acc / tl.maximum(l_i[:, None], 1e-6)
        tl.store(out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, out.to(Q_ptr.dtype.element_ty), mask=mask_m[:, None])
else:
    def _dynamic_sparse_attn_fwd_kernel(*args, **kwargs):
        pass


def triton_dynamic_sparse_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_block_indices: torch.Tensor,
    block_size: int = 32,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for Dynamic Sparse Attention."""
    orig_dtype = q.dtype
    B, H, L, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.zeros_like(v)
    num_q_blocks = (L + block_size - 1) // block_size

    for b in range(B):
        for h in range(H):
            for m in range(num_q_blocks):
                q_st = m * block_size
                q_ed = min(q_st + block_size, L)
                q_sub = q[b, h, q_st:q_ed].float()

                active_k_ids = active_block_indices[b, h, m]
                k_chunks, v_chunks = [], []
                for k_id in active_k_ids:
                    k_val = int(k_id.item())
                    if k_val >= 0:
                        kv_st = k_val * block_size
                        kv_ed = min(kv_st + block_size, L)
                        if kv_st < L:
                            k_chunks.append(k[b, h, kv_st:kv_ed].float())
                            v_chunks.append(v[b, h, kv_st:kv_ed].float())

                if len(k_chunks) > 0:
                    k_sub = torch.cat(k_chunks, dim=0)
                    v_sub = torch.cat(v_chunks, dim=0)
                    scores = torch.matmul(q_sub, k_sub.t()) * sm_scale
                    p = F.softmax(scores, dim=-1)
                    out[b, h, q_st:q_ed] = torch.matmul(p, v_sub).to(orig_dtype)

    return out


class TritonDynamicSparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        active_block_indices: torch.Tensor,
        block_size: int = 32,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        orig_shape = q.shape
        is_3d = (q.ndim == 3)
        if is_3d:
            B, L, D = q.shape
            H = 1
            q_4d = q.view(B, H, L, D)
            k_4d = k.view(B, H, L, D)
            v_4d = v.view(B, H, L, D)
        else:
            B, H, L, D = q.shape
            q_4d, k_4d, v_4d = q, k, v

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        K_ACTIVE = active_block_indices.shape[-1]

        if not HAS_TRITON or not q.is_cuda or D not in (16, 32, 64, 128):
            out_4d = triton_dynamic_sparse_attention_pytorch_ref(
                q_4d, k_4d, v_4d, active_block_indices, block_size=block_size, sm_scale=sm_scale
            )
            ctx.save_for_backward(q_4d, k_4d, v_4d, active_block_indices)
            ctx.block_size = block_size
            ctx.sm_scale = sm_scale
            ctx.is_3d = is_3d
            ctx.orig_shape = orig_shape
            return out_4d.view(orig_shape)

        q_c = q_4d.contiguous()
        k_c = k_4d.contiguous()
        v_c = v_4d.contiguous()
        ab_c = active_block_indices.contiguous().to(torch.int32)
        out_4d = torch.empty_like(v_c)

        grid = (triton.cdiv(L, block_size), B * H)

        _dynamic_sparse_attn_fwd_kernel[grid](
            q_c, k_c, v_c, ab_c, out_4d,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            ab_c.stride(0), ab_c.stride(1), ab_c.stride(2), ab_c.stride(3),
            out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
            float(sm_scale),
            B, H, L,
            D=D,
            K_ACTIVE=K_ACTIVE,
            BLOCK_SIZE=block_size,
            BLOCK_M=block_size,
            BLOCK_N=block_size,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q_c, k_c, v_c, ab_c)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.is_3d = is_3d
        ctx.orig_shape = orig_shape
        return out_4d.view(orig_shape)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, ab = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        B, H, L, D = q.shape
        dout_4d = dout.view(B, H, L, D).contiguous()
        orig_dtype = q.dtype

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        num_q_blocks = (L + block_size - 1) // block_size

        for b in range(B):
            for h in range(H):
                for m in range(num_q_blocks):
                    q_st = m * block_size
                    q_ed = min(q_st + block_size, L)
                    q_sub = q[b, h, q_st:q_ed].float()
                    dout_sub = dout_4d[b, h, q_st:q_ed].float()

                    active_k_ids = ab[b, h, m]
                    k_chunks, v_chunks, ranges = [], [], []
                    for k_id in active_k_ids:
                        k_val = int(k_id.item())
                        if k_val >= 0:
                            kv_st = k_val * block_size
                            kv_ed = min(kv_st + block_size, L)
                            if kv_st < L:
                                k_chunks.append(k[b, h, kv_st:kv_ed].float())
                                v_chunks.append(v[b, h, kv_st:kv_ed].float())
                                ranges.append((kv_st, kv_ed))

                    if len(k_chunks) > 0:
                        k_sub = torch.cat(k_chunks, dim=0)
                        v_sub = torch.cat(v_chunks, dim=0)

                        scores = torch.matmul(q_sub, k_sub.t()) * sm_scale
                        p = F.softmax(scores, dim=-1)
                        o_sub = torch.matmul(p, v_sub)

                        dv_sub = torch.matmul(p.t(), dout_sub)
                        dp = torch.matmul(dout_sub, v_sub.t())
                        D_sub = (dout_sub * o_sub).sum(dim=-1, keepdim=True)
                        ds = p * (dp - D_sub) * sm_scale

                        dq[b, h, q_st:q_ed] += torch.matmul(ds, k_sub)
                        dk_sub = torch.matmul(ds.t(), q_sub)

                        curr_offset = 0
                        for (kv_st, kv_ed) in ranges:
                            chunk_len = kv_ed - kv_st
                            dk[b, h, kv_st:kv_ed] += dk_sub[curr_offset : curr_offset + chunk_len]
                            dv[b, h, kv_st:kv_ed] += dv_sub[curr_offset : curr_offset + chunk_len]
                            curr_offset += chunk_len

        dq_out = dq.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[0] else None
        dk_out = dk.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[1] else None
        dv_out = dv.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[2] else None

        return dq_out, dk_out, dv_out, None, None, None


def triton_dynamic_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_block_indices: torch.Tensor,
    block_size: int = 32,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Dispatched Dynamic Sparse Attention with full Autograd backward support."""
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)
    ):
        return TritonDynamicSparseAttentionFunction.apply(
            q, k, v, active_block_indices, block_size, sm_scale
        )

    orig_shape = q.shape
    is_3d = (q.ndim == 3)
    if is_3d:
        B, L, D = q.shape
        H = 1
        q_4d = q.view(B, H, L, D)
        k_4d = k.view(B, H, L, D)
        v_4d = v.view(B, H, L, D)
    else:
        B, H, L, D = q.shape
        q_4d, k_4d, v_4d = q, k, v

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    K_ACTIVE = active_block_indices.shape[-1]

    if not HAS_TRITON or not q.is_cuda or D not in (16, 32, 64, 128):
        return triton_dynamic_sparse_attention_pytorch_ref(
            q_4d, k_4d, v_4d, active_block_indices, block_size=block_size, sm_scale=sm_scale
        ).view(orig_shape)

    q_c = q_4d.contiguous()
    k_c = k_4d.contiguous()
    v_c = v_4d.contiguous()
    ab_c = active_block_indices.contiguous().to(torch.int32)
    out_4d = torch.empty_like(v_c)

    grid = (triton.cdiv(L, block_size), B * H)

    _dynamic_sparse_attn_fwd_kernel[grid](
        q_c, k_c, v_c, ab_c, out_4d,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        ab_c.stride(0), ab_c.stride(1), ab_c.stride(2), ab_c.stride(3),
        out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
        float(sm_scale),
        B, H, L,
        D=D,
        K_ACTIVE=K_ACTIVE,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        num_warps=4,
        num_stages=2,
    )

    return out_4d.view(orig_shape)


__all__ = [
    "triton_dynamic_sparse_attention",
    "triton_dynamic_sparse_attention_pytorch_ref",
    "TritonDynamicSparseAttentionFunction",
]
