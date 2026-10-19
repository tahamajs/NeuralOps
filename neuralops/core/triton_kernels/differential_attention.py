# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Differential Attention

Computes differential attention maps subtracting noise-cancelling attention patterns with fused streaming softmax.
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
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _diff_attn_fwd_kernel(
        Q1_ptr, K1_ptr, Q2_ptr, K2_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        lambda_val,
        sm_scale,
        B, H, L,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        q1_base = Q1_ptr + b * stride_qb + h * stride_qh
        k1_base = K1_ptr + b * stride_kb + h * stride_kh
        q2_base = Q2_ptr + b * stride_qb + h * stride_qh
        k2_base = K2_ptr + b * stride_kb + h * stride_kh
        v_base  = V_ptr  + b * stride_vb + h * stride_vh
        out_base = Out_ptr + b * stride_ob + h * stride_oh

        mask_m = offs_m < L

        q1 = tl.load(q1_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        q2 = tl.load(q2_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)

        m1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l1 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        m2 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l2 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        max_k_idx = tl.minimum((pid_m + 1) * BLOCK_M, L)

        for start_n in range(0, max_k_idx, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < L

            k1 = tl.load(k1_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
            k2 = tl.load(k2_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
            v  = tl.load(v_base  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

            causal_mask = offs_m[:, None] >= offs_n[None, :]

            s1 = tl.dot(q1, tl.trans(k1)) * sm_scale
            s1 = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], s1, -float("inf"))

            m1_new = tl.maximum(m1, tl.max(s1, 1))
            alpha1 = tl.exp(m1 - m1_new)
            p1 = tl.exp(s1 - m1_new[:, None])
            l1 = l1 * alpha1 + tl.sum(p1, 1)
            acc1 = acc1 * alpha1[:, None] + tl.dot(p1.to(v.dtype), v)
            m1 = m1_new

            s2 = tl.dot(q2, tl.trans(k2)) * sm_scale
            s2 = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], s2, -float("inf"))

            m2_new = tl.maximum(m2, tl.max(s2, 1))
            alpha2 = tl.exp(m2 - m2_new)
            p2 = tl.exp(s2 - m2_new[:, None])
            l2 = l2 * alpha2 + tl.sum(p2, 1)
            acc2 = acc2 * alpha2[:, None] + tl.dot(p2.to(v.dtype), v)
            m2 = m2_new

        out1 = acc1 / tl.maximum(l1[:, None], 1e-6)
        out2 = acc2 / tl.maximum(l2[:, None], 1e-6)
        out = out1 - lambda_val * out2

        tl.store(out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, out.to(Q1_ptr.dtype.element_ty), mask=mask_m[:, None])
else:
    def _diff_attn_fwd_kernel(*args, **kwargs):
        pass


def triton_differential_attention_pytorch_ref(
    q1: torch.Tensor,
    k1: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    lambda_val: float = 0.8,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for Differential Attention."""
    orig_dtype = q1.dtype
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q1.shape[-1])
    out1 = F.scaled_dot_product_attention(q1, k1, v, scale=scale, is_causal=True)
    out2 = F.scaled_dot_product_attention(q2, k2, v, scale=scale, is_causal=True)
    return (out1.float() - float(lambda_val) * out2.float()).to(orig_dtype)


class TritonDifferentialAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q1: torch.Tensor,
        k1: torch.Tensor,
        q2: torch.Tensor,
        k2: torch.Tensor,
        v: torch.Tensor,
        lambda_val: float = 0.8,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        orig_shape = q1.shape
        _orig_dtype = q1.dtype
        is_3d = (q1.ndim == 3)
        if is_3d:
            B, L, D = q1.shape
            H = 1
            q1_4d = q1.view(B, H, L, D)
            k1_4d = k1.view(B, H, L, D)
            q2_4d = q2.view(B, H, L, D)
            k2_4d = k2.view(B, H, L, D)
            v_4d  = v.view(B, H, L, D)
        else:
            B, H, L, D = q1.shape
            q1_4d, k1_4d, q2_4d, k2_4d, v_4d = q1, k1, q2, k2, v

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        if not HAS_TRITON or not q1.is_cuda or D not in (16, 32, 64, 128):
            out_4d = triton_differential_attention_pytorch_ref(
                q1_4d, k1_4d, q2_4d, k2_4d, v_4d, lambda_val=lambda_val, sm_scale=sm_scale
            )
            ctx.save_for_backward(q1_4d, k1_4d, q2_4d, k2_4d, v_4d)
            ctx.lambda_val = float(lambda_val)
            ctx.sm_scale = float(sm_scale)
            ctx.is_3d = is_3d
            ctx.orig_shape = orig_shape
            return out_4d.view(orig_shape)

        q1_c = q1_4d.contiguous()
        k1_c = k1_4d.contiguous()
        q2_c = q2_4d.contiguous()
        k2_c = k2_4d.contiguous()
        v_c  = v_4d.contiguous()
        out_4d = torch.empty_like(v_c)

        grid = lambda META: (triton.cdiv(L, META["BLOCK_M"]), B * H)

        _diff_attn_fwd_kernel[grid](
            q1_c, k1_c, q2_c, k2_c, v_c, out_4d,
            q1_c.stride(0), q1_c.stride(1), q1_c.stride(2), q1_c.stride(3),
            k1_c.stride(0), k1_c.stride(1), k1_c.stride(2), k1_c.stride(3),
            v_c.stride(0),  v_c.stride(1),  v_c.stride(2),  v_c.stride(3),
            out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
            float(lambda_val), float(sm_scale),
            B, H, L,
            D=D,
        )

        ctx.save_for_backward(q1_c, k1_c, q2_c, k2_c, v_c)
        ctx.lambda_val = float(lambda_val)
        ctx.sm_scale = float(sm_scale)
        ctx.is_3d = is_3d
        ctx.orig_shape = orig_shape

        return out_4d.view(orig_shape)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q1, k1, q2, k2, v = ctx.saved_tensors
        lambda_val = ctx.lambda_val
        sm_scale = ctx.sm_scale

        dout_4d = dout.view(q1.shape)
        orig_dtype = q1.dtype

        mask = torch.tril(torch.ones(q1.shape[-2], q1.shape[-2], device=q1.device, dtype=torch.bool))
        s1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1)) * sm_scale
        s1 = torch.where(mask, s1, -1e9)
        p1 = torch.softmax(s1, dim=-1)
        o1 = torch.matmul(p1, v.float())

        s2 = torch.matmul(q2.float(), k2.float().transpose(-2, -1)) * sm_scale
        s2 = torch.where(mask, s2, -1e9)
        p2 = torch.softmax(s2, dim=-1)
        o2 = torch.matmul(p2, v.float())

        dv = torch.matmul(p1.transpose(-2, -1), dout_4d.float()) - lambda_val * torch.matmul(p2.transpose(-2, -1), dout_4d.float())

        D1 = (dout_4d.float() * o1).sum(dim=-1, keepdim=True)
        dP1 = torch.matmul(dout_4d.float(), v.float().transpose(-2, -1))
        dS1 = p1 * (dP1 - D1) * sm_scale
        dq1 = torch.matmul(dS1, k1.float())
        dk1 = torch.matmul(dS1.transpose(-2, -1), q1.float())

        D2 = (-lambda_val * dout_4d.float() * o2).sum(dim=-1, keepdim=True)
        dP2 = -lambda_val * torch.matmul(dout_4d.float(), v.float().transpose(-2, -1))
        dS2 = p2 * (dP2 - D2) * sm_scale
        dq2 = torch.matmul(dS2, k2.float())
        dk2 = torch.matmul(dS2.transpose(-2, -1), q2.float())

        dq1 = dq1.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[0] else None
        dk1 = dk1.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[1] else None
        dq2 = dq2.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[2] else None
        dk2 = dk2.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[3] else None
        dv  = dv.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[4] else None

        return dq1, dk1, dq2, dk2, dv, None, None


def triton_differential_attention(
    q1: torch.Tensor,
    k1: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    lambda_val: float = 0.8,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Dispatched Differential Attention with full Autograd backward support."""
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q1, k1, q2, k2, v) if isinstance(t, torch.Tensor)
    ):
        return TritonDifferentialAttentionFunction.apply(q1, k1, q2, k2, v, lambda_val, sm_scale)

    orig_shape = q1.shape
    is_3d = (q1.ndim == 3)
    if is_3d:
        B, L, D = q1.shape
        H = 1
        q1_4d = q1.view(B, H, L, D)
        k1_4d = k1.view(B, H, L, D)
        q2_4d = q2.view(B, H, L, D)
        k2_4d = k2.view(B, H, L, D)
        v_4d  = v.view(B, H, L, D)
    else:
        B, H, L, D = q1.shape
        q1_4d, k1_4d, q2_4d, k2_4d, v_4d = q1, k1, q2, k2, v

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if not HAS_TRITON or not q1.is_cuda or D not in (16, 32, 64, 128):
        return triton_differential_attention_pytorch_ref(
            q1, k1, q2, k2, v, lambda_val=lambda_val, sm_scale=sm_scale
        )

    q1_c = q1_4d.contiguous()
    k1_c = k1_4d.contiguous()
    q2_c = q2_4d.contiguous()
    k2_c = k2_4d.contiguous()
    v_c  = v_4d.contiguous()
    out_4d = torch.empty_like(v_c)

    grid = lambda META: (triton.cdiv(L, META["BLOCK_M"]), B * H)

    _diff_attn_fwd_kernel[grid](
        q1_c, k1_c, q2_c, k2_c, v_c, out_4d,
        q1_c.stride(0), q1_c.stride(1), q1_c.stride(2), q1_c.stride(3),
        k1_c.stride(0), k1_c.stride(1), k1_c.stride(2), k1_c.stride(3),
        v_c.stride(0),  v_c.stride(1),  v_c.stride(2),  v_c.stride(3),
        out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
        float(lambda_val), float(sm_scale),
        B, H, L,
        D=D,
    )

    return out_4d.view(orig_shape)


__all__ = [
    "triton_differential_attention",
    "triton_differential_attention_pytorch_ref",
    "TritonDifferentialAttentionFunction",
]
