#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Rotary Position Embedding

Applies rotary positional embeddings directly to query and key states with in-place pointer arithmetic.
"""

from __future__ import annotations
import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl
from neuralops.core.utils.rope import fused_rope_embedding


if HAS_TRITON and tl is not None:
    @triton.jit
    def fused_rope_fwd_kernel(
        Q_ptr, K_ptr, COS_ptr, SIN_ptr, OUT_Q_ptr, OUT_K_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_cos_b, stride_cos_n, stride_cos_d,
        stride_sin_b, stride_sin_n, stride_sin_d,
        stride_oqb, stride_oqh, stride_oqn, stride_oqd,
        stride_okb, stride_okh, stride_okn, stride_okd,
        H, N, D: tl.constexpr, HALF_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_n = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d1 = tl.arange(0, HALF_D)
        offs_d2 = HALF_D + tl.arange(0, HALF_D)

        mask_n = offs_n[:, None] < N

        q1 = tl.load(Q_ptr + i_b * stride_qb + i_h * stride_qh + offs_n[:, None] * stride_qn + offs_d1[None, :] * stride_qd, mask=mask_n, other=0.0)
        q2 = tl.load(Q_ptr + i_b * stride_qb + i_h * stride_qh + offs_n[:, None] * stride_qn + offs_d2[None, :] * stride_qd, mask=mask_n, other=0.0)

        k1 = tl.load(K_ptr + i_b * stride_kb + i_h * stride_kh + offs_n[:, None] * stride_kn + offs_d1[None, :] * stride_kd, mask=mask_n, other=0.0)
        k2 = tl.load(K_ptr + i_b * stride_kb + i_h * stride_kh + offs_n[:, None] * stride_kn + offs_d2[None, :] * stride_kd, mask=mask_n, other=0.0)

        cos1 = tl.load(COS_ptr + offs_n[:, None] * stride_cos_n + offs_d1[None, :] * stride_cos_d, mask=mask_n, other=0.0)
        sin1 = tl.load(SIN_ptr + offs_n[:, None] * stride_sin_n + offs_d1[None, :] * stride_sin_d, mask=mask_n, other=0.0)

        oq1 = q1 * cos1 - q2 * sin1
        oq2 = q2 * cos1 + q1 * sin1

        ok1 = k1 * cos1 - k2 * sin1
        ok2 = k2 * cos1 + k1 * sin1

        tl.store(OUT_Q_ptr + i_b * stride_oqb + i_h * stride_oqh + offs_n[:, None] * stride_oqn + offs_d1[None, :] * stride_oqd, oq1, mask=mask_n)
        tl.store(OUT_Q_ptr + i_b * stride_oqb + i_h * stride_oqh + offs_n[:, None] * stride_oqn + offs_d2[None, :] * stride_oqd, oq2, mask=mask_n)

        tl.store(OUT_K_ptr + i_b * stride_okb + i_h * stride_okh + offs_n[:, None] * stride_okn + offs_d1[None, :] * stride_okd, ok1, mask=mask_n)
        tl.store(OUT_K_ptr + i_b * stride_okb + i_h * stride_okh + offs_n[:, None] * stride_okn + offs_d2[None, :] * stride_okd, ok2, mask=mask_n)
else:
    fused_rope_fwd_kernel = None


def _apply_rope_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard RoPE rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]."""
    D = x.shape[-1]
    half_d = D // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    c = cos[..., :half_d]
    s = sin[..., :half_d]
    while c.dim() < x1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat([out1, out2], dim=-1)


apply_rope_rotation = _apply_rope_rotation


def _apply_rope_inverse_rotation(grad: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Inverse RoPE rotation for backward pass:
    Since rotation is orthogonal (R^T = R^(-1)), inverse rotation replaces sin with -sin:
    [g1, g2] -> [g1*cos + g2*sin, g2*cos - g1*sin].
    """
    D = grad.shape[-1]
    half_d = D // 2
    g1 = grad[..., :half_d]
    g2 = grad[..., half_d:]
    c = cos[..., :half_d]
    s = sin[..., :half_d]
    while c.dim() < g1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    dx1 = g1 * c + g2 * s
    dx2 = g2 * c - g1 * s
    return torch.cat([dx1, dx2], dim=-1)


class _FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        ctx.save_for_backward(cos, sin)
        out_q = _apply_rope_rotation(q, cos, sin)
        out_k = _apply_rope_rotation(k, cos, sin)
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        cos, sin = ctx.saved_tensors
        dq = _apply_rope_inverse_rotation(grad_q, cos, sin) if grad_q is not None else None
        dk = _apply_rope_inverse_rotation(grad_k, cos, sin) if grad_k is not None else None
        return dq, dk, None, None


def fused_rope_embedding_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies fused RoPE with full autograd backward support."""
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad:
        return _FusedRoPEFunction.apply(q, k, cos, sin)

    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK
    if (
        HAS_TRITON
        and not FORCE_PYTORCH_FALLBACK
        and q.is_cuda
        and k.is_cuda
        and q.shape[-1] in (32, 64, 128, 256)
        and q.shape[-1] == k.shape[-1]
    ):
        try:
            B, H, N, D = q.shape
            half_d = D // 2
            out_q = torch.empty_like(q)
            out_k = torch.empty_like(k)
            BLOCK_M = 64
            grid = (triton.cdiv(N, BLOCK_M), B * H)
            cos_2d = cos.view(-1, cos.shape[-1]) if cos.dim() >= 2 else cos
            sin_2d = sin.view(-1, sin.shape[-1]) if sin.dim() >= 2 else sin
            fused_rope_fwd_kernel[grid](
                q, k, cos_2d, sin_2d, out_q, out_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                0, cos_2d.stride(0), cos_2d.stride(1),
                0, sin_2d.stride(0), sin_2d.stride(1),
                out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
                out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
                H, N, D=D, HALF_D=half_d,
                BLOCK_M=BLOCK_M,
            )
            return out_q, out_k
        except Exception:
            pass

    return fused_rope_embedding(q, k, cos, sin)


fused_rope_triton = fused_rope_embedding_kernel


def fused_unified_rope_offset_pytorch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Exact PyTorch fallback for RoPE coordinate rotation with past offset."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    c = cos[..., :half]
    s = sin[..., :half]
    while c.dim() < x1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    rx1 = x1 * c - x2 * s
    rx2 = x2 * c + x1 * s
    rx = torch.cat([rx1, rx2], dim=-1)
    if x_pass.numel() > 0:
        return torch.cat([rx, x_pass], dim=-1)
    return rx


if HAS_TRITON and tl is not None:
    @triton.jit
    def _unified_rope_offset_kernel(
        X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
        total_tokens, L, D: tl.constexpr, ROTARY_DIM: tl.constexpr, HALF_ROTARY: tl.constexpr,
        stride_xn, stride_xd,
        stride_cn, stride_cd,
        stride_sn, stride_sd,
        stride_on, stride_od,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= total_tokens:
            return

        pos = pid % L

        offs_d1 = tl.arange(0, HALF_ROTARY)
        offs_d2 = HALF_ROTARY + tl.arange(0, HALF_ROTARY)

        x1 = tl.load(X_ptr + pid * stride_xn + offs_d1 * stride_xd)
        x2 = tl.load(X_ptr + pid * stride_xn + offs_d2 * stride_xd)

        cos = tl.load(COS_ptr + pos * stride_cn + offs_d1 * stride_cd)
        sin = tl.load(SIN_ptr + pos * stride_sn + offs_d1 * stride_sd)

        rx1 = x1 * cos - x2 * sin
        rx2 = x2 * cos + x1 * sin

        tl.store(OUT_ptr + pid * stride_on + offs_d1 * stride_od, rx1)
        tl.store(OUT_ptr + pid * stride_on + offs_d2 * stride_od, rx2)

        if ROTARY_DIM < D:
            offs_pass = ROTARY_DIM + tl.arange(0, BLOCK_D)
            mask_pass = offs_pass < D
            x_pass = tl.load(X_ptr + pid * stride_xn + offs_pass * stride_xd, mask=mask_pass, other=0.0)
            tl.store(OUT_ptr + pid * stride_on + offs_pass * stride_od, x_pass, mask=mask_pass)


def fused_unified_rope_offset_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Fused RoPE coordinate rotation with past offset in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK
    if (
        HAS_TRITON
        and not FORCE_PYTORCH_FALLBACK
        and x.is_cuda
        and rotary_dim % 2 == 0
        and rotary_dim in (16, 32, 64, 128, 256)
    ):
        try:
            orig_shape = x.shape
            D = orig_shape[-1]
            L = orig_shape[-2]
            x_2d = x.contiguous().view(-1, D)
            total_tokens = x_2d.shape[0]
            out_2d = torch.empty_like(x_2d)

            cos_2d = cos.view(-1, cos.shape[-1]).contiguous() if cos.dim() >= 2 else cos.contiguous()
            sin_2d = sin.view(-1, sin.shape[-1]).contiguous() if sin.dim() >= 2 else sin.contiguous()

            half_rotary = rotary_dim // 2
            pass_dim = max(1, D - rotary_dim)
            block_d = triton.next_power_of_2(pass_dim)

            grid = (total_tokens,)
            _unified_rope_offset_kernel[grid](
                x_2d, cos_2d, sin_2d, out_2d,
                total_tokens, L, D=D, ROTARY_DIM=rotary_dim, HALF_ROTARY=half_rotary,
                stride_xn=x_2d.stride(0), stride_xd=x_2d.stride(1),
                stride_cn=cos_2d.stride(0), stride_cd=cos_2d.stride(1),
                stride_sn=sin_2d.stride(0), stride_sd=sin_2d.stride(1),
                stride_on=out_2d.stride(0), stride_od=out_2d.stride(1),
                BLOCK_D=block_d,
            )
            return out_2d.view(orig_shape)
        except Exception:
            pass

    return fused_unified_rope_offset_pytorch(x, cos, sin, rotary_dim)


__all__ = [
    "fused_rope_triton",
    "fused_rope_embedding_kernel",
    "fused_unified_rope_offset_triton",
    "fused_unified_rope_offset_pytorch",
    "fused_rope_fwd_kernel",
    "_apply_rope_rotation",
    "apply_rope_rotation",
]
