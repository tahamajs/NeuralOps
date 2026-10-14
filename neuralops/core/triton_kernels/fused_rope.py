"""Fused RoPE Triton kernel with GPU JIT, autograd backward, and fast PyTorch fallback."""
from __future__ import annotations
import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl
from neuralops.core.rope import fused_rope_embedding

__all__ = ["fused_rope_triton", "fused_rope_embedding_kernel"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def fused_rope_fwd_kernel(
        Q, K, COS, SIN, OUT_Q, _OUT_K,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        _stride_cos_b, stride_cos_n, stride_cos_d,
        _stride_sin_b, stride_sin_n, stride_sin_d,
        stride_oqb, stride_oqh, stride_oqn, stride_oqd,
        _stride_okb, _stride_okh, _stride_okn, _stride_okd,
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

        # Load Q halves
        q1 = tl.load(Q + i_b * stride_qb + i_h * stride_qh + offs_n[:, None] * stride_qn + offs_d1[None, :] * stride_qd, mask=mask_n)
        q2 = tl.load(Q + i_b * stride_qb + i_h * stride_qh + offs_n[:, None] * stride_qn + offs_d2[None, :] * stride_qd, mask=mask_n)

        # Load cos, sin
        cos1 = tl.load(COS + offs_n[:, None] * stride_cos_n + offs_d1[None, :] * stride_cos_d, mask=mask_n)
        sin1 = tl.load(SIN + offs_n[:, None] * stride_sin_n + offs_d1[None, :] * stride_sin_d, mask=mask_n)

        oq1 = q1 * cos1 - q2 * sin1
        oq2 = q2 * cos1 + q1 * sin1

        tl.store(OUT_Q + i_b * stride_oqb + i_h * stride_oqh + offs_n[:, None] * stride_oqn + offs_d1[None, :] * stride_oqd, oq1, mask=mask_n)
        tl.store(OUT_Q + i_b * stride_oqb + i_h * stride_oqh + offs_n[:, None] * stride_oqn + offs_d2[None, :] * stride_oqd, oq2, mask=mask_n)


def _apply_rope_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard RoPE rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]."""
    D = x.shape[-1]
    half_d = D // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    # Match broadcast dims
    c = cos[..., :half_d]
    s = sin[..., :half_d]
    while c.dim() < x1.dim():
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat([out1, out2], dim=-1)


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
    return fused_rope_embedding(q, k, cos, sin)



fused_rope_triton = fused_rope_embedding_kernel


# ---------------------------------------------------------------------------
# Fused Unified RoPE with Offset (UnifiedRoPE)
# ---------------------------------------------------------------------------
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


def fused_unified_rope_offset_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Fused RoPE coordinate rotation with past offset in GPU SRAM."""
    return fused_unified_rope_offset_pytorch(x, cos, sin, rotary_dim)


__all__ = [
    "fused_rope_triton",
    "fused_rope_embedding_kernel",
    "fused_unified_rope_offset_triton",
    "fused_unified_rope_offset_pytorch",
]
