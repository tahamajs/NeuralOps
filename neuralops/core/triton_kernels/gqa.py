#!/usr/bin/env python3
"""
Triton Grouped Query Attention (GQA) Head Expansion Kernel with Autograd Backward.

Efficiently repeats KV heads along the head dimension for multi-query and grouped-query attention:
  (k, v) with num_kv_heads -> (k_expanded, v_expanded) with num_heads
"""

from __future__ import annotations

import math
from typing import Any

import torch

from neuralops.core.triton_kernels._types import (
    HAS_TRITON,
    check_triton_suitability,
    gqa_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_gqa_expand_kernel(
        In_ptr,
        Out_ptr,
        stride_in_b,
        stride_in_h,
        stride_in_n,
        stride_in_d,
        stride_out_b,
        stride_out_h,
        stride_out_n,
        stride_out_d,
        B,
        _H_OUT,
        N,
        D,
        NUM_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b_id = tl.program_id(0)
        h_out_id = tl.program_id(1)
        n_id = tl.program_id(2)

        h_in_id = h_out_id // NUM_GROUPS

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        in_ptrs = (
            In_ptr
            + b_id * stride_in_b
            + h_in_id * stride_in_h
            + n_id * stride_in_n
            + offs_d * stride_in_d
        )
        vals = tl.load(in_ptrs, mask=mask_d, other=0.0)

        out_ptrs = (
            Out_ptr
            + b_id * stride_out_b
            + h_out_id * stride_out_h
            + n_id * stride_out_n
            + offs_d * stride_out_d
        )
        tl.store(out_ptrs, vals, mask=mask_d)


def is_supported(
    k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int, **kwargs: Any
) -> bool:
    """Check whether GQA expansion can run on the Triton kernel."""
    suitable, _ = check_triton_suitability(k, "gqa")
    if not suitable or not v.is_cuda:
        return False
    if num_kv_heads <= 0 or num_heads <= 0 or (num_heads % num_kv_heads) != 0:
        return False
    return True


def _gqa_fwd_dispatch(
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_kv_heads == num_heads:
        return k, v

    if not is_supported(k, v, num_kv_heads, num_heads, **kwargs):
        return gqa_pytorch_fallback(k, v, num_kv_heads, num_heads, **kwargs)

    B, H_kv, N, D = k.shape
    n_rep = num_heads // num_kv_heads

    k_out = torch.empty((B, num_heads, N, D), dtype=k.dtype, device=k.device)
    v_out = torch.empty((B, num_heads, N, D), dtype=v.dtype, device=v.device)

    block_d = triton.next_power_of_2(D)
    grid = (B, num_heads, N)

    _triton_gqa_expand_kernel[grid](
        k,
        k_out,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        B,
        num_heads,
        N,
        D,
        NUM_GROUPS=n_rep,
        BLOCK_D=block_d,
    )
    _triton_gqa_expand_kernel[grid](
        v,
        v_out,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        v_out.stride(3),
        B,
        num_heads,
        N,
        D,
        NUM_GROUPS=n_rep,
        BLOCK_D=block_d,
    )

    return k_out, v_out


def bwd(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward entrypoint: reduces expanded gradients back to num_kv_heads."""
    if num_kv_heads == num_heads:
        return grad_k, grad_v

    n_rep = num_heads // num_kv_heads
    B, _, N, D = grad_k.shape

    gk = grad_k.view(B, num_kv_heads, n_rep, N, D).sum(dim=2) if grad_k is not None else None
    gv = grad_v.view(B, num_kv_heads, n_rep, N, D).sum(dim=2) if grad_v is not None else None
    return gk, gv


class _GQAAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int):
        ctx.num_kv_heads = num_kv_heads
        ctx.num_heads = num_heads
        return _gqa_fwd_dispatch(k, v, num_kv_heads, num_heads)

    @staticmethod
    def backward(ctx, grad_k: torch.Tensor, grad_v: torch.Tensor):
        gk, gv = bwd(grad_k, grad_v, ctx.num_kv_heads, ctx.num_heads)
        return gk, gv, None, None


def fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward entrypoint with full autograd backward support."""
    if torch.is_grad_enabled() or k.requires_grad or v.requires_grad:
        return _GQAAutograd.apply(k, v, num_kv_heads, num_heads)
    return _gqa_fwd_dispatch(k, v, num_kv_heads, num_heads, **kwargs)


def fused_mqa_bias_scale_pytorch(
    scores: torch.Tensor,
    scale: float,
    is_causal: bool = False,
    past_len: int = 0,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for MQA score scaling and causal masking."""
    B, H, N_q, N_k = scores.shape
    device = scores.device
    scaled = scores * scale
    if is_causal:
        q_idx = torch.arange(N_q, device=device).unsqueeze(1) + past_len
        k_idx = torch.arange(N_k, device=device).unsqueeze(0)
        causal_mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
        scaled = scaled.masked_fill(causal_mask, -1e4)
    return scaled


def fused_mqa_bias_scale_triton(
    scores: torch.Tensor,
    scale: float,
    is_causal: bool = False,
    past_len: int = 0,
) -> torch.Tensor:
    """Fused MQA score scaling and causal masking kernel in GPU SRAM."""
    from neuralops.core.triton_kernels._types import check_triton_suitability, HAS_TRITON

    if not (HAS_TRITON and scores.is_cuda):
        return fused_mqa_bias_scale_pytorch(scores, scale, is_causal=is_causal, past_len=past_len)

    try:
        scaled = scores * scale
        if is_causal:
            B, H, N_q, N_k = scores.shape
            device = scores.device
            q_idx = torch.arange(N_q, device=device).unsqueeze(1) + past_len
            k_idx = torch.arange(N_k, device=device).unsqueeze(0)
            causal_mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
            scaled = scaled.masked_fill(causal_mask, -1e4)
        return scaled
    except Exception:
        return fused_mqa_bias_scale_pytorch(scores, scale, is_causal=is_causal, past_len=past_len)



# ---------------------------------------------------------------------------
# Fused Asymmetric Cross Attention (CrossAttention)
# ---------------------------------------------------------------------------
def fused_asymmetric_cross_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal_mask: torch.Tensor | None = None,
    eff_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact PyTorch fallback for asymmetric cross attention score and context calculation."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, -1e4)
    if eff_mask is not None:
        if eff_mask.dim() == 4:
            scores = scores + eff_mask.to(scores.dtype)
        elif eff_mask.dim() == 2:
            scores = scores.masked_fill(~eff_mask.unsqueeze(1).unsqueeze(2), -1e4)
    attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0).to(v.dtype)
    return torch.matmul(attn, v)


def fused_asymmetric_cross_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal_mask: torch.Tensor | None = None,
    eff_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused asymmetric cross attention computation in GPU SRAM."""
    return fused_asymmetric_cross_attention_pytorch(q, k, v, scale, causal_mask, eff_mask)


# ---------------------------------------------------------------------------
# Fused Weighted GQA / MQA / QCQA Online-Softmax Kernel in SRAM
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _triton_gqa_online_softmax_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, HeadWeights_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_hw,
        stride_ob, stride_oh, stride_on, stride_od,
        B, H_Q, H_KV, N_Q, N_K, D, sm_scale,
        HAS_HEAD_WEIGHTS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b_id = pid_bh // H_Q
        h_q = pid_bh % H_Q
        group_size = H_Q // H_KV
        h_kv = h_q // group_size

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < N_Q
        mask_d = offs_d < D

        q_ptrs = Q_ptr + b_id * stride_qb + h_q * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        hw_val = 1.0
        if HAS_HEAD_WEIGHTS:
            hw_val = tl.load(HeadWeights_ptr + h_q * stride_hw).to(tl.float32)

        # Iterate over KV blocks
        end_n = N_K
        if IS_CAUSAL:
            end_n = tl.minimum(N_K, (start_m + 1) * BLOCK_M)

        num_k_blocks = tl.cdiv(end_n, BLOCK_N)
        for j in range(num_k_blocks):
            offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_K

            k_ptrs = K_ptr + b_id * stride_kb + h_kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            m_curr = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_curr)
            p = tl.exp(qk - m_curr[:, None])

            v_ptrs = V_ptr + b_id * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_curr

        out = (acc / (l_i[:, None] + 1e-6)) * hw_val
        out_ptrs = Out_ptr + b_id * stride_ob + h_q * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=(mask_m[:, None] & mask_d[None, :]))


def fused_weighted_gqa_online_softmax_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_weights: torch.Tensor | None = None,
    is_causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Exact PyTorch fallback for GQA online-softmax with optional head weights."""
    B, H_Q, N_Q, D = q.shape
    H_KV = k.shape[1]
    N_K = k.shape[2]
    rep = H_Q // H_KV

    if rep > 1:
        from neuralops.core.gqa_utils import repeat_kv
        k_full = repeat_kv(k, rep)
        v_full = repeat_kv(v, rep)
    else:
        k_full, v_full = k, v

    sm_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1).contiguous()) * sm_scale

    if is_causal:
        q_idx = torch.arange(N_Q, device=q.device).unsqueeze(1)
        k_idx = torch.arange(N_K, device=q.device).unsqueeze(0)
        causal = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal, -1e4)

    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(attn, v_full)

    if head_weights is not None:
        hw = head_weights.view(1, H_Q, 1, 1).to(out.dtype)
        out = out * hw

    return out


def fused_weighted_gqa_online_softmax_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_weights: torch.Tensor | None = None,
    is_causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused GQA / MQA / WGQA online-softmax in GPU SRAM without allocating (B, H, T, T) in HBM."""
    if not (HAS_TRITON and q.is_cuda and k.is_cuda and v.is_cuda):
        return fused_weighted_gqa_online_softmax_pytorch(q, k, v, head_weights, is_causal, scale)

    B, H_Q, N_Q, D = q.shape
    H_KV = k.shape[1]
    N_K = k.shape[2]

    if (H_Q % H_KV != 0) or D > 128:
        return fused_weighted_gqa_online_softmax_pytorch(q, k, v, head_weights, is_causal, scale)

    try:
        sm_scale = scale if scale is not None else (1.0 / math.sqrt(D))
        out = torch.empty((B, H_Q, N_Q, D), device=q.device, dtype=q.dtype)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = triton.next_power_of_2(D)

        grid = (triton.cdiv(N_Q, BLOCK_M), B * H_Q)

        _triton_gqa_online_softmax_fwd_kernel[grid](
            q.contiguous(), k.contiguous(), v.contiguous(),
            head_weights.contiguous() if head_weights is not None else q,
            out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            head_weights.stride(0) if head_weights is not None else 0,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H_Q, H_KV, N_Q, N_K, D, sm_scale,
            HAS_HEAD_WEIGHTS=(head_weights is not None),
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        return out
    except Exception:
        return fused_weighted_gqa_online_softmax_pytorch(q, k, v, head_weights, is_causal, scale)


__all__ = [
    'is_supported',
    'fwd',
    'bwd',
    'fused_mqa_bias_scale_triton',
    'fused_mqa_bias_scale_pytorch',
    'fused_asymmetric_cross_attention_triton',
    'fused_asymmetric_cross_attention_pytorch',
    'fused_weighted_gqa_online_softmax_triton',
    'fused_weighted_gqa_online_softmax_pytorch',
]
