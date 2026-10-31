#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Cross-Attention Triton kernel supporting L_q != L_kv with PyTorch fallback."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

logger = logging.getLogger(__name__)

_INTERPRET_MODE = os.environ.get("TRITON_INTERPRET") == "1"

if HAS_TRITON and triton is not None and tl is not None:
    @triton.jit
    def _cross_attn_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qm,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_om,
        stride_od,
        L_q: tl.constexpr,
        L_kv: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Cross-attention kernel for a single (batch, head) slice with L_q != L_kv."""
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)

        # First pass: find max score
        max_score = -1e9
        for n in range(L_kv):
            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if dot > max_score:
                max_score = dot

        # Second pass: compute softmax weights and accumulate values
        acc = tl.zeros((D,), dtype=tl.float32)
        sum_exp = 0.0
        for n in range(L_kv):
            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            p = tl.exp(dot - max_score)
            sum_exp += p
            v = tl.load(V_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
            acc += p * v

        if sum_exp > 0.0:
            out = acc / sum_exp
        else:
            out = tl.zeros((D,), dtype=tl.float32)

        tl.store(Out_ptr + m_idx * stride_om + d_offs * stride_od, out)

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["L_q", "L_kv"],
    )
    @triton.jit
    def _cross_attn_kernel_autotuned(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qm,
        stride_qd,
        stride_kn,
        stride_kd,
        stride_vn,
        stride_vd,
        stride_om,
        stride_od,
        L_q: tl.constexpr,
        L_kv: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)

        max_score = -1e9
        for n in range(L_kv):
            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if dot > max_score:
                max_score = dot

        acc = tl.zeros((D,), dtype=tl.float32)
        sum_exp = 0.0
        for n in range(L_kv):
            k = tl.load(K_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            p = tl.exp(dot - max_score)
            sum_exp += p
            v = tl.load(V_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
            acc += p * v

        if sum_exp > 0.0:
            out = acc / sum_exp
        else:
            out = tl.zeros((D,), dtype=tl.float32)

        tl.store(Out_ptr + m_idx * stride_om + d_offs * stride_od, out)


def _cross_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of cross-attention with L_q != L_kv."""
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(int(d))

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

    if mask is not None:
        scores = scores + mask.float()

    if is_causal:
        L_q = q.shape[-2]
        L_kv = k.shape[-2]
        causal_mask = torch.ones(L_q, L_kv, device=q.device, dtype=torch.bool).tril(diagonal=L_kv - L_q)
        scores = scores.masked_fill(~causal_mask, -1e4)

    attn = F.softmax(scores, dim=-1).to(q.dtype)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, v.to(q.dtype))
    return out


def cross_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Cross-attention forward operator using Triton kernel with PyTorch fallback.

    Args:
        q: Query tensor of shape (B, H, L_q, D).
        k: Key tensor of shape (B, H, L_kv, D).
        v: Value tensor of shape (B, H, L_kv, D).
        mask: Optional attention mask.
        is_causal: Whether to apply causal mask.
        scale: Optional softmax scale factor (defaults to 1/sqrt(D)).

    Returns:
        Attention output tensor of shape (B, H, L_q, D).
    """
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(q, "cross_attention", allow_fp32=True)
        if not is_suitable or mask is not None or is_causal:
            logger.warning("cross_attention Triton fallback: %s. Using PyTorch reference.", reason if not is_suitable else "mask present")
            return _cross_attention_pytorch(q, k, v, mask=mask, is_causal=is_causal, scale=scale, **kwargs)

    B, H, L_q, D = q.shape
    L_kv = k.shape[-2]
    device = q.device
    dtype = q.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(int(D))

    out = torch.empty(B, H, L_q, D, device=device, dtype=dtype)

    Q_f = q.contiguous().float()
    K_f = k.contiguous().float()
    V_f = v.contiguous().float()

    for b in range(B):
        for h in range(H):
            q_slice = Q_f[b, h]
            k_slice = K_f[b, h]
            v_slice = V_f[b, h]
            out_slice = out[b, h]

            if _INTERPRET_MODE:
                grid = (L_q,)
                _cross_attn_kernel[grid](
                    q_slice,
                    k_slice,
                    v_slice,
                    out_slice,
                    q_slice.stride(0),
                    q_slice.stride(1),
                    k_slice.stride(0),
                    k_slice.stride(1),
                    v_slice.stride(0),
                    v_slice.stride(1),
                    out_slice.stride(0),
                    out_slice.stride(1),
                    L_q=L_q,
                    L_kv=L_kv,
                    D=D,
                    scale=scale,
                )
            else:
                grid = lambda meta: (L_q,)
                _cross_attn_kernel_autotuned[grid](
                    q_slice,
                    k_slice,
                    v_slice,
                    out_slice,
                    q_slice.stride(0),
                    q_slice.stride(1),
                    k_slice.stride(0),
                    k_slice.stride(1),
                    v_slice.stride(0),
                    v_slice.stride(1),
                    out_slice.stride(0),
                    out_slice.stride(1),
                    L_q=L_q,
                    L_kv=L_kv,
                    D=D,
                    scale=scale,
                )

    return out


__all__ = ["cross_attention_forward", "_cross_attention_pytorch"]
