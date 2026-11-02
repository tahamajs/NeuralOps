#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Token-Compressed Attention Triton kernel with PyTorch fallback."""

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
    def _token_compressed_attn_kernel(
        Q_ptr,
        K_comp_ptr,
        V_comp_ptr,
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
        L_proj: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Attention kernel computing query to compressed KV attention for a single (batch, head) slice."""
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)

        # First pass: find max score
        max_score = -1e9
        for n in range(L_proj):
            k = tl.load(K_comp_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if dot > max_score:
                max_score = dot

        # Second pass: compute softmax and accumulate values
        acc = tl.zeros((D,), dtype=tl.float32)
        sum_exp = 0.0
        for n in range(L_proj):
            k = tl.load(K_comp_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            p = tl.math.exp2(1.44269504 * (dot - max_score))
            sum_exp += p
            v = tl.load(V_comp_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
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
        key=["L_q", "L_proj"],
    )
    @triton.jit
    def _token_compressed_attn_kernel_autotuned(
        Q_ptr,
        K_comp_ptr,
        V_comp_ptr,
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
        L_proj: tl.constexpr,
        D: tl.constexpr,
        scale: tl.constexpr,
    ):
        m_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        q = tl.load(Q_ptr + m_idx * stride_qm + d_offs * stride_qd).to(tl.float32)

        max_score = -1e9
        for n in range(L_proj):
            k = tl.load(K_comp_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            if dot > max_score:
                max_score = dot

        acc = tl.zeros((D,), dtype=tl.float32)
        sum_exp = 0.0
        for n in range(L_proj):
            k = tl.load(K_comp_ptr + n * stride_kn + d_offs * stride_kd).to(tl.float32)
            dot = tl.sum(q * k) * scale
            p = tl.math.exp2(1.44269504 * (dot - max_score))
            sum_exp += p
            v = tl.load(V_comp_ptr + n * stride_vn + d_offs * stride_vd).to(tl.float32)
            acc += p * v

        if sum_exp > 0.0:
            out = acc / sum_exp
        else:
            out = tl.zeros((D,), dtype=tl.float32)

        tl.store(Out_ptr + m_idx * stride_om + d_offs * stride_od, out)


def _token_compressed_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_projected_tokens: int = 4,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of token-compressed attention."""
    B, H, L, D = q.shape
    L_proj = min(num_projected_tokens, L)
    dtype = q.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(int(D))

    # Adaptive pooling compression along sequence dimension
    k_comp = F.adaptive_avg_pool1d(k.permute(0, 1, 3, 2).reshape(B * H, D, L), L_proj)
    k_comp = k_comp.reshape(B, H, D, L_proj).permute(0, 1, 3, 2).contiguous()

    v_comp = F.adaptive_avg_pool1d(v.permute(0, 1, 3, 2).reshape(B * H, D, L), L_proj)
    v_comp = v_comp.reshape(B, H, D, L_proj).permute(0, 1, 3, 2).contiguous()

    scores = torch.matmul(q.float(), k_comp.float().transpose(-2, -1).contiguous()) * scale
    attn = F.softmax(scores, dim=-1).to(dtype)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, v_comp.to(dtype))
    return out


def token_compressed_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_projected_tokens: int = 4,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Token-compressed attention forward operator using Triton kernel with PyTorch fallback.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        num_projected_tokens: Target number of compressed tokens L_proj.
        scale: Optional softmax scale factor.

    Returns:
        Attention output tensor of shape (B, H, L, D).
    """
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(q, "token_compressed", allow_fp32=True)
        if not is_suitable:
            logger.warning("token_compressed Triton fallback: %s. Using PyTorch reference.", reason)
            return _token_compressed_attention_pytorch(
                q, k, v, num_projected_tokens=num_projected_tokens, scale=scale, **kwargs
            )

    B, H, L, D = q.shape
    L_proj = min(num_projected_tokens, L)
    device = q.device
    dtype = q.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(int(D))

    # Compress K and V via adaptive average pooling
    k_comp = F.adaptive_avg_pool1d(k.permute(0, 1, 3, 2).reshape(B * H, D, L), L_proj)
    k_comp = k_comp.reshape(B, H, D, L_proj).permute(0, 1, 3, 2).contiguous().float()

    v_comp = F.adaptive_avg_pool1d(v.permute(0, 1, 3, 2).reshape(B * H, D, L), L_proj)
    v_comp = v_comp.reshape(B, H, D, L_proj).permute(0, 1, 3, 2).contiguous().float()

    out = torch.empty(B, H, L, D, device=device, dtype=dtype)
    Q_f = q.contiguous().float()

    for b in range(B):
        for h in range(H):
            q_slice = Q_f[b, h]
            k_slice = k_comp[b, h]
            v_slice = v_comp[b, h]
            out_slice = out[b, h]

            if _INTERPRET_MODE:
                grid = (L,)
                _token_compressed_attn_kernel[grid](
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
                    L_q=L,
                    L_proj=L_proj,
                    D=D,
                    scale=scale,
                )
            else:
                grid = lambda meta: (L,)
                _token_compressed_attn_kernel_autotuned[grid](
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
                    L_q=L,
                    L_proj=L_proj,
                    D=D,
                    scale=scale,
                )

    return out


token_compressed_attention_forward_pytorch = _token_compressed_attention_pytorch

__all__ = ["token_compressed_attention_forward", "_token_compressed_attention_pytorch", "token_compressed_attention_forward_pytorch"]
