#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Rank-Augmented Linear Attention (RALA) Triton kernel with PyTorch fallback."""

from __future__ import annotations

import logging
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
    def _rala_causal_step_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_ql,
        stride_qd,
        stride_kl,
        stride_kd,
        stride_vl,
        stride_vd,
        stride_ol,
        stride_od,
        L: tl.constexpr,
        D: tl.constexpr,
    ):
        """Sequential causal linear recurrence kernel for a single (batch, head) slice."""
        d_offs = tl.arange(0, D)

        S = tl.zeros((D, D), dtype=tl.float32)
        Z = tl.zeros((D,), dtype=tl.float32)

        for i in range(L):
            q = tl.load(Q_ptr + i * stride_ql + d_offs * stride_qd).to(tl.float32)
            k = tl.load(K_ptr + i * stride_kl + d_offs * stride_kd).to(tl.float32)
            v = tl.load(V_ptr + i * stride_vl + d_offs * stride_vd).to(tl.float32)

            # Update S += k[:, None] * v[None, :]
            S += k[:, None] * v[None, :]
            Z += k

            # Compute num = q @ S
            num = tl.sum(q[:, None] * S, axis=0)
            denom = tl.sum(q * Z)
            denom_clamped = tl.maximum(denom, 1e-4)

            out = num / denom_clamped
            tl.store(Out_ptr + i * stride_ol + d_offs * stride_od, out)

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _rala_causal_step_kernel_autotuned(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_ql,
        stride_qd,
        stride_kl,
        stride_kd,
        stride_vl,
        stride_vd,
        stride_ol,
        stride_od,
        L: tl.constexpr,
        D: tl.constexpr,
    ):
        d_offs = tl.arange(0, D)

        S = tl.zeros((D, D), dtype=tl.float32)
        Z = tl.zeros((D,), dtype=tl.float32)

        for i in range(L):
            q = tl.load(Q_ptr + i * stride_ql + d_offs * stride_qd).to(tl.float32)
            k = tl.load(K_ptr + i * stride_kl + d_offs * stride_kd).to(tl.float32)
            v = tl.load(V_ptr + i * stride_vl + d_offs * stride_vd).to(tl.float32)

            S += k[:, None] * v[None, :]
            Z += k

            num = tl.sum(q[:, None] * S, axis=0)
            denom = tl.sum(q * Z)
            denom_clamped = tl.maximum(denom, 1e-4)

            out = num / denom_clamped
            tl.store(Out_ptr + i * stride_ol + d_offs * stride_od, out)


def _rala_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lora_U: torch.Tensor | None = None,
    lora_V: torch.Tensor | None = None,
    is_causal: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Exact reference PyTorch implementation matching RALALinearAttention."""
    B, H, N, d = q.shape
    device = q.device
    q_p = F.softplus(q.float())
    k_p = F.softplus(k.float())
    v_m = v.float()

    if is_causal:
        S = torch.zeros(B, H, d, d, dtype=torch.float32, device=device)
        z = torch.zeros(B, H, d, dtype=torch.float32, device=device)
        num_list = []
        denom_list = []
        for i in range(N):
            q_i = q_p[:, :, i, :]
            k_i = k_p[:, :, i, :]
            v_i = v_m[:, :, i, :]
            S = S + torch.einsum("bhd,bhe->bhde", k_i, v_i)
            z = z + k_i
            num_i = torch.einsum("bhd,bhde->bhe", q_i, S)
            denom_i = torch.einsum("bhd,bhd->bh", q_i, z).unsqueeze(-1)
            num_list.append(num_i)
            denom_list.append(denom_i)
        num = torch.stack(num_list, dim=2)
        denom = torch.stack(denom_list, dim=2).clamp(min=1e-4)
    else:
        kv = torch.einsum("bhnd,bhne->bhde", k_p, v_m)
        z = k_p.sum(dim=2)
        num = torch.einsum("bhnd,bhde->bhne", q_p, kv)
        denom = torch.einsum("bhnd,bhd->bhn", q_p, z).unsqueeze(-1).clamp(min=1e-4)

    lin_out = torch.nan_to_num(num / denom, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)

    if lora_U is not None and lora_V is not None:
        rank_mat = torch.einsum("hdr,her->hde", lora_U.float(), lora_V.float())
        aug_out = torch.nan_to_num(
            torch.einsum("bhnd,hde->bhne", q_p, rank_mat) / denom, nan=0.0, posinf=1e4, neginf=-1e4
        ).to(q.dtype)
        return lin_out + aug_out

    return lin_out


def rala_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lora_U: torch.Tensor | None = None,
    lora_V: torch.Tensor | None = None,
    is_causal: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """RALA forward operator using Triton kernel with PyTorch fallback.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        lora_U: Optional head interaction U matrix (H, D, R).
        lora_V: Optional head interaction V matrix (H, D, R).
        is_causal: Whether to apply causal mask.

    Returns:
        Attention output tensor of shape (B, H, L, D).
    """
    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(q, "rala_attention", allow_fp32=True)
        if not is_suitable:
            logger.warning("rala_attention Triton fallback: %s. Using PyTorch reference.", reason)
            return _rala_attention_pytorch(q, k, v, lora_U=lora_U, lora_V=lora_V, is_causal=is_causal, **kwargs)

    B, H, L, D = q.shape
    device = q.device
    dtype = q.dtype

    q_p = F.softplus(q.float()).contiguous()
    k_p = F.softplus(k.float()).contiguous()
    v_m = v.float().contiguous()

    out = torch.empty(B, H, L, D, device=device, dtype=dtype)

    for b in range(B):
        for h in range(H):
            q_slice = q_p[b, h]
            k_slice = k_p[b, h]
            v_slice = v_m[b, h]
            out_slice = out[b, h]

            if _INTERPRET_MODE:
                _rala_causal_step_kernel[(1,)](
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
                    L=L,
                    D=D,
                )
            else:
                _rala_causal_step_kernel_autotuned[(1,)](
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
                    L=L,
                    D=D,
                )

    if lora_U is not None and lora_V is not None:
        rank_mat = torch.einsum("hdr,her->hde", lora_U.float(), lora_V.float())
        z = k_p.cumsum(dim=2)
        denom = (q_p * z).sum(dim=-1, keepdim=True).clamp(min=1e-4)
        aug_out = torch.nan_to_num(
            torch.einsum("bhnd,hde->bhne", q_p, rank_mat) / denom, nan=0.0, posinf=1e4, neginf=-1e4
        ).to(dtype)
        return out + aug_out

    return out


rala_attention_forward_pytorch = _rala_attention_pytorch

__all__ = ["rala_attention_forward", "_rala_attention_pytorch", "rala_attention_forward_pytorch"]
