#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Dynamic Depthwise Convolution Attention Triton kernel with PyTorch fallback."""

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
    def _dynamic_conv_kernel(
        V_ptr,
        Weights_ptr,
        Out_ptr,
        stride_vt,
        stride_vd,
        stride_wn,
        stride_wk,
        stride_on,
        stride_od,
        N: tl.constexpr,
        K: tl.constexpr,
        D: tl.constexpr,
    ):
        """Dynamic depthwise convolution kernel for a single (batch, head) slice."""
        n_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        acc = tl.zeros((D,), dtype=tl.float32)

        for k in range(K):
            w = tl.load(Weights_ptr + n_idx * stride_wn + k * stride_wk).to(tl.float32)
            v = tl.load(V_ptr + (n_idx + k) * stride_vt + d_offs * stride_vd).to(tl.float32)
            acc += w * v

        tl.store(Out_ptr + n_idx * stride_on + d_offs * stride_od, acc)

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _dynamic_conv_kernel_autotuned(
        V_ptr,
        Weights_ptr,
        Out_ptr,
        stride_vt,
        stride_vd,
        stride_wn,
        stride_wk,
        stride_on,
        stride_od,
        N: tl.constexpr,
        K: tl.constexpr,
        D: tl.constexpr,
    ):
        n_idx = tl.program_id(0)
        d_offs = tl.arange(0, D)

        acc = tl.zeros((D,), dtype=tl.float32)

        for k in range(K):
            w = tl.load(Weights_ptr + n_idx * stride_wn + k * stride_wk).to(tl.float32)
            v = tl.load(V_ptr + (n_idx + k) * stride_vt + d_offs * stride_vd).to(tl.float32)
            acc += w * v

        tl.store(Out_ptr + n_idx * stride_on + d_offs * stride_od, acc)


def _dynamic_conv_pytorch(
    v_full: torch.Tensor,
    dyn_w: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation matching DynamicConvAttention."""
    B, H, T, d = v_full.shape
    N = dyn_w.shape[2]
    K = dyn_w.shape[3]

    v_win_all = v_full.unfold(dimension=2, size=K, step=1)
    out = torch.einsum("bhndk,bhnk->bhnd", v_win_all, dyn_w)
    return out


def dynamic_conv_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    kernel_size: int = 4,
    weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dynamic depthwise convolution forward operator using Triton kernel with PyTorch fallback.

    Args:
        q: Query / input tensor of shape (B, H, N, D).
        k: Optional key tensor (unused or for weight synthesis).
        v: Optional value tensor of shape (B, H, N, D) (defaults to q).
        kernel_size: Convolution kernel width K.
        weights: Optional pre-computed dynamic weights of shape (B, H, N, K).

    Returns:
        Convolved tensor of shape (B, H, N, D).
    """
    if v is None:
        v = q

    B, H, N, D = v.shape
    device = v.device
    dtype = v.dtype
    K = kernel_size

    if weights is None:
        # Synthesize normalized dynamic convolution weights per token
        logits = q.mean(dim=-1, keepdim=True).expand(-1, -1, -1, K)
        weights = F.softmax(logits.float(), dim=-1).to(dtype)

    pad = torch.zeros((B, H, K - 1, D), dtype=dtype, device=device)
    v_full = torch.cat([pad, v], dim=2)

    if not _INTERPRET_MODE:
        is_suitable, reason = check_triton_suitability(v, "dynamic_conv", allow_fp32=True)
        if not is_suitable:
            logger.warning("dynamic_conv Triton fallback: %s. Using PyTorch reference.", reason)
            return _dynamic_conv_pytorch(v_full, weights)

    out = torch.empty((B, H, N, D), device=device, dtype=dtype)
    v_full_f = v_full.contiguous().float()
    weights_f = weights.contiguous().float()

    for b in range(B):
        for h in range(H):
            v_slice = v_full_f[b, h]
            w_slice = weights_f[b, h]
            out_slice = out[b, h]

            if _INTERPRET_MODE:
                grid = (N,)
                _dynamic_conv_kernel[grid](
                    v_slice,
                    w_slice,
                    out_slice,
                    v_slice.stride(0),
                    v_slice.stride(1),
                    w_slice.stride(0),
                    w_slice.stride(1),
                    out_slice.stride(0),
                    out_slice.stride(1),
                    N=N,
                    K=K,
                    D=D,
                )
            else:
                grid = lambda meta: (N,)
                _dynamic_conv_kernel_autotuned[grid](
                    v_slice,
                    w_slice,
                    out_slice,
                    v_slice.stride(0),
                    v_slice.stride(1),
                    w_slice.stride(0),
                    w_slice.stride(1),
                    out_slice.stride(0),
                    out_slice.stride(1),
                    N=N,
                    K=K,
                    D=D,
                )

    return out


__all__ = ["dynamic_conv_attention_forward", "_dynamic_conv_pytorch"]
