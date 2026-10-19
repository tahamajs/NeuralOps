#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HGRN2 Fused Bidirectional Scan

Fuses forward and reverse recurrence passes using directional pointer arithmetic without intermediate tensor reversals.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _hgrn2_bidirectional_scan_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    B,
    H,
    L,
    D: tl.constexpr,
    stride_xb,
    stride_xh,
    stride_xl,
    stride_xd,
    stride_gb,
    stride_gh,
    stride_gl,
    stride_gd,
    stride_yb,
    stride_yh,
    stride_yl,
    stride_yd,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, D)

    h_fwd = tl.zeros([D], dtype=tl.float32)
    for t in range(0, L):
        x_ptrs = x_ptr + pid_b * stride_xb + pid_h * stride_xh + t * stride_xl + offs_d * stride_xd
        g_ptrs = g_ptr + pid_b * stride_gb + pid_h * stride_gh + t * stride_gl + offs_d * stride_gd

        x_t = tl.load(x_ptrs).to(tl.float32)
        g_t = tl.load(g_ptrs).to(tl.float32)

        h_fwd = g_t * h_fwd + (1.0 - g_t) * x_t

        y_ptrs = y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd
        tl.store(y_ptrs, (0.5 * h_fwd).to(x_ptr.dtype.element_ty))

    h_bwd = tl.zeros([D], dtype=tl.float32)
    for t_rev in range(0, L):
        t = L - 1 - t_rev
        x_ptrs = x_ptr + pid_b * stride_xb + pid_h * stride_xh + t * stride_xl + offs_d * stride_xd
        g_ptrs = g_ptr + pid_b * stride_gb + pid_h * stride_gh + t * stride_gl + offs_d * stride_gd

        x_t = tl.load(x_ptrs).to(tl.float32)
        g_t = tl.load(g_ptrs).to(tl.float32)

        h_bwd = g_t * h_bwd + (1.0 - g_t) * x_t

        y_ptrs = y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd
        prev_fwd = tl.load(y_ptrs).to(tl.float32)
        y_fused = prev_fwd + 0.5 * h_bwd
        tl.store(y_ptrs, y_fused.to(x_ptr.dtype.element_ty))


def triton_hgrn2_fused_bwd_scan(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """
    High-performance Triton fused bidirectional HGRN2 recurrent scan.

    Args:
        x: Input activations of shape (B, H, L, D).
        g: Decay gate values of shape (B, H, L, D) in [0, 1].

    Returns:
        Output tensor of shape (B, H, L, D).
    """
    B, H, L, D = x.shape

    if not x.is_cuda or D > 128:
        h_fwd = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        y_fwd = torch.empty_like(x)
        for t in range(L):
            h_fwd = g[:, :, t] * h_fwd + (1.0 - g[:, :, t]) * x[:, :, t]
            y_fwd[:, :, t] = h_fwd

        x_rev = torch.flip(x, dims=[2])
        g_rev = torch.flip(g, dims=[2])
        h_bwd = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        y_bwd_rev = torch.empty_like(x)
        for t in range(L):
            h_bwd = g_rev[:, :, t] * h_bwd + (1.0 - g_rev[:, :, t]) * x_rev[:, :, t]
            y_bwd_rev[:, :, t] = h_bwd

        y_bwd = torch.flip(y_bwd_rev, dims=[2])
        return (0.5 * (y_fwd + y_bwd)).to(x.dtype)

    y = torch.empty_like(x)
    grid = (B, H)

    x_c = x.contiguous()
    g_c = g.contiguous()

    _hgrn2_bidirectional_scan_kernel[grid](
        x_c,
        g_c,
        y,
        B,
        H,
        L,
        D=D,
        stride_xb=x_c.stride(0),
        stride_xh=x_c.stride(1),
        stride_xl=x_c.stride(2),
        stride_xd=x_c.stride(3),
        stride_gb=g_c.stride(0),
        stride_gh=g_c.stride(1),
        stride_gl=g_c.stride(2),
        stride_gd=g_c.stride(3),
        stride_yb=y.stride(0),
        stride_yh=y.stride(1),
        stride_yl=y.stride(2),
        stride_yd=y.stride(3),
    )

    return y
