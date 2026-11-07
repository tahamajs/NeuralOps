#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TTT Linear Attention Step (Level 2 Triton JIT & Level 3 PyTorch Reference).

Fuses online test-time gradient descent on memory matrix S:
    pred_v_t = k_t @ S_{t-1}
    err_t = pred_v_t - v_t
    S_t = S_{t-1} - eta_t * (k_t^T @ err_t)
    y_t = q_t @ S_t
Supports parameterized BLOCK_D compile-time branches with zero tl.make_block_ptr.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False
    triton = None
    tl = None

from neuralops.ops.triton._types import check_triton_suitability

logger = logging.getLogger(__name__)


if HAS_TRITON and triton is not None and tl is not None:

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=2),
        ],
        key=["N", "D"],
    )
    @triton.jit
    def _ttt_linear_step_scan_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Etas_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_eb,
        stride_eh,
        stride_en,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        B,
        H,
        N,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """TTT online gradient descent recurrence kernel."""
        bh_idx = tl.program_id(0)
        b_idx = bh_idx // H
        h_idx = bh_idx % H

        q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
        k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
        v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
        e_base = Etas_ptr + b_idx * stride_eb + h_idx * stride_eh
        o_base = Out_ptr + b_idx * stride_ob + h_idx * stride_oh

        offs_d1 = tl.arange(0, BLOCK_D)
        offs_d2 = tl.arange(0, BLOCK_D)

        mask_d1 = offs_d1 < D
        mask_d2 = offs_d2 < D

        # Recurrent state matrix S in [BLOCK_D, BLOCK_D]
        S = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

        for t in range(N):
            q_ptrs = q_base + t * stride_qn + offs_d1 * stride_qd
            k_ptrs = k_base + t * stride_kn + offs_d1 * stride_kd
            v_ptrs = v_base + t * stride_vn + offs_d2 * stride_vd
            e_ptr = e_base + t * stride_en

            q_t = tl.load(q_ptrs, mask=mask_d1, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptrs, mask=mask_d1, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptrs, mask=mask_d2, other=0.0).to(tl.float32)
            eta_t = tl.load(e_ptr).to(tl.float32)

            # pred_v = k_t @ S (dimension 1 x BLOCK_D)
            # k_t is [BLOCK_D], S is [BLOCK_D, BLOCK_D] -> k_t[:, None] * S summed over axis 0
            pred_v = tl.sum(k_t[:, None] * S, axis=0)

            # err = pred_v - v_t
            err = pred_v - v_t

            # Online gradient step: S = S - eta_t * (k_t[:, None] * err[None, :])
            grad_S = k_t[:, None] * err[None, :]
            S = S - eta_t * grad_S

            # y_t = q_t @ S (dimension 1 x BLOCK_D)
            y_t = tl.sum(q_t[:, None] * S, axis=0)

            o_ptrs = o_base + t * stride_on + offs_d2 * stride_od
            tl.store(o_ptrs, y_t.to(Out_ptr.dtype.element_ty), mask=mask_d2)


def _ttt_linear_step_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    etas: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """PyTorch reference implementation of TTT online gradient descent memory scan."""
    B, H, N, d = q.shape
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    etas_f = etas.float()

    S = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
    outs = []
    for t in range(N):
        qt = q_f[:, :, t : t + 1]
        kt = k_f[:, :, t : t + 1]
        vt = v_f[:, :, t : t + 1]
        eta = etas_f[:, :, t : t + 1]
        if eta.dim() == 3:
            eta = eta.unsqueeze(-1)

        pred_v = torch.matmul(kt, S)
        err = pred_v - vt
        grad_S = torch.matmul(kt.transpose(-1, -2), err)
        S = S - eta * grad_S
        yt = torch.matmul(qt, S)
        outs.append(yt.to(q.dtype))

    return torch.cat(outs, dim=2)


def ttt_linear_step_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    etas: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """TTT Linear Attention forward dispatcher with Level 2 Triton JIT and Level 3 PyTorch fallback."""
    if not HAS_TRITON or not q.is_cuda:
        return _ttt_linear_step_pytorch_ref(q, k, v, etas, **kwargs)

    is_suitable, reason = check_triton_suitability(q, "ttt_linear_step", allow_fp32=True)
    if not is_suitable:
        logger.warning("ttt_linear_step Triton unsuitable: %s. Falling back to PyTorch.", reason)
        return _ttt_linear_step_pytorch_ref(q, k, v, etas, **kwargs)

    B, H, N, D = q.shape

    # Determine BLOCK_D based on compile-time branches
    if D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    elif D <= 128:
        BLOCK_D = 128
    else:
        # Fall back to PyTorch reference if state matrix exceeds 128x128
        return _ttt_linear_step_pytorch_ref(q, k, v, etas, **kwargs)

    Q_f = q.contiguous()
    K_f = k.contiguous()
    V_f = v.contiguous()
    etas_f = etas.contiguous()
    if etas_f.dim() == 4 and etas_f.shape[-1] == 1:
        etas_f = etas_f.squeeze(-1)

    out = torch.empty(B, H, N, D, device=q.device, dtype=q.dtype)

    grid = (B * H,)

    _ttt_linear_step_scan_kernel[grid](
        Q_f,
        K_f,
        V_f,
        etas_f,
        out,
        Q_f.stride(0),
        Q_f.stride(1),
        Q_f.stride(2),
        Q_f.stride(3),
        K_f.stride(0),
        K_f.stride(1),
        K_f.stride(2),
        K_f.stride(3),
        V_f.stride(0),
        V_f.stride(1),
        V_f.stride(2),
        V_f.stride(3),
        etas_f.stride(0),
        etas_f.stride(1),
        etas_f.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B=B,
        H=H,
        N=N,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return out


__all__ = [
    "ttt_linear_step_triton",
    "_ttt_linear_step_pytorch_ref",
]
