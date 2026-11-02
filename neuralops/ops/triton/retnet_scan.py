#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RetNet Multi-Scale Retention Scan

Computes multi-scale retention mechanism with exponential decay matrices:
  S_t = gamma * S_{t-1} + k_t^T v_t
  o_t = q_t S_t
Adapted from Flash Linear Attention (FLA) architecture with 3-level fallback hierarchy.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from neuralops.ops.triton._base import HAS_TRITON, tl, triton
from neuralops.ops.triton._types import check_triton_suitability

logger = logging.getLogger(__name__)


if HAS_TRITON and tl is not None:
    @triton.jit
    def _retnet_recurrent_scan_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        gamma_ptr,
        o_ptr,
        final_state_ptr,
        B,
        H,
        L,
        D: tl.constexpr,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_gh,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        stride_sb,
        stride_sh,
        stride_sk,
        stride_sv,
        SAVE_STATE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_k = tl.arange(0, D)
        offs_v = tl.arange(0, D)

        gamma = tl.load(gamma_ptr + pid_h * stride_gh).to(tl.float32)
        S = tl.zeros((D, D), dtype=tl.float32)

        for t in range(L):
            q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_k * stride_qd
            k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_k * stride_kd
            v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_v * stride_vd

            q_t = tl.load(q_ptrs).to(tl.float32)
            k_t = tl.load(k_ptrs).to(tl.float32)
            v_t = tl.load(v_ptrs).to(tl.float32)

            S = gamma * S + (k_t[:, None] * v_t[None, :])
            o_t = tl.sum(q_t[:, None] * S, axis=0)

            o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ol + offs_v * stride_od
            tl.store(o_ptrs, o_t.to(q_ptr.dtype.element_ty))

        if SAVE_STATE:
            s_ptrs = (
                final_state_ptr
                + pid_b * stride_sb
                + pid_h * stride_sh
                + offs_k[:, None] * stride_sk
                + offs_v[None, :] * stride_sv
            )
            tl.store(s_ptrs, S.to(final_state_ptr.dtype.element_ty))
else:
    _retnet_recurrent_scan_fwd_kernel = None


def retnet_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float | torch.Tensor = 0.9,
    initial_state: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback reference implementation for RetNet retention scan."""
    B, H, L, D = q.shape
    device = q.device
    dtype = q.dtype

    if isinstance(gamma, (float, int)):
        gammas = torch.full((H,), float(gamma), device=device, dtype=torch.float32)
    elif gamma.dim() == 0:
        gammas = gamma.expand(H).float()
    else:
        gammas = gamma.view(H).float()

    if initial_state is not None:
        S = initial_state.clone().float()
    else:
        S = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)

    outputs = []
    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()

        g = gammas.view(1, H, 1, 1)
        kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
        S = g * S + kv

        o_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2)
        outputs.append(o_t)

    out = torch.stack(outputs, dim=2).to(dtype)
    if return_final_state:
        return out, S.to(dtype)
    return out


def retnet_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float | torch.Tensor = 0.9,
    initial_state: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated RetNet retention scan with fail-closed PyTorch fallback."""
    B, H, L, D = q.shape

    if isinstance(gamma, (float, int)):
        gammas = torch.full((H,), float(gamma), device=q.device, dtype=torch.float32)
    elif gamma.dim() == 0:
        gammas = gamma.expand(H).float().to(q.device)
    else:
        gammas = gamma.view(H).float().to(q.device)

    is_suitable, reason = check_triton_suitability(q, "retnet_scan", head_dim=D, seq_len=L, allow_fp32=True)
    if not is_suitable or not HAS_TRITON or initial_state is not None or D not in (16, 32, 64, 128):
        if not is_suitable:
            logger.warning("Triton kernel 'retnet_scan' using PyTorch fallback: %s", reason)
        return retnet_scan_pytorch(
            q, k, v, gamma=gammas, initial_state=initial_state, return_final_state=return_final_state,
        )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = gammas.contiguous()

    out = torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)
    final_s = torch.empty((B, H, D, D), dtype=q.dtype, device=q.device) if return_final_state else out

    grid = (B, H)
    try:
        _retnet_recurrent_scan_fwd_kernel[grid](
            q_c, k_c, v_c, g_c, out, final_s,
            B, H, L,
            D=D,
            stride_qb=q_c.stride(0), stride_qh=q_c.stride(1), stride_ql=q_c.stride(2), stride_qd=q_c.stride(3),
            stride_kb=k_c.stride(0), stride_kh=k_c.stride(1), stride_kl=k_c.stride(2), stride_kd=k_c.stride(3),
            stride_vb=v_c.stride(0), stride_vh=v_c.stride(1), stride_vl=v_c.stride(2), stride_vd=v_c.stride(3),
            stride_gh=g_c.stride(0),
            stride_ob=out.stride(0), stride_oh=out.stride(1), stride_ol=out.stride(2), stride_od=out.stride(3),
            stride_sb=final_s.stride(0), stride_sh=final_s.stride(1), stride_sk=final_s.stride(2), stride_sv=final_s.stride(3),
            SAVE_STATE=return_final_state,
        )
        if return_final_state:
            return out, final_s
        return out
    except Exception as exc:
        logger.warning("Triton kernel 'retnet_scan' failed (%s); falling back to PyTorch reference.", exc)
        return retnet_scan_pytorch(
            q, k, v, gamma=gammas, initial_state=initial_state, return_final_state=return_final_state,
        )


retnet_scan = retnet_scan_triton

__all__ = [
    "retnet_scan",
    "retnet_scan_triton",
    "retnet_scan_pytorch",
]
