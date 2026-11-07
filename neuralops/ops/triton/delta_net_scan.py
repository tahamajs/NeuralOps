#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
DeltaNet Delta-Rule Recurrent Scan

Computes gated DeltaNet linear attention recurrence using on-chip SRAM register state:
  S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
  o_t = S_t q_t
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
    def _delta_net_scan_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        alpha_ptr,
        beta_ptr,
        o_ptr,
        final_state_ptr,
        B,
        H,
        L,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
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
        stride_ab,
        stride_ah,
        stride_al,
        stride_bb,
        stride_bh,
        stride_bl,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        stride_sb,
        stride_sh,
        stride_sv,
        stride_sk,
        SAVE_STATE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_k = tl.arange(0, D_K)
        offs_v = tl.arange(0, D_V)

        # In-register S state matrix: S is [D_V, D_K] maintained in fast on-chip SRAM register tiles.
        # Plastic error-correction delta rule update:
        #   S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        # computed directly inside SRAM register tiles without round-tripping to DRAM.
        S = tl.zeros((D_V, D_K), dtype=tl.float32)

        for t in range(L):
            q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_k * stride_qd
            k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_k * stride_kd
            v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_v * stride_vd

            q_t = tl.load(q_ptrs).to(tl.float32)
            k_t = tl.load(k_ptrs).to(tl.float32)
            v_t = tl.load(v_ptrs).to(tl.float32)

            alpha_val = tl.load(alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al).to(tl.float32)
            beta_val = tl.load(beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl).to(tl.float32)

            # 1. State decay in-register
            S = alpha_val * S

            # 2. Memory readout: retrieval prediction pred_v = S * k_t
            pred_v = tl.sum(S * k_t[None, :], axis=1)

            # 3. Plastic prediction error err_v = v_t - pred_v
            err_v = v_t - pred_v

            # 4. Plastic delta error-correction update directly inside SRAM register tiles
            S = S + beta_val * (err_v[:, None] * k_t[None, :])

            # 5. Associative query retrieval: o_t = S * q_t
            o_t = tl.sum(S * q_t[None, :], axis=1)

            o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ol + offs_v * stride_od
            tl.store(o_ptrs, o_t.to(q_ptr.dtype.element_ty))

        if SAVE_STATE:
            s_ptrs = (
                final_state_ptr
                + pid_b * stride_sb
                + pid_h * stride_sh
                + offs_v[:, None] * stride_sv
                + offs_k[None, :] * stride_sk
            )
            tl.store(s_ptrs, S.to(final_state_ptr.dtype.element_ty))

else:
    _delta_net_scan_fwd_kernel = None


def delta_net_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback reference implementation for DeltaNet recurrent scan."""
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]
    device = q.device
    dtype = q.dtype

    if beta is None:
        beta = torch.sigmoid(q.mean(dim=-1))
    if beta.dim() == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)

    if alpha is None:
        alpha = torch.ones((B, H, L), device=device, dtype=torch.float32)
    elif alpha.dim() == 4 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)

    if initial_state is not None:
        S = initial_state.clone().float()
    else:
        S = torch.zeros((B, H, D_V, D_K), device=device, dtype=torch.float32)

    outputs = []
    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()
        a_t = alpha[:, :, t].float()
        b_t = beta[:, :, t].float()

        S = a_t.unsqueeze(-1).unsqueeze(-1) * S

        pred_v = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)
        err_v = v_t - pred_v

        S = S + b_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err_v.unsqueeze(-1), k_t.unsqueeze(-2))

        o_t = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
        outputs.append(o_t)

    out = torch.stack(outputs, dim=2).to(dtype)
    if return_final_state:
        return out, S.to(dtype)
    return out


def delta_net_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated DeltaNet recurrent scan with fail-closed PyTorch fallback."""
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]

    if beta is None:
        beta = torch.sigmoid(q.mean(dim=-1))
    if beta.dim() == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)

    if alpha is None:
        alpha = torch.ones((B, H, L), device=q.device, dtype=torch.float32)
    elif alpha.dim() == 4 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)

    is_suitable, reason = check_triton_suitability(q, "delta_net_scan", head_dim=D_K, seq_len=L, allow_fp32=True)
    if (
        not is_suitable
        or not HAS_TRITON
        or initial_state is not None
        or D_K not in (16, 32, 64, 128)
        or D_V not in (16, 32, 64, 128)
    ):
        if not is_suitable:
            logger.warning("Triton kernel 'delta_net_scan' using PyTorch fallback: %s", reason)
        return delta_net_scan_pytorch(
            q,
            k,
            v,
            beta=beta,
            alpha=alpha,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    alpha_c = alpha.contiguous().to(torch.float32)
    beta_c = beta.contiguous().to(torch.float32)

    out = torch.empty((B, H, L, D_V), dtype=q.dtype, device=q.device)
    final_s = torch.empty((B, H, D_V, D_K), dtype=q.dtype, device=q.device) if return_final_state else out

    grid = (B, H)
    try:
        _delta_net_scan_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            alpha_c,
            beta_c,
            out,
            final_s,
            B,
            H,
            L,
            D_K=D_K,
            D_V=D_V,
            stride_qb=q_c.stride(0),
            stride_qh=q_c.stride(1),
            stride_ql=q_c.stride(2),
            stride_qd=q_c.stride(3),
            stride_kb=k_c.stride(0),
            stride_kh=k_c.stride(1),
            stride_kl=k_c.stride(2),
            stride_kd=k_c.stride(3),
            stride_vb=v_c.stride(0),
            stride_vh=v_c.stride(1),
            stride_vl=v_c.stride(2),
            stride_vd=v_c.stride(3),
            stride_ab=alpha_c.stride(0),
            stride_ah=alpha_c.stride(1),
            stride_al=alpha_c.stride(2),
            stride_bb=beta_c.stride(0),
            stride_bh=beta_c.stride(1),
            stride_bl=beta_c.stride(2),
            stride_ob=out.stride(0),
            stride_oh=out.stride(1),
            stride_ol=out.stride(2),
            stride_od=out.stride(3),
            stride_sb=final_s.stride(0),
            stride_sh=final_s.stride(1),
            stride_sv=final_s.stride(2),
            stride_sk=final_s.stride(3),
            SAVE_STATE=return_final_state,
        )
        if return_final_state:
            return out, final_s
        return out
    except Exception as exc:
        logger.warning("Triton kernel 'delta_net_scan' failed (%s); falling back to PyTorch reference.", exc)
        return delta_net_scan_pytorch(
            q,
            k,
            v,
            beta=beta,
            alpha=alpha,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )


delta_net_scan = delta_net_scan_triton

__all__ = [
    "delta_net_scan",
    "delta_net_scan_triton",
    "delta_net_scan_pytorch",
]
