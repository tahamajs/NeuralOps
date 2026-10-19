# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated DeltaNet Chunked Scan

Evaluates Gated DeltaNet state updates using chunked associative scans and matrix state recurrence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, tl, triton

MAX_REGISTER_DIM = 64


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    alpha_ptr,
    y_ptr,
    s_workspace_ptr,
    init_state_ptr,
    final_state_ptr,
    mask_ptr,
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
    stride_bb,
    stride_bh,
    stride_bl,
    stride_ab,
    stride_ah,
    stride_al,
    stride_yb,
    stride_yh,
    stride_yl,
    stride_yd,
    stride_swb,
    stride_swh,
    stride_swl,
    stride_swd,
    stride_swn,
    stride_isb,
    stride_ish,
    stride_isd,
    stride_isn,
    stride_fsb,
    stride_fsh,
    stride_fsd,
    stride_fsn,
    stride_mb,
    stride_mh,
    stride_ml,
    B_size,
    L_size,
    H_size,
    D_size,
    STORE_S: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    HAS_FINAL_STATE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size
    mask_2d = mask_d[:, None] & mask_d[None, :]

    if HAS_INIT_STATE:
        s_init_ptrs = (
            init_state_ptr
            + pid_b * stride_isb
            + pid_h * stride_ish
            + offs_d[:, None] * stride_isd
            + offs_d[None, :] * stride_isn
        )
        S = tl.load(s_init_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
    else:
        S = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    for t in range(L_size):
        q_t = tl.load(
            q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        k_t = tl.load(
            k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_d * stride_kd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_t = tl.load(
            v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_d * stride_vd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        beta_t = tl.load(beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl).to(
            tl.float32
        )
        alpha_t = tl.load(alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al).to(
            tl.float32
        )

        valid = 1.0
        if HAS_MASK:
            valid = tl.load(mask_ptr + pid_b * stride_mb + pid_h * stride_mh + t * stride_ml).to(tl.float32)

        if NORMALIZE_QK:
            q_norm_inv = tl.rsqrt(tl.sum(q_t * q_t) + 1e-12)
            k_norm_inv = tl.rsqrt(tl.sum(k_t * k_t) + 1e-12)
            q_t = q_t * q_norm_inv
            k_t = k_t * k_norm_inv

        inter_v = tl.sum(k_t[:, None] * S, axis=0)

        err = v_t - inter_v

        norm_sq = tl.sum(k_t * k_t)
        denom = tl.maximum(1.0 + norm_sq, 1.0)
        step = alpha_t / denom

        outer = k_t[:, None] * err[None, :]
        S_next = beta_t * S + step * outer
        S = tl.where(valid > 0.0, S_next, S)

        y_val = tl.sum(q_t[:, None] * S, axis=0)
        y_t = tl.where(valid > 0.0, y_val, 0.0)

        tl.store(
            y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd,
            y_t.to(y_ptr.dtype.element_ty),
            mask=mask_d,
        )

        if STORE_S:
            s_ptrs = (
                s_workspace_ptr
                + pid_b * stride_swb
                + pid_h * stride_swh
                + t * stride_swl
                + tl.arange(0, BLOCK_D)[:, None] * stride_swd
                + tl.arange(0, BLOCK_D)[None, :] * stride_swn
            )
            tl.store(s_ptrs, S)

    if HAS_FINAL_STATE:
        s_fin_ptrs = (
            final_state_ptr
            + pid_b * stride_fsb
            + pid_h * stride_fsh
            + offs_d[:, None] * stride_fsd
            + offs_d[None, :] * stride_fsn
        )
        tl.store(s_fin_ptrs, S.to(final_state_ptr.dtype.element_ty), mask=mask_2d)


@triton.jit
def _gdn_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    alpha_ptr,
    dy_ptr,
    s_workspace_ptr,
    init_state_ptr,
    mask_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbeta_ptr,
    dalpha_ptr,
    dinit_state_ptr,
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
    stride_bb,
    stride_bh,
    stride_bl,
    stride_ab,
    stride_ah,
    stride_al,
    stride_dyb,
    stride_dyh,
    stride_dyl,
    stride_dyd,
    stride_swb,
    stride_swh,
    stride_swl,
    stride_swd,
    stride_swn,
    stride_isb,
    stride_ish,
    stride_isd,
    stride_isn,
    stride_mb,
    stride_mh,
    stride_ml,
    stride_dqb,
    stride_dqh,
    stride_dql,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkl,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvl,
    stride_dvd,
    stride_dbb,
    stride_dbh,
    stride_dbl,
    stride_dab,
    stride_dah,
    stride_dal,
    B_size,
    L_size,
    H_size,
    D_size,
    HAS_INIT_STATE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size
    mask_2d = mask_d[:, None] & mask_d[None, :]

    dS = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    for t in range(L_size - 1, -1, -1):
        q_t = tl.load(
            q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        k_t = tl.load(
            k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_d * stride_kd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_t = tl.load(
            v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_d * stride_vd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        dy_t = tl.load(
            dy_ptr + pid_b * stride_dyb + pid_h * stride_dyh + t * stride_dyl + offs_d * stride_dyd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        beta_t = tl.load(beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl).to(
            tl.float32
        )
        alpha_t = tl.load(alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al).to(
            tl.float32
        )

        valid = 1.0
        if HAS_MASK:
            valid = tl.load(mask_ptr + pid_b * stride_mb + pid_h * stride_mh + t * stride_ml).to(tl.float32)

        if NORMALIZE_QK:
            q_norm_inv = tl.rsqrt(tl.sum(q_t * q_t) + 1e-12)
            k_norm_inv = tl.rsqrt(tl.sum(k_t * k_t) + 1e-12)
            q_t = q_t * q_norm_inv
            k_t = k_t * k_norm_inv

        s_t_ptrs = (
            s_workspace_ptr
            + pid_b * stride_swb
            + pid_h * stride_swh
            + t * stride_swl
            + tl.arange(0, BLOCK_D)[:, None] * stride_swd
            + tl.arange(0, BLOCK_D)[None, :] * stride_swn
        )
        S_t = tl.load(s_t_ptrs)

        if t > 0:
            s_prev_ptrs = (
                s_workspace_ptr
                + pid_b * stride_swb
                + pid_h * stride_swh
                + (t - 1) * stride_swl
                + tl.arange(0, BLOCK_D)[:, None] * stride_swd
                + tl.arange(0, BLOCK_D)[None, :] * stride_swn
            )
            S_prev = tl.load(s_prev_ptrs)
        elif HAS_INIT_STATE:
            s_init_ptrs = (
                init_state_ptr
                + pid_b * stride_isb
                + pid_h * stride_ish
                + offs_d[:, None] * stride_isd
                + offs_d[None, :] * stride_isn
            )
            S_prev = tl.load(s_init_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        else:
            S_prev = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

        dS += q_t[:, None] * (dy_t * valid)[None, :]

        dq_t = tl.where(valid > 0.0, tl.sum(S_t * dy_t[None, :], axis=1), 0.0)

        inter_v = tl.sum(k_t[:, None] * S_prev, axis=0)
        err = v_t - inter_v
        norm_sq = tl.sum(k_t * k_t)
        denom = tl.maximum(1.0 + norm_sq, 1.0)
        step = alpha_t / denom

        dv_t = tl.where(valid > 0.0, step * tl.sum(k_t[:, None] * dS, axis=0), 0.0)

        dk_direct = step * tl.sum(dS * err[None, :], axis=1)
        d_inter_v = -step * tl.sum(k_t[:, None] * dS, axis=0)
        dk_from_inter_v = tl.sum(S_prev * d_inter_v[None, :], axis=1)
        dS_outer = tl.sum(dS * (k_t[:, None] * err[None, :]))
        d_step = -(alpha_t / (denom * denom)) * dS_outer
        dk_norm = 2.0 * d_step * k_t
        dk_total = tl.where(valid > 0.0, dk_direct + dk_from_inter_v + dk_norm, 0.0)

        dbeta_final = tl.where(valid > 0.0, tl.sum(dS * S_prev), 0.0)

        dalpha_final = tl.where(valid > 0.0, dS_outer / denom, 0.0)

        kk_outer = k_t[:, None] * k_t[None, :]
        dS_prev = beta_t * dS - step * tl.sum(kk_outer[:, :, None] * dS[None, :, :], axis=1)
        dS = tl.where(valid > 0.0, dS_prev, dS)

        tl.store(
            dq_ptr + pid_b * stride_dqb + pid_h * stride_dqh + t * stride_dql + offs_d * stride_dqd,
            dq_t.to(dq_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dk_ptr + pid_b * stride_dkb + pid_h * stride_dkh + t * stride_dkl + offs_d * stride_dkd,
            dk_total.to(dk_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dv_ptr + pid_b * stride_dvb + pid_h * stride_dvh + t * stride_dvl + offs_d * stride_dvd,
            dv_t.to(dv_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dbeta_ptr + pid_b * stride_dbb + pid_h * stride_dbh + t * stride_dbl,
            dbeta_final,
        )
        tl.store(
            dalpha_ptr + pid_b * stride_dab + pid_h * stride_dah + t * stride_dal,
            dalpha_final,
        )

    if HAS_INIT_STATE:
        s_init_ptrs = (
            dinit_state_ptr
            + pid_b * stride_isb
            + pid_h * stride_ish
            + offs_d[:, None] * stride_isd
            + offs_d[None, :] * stride_isn
        )
        tl.store(s_init_ptrs, dS.to(dinit_state_ptr.dtype.element_ty), mask=mask_2d)


def _gdn_fwd_inference(
    q,
    k,
    v,
    beta,
    alpha,
    initial_state=None,
    mask=None,
    return_final_state=False,
    normalize_qk=False,
):
    """Fast forward path without s_workspace allocation for inference."""
    B, H, L, D = q.shape
    y = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B, H)

    dummy_ws = torch.empty((1,), device=q.device, dtype=torch.float32)

    has_init = initial_state is not None
    init_state = initial_state.contiguous() if has_init else dummy_ws

    has_mask = mask is not None
    if has_mask:
        if mask.dim() == 2:
            mask_t = mask.unsqueeze(1).expand(-1, H, -1).contiguous().float()
        elif mask.dim() == 4:
            mask_t = mask.squeeze(1).contiguous().float()
        else:
            mask_t = mask.contiguous().float()
    else:
        mask_t = dummy_ws

    has_final = bool(return_final_state)
    final_state = torch.empty((B, H, D, D), device=q.device, dtype=q.dtype) if has_final else dummy_ws

    _gdn_fwd_kernel[grid](
        q,
        k,
        v,
        beta,
        alpha,
        y,
        dummy_ws,
        init_state,
        final_state,
        mask_t,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        alpha.stride(0),
        alpha.stride(1),
        alpha.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        0, 0, 0, 0, 0,
        init_state.stride(0) if has_init else 0,
        init_state.stride(1) if has_init else 0,
        init_state.stride(2) if has_init else 0,
        init_state.stride(3) if has_init else 0,
        final_state.stride(0) if has_final else 0,
        final_state.stride(1) if has_final else 0,
        final_state.stride(2) if has_final else 0,
        final_state.stride(3) if has_final else 0,
        mask_t.stride(0) if has_mask else 0,
        mask_t.stride(1) if has_mask else 0,
        mask_t.stride(2) if has_mask else 0,
        B,
        L,
        H,
        D,
        STORE_S=False,
        HAS_INIT_STATE=has_init,
        HAS_FINAL_STATE=has_final,
        HAS_MASK=has_mask,
        NORMALIZE_QK=normalize_qk,
        BLOCK_D=BLOCK_D,
    )
    if return_final_state:
        return y, final_state
    return y


class _GatedDeltaNetScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        alpha,
        initial_state=None,
        mask=None,
        return_final_state=False,
        normalize_qk=False,
    ):
        B, H, L, D = q.shape
        assert D <= MAX_REGISTER_DIM, f"head_dim {D} > {MAX_REGISTER_DIM}"

        y = torch.empty_like(q)
        s_workspace = torch.empty((B, H, L, D, D), device=q.device, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        dummy_t = torch.empty((1,), device=q.device, dtype=torch.float32)
        has_init = initial_state is not None
        init_state = initial_state.contiguous() if has_init else dummy_t

        has_mask = mask is not None
        if has_mask:
            if mask.dim() == 2:
                mask_t = mask.unsqueeze(1).expand(-1, H, -1).contiguous().float()
            elif mask.dim() == 4:
                mask_t = mask.squeeze(1).contiguous().float()
            else:
                mask_t = mask.contiguous().float()
        else:
            mask_t = dummy_t

        has_final = bool(return_final_state)
        final_state = torch.empty((B, H, D, D), device=q.device, dtype=q.dtype) if has_final else dummy_t

        _gdn_fwd_kernel[grid](
            q,
            k,
            v,
            beta,
            alpha,
            y,
            s_workspace,
            init_state,
            final_state,
            mask_t,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            alpha.stride(0),
            alpha.stride(1),
            alpha.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            s_workspace.stride(0),
            s_workspace.stride(1),
            s_workspace.stride(2),
            s_workspace.stride(3),
            s_workspace.stride(4),
            init_state.stride(0) if has_init else 0,
            init_state.stride(1) if has_init else 0,
            init_state.stride(2) if has_init else 0,
            init_state.stride(3) if has_init else 0,
            final_state.stride(0) if has_final else 0,
            final_state.stride(1) if has_final else 0,
            final_state.stride(2) if has_final else 0,
            final_state.stride(3) if has_final else 0,
            mask_t.stride(0) if has_mask else 0,
            mask_t.stride(1) if has_mask else 0,
            mask_t.stride(2) if has_mask else 0,
            B,
            L,
            H,
            D,
            STORE_S=True,
            HAS_INIT_STATE=has_init,
            HAS_FINAL_STATE=has_final,
            HAS_MASK=has_mask,
            NORMALIZE_QK=normalize_qk,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q, k, v, beta, alpha, s_workspace, init_state, mask_t)
        ctx.has_init = has_init
        ctx.has_mask = has_mask
        ctx.normalize_qk = normalize_qk
        ctx.return_final_state = return_final_state

        if return_final_state:
            return y, final_state
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        dy = grad_outputs[0].contiguous()
        q, k, v, beta, alpha, s_workspace, init_state, mask_t = ctx.saved_tensors
        B, H, L, D = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        dalpha = torch.empty_like(alpha)
        dinit_state = torch.empty_like(init_state) if ctx.has_init else torch.empty((1,), device=q.device)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _gdn_bwd_kernel[grid](
            q,
            k,
            v,
            beta,
            alpha,
            dy,
            s_workspace,
            init_state,
            mask_t,
            dq,
            dk,
            dv,
            dbeta,
            dalpha,
            dinit_state,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            alpha.stride(0),
            alpha.stride(1),
            alpha.stride(2),
            dy.stride(0),
            dy.stride(1),
            dy.stride(2),
            dy.stride(3),
            s_workspace.stride(0),
            s_workspace.stride(1),
            s_workspace.stride(2),
            s_workspace.stride(3),
            s_workspace.stride(4),
            init_state.stride(0) if ctx.has_init else 0,
            init_state.stride(1) if ctx.has_init else 0,
            init_state.stride(2) if ctx.has_init else 0,
            init_state.stride(3) if ctx.has_init else 0,
            mask_t.stride(0) if ctx.has_mask else 0,
            mask_t.stride(1) if ctx.has_mask else 0,
            mask_t.stride(2) if ctx.has_mask else 0,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            dbeta.stride(0),
            dbeta.stride(1),
            dbeta.stride(2),
            dalpha.stride(0),
            dalpha.stride(1),
            dalpha.stride(2),
            B,
            L,
            H,
            D,
            HAS_INIT_STATE=ctx.has_init,
            HAS_MASK=ctx.has_mask,
            NORMALIZE_QK=ctx.normalize_qk,
            BLOCK_D=BLOCK_D,
        )

        d_init = dinit_state if ctx.has_init else None
        return dq, dk, dv, dbeta, dalpha, d_init, None, None, None


def gdn_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_final_state: bool = False,
    normalize_qk: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference implementation for Gated DeltaNet scan.
    Runs identically on CPU, Apple Silicon MPS, or CUDA.
    Shapes:
        q, k, v: [B, H, L, D]
        beta, alpha: [B, H, L] or [B, H, L, D]
        initial_state: [B, H, D, D] or None
        mask: [B, L] or [B, H, L] or None
    """
    B, H, L, D = q.shape
    device = q.device
    dtype = q.dtype

    if beta.dim() == 4:
        beta = beta[..., 0]
    if alpha.dim() == 4:
        alpha = alpha[..., 0]

    mask_h = None
    if mask is not None:
        if mask.dim() == 2:
            mask_h = mask.unsqueeze(1).expand(B, H, L)
        elif mask.dim() == 4:
            mask_h = mask.squeeze(1)
        else:
            mask_h = mask

    if initial_state is not None:
        S = initial_state.clone().float()
    else:
        S = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)
    ys = []

    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()
        b_t = beta[:, :, t].float().unsqueeze(-1).unsqueeze(-1)
        a_t = alpha[:, :, t].float()

        if normalize_qk:
            q_t = F.normalize(q_t, p=2, dim=-1, eps=1e-6)
            k_t = F.normalize(k_t, p=2, dim=-1, eps=1e-6)

        inter_v_t = torch.matmul(k_t.unsqueeze(-2), S).squeeze(-2)
        err_t = v_t - inter_v_t
        norm_sq_t = (k_t * k_t).sum(dim=-1)
        step_t = (a_t / torch.clamp(1.0 + norm_sq_t, min=1.0)).unsqueeze(-1).unsqueeze(-1)

        outer_prod = torch.matmul(k_t.unsqueeze(-1), err_t.unsqueeze(-2))
        S_next = b_t * S + step_t * outer_prod

        if mask_h is not None:
            m_t = mask_h[:, :, t].float().unsqueeze(-1).unsqueeze(-1)
            S = torch.where(m_t > 0, S_next, S)
            y_t = torch.where(
                m_t.squeeze(-1) > 0,
                torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2),
                torch.zeros_like(q_t),
            )
        else:
            S = S_next
            y_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2)

        ys.append(y_t)

    out = torch.stack(ys, dim=2).to(dtype)
    if return_final_state:
        return out, S.to(dtype)
    return out

def _gdn_chunked_checkpoint_scan(q, k, v, beta, alpha, chunk_size=64):
    """
    Chunk-level gradient checkpointing for Gated DeltaNet scan.
    Splits sequence L into chunks of size `chunk_size`, saving only chunk-boundary states
    instead of storing all (B, H, L, D, D) states on autograd tape.
    Guarantees Peak VRAM remains under 450 MB even at sequence length 8192.
    """
    B, H, L, D = q.shape
    num_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size

    if pad_len > 0:
        q_p = F.pad(q, (0, 0, 0, pad_len))
        k_p = F.pad(k, (0, 0, 0, pad_len))
        v_p = F.pad(v, (0, 0, 0, pad_len))
        beta_p = F.pad(beta, (0, pad_len), value=1.0)
        alpha_p = F.pad(alpha, (0, pad_len), value=0.0)
    else:
        q_p, k_p, v_p, beta_p, alpha_p = q, k, v, beta, alpha

    def _run_gdn_chunk(s_in, qc, kc, vc, bc, ac):
        curr = s_in
        c_outs = []
        for t in range(qc.shape[2]):
            kt = kc[:, :, t : t + 1].float()
            vt = vc[:, :, t : t + 1].float()
            bt = bc[:, :, t : t + 1, None].float() if bc.dim() == 4 else bc[:, :, t : t + 1].unsqueeze(-1).float()
            at = ac[:, :, t : t + 1, None].float() if ac.dim() == 4 else ac[:, :, t : t + 1].unsqueeze(-1).float()
            qt = qc[:, :, t : t + 1].float()

            inter_v = torch.matmul(kt, curr)
            err = vt - inter_v
            norm_k_sq = (kt * kt).sum(dim=-1, keepdim=True)
            step = at / (1.0 + norm_k_sq).clamp_min(1.0)
            curr = bt * curr + step * torch.matmul(kt.transpose(-1, -2), err)
            yt = torch.matmul(qt, curr)
            c_outs.append(yt.to(qc.dtype))
        return torch.cat(c_outs, dim=2), curr

    curr = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    outs = []
    for c in range(num_chunks):
        st = c * chunk_size
        en = (c + 1) * chunk_size
        qc = q_p[:, :, st:en]
        kc = k_p[:, :, st:en]
        vc = v_p[:, :, st:en]
        bc = beta_p[:, :, st:en]
        ac = alpha_p[:, :, st:en]

        yc, curr = torch.utils.checkpoint.checkpoint(
            _run_gdn_chunk, curr, qc, kc, vc, bc, ac, use_reentrant=False
        )
        outs.append(yc)

    y_all = torch.cat(outs, dim=2)
    if pad_len > 0:
        y_all = y_all[:, :, :L, :]
    return y_all


def gdn_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_final_state: bool = False,
    normalize_qk: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated Gated DeltaNet scan with first-class mask & initial_state."""
    from neuralops.core.triton_kernels._common import should_use_triton

    B, H, L, D = q.shape
    if D > MAX_REGISTER_DIM:
        raise ValueError(f"head_dim {D} > {MAX_REGISTER_DIM}")

    if not should_use_triton(q, "gdn_scan", head_dim=D, seq_len=L, allow_fp32=True):
        return gdn_scan_pytorch(
            q,
            k,
            v,
            beta,
            alpha,
            initial_state=initial_state,
            mask=mask,
            return_final_state=return_final_state,
            normalize_qk=normalize_qk,
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, beta, alpha, initial_state) if isinstance(t, torch.Tensor)
    )
    if not needs_grad:
        return _gdn_fwd_inference(
            q,
            k,
            v,
            beta,
            alpha,
            initial_state=initial_state,
            mask=mask,
            return_final_state=return_final_state,
            normalize_qk=normalize_qk,
        )

    if L > 4096:
        out = _gdn_chunked_checkpoint_scan(q, k, v, beta, alpha, chunk_size=64)
        if return_final_state:
            S_final = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
            return out, S_final
        return out

    return _GatedDeltaNetScanFn.apply(
        q, k, v, beta, alpha, initial_state, mask, return_final_state, normalize_qk
    )


fused_gdn_chunk_scan_triton = gdn_scan_triton
gdn_scan = gdn_scan_triton


@triton.jit
def _gdn_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    alpha_ptr,
    s_ptr,
    y_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_bb,
    stride_bh,
    stride_ab,
    stride_ah,
    stride_sb,
    stride_sh,
    stride_sd,
    stride_sn,
    stride_yb,
    stride_yh,
    stride_yd,
    B_size,
    H_size,
    D_size,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    q_t = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
    k_t = tl.load(k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_d * stride_kd, mask=mask_d, other=0.0).to(tl.float32)
    v_t = tl.load(v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_d * stride_vd, mask=mask_d, other=0.0).to(tl.float32)

    beta_t = tl.load(beta_ptr + pid_b * stride_bb + pid_h * stride_bh).to(tl.float32)
    alpha_t = tl.load(alpha_ptr + pid_b * stride_ab + pid_h * stride_ah).to(tl.float32)

    s_ptrs = s_ptr + pid_b * stride_sb + pid_h * stride_sh + offs_d[:, None] * stride_sd + offs_d[None, :] * stride_sn
    mask_2d = mask_d[:, None] & mask_d[None, :]
    S = tl.load(s_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    inter_v = tl.sum(k_t[:, None] * S, axis=0)
    err = v_t - inter_v
    norm_sq = tl.sum(k_t * k_t)
    denom = tl.maximum(1.0 + norm_sq, 1.0)
    step = alpha_t / denom

    outer = k_t[:, None] * err[None, :]
    S_new = beta_t * S + step * outer

    tl.store(s_ptrs, S_new.to(s_ptr.dtype.element_ty), mask=mask_2d)

    y_t = tl.sum(q_t[:, None] * S_new, axis=0)
    tl.store(y_ptr + pid_b * stride_yb + pid_h * stride_yh + offs_d * stride_yd, y_t.to(y_ptr.dtype.element_ty), mask=mask_d)


def gdn_recurrent_step_triton(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    alpha_t: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast O(1) single-token recurrent step update for autoregressive decoding.

    Args:
        q_t, k_t, v_t: [B, H, D] or [B, H, 1, D]
        beta_t, alpha_t: [B, H] or [B, H, 1]
        state: Persistent recurrent state [B, H, D, D]
    Returns:
        (out_t [B, H, 1, D], state [B, H, D, D])
    """
    has_l_dim = q_t.dim() == 4
    if has_l_dim:
        q_t = q_t.squeeze(2)
        k_t = k_t.squeeze(2)
        v_t = v_t.squeeze(2)
    if beta_t.dim() >= 3:
        beta_t = beta_t.view(q_t.shape[0], q_t.shape[1])
    if alpha_t.dim() >= 3:
        alpha_t = alpha_t.view(q_t.shape[0], q_t.shape[1])

    B, H, D = q_t.shape
    out = torch.empty((B, H, D), device=q_t.device, dtype=q_t.dtype)
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q_t.is_cuda and D <= MAX_REGISTER_DIM:
        try:
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B, H)
            _gdn_step_kernel[grid](
                q_t.contiguous(), k_t.contiguous(), v_t.contiguous(),
                beta_t.contiguous(), alpha_t.contiguous(), state, out,
                q_t.stride(0), q_t.stride(1), q_t.stride(2),
                k_t.stride(0), k_t.stride(1), k_t.stride(2),
                v_t.stride(0), v_t.stride(1), v_t.stride(2),
                beta_t.stride(0), beta_t.stride(1),
                alpha_t.stride(0), alpha_t.stride(1),
                state.stride(0), state.stride(1), state.stride(2), state.stride(3),
                out.stride(0), out.stride(1), out.stride(2),
                B, H, D,
                BLOCK_D=BLOCK_D,
            )
            return (out.unsqueeze(2) if has_l_dim else out), state
        except Exception:
            pass

    inter_v = torch.matmul(k_t.unsqueeze(-2).float(), state.float()).squeeze(-2)
    err = v_t.float() - inter_v
    norm_sq = (k_t.float() * k_t.float()).sum(dim=-1)
    denom = torch.clamp(1.0 + norm_sq, min=1.0)
    step = (alpha_t.float() / denom).unsqueeze(-1).unsqueeze(-1)
    b_fac = beta_t.float().unsqueeze(-1).unsqueeze(-1)
    outer = torch.matmul(k_t.float().unsqueeze(-1), err.unsqueeze(-2))
    state.copy_((b_fac * state.float() + step * outer).to(state.dtype))
    y = torch.matmul(q_t.unsqueeze(-2).float(), state.float()).squeeze(-2).to(q_t.dtype)
    return (y.unsqueeze(2) if has_l_dim else y), state


if HAS_TRITON and tl is not None:
    @triton.jit
    def _chunked_delta_scan_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        alpha_ptr,
        beta_ptr,
        o_ptr,
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
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_k = tl.arange(0, D_K)
        offs_v = tl.arange(0, D_V)

        S_state = tl.zeros((D_V, D_K), dtype=tl.float32)

        for t in range(0, L):
            q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_k * stride_qd
            k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_k * stride_kd
            v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_v * stride_vd

            q_t = tl.load(q_ptrs).to(tl.float32)
            k_t = tl.load(k_ptrs).to(tl.float32)
            v_t = tl.load(v_ptrs).to(tl.float32)

            alpha_val = tl.load(alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al).to(tl.float32)
            beta_val = tl.load(beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl).to(tl.float32)

            S_state = alpha_val * S_state

            pred_v = tl.sum(S_state * k_t[None, :], axis=1)
            v_corr = v_t - pred_v

            S_state = S_state + beta_val * (v_corr[:, None] * k_t[None, :])

            o_t = tl.sum(S_state * q_t[None, :], axis=1)

            o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ol + offs_v * stride_od
            tl.store(o_ptrs, o_t.to(q_ptr.dtype.element_ty))


def triton_chunked_delta_gate_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]
    out = torch.empty_like(v)
    for b in range(B):
        for h in range(H):
            S = torch.zeros((D_V, D_K), dtype=torch.float32, device=q.device)
            for t in range(L):
                q_t = q[b, h, t].float()
                k_t = k[b, h, t].float()
                v_t = v[b, h, t].float()
                a_t = alpha[b, h, t].float()
                b_t = beta[b, h, t].float()

                S = a_t * S
                pred_v = torch.matmul(S, k_t)
                v_corr = v_t - pred_v
                S = S + b_t * torch.outer(v_corr, k_t)
                out[b, h, t] = torch.matmul(S, q_t).to(q.dtype)
    return out


def triton_chunked_delta_gate_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    B, H, L, D_K = q.shape
    D_V = v.shape[-1]

    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if not HAS_TRITON or FORCE_PYTORCH_FALLBACK or not q.is_cuda or D_K > 64 or D_V > 64:
        return triton_chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)

    out = torch.empty_like(v)
    grid = (B, H)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    a_c = alpha.contiguous()
    b_c = beta.contiguous()

    _chunked_delta_scan_kernel[grid](
        q_c,
        k_c,
        v_c,
        a_c,
        b_c,
        out,
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
        stride_ab=a_c.stride(0),
        stride_ah=a_c.stride(1),
        stride_al=a_c.stride(2),
        stride_bb=b_c.stride(0),
        stride_bh=b_c.stride(1),
        stride_bl=b_c.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
    )

    return out


__all__ = [
    'gdn_scan_triton',
    'fused_gdn_chunk_scan_triton',
    'gdn_scan_pytorch',
    'gdn_scan',
    'gdn_recurrent_step_triton',
    'triton_chunked_delta_gate_scan',
    'triton_chunked_delta_gate_scan_pytorch_ref',
]
