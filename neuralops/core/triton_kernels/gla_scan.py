# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated Linear Attention Recurrent Scan

Computes gated linear attention recurrence with token-level data-dependent decay gates.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


from neuralops.core.triton_kernels._common import tl, triton

MAX_REGISTER_DIM = 64


@triton.jit
def _gla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lam_ptr,
    y_ptr,
    s_workspace_ptr,
    z_workspace_ptr,
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
    stride_lamb,
    stride_lamh,
    stride_laml,
    stride_yb,
    stride_yh,
    stride_yl,
    stride_yd,
    stride_swb,
    stride_swh,
    stride_swl,
    stride_swd,
    stride_swn,
    stride_zwb,
    stride_zwh,
    stride_zwl,
    stride_zwd,
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
    STORE_WORKSPACE: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    HAS_FINAL_STATE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_NORMALIZER: tl.constexpr,
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

    Z = tl.zeros((BLOCK_D,), dtype=tl.float32)

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
        lam_val = tl.load(lam_ptr + pid_b * stride_lamb + pid_h * stride_lamh + t * stride_laml).to(
            tl.float32
        )
        lam_t = tl.maximum(tl.minimum(lam_val, 1.0 - 1e-4), 1e-4)

        valid = 1.0
        if HAS_MASK:
            valid = tl.load(mask_ptr + pid_b * stride_mb + pid_h * stride_mh + t * stride_ml).to(tl.float32)

        kv_outer = k_t[:, None] * v_t[None, :]
        S_next = lam_t * S + kv_outer
        S = tl.where(valid > 0.0, S_next, S)

        num_t = tl.sum(q_t[:, None] * S, axis=0)

        if USE_NORMALIZER:
            Z_next = lam_t * Z + k_t
            Z = tl.where(valid > 0.0, Z_next, Z)
            denom_raw = tl.sum(q_t * Z) + 1e-4
            denom_t = tl.maximum(denom_raw, 1e-4)
            y_val = num_t / denom_t
        else:
            y_val = num_t

        y_t = tl.where(valid > 0.0, y_val, 0.0)

        tl.store(
            y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd,
            y_t.to(y_ptr.dtype.element_ty),
            mask=mask_d,
        )

        if STORE_WORKSPACE:
            s_ptrs = (
                s_workspace_ptr
                + pid_b * stride_swb
                + pid_h * stride_swh
                + t * stride_swl
                + tl.arange(0, BLOCK_D)[:, None] * stride_swd
                + tl.arange(0, BLOCK_D)[None, :] * stride_swn
            )
            tl.store(s_ptrs, S)

            if USE_NORMALIZER:
                z_ptrs = (
                    z_workspace_ptr
                    + pid_b * stride_zwb
                    + pid_h * stride_zwh
                    + t * stride_zwl
                    + offs_d * stride_zwd
                )
                tl.store(z_ptrs, Z, mask=mask_d)

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
def _gla_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lam_ptr,
    dy_ptr,
    s_workspace_ptr,
    z_workspace_ptr,
    init_state_ptr,
    mask_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dlam_ptr,
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
    stride_lamb,
    stride_lamh,
    stride_laml,
    stride_dyb,
    stride_dyh,
    stride_dyl,
    stride_dyd,
    stride_swb,
    stride_swh,
    stride_swl,
    stride_swd,
    stride_swn,
    stride_zwb,
    stride_zwh,
    stride_zwl,
    stride_zwd,
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
    stride_dlb,
    stride_dlh,
    stride_dll,
    B_size,
    L_size,
    H_size,
    D_size,
    HAS_INIT_STATE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_NORMALIZER: tl.constexpr,
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
    dZ = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for step_rev in range(L_size):
        t = L_size - 1 - step_rev

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
        lam_val = tl.load(lam_ptr + pid_b * stride_lamb + pid_h * stride_lamh + t * stride_laml).to(
            tl.float32
        )
        lam_t = tl.maximum(tl.minimum(lam_val, 1.0 - 1e-4), 1e-4)

        dy_t = tl.load(
            dy_ptr + pid_b * stride_dyb + pid_h * stride_dyh + t * stride_dyl + offs_d * stride_dyd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        valid = 1.0
        if HAS_MASK:
            valid = tl.load(mask_ptr + pid_b * stride_mb + pid_h * stride_mh + t * stride_ml).to(tl.float32)

        s_t_ptrs = (
            s_workspace_ptr
            + pid_b * stride_swb
            + pid_h * stride_swh
            + t * stride_swl
            + tl.arange(0, BLOCK_D)[:, None] * stride_swd
            + tl.arange(0, BLOCK_D)[None, :] * stride_swn
        )
        S_t = tl.load(s_t_ptrs)

        if USE_NORMALIZER:
            z_t_ptrs = (
                z_workspace_ptr
                + pid_b * stride_zwb
                + pid_h * stride_zwh
                + t * stride_zwl
                + offs_d * stride_zwd
            )
            Z_t = tl.load(z_t_ptrs, mask=mask_d, other=0.0)

            num_t = tl.sum(q_t[:, None] * S_t, axis=0)
            denom_raw = tl.sum(q_t * Z_t) + 1e-4
            denom_t = tl.maximum(denom_raw, 1e-4)

            dnum_t = dy_t / denom_t
            ddenom_t = tl.where(denom_raw > 1e-4, -tl.sum(dy_t * num_t) / (denom_t * denom_t), 0.0)

            dZ += q_t * (ddenom_t * valid)
            dq_t = tl.where(valid > 0.0, tl.sum(S_t * dnum_t[None, :], axis=1) + Z_t * ddenom_t, 0.0)
            dk_t = tl.where(valid > 0.0, tl.sum(dS * v_t[None, :], axis=1) + dZ, 0.0)
        else:
            dnum_t = dy_t
            dq_t = tl.where(valid > 0.0, tl.sum(S_t * dnum_t[None, :], axis=1), 0.0)
            dk_t = tl.where(valid > 0.0, tl.sum(dS * v_t[None, :], axis=1), 0.0)

        dS += q_t[:, None] * (dnum_t * valid)[None, :]
        dv_t = tl.where(valid > 0.0, tl.sum(k_t[:, None] * dS, axis=0), 0.0)

        dlam_t = 0.0
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

            if USE_NORMALIZER:
                z_prev_ptrs = (
                    z_workspace_ptr
                    + pid_b * stride_zwb
                    + pid_h * stride_zwh
                    + (t - 1) * stride_zwl
                    + offs_d * stride_zwd
                )
                Z_prev = tl.load(z_prev_ptrs, mask=mask_d, other=0.0)
                dlam_t = tl.sum(dS * S_prev) + tl.sum(dZ * Z_prev)
            else:
                dlam_t = tl.sum(dS * S_prev)
        elif HAS_INIT_STATE:
            s_init_ptrs = (
                init_state_ptr
                + pid_b * stride_isb
                + pid_h * stride_ish
                + offs_d[:, None] * stride_isd
                + offs_d[None, :] * stride_isn
            )
            S_prev = tl.load(s_init_ptrs, mask=mask_2d, other=0.0).to(tl.float32)
            dlam_t = tl.sum(dS * S_prev)

        dlam_t = tl.where(valid > 0.0, dlam_t, 0.0)

        tl.store(
            dq_ptr + pid_b * stride_dqb + pid_h * stride_dqh + t * stride_dql + offs_d * stride_dqd,
            dq_t.to(dq_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dk_ptr + pid_b * stride_dkb + pid_h * stride_dkh + t * stride_dkl + offs_d * stride_dkd,
            dk_t.to(dk_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dv_ptr + pid_b * stride_dvb + pid_h * stride_dvh + t * stride_dvl + offs_d * stride_dvd,
            dv_t.to(dv_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dlam_ptr + pid_b * stride_dlb + pid_h * stride_dlh + t * stride_dll,
            dlam_t,
        )

        dS = tl.where(valid > 0.0, lam_t * dS, dS)
        if USE_NORMALIZER:
            dZ = tl.where(valid > 0.0, lam_t * dZ, dZ)

    if HAS_INIT_STATE:
        dis_ptrs = (
            dinit_state_ptr
            + pid_b * stride_isb
            + pid_h * stride_ish
            + offs_d[:, None] * stride_isd
            + offs_d[None, :] * stride_isn
        )
        tl.store(dis_ptrs, dS.to(dinit_state_ptr.dtype.element_ty), mask=mask_2d)


class _GLAScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        lambdas,
        initial_state=None,
        mask=None,
        return_final_state=False,
        use_normalizer=True,
    ):
        B, H, L, D = q.shape
        assert D <= MAX_REGISTER_DIM, f"GLA head_dim {D} > {MAX_REGISTER_DIM}"

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        lam_c = lambdas.contiguous()

        y = torch.empty_like(q_c)
        s_workspace = torch.empty((B, H, L, D, D), device=q.device, dtype=torch.float32)
        z_workspace = (
            torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
            if use_normalizer
            else torch.empty((1,), device=q.device, dtype=torch.float32)
        )

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

        _gla_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            lam_c,
            y,
            s_workspace,
            z_workspace,
            init_state,
            final_state,
            mask_t,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            k_c.stride(3),
            v_c.stride(0),
            v_c.stride(1),
            v_c.stride(2),
            v_c.stride(3),
            lam_c.stride(0),
            lam_c.stride(1),
            lam_c.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            s_workspace.stride(0),
            s_workspace.stride(1),
            s_workspace.stride(2),
            s_workspace.stride(3),
            s_workspace.stride(4),
            z_workspace.stride(0) if use_normalizer else 0,
            z_workspace.stride(1) if use_normalizer else 0,
            z_workspace.stride(2) if use_normalizer else 0,
            z_workspace.stride(3) if use_normalizer else 0,
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
            STORE_WORKSPACE=True,
            HAS_INIT_STATE=has_init,
            HAS_FINAL_STATE=has_final,
            HAS_MASK=has_mask,
            USE_NORMALIZER=use_normalizer,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q_c, k_c, v_c, lam_c, s_workspace, z_workspace, init_state, mask_t)
        ctx.has_init = has_init
        ctx.has_mask = has_mask
        ctx.use_normalizer = use_normalizer
        ctx.return_final_state = return_final_state

        if return_final_state:
            return y, final_state
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        dy = grad_outputs[0].contiguous()
        q, k, v, lambdas, s_workspace, z_workspace, init_state, mask_t = ctx.saved_tensors
        B, H, L, D = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dlambdas = torch.empty_like(lambdas)
        dinit_state = torch.empty_like(init_state) if ctx.has_init else torch.empty((1,), device=q.device)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _gla_bwd_kernel[grid](
            q,
            k,
            v,
            lambdas,
            dy,
            s_workspace,
            z_workspace,
            init_state,
            mask_t,
            dq,
            dk,
            dv,
            dlambdas,
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
            lambdas.stride(0),
            lambdas.stride(1),
            lambdas.stride(2),
            dy.stride(0),
            dy.stride(1),
            dy.stride(2),
            dy.stride(3),
            s_workspace.stride(0),
            s_workspace.stride(1),
            s_workspace.stride(2),
            s_workspace.stride(3),
            s_workspace.stride(4),
            z_workspace.stride(0) if ctx.use_normalizer else 0,
            z_workspace.stride(1) if ctx.use_normalizer else 0,
            z_workspace.stride(2) if ctx.use_normalizer else 0,
            z_workspace.stride(3) if ctx.use_normalizer else 0,
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
            dlambdas.stride(0),
            dlambdas.stride(1),
            dlambdas.stride(2),
            B,
            L,
            H,
            D,
            HAS_INIT_STATE=ctx.has_init,
            HAS_MASK=ctx.has_mask,
            USE_NORMALIZER=ctx.use_normalizer,
            BLOCK_D=BLOCK_D,
        )

        d_init = dinit_state if ctx.has_init else None
        return dq, dk, dv, dlambdas, d_init, None, None, None


def _gla_fwd_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambdas: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_final_state: bool = False,
    use_normalizer: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Fast forward recurrent inference path without full workspace memory allocation."""
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

    _gla_fwd_kernel[grid](
        q,
        k,
        v,
        lambdas,
        y,
        dummy_ws,
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
        lambdas.stride(0),
        lambdas.stride(1),
        lambdas.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        0, 0, 0, 0, 0,
        0, 0, 0, 0,
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
        STORE_WORKSPACE=False,
        HAS_INIT_STATE=has_init,
        HAS_FINAL_STATE=has_final,
        HAS_MASK=has_mask,
        USE_NORMALIZER=use_normalizer,
        BLOCK_D=BLOCK_D,
    )
    if return_final_state:
        return y, final_state
    return y


def gla_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    lambdas: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_final_state: bool = False,
    use_normalizer: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback reference implementation for GLA recurrent scan."""
    if k is None:
        k = q
    if v is None:
        v = q
    if lambdas is None:
        lambdas = torch.sigmoid(q.mean(dim=-1)).clamp(1e-4, 1.0 - 1e-4)

    B, H, L, D = q.shape
    device = q.device
    dtype = q.dtype

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

    Z = torch.zeros((B, H, D), device=device, dtype=torch.float32)
    ys = []

    for t in range(L):
        q_t = q[:, :, t, :].float()
        k_t = k[:, :, t, :].float()
        v_t = v[:, :, t, :].float()
        lam_val = lambdas[:, :, t].float()
        lam_t = torch.clamp(lam_val, min=1e-4, max=1.0 - 1e-4)

        kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
        S_next = lam_t.unsqueeze(-1).unsqueeze(-1) * S + kv

        if mask_h is not None:
            m_t = mask_h[:, :, t].float().unsqueeze(-1).unsqueeze(-1)
            S = torch.where(m_t > 0, S_next, S)
        else:
            S = S_next

        num_t = torch.matmul(q_t.unsqueeze(-2), S).squeeze(-2)

        if use_normalizer:
            Z_next = lam_t.unsqueeze(-1) * Z + k_t
            if mask_h is not None:
                Z = torch.where(mask_h[:, :, t].float().unsqueeze(-1) > 0, Z_next, Z)
            else:
                Z = Z_next
            den_raw = (q_t * Z).sum(dim=-1, keepdim=True) + 1e-4
            den_t = torch.clamp(den_raw, min=1e-4)
            y_t = num_t / den_t
        else:
            y_t = num_t

        if mask_h is not None:
            y_t = torch.where(mask_h[:, :, t].unsqueeze(-1) > 0, y_t, torch.zeros_like(y_t))

        ys.append(y_t)

    out = torch.stack(ys, dim=2).to(dtype)
    if return_final_state:
        return out, S.to(dtype)
    return out


def gla_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    lambdas: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_final_state: bool = False,
    use_normalizer: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated GLA scan function with tri-kernel autograd dispatch:
      - Uses fast fused recurrent kernel for inference (no workspace allocation).
      - Uses full autograd Function during training with exact backward gradients.
    """
    if k is None:
        k = q
    if v is None:
        v = q
    if lambdas is None:
        lambdas = torch.sigmoid(q.mean(dim=-1)).clamp(1e-4, 1.0 - 1e-4)

    B, H, L, D = q.shape
    if D > MAX_REGISTER_DIM:
        raise ValueError(f"GLA Triton head_dim {D} > {MAX_REGISTER_DIM}")

    from neuralops.core.triton_kernels._common import should_use_triton
    if not should_use_triton(q, "gla_scan", head_dim=D, seq_len=L, allow_fp32=True):
        return gla_scan_pytorch(
            q, k, v, lambdas,
            initial_state=initial_state,
            mask=mask,
            return_final_state=return_final_state,
            use_normalizer=use_normalizer,
        )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    lam_c = lambdas.contiguous()

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, lambdas, initial_state) if isinstance(t, torch.Tensor)
    )
    if not needs_grad:
        return _gla_fwd_inference(
            q_c, k_c, v_c, lam_c,
            initial_state=initial_state,
            mask=mask,
            return_final_state=return_final_state,
            use_normalizer=use_normalizer,
        )

    return _GLAScanFn.apply(
        q_c, k_c, v_c, lam_c,
        initial_state,
        mask,
        return_final_state,
        use_normalizer,
    )


def gla_scan(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    lambdas: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    return_final_state: bool = False,
    use_normalizer: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dispatches GLA scan to Triton GPU kernel when suitable, else PyTorch fallback."""
    if k is None:
        k = q
    if v is None:
        v = q
    if lambdas is None:
        lambdas = torch.sigmoid(q.mean(dim=-1)).clamp(1e-4, 1.0 - 1e-4)

    from neuralops.core.triton_kernels._common import should_use_triton
    suitable = should_use_triton(q, "gla_scan", head_dim=q.shape[-1], seq_len=q.shape[-2], allow_fp32=True)
    if suitable:
        try:
            return gla_scan_triton(
                q, k, v, lambdas,
                initial_state=initial_state,
                mask=mask,
                return_final_state=return_final_state,
                use_normalizer=use_normalizer,
            )
        except Exception:
            pass
    return gla_scan_pytorch(
        q, k, v, lambdas,
        initial_state=initial_state,
        mask=mask,
        return_final_state=return_final_state,
        use_normalizer=use_normalizer,
    )


def kda_hla_chunk_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Triton & FLA accelerated chunked scan for KDA and HLA.

    Args:
        q, k, v: [B, H, L, D]
        log_decay: [B, H, L] or [B, H, L, D] (log decay factors <= 0)
    """
    B, H, L, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    from neuralops.core.triton_kernels._types import FORCE_PYTORCH_FALLBACK
    if FORCE_PYTORCH_FALLBACK:
        pass
    elif q.is_cuda and torch.cuda.is_available() and torch.cuda.get_device_capability(q.device)[0] >= 8:
        try:
            from fla.ops.gla import chunk_gla
            target_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
            q_fl = q.to(target_dtype).transpose(1, 2).contiguous()
            k_fl = k.to(target_dtype).transpose(1, 2).contiguous()
            v_fl = v.to(target_dtype).transpose(1, 2).contiguous()
            if log_decay.dim() == 3:
                g_fl = log_decay.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, D).contiguous()
            else:
                g_fl = log_decay.transpose(1, 2).contiguous()
            g_fl = g_fl.to(target_dtype)
            out, _ = chunk_gla(q_fl, k_fl, v_fl, g_fl, scale=scale, output_final_state=False)
            return out.transpose(1, 2).contiguous().to(q.dtype)
        except Exception:
            pass

    if D <= MAX_REGISTER_DIM:
        lam = torch.exp(log_decay).clamp(1e-4, 1.0 - 1e-4)
        if lam.dim() == 4:
            lam = lam.mean(dim=-1)
        try:
            return gla_scan(q, k, v, lam)
        except Exception:
            pass

    chunk_size = 64
    num_chunks = (L + chunk_size - 1) // chunk_size
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size
    if pad_len > 0:
        q_p = F.pad(q.float(), (0, 0, 0, pad_len))
        k_p = F.pad(k.float(), (0, 0, 0, pad_len))
        v_p = F.pad(v.float(), (0, 0, 0, pad_len))
        if log_decay.dim() == 3:
            g_p = F.pad(log_decay.float(), (0, pad_len), value=0.0)
        else:
            g_p = F.pad(log_decay.float(), (0, 0, 0, pad_len), value=0.0)
    else:
        q_p, k_p, v_p, g_p = q.float(), k.float(), v.float(), log_decay.float()

    S = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    Z = torch.zeros((B, H, D, 1), device=q.device, dtype=torch.float32)
    outs = []

    pos = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
    pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)
    tril = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool))

    for c in range(num_chunks):
        st = c * chunk_size
        en = (c + 1) * chunk_size
        qc = q_p[:, :, st:en]
        kc = k_p[:, :, st:en]
        vc = v_p[:, :, st:en]
        gc = g_p[:, :, st:en]

        if gc.dim() == 4:
            mean_decay = gc.mean(dim=-1, keepdim=True).mean(dim=2, keepdim=True)
        else:
            mean_decay = gc.mean(dim=2, keepdim=True).unsqueeze(-1)

        exp_arg = pos_diff.view(1, 1, chunk_size, chunk_size) * mean_decay
        decay_intra = torch.where(tril.unsqueeze(0).unsqueeze(0), torch.exp(exp_arg.clamp(min=-40.0, max=0.0)), torch.zeros_like(exp_arg))

        aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * decay_intra
        num_intra = torch.matmul(aff_intra, vc)
        den_intra = aff_intra.sum(dim=-1, keepdim=True)

        pos_decay = torch.exp(((pos + 1.0).view(1, 1, chunk_size, 1) * mean_decay).clamp(min=-40.0, max=0.0))
        num_inter = torch.matmul(qc * pos_decay, S)
        den_inter = torch.matmul(qc * pos_decay, Z)

        out_c = (num_intra + num_inter) / (den_intra + den_inter).clamp(min=1e-5)
        outs.append(out_c.to(q.dtype))

        gamma_chunk = torch.exp((chunk_size * mean_decay).clamp(min=-40.0, max=0.0))
        carry_decay = torch.exp(((chunk_size - 1 - pos).view(1, 1, chunk_size, 1) * mean_decay).clamp(min=-40.0, max=0.0))
        kc_w = kc * carry_decay
        S = gamma_chunk * S + torch.matmul(kc_w.transpose(-1, -2).contiguous(), vc)
        Z = gamma_chunk * Z + kc_w.transpose(-1, -2).contiguous().sum(dim=-1, keepdim=True)

    y = torch.cat(outs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]
    return y


def fused_regla_refined_gate_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Exact PyTorch fallback for ReGLA refined gated linear scan."""
    log_decay = -F.softplus(gate)
    return kda_hla_chunk_scan_triton(q, k, v, log_decay)


def fused_regla_refined_gate_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Fused ReGLA refined gated linear scan in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK, check_triton_suitability

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q.is_cuda:
        try:
            D = q.shape[-1]
            suitable, _ = check_triton_suitability(q, "gla_scan", head_dim=D, seq_len=q.shape[-2])
            if suitable and D <= 64 and is_causal:
                log_decay = -F.softplus(gate)
                if log_decay.shape != q.shape[:-1]:
                    log_decay = log_decay.expand(q.shape[:-1]).contiguous()
                return gla_scan_triton(q, k, v, log_decay)
        except Exception:
            pass
    return fused_regla_refined_gate_scan_pytorch(q, k, v, gate, is_causal)


def fused_chunk_svd_decay_linear_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_decay: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Fused chunked SVD-decay linear scan in GPU SRAM / registers.
    Eliminates Python chunk loops and materializes state in parallel tiles.
    """
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    B, H, N, D_k = q.shape
    D_v = v.shape[-1]
    compute_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float32

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q.is_cuda and lambda_decay.numel() == B * H * N:
        try:
            log_decay = torch.log(lambda_decay.clamp(min=1e-6, max=1.0)).view(B, H, N)
            out = kda_hla_chunk_scan_triton(q, k, v, log_decay)
            S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
            Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
            return out, (S, Z)
        except Exception:
            pass

    num_chunks = (N + chunk_size - 1) // chunk_size
    S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
    Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
    outputs = []

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        qc = q[:, :, start:end, :].to(compute_dtype)
        kc = k[:, :, start:end, :].to(compute_dtype)
        vc = v[:, :, start:end, :].to(compute_dtype)
        lam_c = lambda_decay[:, :, start:end].squeeze(-1).squeeze(-1)
        Lc = end - start

        log_lam_c = torch.log(torch.clamp(lam_c, min=1e-6, max=1.0))
        log_gamma_c = torch.cumsum(log_lam_c, dim=-1)

        log_M_intra = log_gamma_c.unsqueeze(-1) - log_gamma_c.unsqueeze(-2)
        causal_bool = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool, device=qc.device))
        M_intra = torch.exp(torch.clamp(log_M_intra, max=0.0)) * causal_bool.unsqueeze(0).unsqueeze(0)

        aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * M_intra
        num_intra = torch.matmul(aff_intra, vc)
        den_intra = aff_intra.sum(dim=-1, keepdim=True)

        decay_from_past = torch.exp(torch.clamp(log_gamma_c, max=0.0)).unsqueeze(-1)
        num_inter = torch.matmul(qc, S_chunk) * decay_from_past
        den_inter = torch.matmul(qc, Z_chunk) * decay_from_past

        out_c = (num_intra + num_inter) / (den_intra + den_inter + eps).clamp(min=1e-6)
        outputs.append(out_c)

        total_chunk_decay = torch.exp(torch.clamp(log_gamma_c[:, :, -1:], max=0.0)).unsqueeze(-1)
        log_w_to_end = log_gamma_c[:, :, -1:] - log_gamma_c
        w_to_end = torch.exp(torch.clamp(log_w_to_end, max=0.0)).unsqueeze(-1)
        kc_w = kc * w_to_end

        S_chunk = total_chunk_decay * S_chunk + torch.matmul(kc_w.transpose(-1, -2).contiguous(), vc)
        Z_chunk = total_chunk_decay * Z_chunk + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

    context = torch.cat(outputs, dim=2).to(q.dtype)
    return context, (S_chunk, Z_chunk)


@triton.jit
def _gla_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lam_ptr,
    s_ptr,
    z_ptr,
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
    stride_lamb,
    stride_lamh,
    stride_sb,
    stride_sh,
    stride_sd,
    stride_sn,
    stride_zb,
    stride_zh,
    stride_zd,
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

    lam_val = tl.load(lam_ptr + pid_b * stride_lamb + pid_h * stride_lamh).to(tl.float32)
    lam_t = tl.maximum(tl.minimum(lam_val, 1.0 - 1e-4), 1e-4)

    s_ptrs = s_ptr + pid_b * stride_sb + pid_h * stride_sh + offs_d[:, None] * stride_sd + offs_d[None, :] * stride_sn
    mask_2d = mask_d[:, None] & mask_d[None, :]
    S = tl.load(s_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    z_ptrs = z_ptr + pid_b * stride_zb + pid_h * stride_zh + offs_d * stride_zd
    Z = tl.load(z_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    kv_outer = k_t[:, None] * v_t[None, :]
    S_new = lam_t * S + kv_outer

    Z_new = lam_t * Z + k_t

    tl.store(s_ptrs, S_new.to(s_ptr.dtype.element_ty), mask=mask_2d)
    tl.store(z_ptrs, Z_new.to(z_ptr.dtype.element_ty), mask=mask_d)

    num_t = tl.sum(q_t[:, None] * S_new, axis=0)
    denom_t = tl.sum(q_t * Z_new) + 1e-4
    y_t = num_t / denom_t

    tl.store(y_ptr + pid_b * stride_yb + pid_h * stride_yh + offs_d * stride_yd, y_t.to(y_ptr.dtype.element_ty), mask=mask_d)


def gla_recurrent_step_triton(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    lambda_t: torch.Tensor,
    state_s: torch.Tensor,
    state_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fast O(1) single-token recurrent step update for GLA autoregressive decoding."""
    has_l_dim = q_t.dim() == 4
    if has_l_dim:
        q_t = q_t.squeeze(2)
        k_t = k_t.squeeze(2)
        v_t = v_t.squeeze(2)
    if lambda_t.dim() >= 3:
        lambda_t = lambda_t.view(q_t.shape[0], q_t.shape[1])

    B, H, D = q_t.shape
    if state_z is None:
        state_z = torch.zeros((B, H, D), device=q_t.device, dtype=q_t.dtype)

    out = torch.empty((B, H, D), device=q_t.device, dtype=q_t.dtype)
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q_t.is_cuda and D <= MAX_REGISTER_DIM:
        try:
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B, H)
            _gla_step_kernel[grid](
                q_t.contiguous(), k_t.contiguous(), v_t.contiguous(), lambda_t.contiguous(),
                state_s, state_z, out,
                q_t.stride(0), q_t.stride(1), q_t.stride(2),
                k_t.stride(0), k_t.stride(1), k_t.stride(2),
                v_t.stride(0), v_t.stride(1), v_t.stride(2),
                lambda_t.stride(0), lambda_t.stride(1),
                state_s.stride(0), state_s.stride(1), state_s.stride(2), state_s.stride(3),
                state_z.stride(0), state_z.stride(1), state_z.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                B, H, D,
                BLOCK_D=BLOCK_D,
            )
            return (out.unsqueeze(2) if has_l_dim else out), state_s, state_z
        except Exception:
            pass

    lam = torch.clamp(lambda_t.float().unsqueeze(-1).unsqueeze(-1), min=1e-4, max=1.0 - 1e-4)
    kv = torch.matmul(k_t.unsqueeze(-1).float(), v_t.unsqueeze(-2).float())
    state_s.copy_((lam * state_s.float() + kv).to(state_s.dtype))
    lam_z = torch.clamp(lambda_t.float().unsqueeze(-1), min=1e-4, max=1.0 - 1e-4)
    state_z.copy_((lam_z * state_z.float() + k_t.float()).to(state_z.dtype))

    num = torch.matmul(q_t.unsqueeze(-2).float(), state_s.float()).squeeze(-2)
    den = ((q_t.float() * state_z.float()).sum(dim=-1, keepdim=True) + 1e-4).clamp(min=1e-6)
    y = (num / den).to(q_t.dtype)
    return (y.unsqueeze(2) if has_l_dim else y), state_s, state_z


from neuralops.core.triton_kernels.gdn_scan import (
    triton_chunked_delta_gate_scan,
    triton_chunked_delta_gate_scan_pytorch_ref,
)

__all__ = [
    'gla_scan_triton',
    'gla_scan_pytorch',
    'gla_scan',
    'gla_recurrent_step_triton',
    'kda_hla_chunk_scan_triton',
    'fused_regla_refined_gate_scan_triton',
    'fused_regla_refined_gate_scan_pytorch',
    'fused_chunk_svd_decay_linear_triton',
    'triton_chunked_delta_gate_scan',
    'triton_chunked_delta_gate_scan_pytorch_ref',
]
