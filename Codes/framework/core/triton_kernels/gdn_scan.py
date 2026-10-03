"""Triton-accelerated Gated DeltaNet chunked scan.

Recurrence:
    inter_v_t = k_t @ S_{t-1}
    err_t     = v_t - inter_v_t
    norm_sq_t = sum(k_t^2)
    step_t    = alpha_t / max(1 + norm_sq_t, 1)
    S_t       = beta_t * S_{t-1} + step_t * k_t^T @ err_t
    y_t       = q_t @ S_t

All ops are per-(batch, head). S is a (d_k, d_v) matrix kept in registers.
Supports d_k = d_v <= 64.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

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
    B_size,
    L_size,
    H_size,
    D_size,
    STORE_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    # S in registers (BLOCK_D x BLOCK_D)
    S = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    for t in range(L_size):
        # Load q, k, v
        q_t = tl.load(
            q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + t * stride_ql
            + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        k_t = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + t * stride_kl
            + offs_d * stride_kd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_t = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + t * stride_vl
            + offs_d * stride_vd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        beta_t = tl.load(
            beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl
        ).to(tl.float32)
        alpha_t = tl.load(
            alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al
        ).to(tl.float32)

        # inter_v = k_t @ S -> (D,)
        # k_t is (D,) and S is (D, D). Result = sum_i k_t[i] * S[i, :]
        inter_v = tl.sum(k_t[:, None] * S, axis=0)  # (D,)

        err = v_t - inter_v  # (D,)

        norm_sq = tl.sum(k_t * k_t)
        denom = tl.maximum(1.0 + norm_sq, 1.0)
        step = alpha_t / denom

        # Update S: S_t = beta_t * S + step * k_t^T @ err
        # k_t^T @ err is outer product: shape (D, D)
        outer = k_t[:, None] * err[None, :]  # (D, D)
        S = beta_t * S + step * outer

        # y_t = q_t @ S -> (D,)
        y_t = tl.sum(q_t[:, None] * S, axis=0)

        # Store y
        tl.store(
            y_ptr
            + pid_b * stride_yb
            + pid_h * stride_yh
            + t * stride_yl
            + offs_d * stride_yd,
            y_t.to(y_ptr.dtype.element_ty),
            mask=mask_d,
        )

        # Optionally store S for backward (expensive: L * D * D per head)
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


@triton.jit
def _gdn_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    alpha_ptr,
    dy_ptr,
    s_workspace_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dbeta_ptr,
    dalpha_ptr,
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
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    # Adjoint state dS (D, D), initialized to 0
    dS = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    # Iterate backward
    for t in range(L_size - 1, -1, -1):
        # Load inputs
        q_t = tl.load(
            q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + t * stride_ql
            + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        k_t = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + t * stride_kl
            + offs_d * stride_kd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_t = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + t * stride_vl
            + offs_d * stride_vd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        dy_t = tl.load(
            dy_ptr
            + pid_b * stride_dyb
            + pid_h * stride_dyh
            + t * stride_dyl
            + offs_d * stride_dyd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        beta_t = tl.load(
            beta_ptr + pid_b * stride_bb + pid_h * stride_bh + t * stride_bl
        ).to(tl.float32)
        alpha_t = tl.load(
            alpha_ptr + pid_b * stride_ab + pid_h * stride_ah + t * stride_al
        ).to(tl.float32)

        # Load S_t and S_{t-1} from workspace
        s_t_ptrs = (
            s_workspace_ptr
            + pid_b * stride_swb
            + pid_h * stride_swh
            + t * stride_swl
            + tl.arange(0, BLOCK_D)[:, None] * stride_swd
            + tl.arange(0, BLOCK_D)[None, :] * stride_swn
        )
        S_t = tl.load(s_t_ptrs)  # (D, D)

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
        else:
            S_prev = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

        # === Adjoint update ===
        # dS += q_t^T @ dy_t = outer product
        dS += q_t[:, None] * dy_t[None, :]

        # === dq_t = S_t @ dy_t  → (D,) ===
        dq_t = tl.sum(S_t * dy_t[None, :], axis=1)

        # === Recompute intermediates ===
        inter_v = tl.sum(k_t[:, None] * S_prev, axis=0)  # (D,)
        err = v_t - inter_v  # (D,)
        norm_sq = tl.sum(k_t * k_t)
        denom = tl.maximum(1.0 + norm_sq, 1.0)
        step = alpha_t / denom

        # === dv_t = step * k_t @ dS^T → (D,) ===
        # dS has shape (D, D). (k_t @ dS) gives (D,). But careful with axes:
        # dS[m, n] = dL/dS_t[m, n]
        # dv_t[n] = sum_m dS[m, n] * step * k_t[m]
        dv_t = step * tl.sum(k_t[:, None] * dS, axis=0)  # (D,)

        # === dk_t = dk_direct + dk_from_inter_v + dk_norm ===
        dk_direct = step * tl.sum(dS * err[None, :], axis=1)  # (D,)

        # Path through inter_v = k_t @ S_prev:
        d_inter_v = -step * tl.sum(k_t[:, None] * dS, axis=0)  # (D,)
        dk_from_inter_v = tl.sum(S_prev * d_inter_v[None, :], axis=1)  # (D,)

        # Path through norm_sq in step = alpha_t / denom:
        dS_outer = tl.sum(dS * (k_t[:, None] * err[None, :]))
        d_step = -(alpha_t / (denom * denom)) * dS_outer
        dk_norm = 2.0 * d_step * k_t

        dk_total = dk_direct + dk_from_inter_v + dk_norm

        # === dbeta_t = sum(dS * S_prev) elementwise ===
        dbeta_final = tl.sum(dS * S_prev)

        # === dalpha_t ===
        dalpha_final = dS_outer / denom

        # === Compute dS_prev for the next (earlier) iteration ===
        # dS_prev = beta_t * dS - step * k_t @ k_t^T @ dS
        kk_outer = k_t[:, None] * k_t[None, :]  # (D, D)
        dS_prev = beta_t * dS - step * tl.sum(
            kk_outer[:, :, None] * dS[None, :, :], axis=1
        )

        # Store gradients
        tl.store(
            dq_ptr
            + pid_b * stride_dqb
            + pid_h * stride_dqh
            + t * stride_dql
            + offs_d * stride_dqd,
            dq_t.to(dq_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dk_ptr
            + pid_b * stride_dkb
            + pid_h * stride_dkh
            + t * stride_dkl
            + offs_d * stride_dkd,
            dk_total.to(dk_ptr.dtype.element_ty),
            mask=mask_d,
        )
        tl.store(
            dv_ptr
            + pid_b * stride_dvb
            + pid_h * stride_dvh
            + t * stride_dvl
            + offs_d * stride_dvd,
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

        # Propagate
        dS = dS_prev


class _GatedDeltaNetScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, alpha):
        B, H, L, D = q.shape
        assert D <= MAX_REGISTER_DIM, f"head_dim {D} > {MAX_REGISTER_DIM}"

        y = torch.empty_like(q)
        # Workspace for S_t: (B, H, L, D, D)
        s_workspace = torch.empty((B, H, L, D, D), device=q.device, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _gdn_fwd_kernel[grid](
            q,
            k,
            v,
            beta,
            alpha,
            y,
            s_workspace,
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
            B,
            L,
            H,
            D,
            STORE_S=True,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q, k, v, beta, alpha, s_workspace)
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, beta, alpha, s_workspace = ctx.saved_tensors
        B, H, L, D = q.shape

        dy = dy.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        dalpha = torch.empty_like(alpha)

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
            dq,
            dk,
            dv,
            dbeta,
            dalpha,
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
            BLOCK_D=BLOCK_D,
        )

        return dq, dk, dv, dbeta, dalpha


def gdn_scan_triton(q, k, v, beta, alpha):
    """Triton-accelerated Gated DeltaNet scan. head_dim must be <= 64."""
    B, H, L, D = q.shape
    if D > MAX_REGISTER_DIM:
        raise ValueError(f"head_dim {D} > {MAX_REGISTER_DIM}")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    return _GatedDeltaNetScanFn.apply(q, k, v, beta, alpha)
