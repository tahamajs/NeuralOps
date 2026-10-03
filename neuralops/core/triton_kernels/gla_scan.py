"""Triton-accelerated Gated Linear Attention (GLA) recurrent scan (Forward & Backward).

Recurrence (dynamic per-token decay gating):
    q_t, k_t, v_t in R^{B x H x L x D}
    lam_t in R^{B x H x L} (sigmoid decay values clamped to [1e-4, 1 - 1e-4])

    S_t = lam_t * S_{t-1} + k_t^T @ v_t    # (D, D) matrix
    Z_t = lam_t * Z_{t-1} + k_t            # (D,) vector
    num_t = q_t @ S_t                      # (D,)
    den_t = q_t @ Z_t + 1e-4               # scalar
    y_t = num_t / den_t                    # (D,)

All ops are per-(batch, head) stream with S and Z maintained in registers.
Supports head_dim D <= 64.
"""

from __future__ import annotations

import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = type("triton", (), {"jit": lambda f: f, "heuristics": lambda *a, **kw: lambda f: f, "autotune": lambda *a, **kw: lambda f: f})
    tl = None

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
    B_size,
    L_size,
    H_size,
    D_size,
    STORE_WORKSPACE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    S = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    Z = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(L_size):
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
        lam_val = tl.load(
            lam_ptr + pid_b * stride_lamb + pid_h * stride_lamh + t * stride_laml
        ).to(tl.float32)
        lam_t = tl.maximum(tl.minimum(lam_val, 1.0 - 1e-4), 1e-4)

        # S_t = lam_t * S_{t-1} + k_t^T @ v_t
        kv_outer = k_t[:, None] * v_t[None, :]
        S = lam_t * S + kv_outer

        # Z_t = lam_t * Z_{t-1} + k_t
        Z = lam_t * Z + k_t

        num_t = tl.sum(q_t[:, None] * S, axis=0)
        denom_raw = tl.sum(q_t * Z)
        denom_t = tl.maximum(denom_raw, 1e-4)

        y_t = num_t / denom_t

        tl.store(
            y_ptr
            + pid_b * stride_yb
            + pid_h * stride_yh
            + t * stride_yl
            + offs_d * stride_yd,
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

            z_ptrs = (
                z_workspace_ptr
                + pid_b * stride_zwb
                + pid_h * stride_zwh
                + t * stride_zwl
                + offs_d * stride_zwd
            )
            tl.store(z_ptrs, Z, mask=mask_d)


@triton.jit
def _gla_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lam_ptr,
    dy_ptr,
    s_workspace_ptr,
    z_workspace_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dlam_ptr,
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
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    dS = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    dZ = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for step_rev in range(L_size):
        t = L_size - 1 - step_rev

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
        lam_val = tl.load(
            lam_ptr + pid_b * stride_lamb + pid_h * stride_lamh + t * stride_laml
        ).to(tl.float32)
        lam_t = tl.maximum(tl.minimum(lam_val, 1.0 - 1e-4), 1e-4)

        dy_t = tl.load(
            dy_ptr
            + pid_b * stride_dyb
            + pid_h * stride_dyh
            + t * stride_dyl
            + offs_d * stride_dyd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        # Load S_t and Z_t
        s_t_ptrs = (
            s_workspace_ptr
            + pid_b * stride_swb
            + pid_h * stride_swh
            + t * stride_swl
            + tl.arange(0, BLOCK_D)[:, None] * stride_swd
            + tl.arange(0, BLOCK_D)[None, :] * stride_swn
        )
        S_t = tl.load(s_t_ptrs)

        z_t_ptrs = (
            z_workspace_ptr
            + pid_b * stride_zwb
            + pid_h * stride_zwh
            + t * stride_zwl
            + offs_d * stride_zwd
        )
        Z_t = tl.load(z_t_ptrs, mask=mask_d, other=0.0)

        num_t = tl.sum(q_t[:, None] * S_t, axis=0)
        denom_raw = tl.sum(q_t * Z_t)
        denom_t = tl.maximum(denom_raw, 1e-4)

        dnum_t = dy_t / denom_t
        # If denom_raw <= 1e-4, clamp_min is flat => gradient through denom_raw is 0
        ddenom_t = tl.where(
            denom_raw > 1e-4, -tl.sum(dy_t * num_t) / (denom_t * denom_t), 0.0
        )

        dS += q_t[:, None] * dnum_t[None, :]
        dZ += q_t * ddenom_t

        dq_t = tl.sum(S_t * dnum_t[None, :], axis=1) + Z_t * ddenom_t
        dv_t = tl.sum(k_t[:, None] * dS, axis=0)
        dk_t = tl.sum(dS * v_t[None, :], axis=1) + dZ

        # Load S_{t-1} and Z_{t-1} for dlam_t
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

            z_prev_ptrs = (
                z_workspace_ptr
                + pid_b * stride_zwb
                + pid_h * stride_zwh
                + (t - 1) * stride_zwl
                + offs_d * stride_zwd
            )
            Z_prev = tl.load(z_prev_ptrs, mask=mask_d, other=0.0)

            dlam_t = tl.sum(dS * S_prev) + tl.sum(dZ * Z_prev)

        # Store grads
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
            dk_t.to(dk_ptr.dtype.element_ty),
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
            dlam_ptr + pid_b * stride_dlb + pid_h * stride_dlh + t * stride_dll,
            dlam_t,
        )

        dS = lam_t * dS
        dZ = lam_t * dZ


class _GLAScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, lambdas):
        B, H, L, D = q.shape
        assert D <= MAX_REGISTER_DIM, f"GLA head_dim {D} > {MAX_REGISTER_DIM}"

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        lam_c = lambdas.contiguous()

        y = torch.empty_like(q_c)
        s_workspace = torch.empty((B, H, L, D, D), device=q.device, dtype=torch.float32)
        z_workspace = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _gla_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            lam_c,
            y,
            s_workspace,
            z_workspace,
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
            z_workspace.stride(0),
            z_workspace.stride(1),
            z_workspace.stride(2),
            z_workspace.stride(3),
            B,
            L,
            H,
            D,
            STORE_WORKSPACE=True,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q_c, k_c, v_c, lam_c, s_workspace, z_workspace)
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, lambdas, s_workspace, z_workspace = ctx.saved_tensors
        B, H, L, D = q.shape

        dy_c = dy.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dlambdas = torch.empty_like(lambdas)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _gla_bwd_kernel[grid](
            q,
            k,
            v,
            lambdas,
            dy_c,
            s_workspace,
            z_workspace,
            dq,
            dk,
            dv,
            dlambdas,
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
            dy_c.stride(0),
            dy_c.stride(1),
            dy_c.stride(2),
            dy_c.stride(3),
            s_workspace.stride(0),
            s_workspace.stride(1),
            s_workspace.stride(2),
            s_workspace.stride(3),
            s_workspace.stride(4),
            z_workspace.stride(0),
            z_workspace.stride(1),
            z_workspace.stride(2),
            z_workspace.stride(3),
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
            BLOCK_D=BLOCK_D,
        )

        return dq, dk, dv, dlambdas


def gla_scan_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lambdas: torch.Tensor
) -> torch.Tensor:
    """Triton-accelerated GLA scan function. head_dim must be <= 64."""
    B, H, L, D = q.shape
    if D > MAX_REGISTER_DIM:
        raise ValueError(f"GLA Triton head_dim {D} > {MAX_REGISTER_DIM}")
    return _GLAScanFn.apply(q, k, v, lambdas)
