"""Triton-accelerated Factorised Matrix State (FMS) recurrent scan.

Recurrence (associative matrix memory with constant decay per head):
    q_feat = ELU(q) + 1.0
    k_feat = (ELU(k) + 1.0) / sqrt(D)
    v_t    = v

    S_t    = decay_h * S_{t-1} + k_feat_t^T @ v_t    # (D, D) matrix state
    Z_t    = decay_h * Z_{t-1} + k_feat_t            # (D,) vector normalizer
    num_t  = q_feat_t @ S_t                          # (D,)
    den_t  = q_feat_t @ Z_t + 1e-5                   # scalar
    y_t    = num_t / den_t                           # (D,)

All ops are per-(batch, head). S and Z are kept in SRAM registers during recurrence.
Supports head_dim D <= 64.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


MAX_REGISTER_DIM = 64


@triton.jit
def _fms_fwd_kernel(
    q_ptr, k_ptr, v_ptr, decay_ptr,
    y_ptr, s_workspace_ptr, z_workspace_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_dh,
    stride_yb, stride_yh, stride_yl, stride_yd,
    stride_swb, stride_swh, stride_swl, stride_swd, stride_swn,
    stride_zwb, stride_zwh, stride_zwl, stride_zwd,
    B_size, L_size, H_size, D_size,
    STORE_WORKSPACE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    # Load constant decay for this head: decay_h in (0, 1)
    decay_val = tl.load(decay_ptr + pid_h * stride_dh).to(tl.float32)
    inv_sqrt_d = 1.0 / tl.sqrt(D_size.to(tl.float32))

    # Recurrent state in SRAM registers
    # S is (BLOCK_D, BLOCK_D), Z is (BLOCK_D,)
    S = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    Z = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(L_size):
        # Load raw inputs q, k, v
        q_raw = tl.load(
            q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_d * stride_qd,
            mask=mask_d, other=0.0
        ).to(tl.float32)
        k_raw = tl.load(
            k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_d * stride_kd,
            mask=mask_d, other=0.0
        ).to(tl.float32)
        v_raw = tl.load(
            v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_d * stride_vd,
            mask=mask_d, other=0.0
        ).to(tl.float32)

        # Feature maps: ELU + 1
        q_f = tl.where(q_raw > 0.0, q_raw + 1.0, tl.exp(q_raw))
        k_elu = tl.where(k_raw > 0.0, k_raw + 1.0, tl.exp(k_raw))
        k_f = k_elu * inv_sqrt_d
        v_f = v_raw

        # Zero out padding elements
        q_f = tl.where(mask_d, q_f, 0.0)
        k_f = tl.where(mask_d, k_f, 0.0)
        v_f = tl.where(mask_d, v_f, 0.0)

        # Recurrent state update:
        # S_t = decay_val * S_{t-1} + k_f^T @ v_f (outer product: [i, j] = k_f[i] * v_f[j])
        kv_outer = k_f[:, None] * v_f[None, :]
        S = decay_val * S + kv_outer

        # Z_t = decay_val * Z_{t-1} + k_f
        Z = decay_val * Z + k_f

        # Output projection:
        # num = q_f @ S -> num[j] = sum_i q_f[i] * S[i, j]
        num_t = tl.sum(q_f[:, None] * S, axis=0)  # (BLOCK_D,)
        # denom = q_f @ Z -> scalar
        denom_raw = tl.sum(q_f * Z)
        denom_t = tl.maximum(denom_raw, 1e-5)

        y_t = num_t / denom_t

        # Store y
        tl.store(
            y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd,
            y_t.to(y_ptr.dtype.element_ty),
            mask=mask_d,
        )

        # Save S and Z in workspace for backward pass
        if STORE_WORKSPACE:
            s_ptrs = (
                s_workspace_ptr
                + pid_b * stride_swb + pid_h * stride_swh + t * stride_swl
                + tl.arange(0, BLOCK_D)[:, None] * stride_swd
                + tl.arange(0, BLOCK_D)[None, :] * stride_swn
            )
            tl.store(s_ptrs, S)

            z_ptrs = (
                z_workspace_ptr
                + pid_b * stride_zwb + pid_h * stride_zwh + t * stride_zwl
                + offs_d * stride_zwd
            )
            tl.store(z_ptrs, Z, mask=mask_d)


@triton.jit
def _fms_bwd_kernel(
    q_ptr, k_ptr, v_ptr, decay_ptr, dy_ptr,
    s_workspace_ptr, z_workspace_ptr,
    dq_ptr, dk_ptr, dv_ptr, ddecay_acc_ptr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_kb, stride_kh, stride_kl, stride_kd,
    stride_vb, stride_vh, stride_vl, stride_vd,
    stride_dh,
    stride_dyb, stride_dyh, stride_dyl, stride_dyd,
    stride_swb, stride_swh, stride_swl, stride_swd, stride_swn,
    stride_zwb, stride_zwh, stride_zwl, stride_zwd,
    stride_dqb, stride_dqh, stride_dql, stride_dqd,
    stride_dkb, stride_dkh, stride_dkl, stride_dkd,
    stride_dvb, stride_dvh, stride_dvl, stride_dvd,
    stride_dah,
    B_size, L_size, H_size, D_size,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B_size or pid_h >= H_size:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_size

    decay_val = tl.load(decay_ptr + pid_h * stride_dh).to(tl.float32)
    inv_sqrt_d = 1.0 / tl.sqrt(D_size.to(tl.float32))

    # Adjoint states in registers: dS is (BLOCK_D, BLOCK_D), dZ is (BLOCK_D,)
    dS = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    dZ = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Accumulator for decay gradient for this head from this batch item
    ddecay_local = 0.0

    # Reverse pass over tokens
    for step_rev in range(L_size):
        t = L_size - 1 - step_rev

        # Load inputs at step t
        q_raw = tl.load(
            q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_d * stride_qd,
            mask=mask_d, other=0.0
        ).to(tl.float32)
        k_raw = tl.load(
            k_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_d * stride_kd,
            mask=mask_d, other=0.0
        ).to(tl.float32)
        v_raw = tl.load(
            v_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_d * stride_vd,
            mask=mask_d, other=0.0
        ).to(tl.float32)
        dy_t = tl.load(
            dy_ptr + pid_b * stride_dyb + pid_h * stride_dyh + t * stride_dyl + offs_d * stride_dyd,
            mask=mask_d, other=0.0
        ).to(tl.float32)

        # Feature activations
        q_f = tl.where(q_raw > 0.0, q_raw + 1.0, tl.exp(q_raw))
        k_elu = tl.where(k_raw > 0.0, k_raw + 1.0, tl.exp(k_raw))
        k_f = k_elu * inv_sqrt_d
        v_f = v_raw

        q_f = tl.where(mask_d, q_f, 0.0)
        k_f = tl.where(mask_d, k_f, 0.0)
        v_f = tl.where(mask_d, v_f, 0.0)

        # Load S_t and Z_t
        s_t_ptrs = (
            s_workspace_ptr
            + pid_b * stride_swb + pid_h * stride_swh + t * stride_swl
            + tl.arange(0, BLOCK_D)[:, None] * stride_swd
            + tl.arange(0, BLOCK_D)[None, :] * stride_swn
        )
        S_t = tl.load(s_t_ptrs)

        z_t_ptrs = (
            z_workspace_ptr
            + pid_b * stride_zwb + pid_h * stride_zwh + t * stride_zwl
            + offs_d * stride_zwd
        )
        Z_t = tl.load(z_t_ptrs, mask=mask_d, other=0.0)

        # Recompute forward intermediates num_t and den_t
        num_t = tl.sum(q_f[:, None] * S_t, axis=0)  # (BLOCK_D,)
        denom_raw = tl.sum(q_f * Z_t)
        denom_t = tl.maximum(denom_raw, 1e-5)

        # Gradients from y_t = num_t / denom_t
        # dL/dnum_t = dy_t / denom_t
        dnum_t = dy_t / denom_t
        # dL/ddenom_t = - sum(dy_t * num_t) / (denom_t * denom_t)
        ddenom_t = -tl.sum(dy_t * num_t) / (denom_t * denom_t)

        # Adjoint accumulation into dS and dZ at step t:
        # num_t = q_f @ S_t  =>  dL/dS_t += q_f^T @ dnum_t
        dS += q_f[:, None] * dnum_t[None, :]
        # denom_t = q_f @ Z_t => dL/dZ_t += q_f * ddenom_t
        dZ += q_f * ddenom_t

        # dq_f:
        # from num_t: sum_j S_t[i, j] * dnum_t[j]
        dq_from_num = tl.sum(S_t * dnum_t[None, :], axis=1)
        # from denom_t: Z_t * ddenom_t
        dq_from_den = Z_t * ddenom_t
        dq_f = dq_from_num + dq_from_den

        # Gradient through ELU + 1: dELU(x)/dx = 1 if x > 0 else exp(x)
        dq_t = tl.where(q_raw > 0.0, dq_f, dq_f * tl.exp(q_raw))
        dq_t = tl.where(mask_d, dq_t, 0.0)

        # dv_t from S_t = decay * S_{t-1} + k_f^T @ v_f:
        # dL/dv_f[j] = sum_i k_f[i] * dS[i, j]
        dv_f = tl.sum(k_f[:, None] * dS, axis=0)
        dv_t = tl.where(mask_d, dv_f, 0.0)

        # dk_f from S_t and Z_t:
        # from S_t: sum_j dS[i, j] * v_f[j]
        # from Z_t: dZ[i]
        dk_from_S = tl.sum(dS * v_f[None, :], axis=1)
        dk_from_Z = dZ
        dk_f = (dk_from_S + dk_from_Z) * inv_sqrt_d

        dk_t = tl.where(k_raw > 0.0, dk_f, dk_f * tl.exp(k_raw))
        dk_t = tl.where(mask_d, dk_t, 0.0)

        # Store input gradients
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

        # Decay gradient accumulation:
        # S_t = decay * S_{t-1} + ... => dL/ddecay += sum(dS * S_{t-1})
        # Z_t = decay * Z_{t-1} + ... => dL/ddecay += sum(dZ * Z_{t-1})
        if t > 0:
            s_prev_ptrs = (
                s_workspace_ptr
                + pid_b * stride_swb + pid_h * stride_swh + (t - 1) * stride_swl
                + tl.arange(0, BLOCK_D)[:, None] * stride_swd
                + tl.arange(0, BLOCK_D)[None, :] * stride_swn
            )
            S_prev = tl.load(s_prev_ptrs)

            z_prev_ptrs = (
                z_workspace_ptr
                + pid_b * stride_zwb + pid_h * stride_zwh + (t - 1) * stride_zwl
                + offs_d * stride_zwd
            )
            Z_prev = tl.load(z_prev_ptrs, mask=mask_d, other=0.0)

            ddecay_local += tl.sum(dS * S_prev) + tl.sum(dZ * Z_prev)

        # Backward propagation to previous step:
        # dS_{t-1} = decay * dS_t, dZ_{t-1} = decay * dZ_t
        dS = decay_val * dS
        dZ = decay_val * dZ

    # Atomically accumulate ddecay into global head buffer
    tl.atomic_add(ddecay_acc_ptr + pid_h * stride_dah, ddecay_local)


class _FMSScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, decay):
        B, H, L, D = q.shape
        assert D <= MAX_REGISTER_DIM, f"head_dim {D} > {MAX_REGISTER_DIM}"

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        decay_c = decay.contiguous()

        y = torch.empty_like(q_c)
        s_workspace = torch.empty((B, H, L, D, D), device=q.device, dtype=torch.float32)
        z_workspace = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _fms_fwd_kernel[grid](
            q_c, k_c, v_c, decay_c,
            y, s_workspace, z_workspace,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            decay_c.stride(0),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            s_workspace.stride(0), s_workspace.stride(1), s_workspace.stride(2),
            s_workspace.stride(3), s_workspace.stride(4),
            z_workspace.stride(0), z_workspace.stride(1), z_workspace.stride(2),
            z_workspace.stride(3),
            B, L, H, D,
            STORE_WORKSPACE=True,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q_c, k_c, v_c, decay_c, s_workspace, z_workspace)
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, decay, s_workspace, z_workspace = ctx.saved_tensors
        B, H, L, D = q.shape

        dy_c = dy.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        ddecay_acc = torch.zeros_like(decay, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H)

        _fms_bwd_kernel[grid](
            q, k, v, decay, dy_c,
            s_workspace, z_workspace,
            dq, dk, dv, ddecay_acc,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            decay.stride(0),
            dy_c.stride(0), dy_c.stride(1), dy_c.stride(2), dy_c.stride(3),
            s_workspace.stride(0), s_workspace.stride(1), s_workspace.stride(2),
            s_workspace.stride(3), s_workspace.stride(4),
            z_workspace.stride(0), z_workspace.stride(1), z_workspace.stride(2),
            z_workspace.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            ddecay_acc.stride(0),
            B, L, H, D,
            BLOCK_D=BLOCK_D,
        )

        return dq, dk, dv, ddecay_acc.to(decay.dtype)


def fms_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated FMS scan function. head_dim must be <= 64."""
    B, H, L, D = q.shape
    if D > MAX_REGISTER_DIM:
        raise ValueError(f"FMS Triton head_dim {D} > {MAX_REGISTER_DIM}")

    # Ensure decay is (H,)
    if decay.dim() > 1:
        decay = decay.squeeze()
    elif decay.dim() == 0:
        decay = decay.repeat(H)

    if decay.shape[0] != H:
        raise ValueError(f"decay shape {decay.shape} does not match H={H}")

    return _FMSScanFn.apply(q, k, v, decay)
