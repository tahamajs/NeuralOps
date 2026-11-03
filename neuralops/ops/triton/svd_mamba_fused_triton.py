#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD-Mamba Fused State Space Operator (Triton JIT Implementation)

High-throughput Triton kernel fusing continuous-time SSM discretization,
parallel associative scan across sequence dimension, feedthrough D, silu(z) gating,
and low-rank global skip connection without deprecated block pointer constructs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuralops.ops.triton._common import HAS_TRITON, tl, triton


@triton.jit
def _svd_mamba_fused_fwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    global_ptr,
    out_ptr,
    stride_ub,
    stride_ul,
    stride_ud,
    stride_deltab,
    stride_deltal,
    stride_deltad,
    stride_ad,
    stride_an,
    stride_bb,
    stride_bl,
    stride_bn,
    stride_cb,
    stride_cl,
    stride_cn,
    stride_zb,
    stride_zl,
    stride_zd,
    stride_gb,
    stride_gl,
    stride_gd,
    stride_ob,
    stride_ol,
    stride_od,
    B_sz,
    L_sz,
    D_dim,
    N_STATE: tl.constexpr,
    A_HAS_D: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_GLOBAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_b >= B_sz or pid_d >= D_dim:
        return

    n_offsets = tl.arange(0, N_STATE)

    if A_HAS_D:
        A_ptrs = A_ptr + pid_d * stride_ad + n_offsets * stride_an
    else:
        A_ptrs = A_ptr + n_offsets * stride_an
    a_val = tl.load(A_ptrs).to(tl.float32)

    d_val = 0.0
    if HAS_D:
        d_val = tl.load(D_ptr + pid_d).to(tl.float32)

    h_num = tl.zeros((N_STATE,), dtype=tl.float32)
    h_den = tl.zeros((N_STATE,), dtype=tl.float32)

    for l in range(L_sz):
        u_p = u_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        u_val = tl.load(u_p).to(tl.float32)

        delta_p = delta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        delta_val = tl.load(delta_p).to(tl.float32)

        B_ptrs = B_ptr + pid_b * stride_bb + l * stride_bl + n_offsets * stride_bn
        b_val = tl.load(B_ptrs).to(tl.float32)

        C_ptrs = C_ptr + pid_b * stride_cb + l * stride_cl + n_offsets * stride_cn
        c_val = tl.load(C_ptrs).to(tl.float32)

        dt_a = delta_val * a_val
        # Clamp between -25.0 and 0.0
        dt_a_clamped = tl.minimum(tl.maximum(dt_a, -25.0), 0.0)
        decay = tl.math.exp2(1.44269504 * dt_a_clamped)

        du_num = delta_val * b_val * u_val
        du_den = delta_val * tl.abs(b_val)

        h_num = decay * h_num + du_num
        h_num = tl.minimum(tl.maximum(h_num, -1e4), 1e4)

        h_den = decay * h_den + du_den
        h_den = tl.minimum(tl.maximum(h_den, 0.0), 1e4)

        num_sum = tl.sum(c_val * h_num, axis=0)
        den_sum = tl.sum(tl.abs(c_val) * h_den, axis=0)
        den_clamped = tl.maximum(den_sum, 1e-4)

        y_val = (num_sum / den_clamped) + (d_val * u_val)

        if HAS_Z:
            z_p = z_ptr + pid_b * stride_zb + l * stride_zl + pid_d * stride_zd
            z_val = tl.load(z_p).to(tl.float32)
            silu_z = z_val / (1.0 + tl.math.exp2(-1.44269504 * z_val))
            y_val = y_val * silu_z

        if HAS_GLOBAL:
            g_p = global_ptr + pid_b * stride_gb + l * stride_gl + pid_d * stride_gd
            g_val = tl.load(g_p).to(tl.float32)
            y_val = y_val + g_val

        out_p = out_ptr + pid_b * stride_ob + l * stride_ol + pid_d * stride_od
        tl.store(out_p, y_val)


def svd_mamba_fused_triton(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    global_mix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation of SVD-Mamba fused state-space recurrence."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in the current runtime environment.")

    B_sz, L_sz, D_dim = u.shape
    d_state = B_ssm.shape[-1]
    if d_state not in (8, 16, 32, 64):
        # Round up power of two for Triton arange
        raise ValueError(f"Unsupported d_state={d_state}. Must be one of 8, 16, 32, 64.")

    u_c = u.contiguous()
    delta_c = delta.contiguous()
    A_c = A.contiguous()
    B_c = B_ssm.contiguous()
    C_c = C_ssm.contiguous()
    D_c = D.contiguous() if D is not None and D.numel() > 0 else u_c
    z_c = z.contiguous() if z is not None and z.numel() > 0 else u_c
    g_c = global_mix.contiguous() if global_mix is not None and global_mix.numel() > 0 else u_c

    out = torch.empty_like(u_c)
    grid = (B_sz, D_dim)

    a_has_d = (A_c.dim() == 2 and A_c.size(0) == D_dim)
    stride_ad = A_c.stride(0) if a_has_d else 0
    stride_an = A_c.stride(1) if a_has_d else A_c.stride(0)

    _svd_mamba_fused_fwd_kernel[grid](
        u_c,
        delta_c,
        A_c,
        B_c,
        C_c,
        D_c,
        z_c,
        g_c,
        out,
        u_c.stride(0),
        u_c.stride(1),
        u_c.stride(2),
        delta_c.stride(0),
        delta_c.stride(1),
        delta_c.stride(2),
        stride_ad,
        stride_an,
        B_c.stride(0),
        B_c.stride(1),
        B_c.stride(2),
        C_c.stride(0),
        C_c.stride(1),
        C_c.stride(2),
        z_c.stride(0) if z is not None else 0,
        z_c.stride(1) if z is not None else 0,
        z_c.stride(2) if z is not None else 0,
        g_c.stride(0) if global_mix is not None else 0,
        g_c.stride(1) if global_mix is not None else 0,
        g_c.stride(2) if global_mix is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B_sz,
        L_sz,
        D_dim,
        N_STATE=d_state,
        A_HAS_D=a_has_d,
        HAS_D=D is not None and D.numel() > 0,
        HAS_Z=z is not None and z.numel() > 0,
        HAS_GLOBAL=global_mix is not None and global_mix.numel() > 0,
    )
    return out
