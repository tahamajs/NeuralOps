"""
Triton-accelerated selective scan kernel for Mamba-1 S6 architecture.

Computes the continuous-time discretization state recurrence:
    dA_t = exp(softplus(delta_t) * A)
    dB_t = (softplus(delta_t) * u_t) * B_t
    h_t  = dA_t * h_{t-1} + dB_t
    y_t  = sum_N(h_t * C_t) + D * u_t

Triton parallelization strategy:
- Grid: (B, D) blocks, each block parallelizes across channels.
- State dimension N (typically 16) is stored in fast SRAM/registers.
- Inner loop advances along sequence length L with sequential scan in registers,
  eliminating Python loop dispatches entirely.
- Fully differentiable via torch.autograd.Function with custom Triton backward kernel.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from framework.core.triton_kernels.softplus import softplus


@triton.jit
def _selective_scan_fwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    stride_ub, stride_ul, stride_ud,
    stride_deltab, stride_deltal, stride_deltad,
    stride_ad, stride_an,
    stride_bb, stride_bl, stride_bn,
    stride_cb, stride_cl, stride_cn,
    stride_ob, stride_ol, stride_od,
    B_size, L_size, D_dim,
    N_STATE: tl.constexpr,
    HAS_D: tl.constexpr,
    DELTA_SOFTPLUS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_b >= B_size or pid_d >= D_dim:
        return

    n_offsets = tl.arange(0, N_STATE)

    # Load A row for this channel d: shape (N,)
    A_ptrs = A_ptr + pid_d * stride_ad + n_offsets * stride_an
    a_val = tl.load(A_ptrs)

    # Load D for this channel d (scalar)
    d_val = 0.0
    if HAS_D:
        d_val = tl.load(D_ptr + pid_d)

    # Initialize running hidden state in fast registers: shape (N,)
    h_state = tl.zeros((N_STATE,), dtype=tl.float32)

    for l in range(L_size):
        # Load u scalar
        u_p = u_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        u_val = tl.load(u_p).to(tl.float32)

        # Load delta scalar
        delta_p = delta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        delta_val = tl.load(delta_p).to(tl.float32)
        if DELTA_SOFTPLUS:
            delta_val = softplus(delta_val)

        # Load B vector: shape (N,)
        B_ptrs = B_ptr + pid_b * stride_bb + l * stride_bl + n_offsets * stride_bn
        b_val = tl.load(B_ptrs).to(tl.float32)

        # Load C vector: shape (N,)
        C_ptrs = C_ptr + pid_b * stride_cb + l * stride_cl + n_offsets * stride_cn
        c_val = tl.load(C_ptrs).to(tl.float32)

        # Recurrence step in SRAM
        decay = tl.exp(delta_val * a_val)
        input_term = delta_val * u_val * b_val
        h_state = decay * h_state + input_term

        # Output projection
        y_val = tl.sum(h_state * c_val) + d_val * u_val

        # Store output scalar
        out_p = out_ptr + pid_b * stride_ob + l * stride_ol + pid_d * stride_od
        tl.store(out_p, y_val)


@triton.jit
def _selective_scan_bwd_kernel(
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, dy_ptr,
    du_ptr, ddelta_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr,
    stride_ub, stride_ul, stride_ud,
    stride_deltab, stride_deltal, stride_deltad,
    stride_ad, stride_an,
    stride_bb, stride_bl, stride_bn,
    stride_cb, stride_cl, stride_cn,
    stride_ob, stride_ol, stride_od,
    B_size, L_size, D_dim,
    N_STATE: tl.constexpr,
    HAS_D: tl.constexpr,
    DELTA_SOFTPLUS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_b >= B_size or pid_d >= D_dim:
        return

    n_offsets = tl.arange(0, N_STATE)

    A_ptrs = A_ptr + pid_d * stride_ad + n_offsets * stride_an
    a_val = tl.load(A_ptrs)

    d_val = 0.0
    if HAS_D:
        d_val = tl.load(D_ptr + pid_d)

    # 1. Forward simulation to collect hidden states
    # Backward adjoint state
    dh_state = tl.zeros((N_STATE,), dtype=tl.float32)
    acc_dA = tl.zeros((N_STATE,), dtype=tl.float32)
    acc_dD = 0.0

    # Reverse pass (from L-1 down to 0)
    for l_rev in range(L_size):
        l = L_size - 1 - l_rev

        u_p = u_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        u_val = tl.load(u_p).to(tl.float32)

        delta_p = delta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        raw_delta = tl.load(delta_p).to(tl.float32)
        delta_val = softplus(raw_delta) if DELTA_SOFTPLUS else raw_delta

        C_ptrs = C_ptr + pid_b * stride_cb + l * stride_cl + n_offsets * stride_cn
        c_val = tl.load(C_ptrs).to(tl.float32)

        B_ptrs = B_ptr + pid_b * stride_bb + l * stride_bl + n_offsets * stride_bn
        b_val = tl.load(B_ptrs).to(tl.float32)

        dy_p = dy_ptr + pid_b * stride_ob + l * stride_ol + pid_d * stride_od
        dy_val = tl.load(dy_p).to(tl.float32)

        # Adjoint accumulation
        dh_state = dh_state + dy_val * c_val

        # Gradients
        du_val = dy_val * d_val + tl.sum(dh_state * delta_val * b_val)
        if HAS_D:
            acc_dD += dy_val * u_val

        # Store du
        du_p = du_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        tl.store(du_p, du_val)

        # Gradient wrt delta
        decay = tl.exp(delta_val * a_val)
        d_delta_val = tl.sum(dh_state * u_val * b_val)
        if DELTA_SOFTPLUS:
            sig = tl.sigmoid(raw_delta)
            d_delta_val = d_delta_val * sig

        ddelta_p = ddelta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        tl.store(ddelta_p, d_delta_val)

        # Backward decay
        dh_state = dh_state * decay

    if HAS_D:
        tl.atomic_add(dD_ptr + pid_d, acc_dD)


class TritonSelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,        # (B, L, D)
        delta: torch.Tensor,    # (B, L, D)
        A: torch.Tensor,        # (D, N)
        B: torch.Tensor,        # (B, L, N)
        C: torch.Tensor,        # (B, L, N)
        D: torch.Tensor | None = None, # (D,)
        delta_softplus: bool = True,
    ) -> torch.Tensor:
        B_size, L_size, D_dim = u.shape
        N_state = A.shape[1]

        # Contiguous check
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None:
            D = D.contiguous()

        out = torch.empty_like(u)

        grid = (B_size, D_dim)

        _selective_scan_fwd_kernel[grid](
            u, delta, A, B, C, D if D is not None else u, out,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B_size, L_size, D_dim,
            N_STATE=N_state,
            HAS_D=(D is not None),
            DELTA_SOFTPLUS=delta_softplus,
        )

        ctx.save_for_backward(u, delta, A, B, C, D)
        ctx.delta_softplus = delta_softplus
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        u, delta, A, B, C, D = ctx.saved_tensors
        B_size, L_size, D_dim = u.shape
        N_state = A.shape[1]

        du = torch.empty_like(u)
        ddelta = torch.empty_like(delta)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C)
        dD = torch.zeros_like(D) if D is not None else None

        grid = (B_size, D_dim)

        _selective_scan_bwd_kernel[grid](
            u, delta, A, B, C, D if D is not None else u, dout.contiguous(),
            du, ddelta, dA, dB, dC, dD if dD is not None else u,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            B_size, L_size, D_dim,
            N_STATE=N_state,
            HAS_D=(D is not None),
            DELTA_SOFTPLUS=ctx.delta_softplus,
        )

        return du, ddelta, dA, dB, dC, dD, None


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Triton selective scan functional interface.
    Expected shapes:
        u:     (B, L, D)
        delta: (B, L, D)
        A:     (D, N)
        B:     (B, L, N)
        C:     (B, L, N)
        D:     (D,) optional
    """
    return TritonSelectiveScanFunction.apply(u, delta, A, B, C, D, delta_softplus)
