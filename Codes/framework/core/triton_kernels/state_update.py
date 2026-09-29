#!/usr/bin/env python3
"""
Triton State-Space Model (SSM) & RWKV State Update Kernel.

Computes discrete sequential recurrence:
  h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
  y_t = (h_t @ C_t) + D * u_t

Used by:
  - Mamba, S4, S6, RWKV, HGRN2, Jamba
  - Hybrid SSM-Attention architectures
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from framework.core.triton_kernels._common import (
    HAS_TRITON,
    check_triton_suitability,
    state_update_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_ssm_recurrence_kernel(
        U_ptr, Delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, State_in_ptr, Out_ptr, State_out_ptr,
        stride_ub, stride_ul, stride_ud,
        stride_deltab, stride_deltal, stride_deltad,
        stride_ad, stride_an,
        stride_bb, stride_bl, stride_bn,
        stride_cb, stride_cl, stride_cn,
        stride_yb, stride_yl, stride_yd,
        stride_sb, stride_sd, stride_sn,
        B_SZ, L_SZ, D_SZ,
        N_SZ: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_INIT_STATE: tl.constexpr,
    ):
        b_id = tl.program_id(0)
        d_id = tl.program_id(1)

        offs_n = tl.arange(0, N_SZ)

        # Load A for this channel: [N_SZ]
        a_ptrs = A_ptr + d_id * stride_ad + offs_n * stride_an
        a_vec = tl.load(a_ptrs)

        d_val = tl.load(D_ptr + d_id) if HAS_D else 0.0

        # Load or initialize state vector h \in [N_SZ]
        if HAS_INIT_STATE:
            s_in_ptrs = State_in_ptr + b_id * stride_sb + d_id * stride_sd + offs_n * stride_sn
            h = tl.load(s_in_ptrs).to(tl.float32)
        else:
            h = tl.zeros((N_SZ,), dtype=tl.float32)

        for t in range(0, L_SZ):
            # Load inputs at step t
            u_val = tl.load(U_ptr + b_id * stride_ub + t * stride_ul + d_id * stride_ud).to(tl.float32)
            delta_val = tl.load(Delta_ptr + b_id * stride_deltab + t * stride_deltal + d_id * stride_deltad).to(tl.float32)

            b_ptrs = B_ptr + b_id * stride_bb + t * stride_bl + offs_n * stride_bn
            c_ptrs = C_ptr + b_id * stride_cb + t * stride_cl + offs_n * stride_cn
            b_vec = tl.load(b_ptrs).to(tl.float32)
            c_vec = tl.load(c_ptrs).to(tl.float32)

            # Discretize: dA = exp(delta * A), dB = delta * B
            dA = tl.exp(delta_val * a_vec)
            dB = delta_val * b_vec

            # Update state: h = h * dA + u * dB
            h = h * dA + u_val * dB

            # Output: y = sum(h * C) + D * u
            y_val = tl.sum(h * c_vec) + (d_val * u_val if HAS_D else 0.0)

            # Store output
            y_ptr = Out_ptr + b_id * stride_yb + t * stride_yl + d_id * stride_yd
            tl.store(y_ptr, y_val.to(Out_ptr.dtype.element_ty))

        # Store final state
        s_out_ptrs = State_out_ptr + b_id * stride_sb + d_id * stride_sd + offs_n * stride_sn
        tl.store(s_out_ptrs, h.to(State_out_ptr.dtype.element_ty))


class _TritonStateUpdateAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B_sz, L_sz, D_sz = u.shape
        N_sz = A.shape[-1]

        # Clamp N_sz to nearest power of 2 for Triton compile: 16, 32, 64
        if N_sz not in (8, 16, 32, 64):
            return state_update_pytorch_fallback(u, delta, A, B, C, D=D, state=state, eps=eps)

        y = torch.empty((B_sz, L_sz, D_sz), dtype=u.dtype, device=u.device)
        new_state = torch.empty((B_sz, D_sz, N_sz), dtype=u.dtype, device=u.device)

        has_d = D is not None
        d_tensor = D.contiguous() if has_d else torch.empty((1,), dtype=u.dtype, device=u.device)

        has_init = state is not None
        init_state = state.contiguous() if has_init else torch.empty((1,), dtype=u.dtype, device=u.device)

        # Delta stride handling
        delta_c = delta.expand(B_sz, L_sz, D_sz).contiguous() if delta.dim() == 2 or delta.shape[-1] != D_sz else delta.contiguous()
        b_c = B.expand(B_sz, L_sz, N_sz).contiguous() if B.dim() == 2 else B.contiguous()
        c_c = C.expand(B_sz, L_sz, N_sz).contiguous() if C.dim() == 2 else C.contiguous()

        grid = (B_sz, D_sz)
        _triton_ssm_recurrence_kernel[grid](
            u.contiguous(), delta_c, A.contiguous(), b_c, c_c, d_tensor, init_state, y, new_state,
            u.stride(0), u.stride(1), u.stride(2),
            delta_c.stride(0), delta_c.stride(1), delta_c.stride(2),
            A.stride(0), A.stride(1),
            b_c.stride(0), b_c.stride(1), b_c.stride(2),
            c_c.stride(0), c_c.stride(1), c_c.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            new_state.stride(0), new_state.stride(1), new_state.stride(2),
            B_sz, L_sz, D_sz,
            N_SZ=N_sz,
            HAS_D=has_d,
            HAS_INIT_STATE=has_init,
        )

        ctx.save_for_backward(u, delta, A, B, C, D, state)
        ctx.eps = eps
        return y, new_state

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor, grad_state: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        u, delta, A, B, C, D, state = ctx.saved_tensors
        u_req = u.detach().requires_grad_(True)
        delta_req = delta.detach().requires_grad_(True)
        A_req = A.detach().requires_grad_(True)
        B_req = B.detach().requires_grad_(True)
        C_req = C.detach().requires_grad_(True)
        D_req = D.detach().requires_grad_(True) if D is not None else None
        state_req = state.detach().requires_grad_(True) if state is not None else None

        with torch.enable_grad():
            out_y, out_state = state_update_pytorch_fallback(
                u_req, delta_req, A_req, B_req, C_req, D=D_req, state=state_req, eps=ctx.eps
            )
            diff_vars = [u_req, delta_req, A_req, B_req, C_req]
            grad_outputs = [grad_y]
            if D_req is not None:
                diff_vars.append(D_req)
            if state_req is not None:
                diff_vars.append(state_req)
                grad_outputs.append(grad_state)

            grads = torch.autograd.grad(out_y, diff_vars, grad_y, retain_graph=False, allow_unused=True)

        g_u = grads[0] if ctx.needs_input_grad[0] else None
        g_delta = grads[1] if ctx.needs_input_grad[1] else None
        g_A = grads[2] if ctx.needs_input_grad[2] else None
        g_B = grads[3] if ctx.needs_input_grad[3] else None
        g_C = grads[4] if ctx.needs_input_grad[4] else None
        idx = 5
        g_D = grads[idx] if D_req is not None and ctx.needs_input_grad[5] else None
        if D_req is not None:
            idx += 1
        g_state = grads[idx] if state_req is not None and ctx.needs_input_grad[6] else None
        return g_u, g_delta, g_A, g_B, g_C, g_D, g_state, None


def is_supported(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> bool:
    """Check whether SSM state update can run on the Triton kernel."""
    suitable, _ = check_triton_suitability(u, "state_update")
    if not suitable:
        return False
    N_sz = A.shape[-1] if A.dim() >= 2 else 0
    return N_sz in (8, 16, 32, 64) and A.is_cuda and B.is_cuda and C.is_cuda


def fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward entrypoint for fused Triton SSM state updates: (y, new_state)."""
    if not is_supported(u, delta, A, B, C, D=D, **kwargs):
        return state_update_pytorch_fallback(u, delta, A, B, C, D=D, state=state, eps=eps, **kwargs)
    return _TritonStateUpdateAutogradFunction.apply(u, delta, A, B, C, D, state, eps)


def bwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    grad_out: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    """Backward entrypoint computing gradients for SSM inputs."""
    u_req = u.detach().requires_grad_(True)
    delta_req = delta.detach().requires_grad_(True)
    A_req = A.detach().requires_grad_(True)
    B_req = B.detach().requires_grad_(True)
    C_req = C.detach().requires_grad_(True)
    with torch.enable_grad():
        out, _ = fwd(u_req, delta_req, A_req, B_req, C_req, D=D, state=state, eps=eps, **kwargs)
        grads = torch.autograd.grad(out, [u_req, delta_req, A_req, B_req, C_req], grad_out)
    return grads
