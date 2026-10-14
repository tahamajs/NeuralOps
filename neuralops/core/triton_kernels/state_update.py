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

from typing import Any

import torch

from neuralops.core.triton_kernels._types import (
    HAS_TRITON,
    check_triton_suitability,
    state_update_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_ssm_recurrence_kernel(
        U_ptr,
        Delta_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        State_in_ptr,
        Out_ptr,
        State_out_ptr,
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
        stride_yb,
        stride_yl,
        stride_yd,
        stride_sb,
        stride_sd,
        stride_sn,
        _B_SZ,
        L_SZ,
        _D_SZ,
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
            u_val = tl.load(U_ptr + b_id * stride_ub + t * stride_ul + d_id * stride_ud).to(
                tl.float32
            )
            delta_val = tl.load(
                Delta_ptr + b_id * stride_deltab + t * stride_deltal + d_id * stride_deltad
            ).to(tl.float32)

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
        D: torch.Tensor | None,
        state: torch.Tensor | None,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        init_state = (
            state.contiguous() if has_init else torch.empty((1,), dtype=u.dtype, device=u.device)
        )

        # Delta stride handling
        delta_c = (
            delta.expand(B_sz, L_sz, D_sz).contiguous()
            if delta.dim() == 2 or delta.shape[-1] != D_sz
            else delta.contiguous()
        )
        b_c = B.expand(B_sz, L_sz, N_sz).contiguous() if B.dim() == 2 else B.contiguous()
        c_c = C.expand(B_sz, L_sz, N_sz).contiguous() if C.dim() == 2 else C.contiguous()

        grid = (B_sz, D_sz)
        _triton_ssm_recurrence_kernel[grid](
            u.contiguous(),
            delta_c,
            A.contiguous(),
            b_c,
            c_c,
            d_tensor,
            init_state,
            y,
            new_state,
            u.stride(0),
            u.stride(1),
            u.stride(2),
            delta_c.stride(0),
            delta_c.stride(1),
            delta_c.stride(2),
            A.stride(0),
            A.stride(1),
            b_c.stride(0),
            b_c.stride(1),
            b_c.stride(2),
            c_c.stride(0),
            c_c.stride(1),
            c_c.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            new_state.stride(0),
            new_state.stride(1),
            new_state.stride(2),
            B_sz,
            L_sz,
            D_sz,
            N_SZ=N_sz,
            HAS_D=has_d,
            HAS_INIT_STATE=has_init,
        )

        ctx.save_for_backward(u, delta, A, B, C, D, state)
        ctx.eps = eps
        return y, new_state

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor, grad_state: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
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
                u_req,
                delta_req,
                A_req,
                B_req,
                C_req,
                D=D_req,
                state=state_req,
                eps=ctx.eps,
            )
            diff_vars = [u_req, delta_req, A_req, B_req, C_req]
            grad_outputs = [grad_y]
            if D_req is not None:
                diff_vars.append(D_req)
            if state_req is not None:
                diff_vars.append(state_req)
                grad_outputs.append(grad_state)

            grads = torch.autograd.grad(
                out_y, diff_vars, grad_y, retain_graph=False, allow_unused=True
            )

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
    D: torch.Tensor | None = None,
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
    D: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    D: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
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


# ===========================================================================
# Hardware-Adaptive Single-Step Recurrent Kernels (SM >= 7.0 & CPU/Fallback)
# ===========================================================================

if HAS_TRITON:
    @triton.jit
    def _triton_recurrent_linear_step_kernel(
        Q_ptr, K_ptr, V_ptr, KV_ptr, Z_ptr, Out_ptr,
        B, H, Dk: tl.constexpr, Dv: tl.constexpr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_kvb, stride_kvh, stride_kvk, stride_kvv,
        stride_zb, stride_zh, stride_zk,
        stride_ob, stride_oh, stride_ov,
        eps,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_k = tl.arange(0, Dk)
        offs_v = tl.arange(0, Dv)

        q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_k * stride_qd
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_k * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_v * stride_vd

        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)

        kv_ptrs = KV_ptr + pid_b * stride_kvb + pid_h * stride_kvh + offs_k[:, None] * stride_kvk + offs_v[None, :] * stride_kvv
        kv = tl.load(kv_ptrs).to(tl.float32)

        # S_t = S_{t-1} + k_t \otimes v_t
        kv_new = kv + k[:, None] * v[None, :]
        tl.store(kv_ptrs, kv_new.to(KV_ptr.dtype.element_ty))

        z_ptrs = Z_ptr + pid_b * stride_zb + pid_h * stride_zh + offs_k * stride_zk
        z = tl.load(z_ptrs).to(tl.float32)
        z_new = z + k
        tl.store(z_ptrs, z_new.to(Z_ptr.dtype.element_ty))

        # o_t = S_t^T q_t / (<q_t, z_t> + eps)
        # S_t has shape [Dk, Dv] -> q_t [Dk] @ S_t [Dk, Dv] -> [Dv]
        num = tl.sum(q[:, None] * kv_new, axis=0)
        denom = tl.sum(q * z_new) + eps
        out = num / denom

        out_ptrs = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_v * stride_ov
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty))


    @triton.jit
    def _triton_recurrent_gla_step_kernel(
        Q_ptr, K_ptr, V_ptr, Gk_ptr, KV_ptr, Out_ptr,
        B, H, Dk: tl.constexpr, Dv: tl.constexpr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_gkb, stride_gkh, stride_gkd,
        stride_kvb, stride_kvh, stride_kvk, stride_kvv,
        stride_ob, stride_oh, stride_ov,
        scale, IS_VECTOR_GATE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_k = tl.arange(0, Dk)
        offs_v = tl.arange(0, Dv)

        q = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_k * stride_qd).to(tl.float32)
        k = tl.load(K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_k * stride_kd).to(tl.float32)
        v = tl.load(V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_v * stride_vd).to(tl.float32)

        if IS_VECTOR_GATE:
            gk = tl.load(Gk_ptr + pid_b * stride_gkb + pid_h * stride_gkh + offs_k * stride_gkd).to(tl.float32)
            decay = tl.exp(gk)  # [Dk]
        else:
            g = tl.load(Gk_ptr + pid_b * stride_gkb + pid_h * stride_gkh).to(tl.float32)
            decay = tl.exp(g)

        kv_ptrs = KV_ptr + pid_b * stride_kvb + pid_h * stride_kvh + offs_k[:, None] * stride_kvk + offs_v[None, :] * stride_kvv
        kv = tl.load(kv_ptrs).to(tl.float32)

        # S_t = decay * S_{t-1} + k_t \otimes v_t
        if IS_VECTOR_GATE:
            kv_new = decay[:, None] * kv + k[:, None] * v[None, :]
        else:
            kv_new = decay * kv + k[:, None] * v[None, :]
        tl.store(kv_ptrs, kv_new.to(KV_ptr.dtype.element_ty))

        out = tl.sum(q[:, None] * kv_new, axis=0) * scale
        tl.store(Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_v * stride_ov, out.to(Out_ptr.dtype.element_ty))


def fused_recurrent_linear_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    kv_state: torch.Tensor,
    z_state: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step recurrent linear attention update: O(1) time and memory.

    Args:
        q_t, k_t: [B, H, Dk]
        v_t: [B, H, Dv]
        kv_state: [B, H, Dk, Dv]
        z_state: [B, H, Dk]
    Returns:
        (o_t, kv_state, z_state) where o_t is [B, H, Dv]
    """
    B, H, Dk = q_t.shape
    Dv = v_t.shape[-1]
    suitable, _ = check_triton_suitability(q_t, "state_update")

    if suitable and HAS_TRITON and Dk in (16, 32, 64, 128) and Dv in (16, 32, 64, 128):
        out = torch.empty((B, H, Dv), dtype=q_t.dtype, device=q_t.device)
        grid = (B, H)
        _triton_recurrent_linear_step_kernel[grid](
            q_t, k_t, v_t, kv_state, z_state, out,
            B, H, Dk, Dv,
            q_t.stride(0), q_t.stride(1), q_t.stride(2),
            k_t.stride(0), k_t.stride(1), k_t.stride(2),
            v_t.stride(0), v_t.stride(1), v_t.stride(2),
            kv_state.stride(0), kv_state.stride(1), kv_state.stride(2), kv_state.stride(3),
            z_state.stride(0), z_state.stride(1), z_state.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            eps,
        )
        return out, kv_state, z_state

    # Exact PyTorch Vectorized Fallback
    # S_t = S_{t-1} + k_t^T @ v_t
    kv_state = kv_state + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float()).to(kv_state.dtype)
    z_state = z_state + k_t
    num = torch.einsum("bhk,bhkv->bhv", q_t.float(), kv_state.float())
    denom = torch.einsum("bhk,bhk->bh", q_t.float(), z_state.float()).unsqueeze(-1) + eps
    out = (num / denom).to(q_t.dtype)
    return out, kv_state, z_state


def fused_recurrent_gla_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    gk_t: torch.Tensor,
    kv_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrent Gated Linear Attention update with elementwise or scalar decay.

    Args:
        q_t, k_t: [B, H, Dk]
        v_t: [B, H, Dv]
        gk_t: [B, H, Dk] or [B, H, 1] (log decay or decay factor)
        kv_state: [B, H, Dk, Dv]
    """
    B, H, Dk = q_t.shape
    Dv = v_t.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(Dk)

    is_vec = gk_t.shape[-1] == Dk
    suitable, _ = check_triton_suitability(q_t, "state_update")

    if suitable and HAS_TRITON and Dk in (16, 32, 64) and Dv in (16, 32, 64):
        out = torch.empty((B, H, Dv), dtype=q_t.dtype, device=q_t.device)
        grid = (B, H)
        _triton_recurrent_gla_step_kernel[grid](
            q_t, k_t, v_t, gk_t, kv_state, out,
            B, H, Dk, Dv,
            q_t.stride(0), q_t.stride(1), q_t.stride(2),
            k_t.stride(0), k_t.stride(1), k_t.stride(2),
            v_t.stride(0), v_t.stride(1), v_t.stride(2),
            gk_t.stride(0), gk_t.stride(1), (gk_t.stride(2) if is_vec else 0),
            kv_state.stride(0), kv_state.stride(1), kv_state.stride(2), kv_state.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            scale, IS_VECTOR_GATE=is_vec,
        )
        return out, kv_state

    # Exact PyTorch Fallback
    decay = torch.exp(gk_t.float()) if gk_t.dtype in (torch.float32, torch.float16, torch.bfloat16) else gk_t.float()
    if is_vec:
        decay = decay.unsqueeze(-1)  # [B, H, Dk, 1]
    else:
        decay = decay.view(B, H, 1, 1)

    kv_state = (decay * kv_state.float() + torch.einsum("bhk,bhv->bhkv", k_t.float(), v_t.float())).to(kv_state.dtype)
    out = (scale * torch.einsum("bhk,bhkv->bhv", q_t.float(), kv_state.float())).to(q_t.dtype)
    return out, kv_state


def fused_recurrent_gdn_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    kv_state: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrent Gated DeltaNet Householder update in O(1).

    Recurrence:
        v_pred = S_{t-1}^T @ k_t
        S_t = g_t * S_{t-1} + beta_t * k_t \otimes (v_t - v_pred)
        o_t = scale * (S_t^T @ q_t)
    """
    B, H, Dk = q_t.shape
    Dv = v_t.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(Dk)

    g = g_t.float()
    if g.dim() == 2:
        g = g.view(B, H, 1, 1)
    elif g.dim() == 3:
        g = g.unsqueeze(-1)

    beta = beta_t.float()
    if beta.dim() == 2:
        beta = beta.view(B, H, 1, 1)
    elif beta.dim() == 3:
        beta = beta.unsqueeze(-1)

    # v_pred: [B, H, Dv] = sum_k S_{t-1}[k, v] * k_t[k]
    s_float = kv_state.float()
    v_pred = torch.einsum("bhk,bhkv->bhv", k_t.float(), s_float)
    delta_v = v_t.float() - v_pred

    s_decayed = g * s_float
    s_new = s_decayed + beta * torch.einsum("bhk,bhv->bhkv", k_t.float(), delta_v)
    kv_state = s_new.to(kv_state.dtype)

    out = (scale * torch.einsum("bhk,bhkv->bhv", q_t.float(), s_new)).to(q_t.dtype)
    return out, kv_state


def fused_recurrent_hgrn2_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    f_t: torch.Tensor,
    kv_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step recurrent HGRN2 lower-bound gated update:
    S_t = f_t * S_{t-1} + ((1 - f_t) * k_t) \otimes v_t
    o_t = S_t^T @ q_t
    """
    B, H, Dk = q_t.shape
    Dv = v_t.shape[-1]

    f = f_t.float()
    if f.dim() == 3 and f.shape[-1] == Dk:
        f = f.unsqueeze(-1)  # [B, H, Dk, 1]
    elif f.dim() == 2:
        f = f.view(B, H, 1, 1)

    s_float = kv_state.float()
    k_gated = (1.0 - f) * k_t.float().unsqueeze(-1)  # [B, H, Dk, 1]
    update = k_gated * v_t.float().unsqueeze(-2)      # [B, H, Dk, Dv]

    s_new = f * s_float + update
    kv_state = s_new.to(kv_state.dtype)

    out = torch.einsum("bhk,bhkv->bhv", q_t.float(), s_new).to(q_t.dtype)
    return out, kv_state


def causal_conv1d_step(
    x_t: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step 1D depthwise causal convolution state update:
    conv_state: [B, D_inner, kernel_size - 1]
    x_t: [B, D_inner] or [B, 1, D_inner]
    """
    if x_t.dim() == 3:
        x_t = x_t.squeeze(1)

    # conv_weight: [D_inner, 1, K] or [D_inner, K]
    w = conv_weight.squeeze(1) if conv_weight.dim() == 3 else conv_weight
    K = w.shape[-1]

    # Shift state: [B, D_inner, K-1]
    state_seq = torch.cat([conv_state, x_t.unsqueeze(-1)], dim=-1)  # [B, D_inner, K]
    next_state = state_seq[:, :, 1:]                               # [B, D_inner, K-1]

    out = (state_seq * w.unsqueeze(0)).sum(dim=-1)
    if conv_bias is not None:
        out = out + conv_bias
    return out, next_state


def fused_recurrent_ssm_step(
    x_t: torch.Tensor,
    dt_t: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    D_param: torch.Tensor | None,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step Selective State Space Model update:
    h_t = exp(dt * A) * h_{t-1} + (dt * B_t) * x_t
    y_t = sum(h_t * C_t) + D * x_t
    """
    if x_t.dim() == 3:
        x_t = x_t.squeeze(1)
    if dt_t.dim() == 3:
        dt_t = dt_t.squeeze(1)

    # A: [D_inner, D_state] or [D_inner]
    # dt_t: [B, D_inner]
    dt = dt_t.float()
    if A.dim() == 2:
        dA = torch.exp(dt.unsqueeze(-1) * A.float().unsqueeze(0))  # [B, D_inner, D_state]
    else:
        dA = torch.exp(dt * A.float()).unsqueeze(-1)

    # B_t: [B, D_state] or [B, G, D_state]
    if B_t.dim() == 2:
        dB = dt.unsqueeze(-1) * B_t.float().unsqueeze(1)  # [B, D_inner, D_state]
    else:
        dB = dt.unsqueeze(-1) * B_t.float()

    h_new = dA * ssm_state.float() + dB * x_t.float().unsqueeze(-1)
    ssm_state = h_new.to(ssm_state.dtype)

    if C_t.dim() == 2:
        y = (h_new * C_t.float().unsqueeze(1)).sum(dim=-1)  # [B, D_inner]
    else:
        y = (h_new * C_t.float()).sum(dim=-1)

    if D_param is not None:
        y = y + D_param.float() * x_t.float()

    return y.to(x_t.dtype), ssm_state


__all__ = [
    'is_supported',
    'fwd',
    'bwd',
    'fused_recurrent_linear_step',
    'fused_recurrent_gla_step',
    'fused_recurrent_gdn_step',
    'fused_recurrent_hgrn2_step',
    'causal_conv1d_step',
    'fused_recurrent_ssm_step',
]
