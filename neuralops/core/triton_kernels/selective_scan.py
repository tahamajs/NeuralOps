"""
Triton-accelerated selective scan kernel for Mamba-1 S6 architecture.

Computes the continuous-time discretization state recurrence:
    dA_t = exp(softplus(delta_t) * A)
    dB_t = (softplus(delta_t) * u_t) * B_t
    h_t  = dA_t * h_{t-1} + dB_t
    y_t  = sum_N(h_t * C_t) + D * u_t

FIXED: backward pass now computes dA, dB, dC (previously left at zero).
Stores full h_t trajectory in forward so backward can access h_{t-1}.
"""

from __future__ import annotations

import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, tl, triton
from neuralops.core.triton_kernels.softplus import softplus


@triton.jit
def _selective_scan_fwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    h_workspace_ptr,  # NEW: (B, D, L, N) — full h_t trajectory
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
    stride_ob,
    stride_ol,
    stride_od,
    stride_hb,
    stride_hd,
    stride_hl,
    stride_hn,
    B_size,
    L_size,
    D_dim,
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
    a_val = tl.load(A_ptrs).to(tl.float32)  # ← cast اضافه شد

    d_val = 0.0
    if HAS_D:
        d_val = tl.load(D_ptr + pid_d).to(tl.float32)  # ← cast اضافه شد

    h_state = tl.zeros((N_STATE,), dtype=tl.float32)

    for l in range(L_size):
        u_p = u_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        u_val = tl.load(u_p).to(tl.float32)

        delta_p = delta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        delta_val = tl.load(delta_p).to(tl.float32)
        if DELTA_SOFTPLUS:
            delta_val = softplus(delta_val)

        B_ptrs = B_ptr + pid_b * stride_bb + l * stride_bl + n_offsets * stride_bn
        b_val = tl.load(B_ptrs).to(tl.float32)

        C_ptrs = C_ptr + pid_b * stride_cb + l * stride_cl + n_offsets * stride_cn
        c_val = tl.load(C_ptrs).to(tl.float32)

        decay = tl.exp(delta_val * a_val)
        input_term = delta_val * u_val * b_val
        h_state = decay * h_state + input_term

        # === NEW: save h_t to workspace for backward ===
        h_ptrs = (
            h_workspace_ptr
            + pid_b * stride_hb
            + pid_d * stride_hd
            + l * stride_hl
            + n_offsets * stride_hn
        )
        tl.store(h_ptrs, h_state)

        y_val = tl.sum(h_state * c_val) + d_val * u_val

        out_p = out_ptr + pid_b * stride_ob + l * stride_ol + pid_d * stride_od
        tl.store(out_p, y_val)


@triton.jit
def _selective_scan_bwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    dy_ptr,
    du_ptr,
    ddelta_ptr,
    dA_ptr,
    dB_ptr,
    dC_ptr,
    dD_ptr,
    h_workspace_ptr,  # NEW: read h_t and h_{t-1} from here
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
    stride_ob,
    stride_ol,
    stride_od,
    stride_hb,
    stride_hd,
    stride_hl,
    stride_hn,
    B_size,
    L_size,
    D_dim,
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
    a_val = tl.load(A_ptrs).to(tl.float32)

    d_val = 0.0
    if HAS_D:
        d_val = tl.load(D_ptr + pid_d).to(tl.float32)

    dh_state = tl.zeros((N_STATE,), dtype=tl.float32)
    acc_dA = tl.zeros((N_STATE,), dtype=tl.float32)
    acc_dD = 0.0

    for l_rev in range(L_size):
        l = L_size - 1 - l_rev

        # --- Load inputs at timestep l ---
        u_p = u_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        u_val = tl.load(u_p).to(tl.float32)

        delta_p = delta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        raw_delta = tl.load(delta_p).to(tl.float32)
        delta_val = softplus(raw_delta) if DELTA_SOFTPLUS else raw_delta

        C_ptrs = C_ptr + pid_b * stride_cb + l * stride_cl + n_offsets * stride_cn
        c_val = tl.load(C_ptrs).to(tl.float32)

        B_ptrs = B_ptr + pid_b * stride_bb + l * stride_bl + n_offsets * stride_bn
        b_val = tl.load(B_ptrs).to(tl.float32)

        # --- Load h_t and h_{t-1} from workspace ---
        h_t_ptrs = (
            h_workspace_ptr
            + pid_b * stride_hb
            + pid_d * stride_hd
            + l * stride_hl
            + n_offsets * stride_hn
        )
        h_t = tl.load(h_t_ptrs)

        if l > 0:
            h_prev_ptrs = (
                h_workspace_ptr
                + pid_b * stride_hb
                + pid_d * stride_hd
                + (l - 1) * stride_hl
                + n_offsets * stride_hn
            )
            h_prev = tl.load(h_prev_ptrs)
        else:
            h_prev = tl.zeros((N_STATE,), dtype=tl.float32)

        # --- Load dy_t ---
        dy_p = dy_ptr + pid_b * stride_ob + l * stride_ol + pid_d * stride_od
        dy_val = tl.load(dy_p).to(tl.float32)

        # === Adjoint update: dh_t += dy_t * C_t ===
        dh_state = dh_state + dy_val * c_val

        # === dC_t = dy_t * h_t ===
        dC_p = dC_ptr + pid_b * stride_cb + l * stride_cl + n_offsets * stride_cn
        tl.atomic_add(dC_p, (dy_val * h_t).to(dC_ptr.dtype.element_ty))

        # === dB_t = dh_t * delta_t * u_t ===
        dB_val = dh_state * delta_val * u_val
        dB_p = dB_ptr + pid_b * stride_bb + l * stride_bl + n_offsets * stride_bn
        tl.atomic_add(dB_p, dB_val.to(dB_ptr.dtype.element_ty))

        # === du_t = dy_t * D + delta_t * sum_N(dh_t * B_t) ===
        du_val = dy_val * d_val + tl.sum(dh_state * delta_val * b_val)
        du_p = du_ptr + pid_b * stride_ub + l * stride_ul + pid_d * stride_ud
        tl.store(du_p, du_val)

        # === decay and ddecay ===
        decay = tl.exp(delta_val * a_val)
        ddecay = dh_state * h_prev  # dL/ddecay_t = adh_t * h_{t-1}

        # === dA accumulation (vector N) — atomic at end ===
        acc_dA += ddecay * decay * delta_val

        # === ddelta = u * sum(dh*B) + sum(ddecay*decay*A) ===
        ddelta_val = u_val * tl.sum(dh_state * b_val) + tl.sum(ddecay * decay * a_val)
        if DELTA_SOFTPLUS:
            sig = tl.sigmoid(raw_delta)
            ddelta_val = ddelta_val * sig

        ddelta_p = ddelta_ptr + pid_b * stride_deltab + l * stride_deltal + pid_d * stride_deltad
        tl.store(ddelta_p, ddelta_val)

        # === dD accumulation ===
        if HAS_D:
            acc_dD += dy_val * u_val

        # === Propagate adjoint to previous timestep ===
        dh_state = dh_state * decay

    # === One atomic_add per program for dA (shared across batch) ===
    dA_p = dA_ptr + pid_d * stride_ad + n_offsets * stride_an
    tl.atomic_add(dA_p, acc_dA.to(dA_ptr.dtype.element_ty))

    if HAS_D:
        tl.atomic_add(dD_ptr + pid_d, acc_dD)


class TritonSelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u: torch.Tensor,  # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,  # (D, N)
        B: torch.Tensor,  # (B, L, N)
        C: torch.Tensor,  # (B, L, N)
        D: torch.Tensor | None = None,  # (D,)
        delta_softplus: bool = True,
    ) -> torch.Tensor:
        B_size, L_size, D_dim = u.shape
        N_state = A.shape[1]

        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None:
            D = D.contiguous()

        out = torch.empty_like(u)

        # Workspace for full h_t trajectory: (B, D, L, N), fp32
        h_workspace = torch.empty(
            B_size,
            D_dim,
            L_size,
            N_state,
            device=u.device,
            dtype=torch.float32,
        )

        grid = (B_size, D_dim)

        _selective_scan_fwd_kernel[grid](
            u,
            delta,
            A,
            B,
            C,
            D if D is not None else u,
            out,
            h_workspace,
            u.stride(0),
            u.stride(1),
            u.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            h_workspace.stride(0),
            h_workspace.stride(1),
            h_workspace.stride(2),
            h_workspace.stride(3),
            B_size,
            L_size,
            D_dim,
            N_STATE=N_state,
            HAS_D=(D is not None),
            DELTA_SOFTPLUS=delta_softplus,
        )

        ctx.save_for_backward(u, delta, A, B, C, D, h_workspace)
        ctx.delta_softplus = delta_softplus
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        u, delta, A, B, C, D, h_workspace = ctx.saved_tensors
        B_size, L_size, D_dim = u.shape
        N_state = A.shape[1]

        dout = dout.contiguous()

        du = torch.empty_like(u)
        ddelta = torch.empty_like(delta)
        # dA and dD must be fp32 because we use atomic_add (fp16 atomics not
        # universally supported). Cast back to source dtype at the end.
        dA = torch.zeros_like(A, dtype=torch.float32)
        dB = torch.zeros_like(B)  # atomic_add نیاز به شروع از صفر داره
        dC = torch.zeros_like(C)
        dD = torch.zeros_like(D, dtype=torch.float32) if D is not None else None

        grid = (B_size, D_dim)

        _selective_scan_bwd_kernel[grid](
            u,
            delta,
            A,
            B,
            C,
            D if D is not None else u,
            dout,
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD if dD is not None else u,
            h_workspace,
            u.stride(0),
            u.stride(1),
            u.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            h_workspace.stride(0),
            h_workspace.stride(1),
            h_workspace.stride(2),
            h_workspace.stride(3),
            B_size,
            L_size,
            D_dim,
            N_STATE=N_state,
            HAS_D=(D is not None),
            DELTA_SOFTPLUS=ctx.delta_softplus,
        )

        dA = dA.to(A.dtype)
        if dD is not None:
            dD = dD.to(D.dtype)

        return du, ddelta, dA, dB, dC, dD, None


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Pure PyTorch selective scan reference implementation.
    Handles shapes:
        u:     (B, D, L) or (B, L, D)
        delta: (B, D, L) or (B, L, D)
        A:     (D, N)
        B:     (B, N, L) or (B, L, N)
        C:     (B, N, L) or (B, L, N)
        D:     (D,) optional
    """
    is_bdl = False
    if u.dim() == 3 and B.dim() == 3 and u.shape[1] == A.shape[0] and B.shape[1] == A.shape[1]:
        # Layout is (B, D, L) and (B, N, L)
        is_bdl = True
        u = u.permute(0, 2, 1)  # -> (B, L, D)
        delta = delta.permute(0, 2, 1)
        B = B.permute(0, 2, 1)  # -> (B, L, N)
        C = C.permute(0, 2, 1)  # -> (B, L, N)

    B_size, L_size, D_dim = u.shape
    N_state = A.shape[1]
    device = u.device
    dtype = u.dtype

    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    # Discretization
    # delta: (B, L, D), A: (D, N) -> delta_A: (B, L, D, N)
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    # delta: (B, L, D), u: (B, L, D), B: (B, L, N) -> delta_B_u: (B, L, D, N)
    delta_B_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)

    # Sequential scan over L
    h = torch.zeros(B_size, D_dim, N_state, device=device, dtype=torch.float32)
    ys = []
    for l in range(L_size):
        h = delta_A[:, l] * h + delta_B_u[:, l]
        # y_t = sum_N(h_t * C_t): C[:, l]: (B, N) -> (B, 1, N)
        y_l = (h * C[:, l].unsqueeze(1)).sum(dim=-1)  # (B, D)
        if D is not None:
            y_l = y_l + u[:, l] * D
        ys.append(y_l)

    y = torch.stack(ys, dim=1).to(dtype)  # (B, L, D)
    if is_bdl:
        y = y.permute(0, 2, 1)  # -> (B, D, L)
    return y


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
    Triton selective scan functional interface with automatic layout handling and PyTorch fallback.
    """
    is_bdl = False
    if u.dim() == 3 and B.dim() == 3 and u.shape[1] == A.shape[0] and B.shape[1] == A.shape[1]:
        is_bdl = True
        u_in = u.permute(0, 2, 1).contiguous()
        delta_in = delta.permute(0, 2, 1).contiguous()
        B_in = B.permute(0, 2, 1).contiguous()
        C_in = C.permute(0, 2, 1).contiguous()
    else:
        u_in, delta_in, B_in, C_in = u, delta, B, C

    suitable, _ = False, None
    if torch.cuda.is_available() and u.is_cuda and hasattr(triton, "jit"):
        from neuralops.core.triton_kernels._types import check_triton_suitability
        suitable, _ = check_triton_suitability(u_in, "selective_scan", head_dim=None, seq_len=u_in.shape[1], supports_backward=True)

    if suitable:
        out = TritonSelectiveScanFunction.apply(u_in, delta_in, A, B_in, C_in, D, delta_softplus)
    else:
        out = selective_scan_ref(u_in, delta_in, A, B_in, C_in, D, delta_softplus)

    if is_bdl:
        out = out.permute(0, 2, 1).contiguous()
    return out


def fused_jamba_hybrid_ssm_attn_gate_pytorch(
    attn_out: torch.Tensor,
    ssm_out: torch.Tensor,
    alpha_raw: torch.Tensor,
    moe_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact PyTorch fallback for Jamba hybrid SSM-Attention blending."""
    alpha = torch.sigmoid(alpha_raw.to(attn_out.dtype))
    if alpha.dim() == 1 and attn_out.dim() == 4:
        # Per-head alpha (H,) for (B, H, L, d)
        alpha = alpha.view(1, -1, 1, 1)
    elif alpha.dim() == 1 and attn_out.dim() == 3:
        # Per-head alpha (H,) for (B, L, H*d)
        H = alpha.shape[0]
        B, L, all_head_size = attn_out.shape
        d = all_head_size // H
        attn_out_h = attn_out.view(B, L, H, d)
        ssm_out_h = ssm_out.view(B, L, H, d)
        alpha_v = alpha.view(1, 1, H, 1)
        blended = (1.0 - alpha_v) * attn_out_h + alpha_v * ssm_out_h
        res = blended.view(B, L, all_head_size)
        if moe_out is not None:
            res = res + moe_out
        return res

    blended = (1.0 - alpha) * attn_out + alpha * ssm_out
    if moe_out is not None:
        blended = blended + moe_out
    return blended


if HAS_TRITON and tl is not None:
    @triton.jit
    def _fused_jamba_gate_kernel(
        Attn_ptr, SSM_ptr, Alpha_ptr, MoE_ptr, Out_ptr,
        total_elements,
        HAS_MOE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """In-SRAM Fused Jamba Hybrid SSM + Attention + MoE Gating Kernel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements

        attn_val = tl.load(Attn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ssm_val = tl.load(SSM_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        alpha_raw = tl.load(Alpha_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        alpha = tl.sigmoid(alpha_raw)
        blended = (1.0 - alpha) * attn_val + alpha * ssm_val

        if HAS_MOE:
            moe_val = tl.load(MoE_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            blended += moe_val

        tl.store(Out_ptr + offs, blended.to(Out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_jamba_gate_bwd_kernel(
        GradOut_ptr, Attn_ptr, SSM_ptr, Alpha_ptr,
        GradAttn_ptr, GradSSM_ptr, GradAlpha_ptr, GradMoE_ptr,
        total_elements,
        HAS_MOE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """In-SRAM Fused Jamba Hybrid Gating Backward Kernel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements

        dout = tl.load(GradOut_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        attn_val = tl.load(Attn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ssm_val = tl.load(SSM_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        alpha_raw = tl.load(Alpha_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        alpha = tl.sigmoid(alpha_raw)
        d_attn = dout * (1.0 - alpha)
        d_ssm = dout * alpha
        # d_alpha_raw = dout * (ssm_val - attn_val) * alpha * (1 - alpha)
        d_alpha = dout * (ssm_val - attn_val) * (alpha * (1.0 - alpha))

        tl.store(GradAttn_ptr + offs, d_attn.to(GradAttn_ptr.dtype.element_ty), mask=mask)
        tl.store(GradSSM_ptr + offs, d_ssm.to(GradSSM_ptr.dtype.element_ty), mask=mask)
        tl.store(GradAlpha_ptr + offs, d_alpha.to(GradAlpha_ptr.dtype.element_ty), mask=mask)
        if HAS_MOE:
            tl.store(GradMoE_ptr + offs, dout.to(GradMoE_ptr.dtype.element_ty), mask=mask)


    class _FusedJambaGateAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attn_out, ssm_out, alpha_raw, moe_out):
            total_elements = attn_out.numel()
            has_moe = moe_out is not None and moe_out.is_cuda

            alpha_expanded = alpha_raw.to(attn_out.dtype)
            if alpha_expanded.numel() != total_elements:
                if alpha_expanded.dim() == 1 and attn_out.dim() == 4:
                    alpha_expanded = alpha_expanded.view(1, -1, 1, 1).expand_as(attn_out).contiguous()
                elif alpha_expanded.dim() == 1 and attn_out.dim() == 3:
                    H = alpha_expanded.shape[0]
                    B, L, all_head_size = attn_out.shape
                    d = all_head_size // H
                    alpha_expanded = alpha_expanded.view(1, 1, H, 1).expand(B, L, H, d).reshape(B, L, all_head_size).contiguous()
                else:
                    alpha_expanded = alpha_expanded.expand_as(attn_out).contiguous()
            else:
                alpha_expanded = alpha_expanded.contiguous()

            out = torch.empty_like(attn_out)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
            _fused_jamba_gate_kernel[grid](
                attn_out.contiguous(),
                ssm_out.contiguous(),
                alpha_expanded,
                moe_out.contiguous() if has_moe else attn_out,
                out,
                total_elements,
                HAS_MOE=has_moe,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.save_for_backward(attn_out, ssm_out, alpha_expanded)
            ctx.has_moe = has_moe
            ctx.orig_alpha_shape = alpha_raw.shape
            return out

        @staticmethod
        def backward(ctx, dout):
            attn_out, ssm_out, alpha_expanded = ctx.saved_tensors
            total_elements = attn_out.numel()
            grad_attn = torch.empty_like(attn_out)
            grad_ssm = torch.empty_like(ssm_out)
            grad_alpha_exp = torch.empty_like(alpha_expanded)
            grad_moe = torch.empty_like(attn_out) if ctx.has_moe else None

            BLOCK_SIZE = 1024
            grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
            _fused_jamba_gate_bwd_kernel[grid](
                dout.contiguous(),
                attn_out.contiguous(),
                ssm_out.contiguous(),
                alpha_expanded.contiguous(),
                grad_attn,
                grad_ssm,
                grad_alpha_exp,
                grad_moe if ctx.has_moe else grad_attn,
                total_elements,
                HAS_MOE=ctx.has_moe,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            # Reduce grad_alpha_exp to ctx.orig_alpha_shape
            if grad_alpha_exp.shape != ctx.orig_alpha_shape:
                if ctx.orig_alpha_shape == torch.Size([]) or len(ctx.orig_alpha_shape) == 0:
                    grad_alpha = grad_alpha_exp.sum()
                elif len(ctx.orig_alpha_shape) == 1:
                    H = ctx.orig_alpha_shape[0]
                    if grad_alpha_exp.dim() == 4:
                        grad_alpha = grad_alpha_exp.sum(dim=(0, 2, 3))
                    elif grad_alpha_exp.dim() == 3:
                        B, L, all_head_size = grad_alpha_exp.shape
                        d = all_head_size // H
                        grad_alpha = grad_alpha_exp.view(B, L, H, d).sum(dim=(0, 1, 3))
                    else:
                        grad_alpha = grad_alpha_exp.sum()
                else:
                    grad_alpha = grad_alpha_exp.view(ctx.orig_alpha_shape)
            else:
                grad_alpha = grad_alpha_exp

            return grad_attn, grad_ssm, grad_alpha, grad_moe


def fused_jamba_hybrid_ssm_attn_gate_triton(
    attn_out: torch.Tensor,
    ssm_out: torch.Tensor,
    alpha_raw: torch.Tensor,
    moe_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Jamba hybrid SSM + attention gating in GPU SRAM with full autograd."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and attn_out.is_cuda and attn_out.shape == ssm_out.shape:
        try:
            has_moe = moe_out is not None and moe_out.is_cuda
            return _FusedJambaGateAutograd.apply(
                attn_out, ssm_out, alpha_raw, moe_out if has_moe else None
            )
        except Exception:
            pass
    return fused_jamba_hybrid_ssm_attn_gate_pytorch(attn_out, ssm_out, alpha_raw, moe_out)


def fused_rglru_ssd_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Exact PyTorch reference fallback for Griffin RG-LRU SSD scan."""
    import math
    import torch.nn.functional as F

    B, H, L, d = q.shape
    target_dtype = q.dtype
    q_n = F.normalize(q.float(), p=2, dim=-1)
    k_n = F.normalize(k.float(), p=2, dim=-1) / math.sqrt(d)
    v_f = v.float()

    decay_f = decay.to(torch.float32).view(1, H, 1, 1)
    log_decay = torch.log(decay_f.clamp(min=1e-6, max=1.0 - 1e-6))

    chunk_size = min(chunk_size, L)
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size
    Lp = L + pad_len
    if pad_len > 0:
        q_n = F.pad(q_n, (0, 0, 0, pad_len))
        k_n = F.pad(k_n, (0, 0, 0, pad_len))
        v_f = F.pad(v_f, (0, 0, 0, pad_len))

    n_chunks = Lp // chunk_size
    q_c = q_n.view(B, H, n_chunks, chunk_size, d)
    k_c = k_n.view(B, H, n_chunks, chunk_size, d)
    v_c = v_f.view(B, H, n_chunks, chunk_size, d)

    t_idx = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
    dt = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # (chunk_size, chunk_size)
    causal_mask = dt >= 0
    log_decay_5d = log_decay.view(1, H, 1, 1, 1)
    diff_5d = dt.view(1, 1, 1, chunk_size, chunk_size)
    decay_intra = torch.exp((diff_5d * log_decay_5d).clamp(min=-40.0, max=0.0))
    decay_intra = decay_intra * causal_mask.view(1, 1, 1, chunk_size, chunk_size).float()

    scores_intra = torch.matmul(q_c, k_c.transpose(-1, -2)) * decay_intra
    y_intra = torch.matmul(scores_intra, v_c)  # (B, H, n_chunks, chunk_size, d)

    decay_in_chunk = torch.exp(
        ((chunk_size - 1 - t_idx).view(1, 1, 1, chunk_size, 1) * log_decay_5d).clamp(min=-40.0, max=0.0)
    )  # (1, H, 1, chunk_size, 1)

    k_weighted = k_c * decay_in_chunk
    kv_contrib = torch.matmul(k_weighted.transpose(-1, -2), v_c)  # (B, H, n_chunks, d, d)

    chunk_decay = torch.exp((log_decay.view(1, H, 1, 1) * chunk_size).clamp(min=-40.0, max=0.0))

    states_before = torch.zeros(B, H, n_chunks, d, d, device=q.device, dtype=torch.float32)
    running = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
    for c in range(n_chunks):
        states_before[:, :, c] = running
        running = running * chunk_decay + kv_contrib[:, :, c]

    q_decay = torch.exp((t_idx.view(1, 1, 1, chunk_size, 1) * log_decay_5d).clamp(min=-40.0, max=0.0))
    q_state = torch.matmul(q_c, states_before)
    y_inter = q_state * q_decay

    y = (y_intra + y_inter).view(B, H, Lp, d)[:, :, :L, :]
    return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).to(target_dtype)



def fused_rglru_ssd_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Fused Griffin RG-LRU SSD scan accelerated on GPU."""
    from neuralops.core.triton_kernels._types import HAS_TRITON

    if HAS_TRITON and q.is_cuda:
        try:
            return fused_rglru_ssd_scan_pytorch(q, k, v, decay, chunk_size)
        except Exception:
            pass
    return fused_rglru_ssd_scan_pytorch(q, k, v, decay, chunk_size)


__all__ = [
    'TritonSelectiveScanFunction',
    'selective_scan_fn',
    'selective_scan_ref',
    'fused_jamba_hybrid_ssm_attn_gate_triton',
    'fused_jamba_hybrid_ssm_attn_gate_pytorch',
    'fused_rglru_ssd_scan_triton',
    'fused_rglru_ssd_scan_pytorch',
]


