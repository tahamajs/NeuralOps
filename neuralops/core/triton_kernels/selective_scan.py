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

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = type(
        "triton",
        (),
        {
            "jit": lambda f: f,
            "heuristics": lambda *a, **kw: lambda f: f,
            "autotune": lambda *a, **kw: lambda f: f,
        },
    )
    tl = None
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
    Shapes:
        u:     (B, L, D)
        delta: (B, L, D)
        A:     (D, N)
        B:     (B, L, N)
        C:     (B, L, N)
        D:     (D,) optional
    """
    return TritonSelectiveScanFunction.apply(u, delta, A, B, C, D, delta_softplus)
