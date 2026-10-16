"""Triton-accelerated HGRN2 selective scan.

Element-wise recurrence:
    h_t = f_t * h_{t-1} + iv_t
    y_t = h_t

Backward:
    dh_t = d_h_t + f_{t+1} * dh_{t+1}    (adjoint recurrence)
    d_f_t = dh_t * h_{t-1}
    d_iv_t = dh_t

Note: f, iv, h all have shape (B, L, D). No state-dim mixing.
"""

from __future__ import annotations

import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, tl, triton


@triton.jit
def _hgrn2_fwd_kernel(
    f_ptr,
    iv_ptr,
    h_out_ptr,
    h_workspace_ptr,
    stride_fb,
    stride_fl,
    stride_fd,
    stride_ivb,
    stride_ivl,
    stride_ivd,
    stride_hob,
    stride_hol,
    stride_hod,
    stride_hwb,
    stride_hwl,
    stride_hwd,
    B_size,
    L_size,
    D_dim,
    BLOCK_D: tl.constexpr,
    DO_CLIP: tl.constexpr,
    CLIP_VAL: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B_size:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_dim

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for l in range(L_size):
        f_ptrs = f_ptr + pid_b * stride_fb + l * stride_fl + offs_d * stride_fd
        iv_ptrs = iv_ptr + pid_b * stride_ivb + l * stride_ivl + offs_d * stride_ivd

        f_t = tl.load(f_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        iv_t = tl.load(iv_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        h = f_t * h + iv_t

        if DO_CLIP:
            h = tl.minimum(tl.maximum(h, -CLIP_VAL), CLIP_VAL)

        hw_ptrs = h_workspace_ptr + pid_b * stride_hwb + l * stride_hwl + offs_d * stride_hwd
        tl.store(hw_ptrs, h, mask=mask_d)

        ho_ptrs = h_out_ptr + pid_b * stride_hob + l * stride_hol + offs_d * stride_hod
        tl.store(ho_ptrs, h.to(h_out_ptr.dtype.element_ty), mask=mask_d)


@triton.jit
def _hgrn2_bwd_kernel(
    f_ptr,
    h_workspace_ptr,
    dh_fwd_ptr,
    df_ptr,
    div_ptr,
    stride_fb,
    stride_fl,
    stride_fd,
    stride_hwb,
    stride_hwl,
    stride_hwd,
    stride_dhb,
    stride_dhl,
    stride_dhd,
    stride_dfb,
    stride_dfl,
    stride_dfd,
    stride_divb,
    stride_divl,
    stride_divd,
    B_size,
    L_size,
    D_dim,
    BLOCK_D: tl.constexpr,
    DO_CLIP: tl.constexpr,
    CLIP_VAL: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B_size:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_dim

    dh_clipped = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for l_rev in range(L_size):
        l = L_size - 1 - l_rev

        # External gradient at timestep l
        dh_ptrs = dh_fwd_ptr + pid_b * stride_dhb + l * stride_dhl + offs_d * stride_dhd
        dh_ext = tl.load(dh_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dh_clipped = dh_clipped + dh_ext

        # Undo the clip: gradient flows only through un-clipped entries
        if DO_CLIP:
            h_l_ptrs = h_workspace_ptr + pid_b * stride_hwb + l * stride_hwl + offs_d * stride_hwd
            h_l = tl.load(h_l_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            clip_mask = (tl.abs(h_l) < CLIP_VAL).to(tl.float32)
            dh_raw = dh_clipped * clip_mask
        else:
            dh_raw = dh_clipped

        # Load f_l
        f_ptrs = f_ptr + pid_b * stride_fb + l * stride_fl + offs_d * stride_fd
        f_l = tl.load(f_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Load h_{l-1}
        if l > 0:
            h_prev_ptrs = (
                h_workspace_ptr + pid_b * stride_hwb + (l - 1) * stride_hwl + offs_d * stride_hwd
            )
            h_prev = tl.load(h_prev_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        else:
            h_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Gradients
        df_val = dh_raw * h_prev
        div_val = dh_raw

        df_ptrs = df_ptr + pid_b * stride_dfb + l * stride_dfl + offs_d * stride_dfd
        div_ptrs = div_ptr + pid_b * stride_divb + l * stride_divl + offs_d * stride_divd
        tl.store(df_ptrs, df_val.to(df_ptr.dtype.element_ty), mask=mask_d)
        tl.store(div_ptrs, div_val.to(div_ptr.dtype.element_ty), mask=mask_d)

        # Propagate adjoint to t-1
        dh_clipped = dh_raw * f_l


class _HGRN2ScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, iv, res_clip):
        B, L, D = f.shape
        out = torch.empty_like(iv)
        h_workspace = torch.empty((B, L, D), device=f.device, dtype=torch.float32)

        BLOCK_D = 64
        num_d_blocks = triton.cdiv(D, BLOCK_D)
        grid = (num_d_blocks, B)

        do_clip = res_clip is not None and res_clip > 0
        clip_val = float(res_clip) if do_clip else 1.0

        _hgrn2_fwd_kernel[grid](
            f,
            iv,
            out,
            h_workspace,
            f.stride(0),
            f.stride(1),
            f.stride(2),
            iv.stride(0),
            iv.stride(1),
            iv.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            h_workspace.stride(0),
            h_workspace.stride(1),
            h_workspace.stride(2),
            B,
            L,
            D,
            BLOCK_D=BLOCK_D,
            DO_CLIP=do_clip,
            CLIP_VAL=clip_val,
        )

        ctx.save_for_backward(f, h_workspace)
        ctx.res_clip = res_clip
        return out

    @staticmethod
    def backward(ctx, d_out):
        f, h_workspace = ctx.saved_tensors
        B, L, D = f.shape
        d_out = d_out.contiguous()

        df = torch.empty_like(f)
        div = torch.empty_like(f)

        BLOCK_D = 64
        num_d_blocks = triton.cdiv(D, BLOCK_D)
        grid = (num_d_blocks, B)

        do_clip = ctx.res_clip is not None and ctx.res_clip > 0
        clip_val = float(ctx.res_clip) if do_clip else 1.0

        _hgrn2_bwd_kernel[grid](
            f,
            h_workspace,
            d_out,
            df,
            div,
            f.stride(0),
            f.stride(1),
            f.stride(2),
            h_workspace.stride(0),
            h_workspace.stride(1),
            h_workspace.stride(2),
            d_out.stride(0),
            d_out.stride(1),
            d_out.stride(2),
            df.stride(0),
            df.stride(1),
            df.stride(2),
            div.stride(0),
            div.stride(1),
            div.stride(2),
            B,
            L,
            D,
            BLOCK_D=BLOCK_D,
            DO_CLIP=do_clip,
            CLIP_VAL=clip_val,
        )

        return df, div, None


def hgrn2_scan_pytorch(f: torch.Tensor, iv: torch.Tensor, res_clip: float | None = None) -> torch.Tensor:
    """
    Pure PyTorch reference implementation for HGRN2 selective scan.
    Recurrence: h_t = f_t * h_{t-1} + iv_t
    """
    B, L, D = f.shape
    device = f.device
    dtype = f.dtype

    # Transpose to (B, D, L) for sequential time step iteration
    h = torch.zeros((B, D), device=device, dtype=torch.float32)
    ys = []

    for t in range(L):
        f_t = f[:, t, :].float()
        iv_t = iv[:, t, :].float()
        h = f_t * h + iv_t
        if res_clip is not None and res_clip > 0:
            h = torch.clamp(h, -float(res_clip), float(res_clip))
        ys.append(h)

    return torch.stack(ys, dim=1).to(dtype)


def hgrn2_scan_triton(f: torch.Tensor, iv: torch.Tensor, res_clip=None) -> torch.Tensor:
    """Triton HGRN2 selective scan. f, iv: (B, L, D). Returns h: (B, L, D)."""
    return _HGRN2ScanFn.apply(f, iv, res_clip)


def hgrn2_scan(f: torch.Tensor, iv: torch.Tensor, res_clip=None) -> torch.Tensor:
    """Dispatches HGRN2 scan to Triton GPU kernel when suitable, else PyTorch fallback."""
    from neuralops.core.triton_kernels._types import check_triton_suitability
    suitable, _ = check_triton_suitability(f, "hgrn2_scan", head_dim=f.shape[-1], seq_len=f.shape[1], supports_backward=True)
    if suitable and hasattr(triton, "jit"):
        try:
            return hgrn2_scan_triton(f, iv, res_clip=res_clip)
        except Exception:
            pass
    return hgrn2_scan_pytorch(f, iv, res_clip=res_clip)


# ==============================================================================
# PHASE 3 / KERNEL 10: Fused Bidirectional HGRN2 Recurrent Scan Kernel
# Target: hgrn2_attention, mamba_block, bidirectional SSM
# ==============================================================================

if HAS_TRITON and tl is not None:
    @triton.jit
    def _hgrn2_bidirectional_scan_kernel(
        x_ptr,
        g_ptr,
        y_ptr,
        B,
        H,
        L,
        D: tl.constexpr,
        stride_xb,
        stride_xh,
        stride_xl,
        stride_xd,
        stride_gb,
        stride_gh,
        stride_gl,
        stride_gd,
        stride_yb,
        stride_yh,
        stride_yl,
        stride_yd,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_d = tl.arange(0, D)

        h_fwd = tl.zeros([D], dtype=tl.float32)
        for t in range(0, L):
            x_ptrs = x_ptr + pid_b * stride_xb + pid_h * stride_xh + t * stride_xl + offs_d * stride_xd
            g_ptrs = g_ptr + pid_b * stride_gb + pid_h * stride_gh + t * stride_gl + offs_d * stride_gd

            x_t = tl.load(x_ptrs).to(tl.float32)
            g_t = tl.load(g_ptrs).to(tl.float32)

            h_fwd = g_t * h_fwd + (1.0 - g_t) * x_t

            y_ptrs = y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd
            tl.store(y_ptrs, (0.5 * h_fwd).to(x_ptr.dtype.element_ty))

        h_bwd = tl.zeros([D], dtype=tl.float32)
        for t_rev in range(0, L):
            t = L - 1 - t_rev
            x_ptrs = x_ptr + pid_b * stride_xb + pid_h * stride_xh + t * stride_xl + offs_d * stride_xd
            g_ptrs = g_ptr + pid_b * stride_gb + pid_h * stride_gh + t * stride_gl + offs_d * stride_gd

            x_t = tl.load(x_ptrs).to(tl.float32)
            g_t = tl.load(g_ptrs).to(tl.float32)

            h_bwd = g_t * h_bwd + (1.0 - g_t) * x_t

            y_ptrs = y_ptr + pid_b * stride_yb + pid_h * stride_yh + t * stride_yl + offs_d * stride_yd
            prev_fwd = tl.load(y_ptrs).to(tl.float32)
            y_fused = prev_fwd + 0.5 * h_bwd
            tl.store(y_ptrs, y_fused.to(x_ptr.dtype.element_ty))


def triton_hgrn2_fused_bwd_scan_pytorch_ref(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    B, H, L, D = x.shape
    h_fwd = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
    y_fwd = torch.empty_like(x)
    for t in range(L):
        h_fwd = g[:, :, t] * h_fwd + (1.0 - g[:, :, t]) * x[:, :, t]
        y_fwd[:, :, t] = h_fwd

    x_rev = torch.flip(x, dims=[2])
    g_rev = torch.flip(g, dims=[2])
    h_bwd = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
    y_bwd_rev = torch.empty_like(x)
    for t in range(L):
        h_bwd = g_rev[:, :, t] * h_bwd + (1.0 - g_rev[:, :, t]) * x_rev[:, :, t]
        y_bwd_rev[:, :, t] = h_bwd

    y_bwd = torch.flip(y_bwd_rev, dims=[2])
    return (0.5 * (y_fwd + y_bwd)).to(x.dtype)


def triton_hgrn2_fused_bwd_scan(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    B, H, L, D = x.shape

    if not HAS_TRITON or not x.is_cuda or D > 128:
        return triton_hgrn2_fused_bwd_scan_pytorch_ref(x, g)

    y = torch.empty_like(x)
    grid = (B, H)

    x_c = x.contiguous()
    g_c = g.contiguous()

    _hgrn2_bidirectional_scan_kernel[grid](
        x_c,
        g_c,
        y,
        B,
        H,
        L,
        D=D,
        stride_xb=x_c.stride(0),
        stride_xh=x_c.stride(1),
        stride_xl=x_c.stride(2),
        stride_xd=x_c.stride(3),
        stride_gb=g_c.stride(0),
        stride_gh=g_c.stride(1),
        stride_gl=g_c.stride(2),
        stride_gd=g_c.stride(3),
        stride_yb=y.stride(0),
        stride_yh=y.stride(1),
        stride_yl=y.stride(2),
        stride_yd=y.stride(3),
    )

    return y


__all__ = [
    'hgrn2_scan_triton',
    'hgrn2_scan_pytorch',
    'hgrn2_scan',
    'triton_hgrn2_fused_bwd_scan',
    'triton_hgrn2_fused_bwd_scan_pytorch_ref',
]

