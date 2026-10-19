# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HGRN2 Selective Scan

Computes hierarchical gated recurrent unit state transitions with data-controlled forgetting gates.
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
    init_state_ptr,
    final_state_ptr,
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
    stride_isb,
    stride_isd,
    stride_fsb,
    stride_fsd,
    B_size,
    L_size,
    D_dim,
    BLOCK_D: tl.constexpr,
    DO_CLIP: tl.constexpr,
    CLIP_VAL: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    HAS_FINAL_STATE: tl.constexpr,
    STORE_WS: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B_size:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_dim

    if HAS_INIT_STATE:
        init_ptrs = init_state_ptr + pid_b * stride_isb + offs_d * stride_isd
        h = tl.load(init_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    else:
        h = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for l in range(L_size):
        f_ptrs = f_ptr + pid_b * stride_fb + l * stride_fl + offs_d * stride_fd
        iv_ptrs = iv_ptr + pid_b * stride_ivb + l * stride_ivl + offs_d * stride_ivd

        f_t = tl.load(f_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        iv_t = tl.load(iv_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        h = f_t * h + iv_t

        if DO_CLIP:
            h = tl.minimum(tl.maximum(h, -CLIP_VAL), CLIP_VAL)

        if STORE_WS:
            hw_ptrs = h_workspace_ptr + pid_b * stride_hwb + l * stride_hwl + offs_d * stride_hwd
            tl.store(hw_ptrs, h, mask=mask_d)

        ho_ptrs = h_out_ptr + pid_b * stride_hob + l * stride_hol + offs_d * stride_hod
        tl.store(ho_ptrs, h.to(h_out_ptr.dtype.element_ty), mask=mask_d)

    if HAS_FINAL_STATE:
        fin_ptrs = final_state_ptr + pid_b * stride_fsb + offs_d * stride_fsd
        tl.store(fin_ptrs, h.to(final_state_ptr.dtype.element_ty), mask=mask_d)


@triton.jit
def _hgrn2_bwd_kernel(
    f_ptr,
    h_workspace_ptr,
    init_state_ptr,
    dh_fwd_ptr,
    df_ptr,
    div_ptr,
    dinit_state_ptr,
    stride_fb,
    stride_fl,
    stride_fd,
    stride_hwb,
    stride_hwl,
    stride_hwd,
    stride_isb,
    stride_isd,
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
    HAS_INIT_STATE: tl.constexpr,
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

        dh_ptrs = dh_fwd_ptr + pid_b * stride_dhb + l * stride_dhl + offs_d * stride_dhd
        dh_ext = tl.load(dh_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dh_clipped = dh_clipped + dh_ext

        if DO_CLIP:
            h_l_ptrs = h_workspace_ptr + pid_b * stride_hwb + l * stride_hwl + offs_d * stride_hwd
            h_l = tl.load(h_l_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            clip_mask = (tl.abs(h_l) < CLIP_VAL).to(tl.float32)
            dh_raw = dh_clipped * clip_mask
        else:
            dh_raw = dh_clipped

        f_ptrs = f_ptr + pid_b * stride_fb + l * stride_fl + offs_d * stride_fd
        f_l = tl.load(f_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        if l > 0:
            h_prev_ptrs = (
                h_workspace_ptr + pid_b * stride_hwb + (l - 1) * stride_hwl + offs_d * stride_hwd
            )
            h_prev = tl.load(h_prev_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        elif HAS_INIT_STATE:
            init_ptrs = init_state_ptr + pid_b * stride_isb + offs_d * stride_isd
            h_prev = tl.load(init_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        else:
            h_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)

        df_val = dh_raw * h_prev
        div_val = dh_raw

        df_ptrs = df_ptr + pid_b * stride_dfb + l * stride_dfl + offs_d * stride_dfd
        div_ptrs = div_ptr + pid_b * stride_divb + l * stride_divl + offs_d * stride_divd
        tl.store(df_ptrs, df_val.to(df_ptr.dtype.element_ty), mask=mask_d)
        tl.store(div_ptrs, div_val.to(div_ptr.dtype.element_ty), mask=mask_d)

        dh_clipped = dh_raw * f_l

    if HAS_INIT_STATE:
        dis_ptrs = dinit_state_ptr + pid_b * stride_isb + offs_d * stride_isd
        tl.store(dis_ptrs, dh_clipped.to(dinit_state_ptr.dtype.element_ty), mask=mask_d)


def _hgrn2_fwd_inference(f, iv, res_clip=None, initial_state=None, return_final_state=False):
    B, L, D = f.shape
    out = torch.empty_like(iv)
    dummy_ws = torch.empty((1,), device=f.device, dtype=torch.float32)

    has_init = initial_state is not None
    init_state = initial_state.contiguous() if has_init else dummy_ws

    has_final = bool(return_final_state)
    final_state = torch.empty((B, D), device=f.device, dtype=f.dtype) if has_final else dummy_ws

    BLOCK_D = 64
    num_d_blocks = triton.cdiv(D, BLOCK_D)
    grid = (num_d_blocks, B)

    do_clip = res_clip is not None and res_clip > 0
    clip_val = float(res_clip) if do_clip else 1.0

    _hgrn2_fwd_kernel[grid](
        f,
        iv,
        out,
        dummy_ws,
        init_state,
        final_state,
        f.stride(0),
        f.stride(1),
        f.stride(2),
        iv.stride(0),
        iv.stride(1),
        iv.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        0, 0, 0,
        init_state.stride(0) if has_init else 0,
        init_state.stride(1) if has_init else 0,
        final_state.stride(0) if has_final else 0,
        final_state.stride(1) if has_final else 0,
        B,
        L,
        D,
        BLOCK_D=BLOCK_D,
        DO_CLIP=do_clip,
        CLIP_VAL=clip_val,
        HAS_INIT_STATE=has_init,
        HAS_FINAL_STATE=has_final,
        STORE_WS=False,
    )
    if return_final_state:
        return out, final_state
    return out


class _HGRN2ScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, iv, res_clip, initial_state=None, return_final_state=False):
        B, L, D = f.shape
        out = torch.empty_like(iv)
        h_workspace = torch.empty((B, L, D), device=f.device, dtype=torch.float32)

        dummy_t = torch.empty((1,), device=f.device, dtype=torch.float32)
        has_init = initial_state is not None
        init_state = initial_state.contiguous() if has_init else dummy_t

        has_final = bool(return_final_state)
        final_state = torch.empty((B, D), device=f.device, dtype=f.dtype) if has_final else dummy_t

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
            init_state,
            final_state,
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
            init_state.stride(0) if has_init else 0,
            init_state.stride(1) if has_init else 0,
            final_state.stride(0) if has_final else 0,
            final_state.stride(1) if has_final else 0,
            B,
            L,
            D,
            BLOCK_D=BLOCK_D,
            DO_CLIP=do_clip,
            CLIP_VAL=clip_val,
            HAS_INIT_STATE=has_init,
            HAS_FINAL_STATE=has_final,
            STORE_WS=True,
        )

        ctx.save_for_backward(f, h_workspace, init_state)
        ctx.res_clip = res_clip
        ctx.has_init = has_init
        ctx.return_final_state = return_final_state

        if return_final_state:
            return out, final_state
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        d_out = grad_outputs[0].contiguous()
        f, h_workspace, init_state = ctx.saved_tensors
        B, L, D = f.shape

        df = torch.empty_like(f)
        div = torch.empty_like(f)
        dinit_state = torch.empty_like(init_state) if ctx.has_init else torch.empty((1,), device=f.device)

        BLOCK_D = 64
        num_d_blocks = triton.cdiv(D, BLOCK_D)
        grid = (num_d_blocks, B)

        do_clip = ctx.res_clip is not None and ctx.res_clip > 0
        clip_val = float(ctx.res_clip) if do_clip else 1.0

        _hgrn2_bwd_kernel[grid](
            f,
            h_workspace,
            init_state,
            d_out,
            df,
            div,
            dinit_state,
            f.stride(0),
            f.stride(1),
            f.stride(2),
            h_workspace.stride(0),
            h_workspace.stride(1),
            h_workspace.stride(2),
            init_state.stride(0) if ctx.has_init else 0,
            init_state.stride(1) if ctx.has_init else 0,
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
            HAS_INIT_STATE=ctx.has_init,
        )

        d_init = dinit_state if ctx.has_init else None
        return df, div, None, d_init, None


def hgrn2_scan_pytorch(
    f: torch.Tensor,
    iv: torch.Tensor,
    res_clip: float | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference implementation for HGRN2 selective scan.
    Recurrence: h_t = f_t * h_{t-1} + iv_t
    """
    B, L, D = f.shape
    device = f.device
    dtype = f.dtype

    if initial_state is not None:
        h = initial_state.clone().float()
    else:
        h = torch.zeros((B, D), device=device, dtype=torch.float32)
    ys = []

    for t in range(L):
        f_t = f[:, t, :].float()
        iv_t = iv[:, t, :].float()
        h = f_t * h + iv_t
        if res_clip is not None and res_clip > 0:
            h = torch.clamp(h, -float(res_clip), float(res_clip))
        ys.append(h)

    out = torch.stack(ys, dim=1).to(dtype)
    if return_final_state:
        return out, h.to(dtype)
    return out


def hgrn2_scan_triton(
    f: torch.Tensor,
    iv: torch.Tensor,
    res_clip: float | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton HGRN2 selective scan with initial_state and zero-allocation inference."""
    from neuralops.core.triton_kernels._common import should_use_triton

    B, L, D = f.shape
    if not should_use_triton(f, "hgrn2_scan", head_dim=D, seq_len=L, allow_fp32=True):
        return hgrn2_scan_pytorch(
            f, iv, res_clip=res_clip, initial_state=initial_state, return_final_state=return_final_state
        )

    f = f.contiguous()
    iv = iv.contiguous()

    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (f, iv, initial_state) if isinstance(t, torch.Tensor)
    )
    if not needs_grad:
        return _hgrn2_fwd_inference(
            f, iv, res_clip=res_clip, initial_state=initial_state, return_final_state=return_final_state
        )

    return _HGRN2ScanFn.apply(f, iv, res_clip, initial_state, return_final_state)


def hgrn2_scan(
    f: torch.Tensor,
    iv: torch.Tensor,
    res_clip: float | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dispatches HGRN2 scan to Triton GPU kernel when suitable, else PyTorch fallback."""
    from neuralops.core.triton_kernels._common import should_use_triton
    suitable = should_use_triton(f, "hgrn2_scan", head_dim=f.shape[-1], seq_len=f.shape[1], allow_fp32=True)
    if suitable and hasattr(triton, "jit"):
        try:
            return hgrn2_scan_triton(
                f, iv, res_clip=res_clip, initial_state=initial_state, return_final_state=return_final_state
            )
        except Exception:
            pass
    return hgrn2_scan_pytorch(
        f, iv, res_clip=res_clip, initial_state=initial_state, return_final_state=return_final_state
    )


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
