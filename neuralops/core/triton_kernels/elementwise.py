"""Production Triton kernels for elementwise SwiGLU, Fast GELU, and RMSNorm.

Falls back transparently to PyTorch when Triton or CUDA is unavailable.
Public API:
    fused_swiglu(x, gate)              -> SiLU(gate) * x
    fast_gelu(x)                       -> tanh-approximated GELU
    fused_rmsnorm(x, weight, eps=1e-6) -> x / sqrt(mean(x^2) + eps) * weight

Backward-compatible aliases are kept at the bottom of this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = [
    "fused_swiglu",
    "fast_gelu",
    "fused_rmsnorm",
    # backward-compat aliases
    "fused_fast_gelu",
]


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_SWIGLU_BLOCK = 1024
_RMSNORM_BLOCK = 1024


def _cuda_ok(*tensors: torch.Tensor) -> bool:
    """True if Triton is available, GPU supports Triton JIT, and every tensor is a valid CUDA float tensor."""
    if not (HAS_TRITON and tl is not None and triton is not None):
        return False
    if not all(t.is_cuda and t.dtype in _FLOAT_DTYPES for t in tensors):
        return False
    try:
        from neuralops.core.triton_kernels.hardware_profile import get_device_hardware_profile

        profile = get_device_hardware_profile(tensors[0].device)
        if not profile.supports_triton_jit:
            return False
        # If any tensor is bf16, verify hardware supports bf16
        if any(t.dtype == torch.bfloat16 for t in tensors) and not profile.supports_bf16:
            return False
    except Exception:
        pass
    return True



# ═══════════════════════════════════════════════════════════════════
#  SwiGLU
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON and tl is not None:

    @triton.jit
    def _swiglu_kernel(
        X, GATE, OUT,
        N_ELEMENTS,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_ELEMENTS

        x = tl.load(X + offs, mask=mask, other=0.0)
        g = tl.load(GATE + offs, mask=mask, other=0.0)

        # Numerically stable SiLU: g * sigmoid(g) = g / (1.0 + exp(-g))
        g_f = g.to(tl.float32)
        silu = g_f / (1.0 + tl.exp(-g_f))
        out = (silu * x.to(tl.float32)).to(OUT.dtype.element_ty)

        tl.store(OUT + offs, out, mask=mask)



def fused_swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute SiLU(gate) * x elementwise.

    Uses a Triton kernel when both tensors are CUDA floats; otherwise
    falls back to ``F.silu(gate) * x`` in pure PyTorch.
    """
    if x.shape != gate.shape:
        raise ValueError(
            f"fused_swiglu: shape mismatch x={tuple(x.shape)} gate={tuple(gate.shape)}"
        )

    if _cuda_ok(x, gate):
        x_c = x.contiguous()
        g_c = gate.contiguous()
        out = torch.empty_like(x_c)
        n = x_c.numel()
        grid = (triton.cdiv(n, _SWIGLU_BLOCK),)
        _swiglu_kernel[grid](x_c, g_c, out, n, BLOCK_SIZE=_SWIGLU_BLOCK)
        return out

    return F.silu(gate) * x


# ═══════════════════════════════════════════════════════════════════
#  Fast GELU
# ═══════════════════════════════════════════════════════════════════

def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh-approximated GELU.

    Matches ``F.gelu(x, approximate='tanh')``. Kept as a thin wrapper
    rather than a Triton kernel because the operation is trivially
    memory-bound and PyTorch already emits a single fused kernel for it.
    """
    return F.gelu(x, approximate="tanh")


# ═══════════════════════════════════════════════════════════════════
#  RMSNorm
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON and tl is not None:

    @triton.jit
    def _rmsnorm_kernel(
        X, W, OUT,
        N,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X + row * N + cols, mask=mask, other=0.0)

        # Single-pass reduction: sum of squares.
        sq_sum = tl.sum(x * x, axis=0)
        inv_rms = 1.0 / tl.sqrt(sq_sum / N + eps)

        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=mask, other=0.0)
            out = x * inv_rms * w
        else:
            out = x * inv_rms

        tl.store(OUT + row * N + cols, out, mask=mask)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight.

    Normalizes over the last dimension. ``weight`` is optional.

    Uses a Triton kernel when x is a 2-D CUDA float tensor with a
    power-of-two-friendly last dim and Triton is available. Otherwise
    falls back to a pure PyTorch implementation.
    """

    orig_shape = x.shape
    if x.dim() < 2:
        x2 = x.unsqueeze(0)
    else:
        x2 = x
    N = x2.shape[-1]
    M = x2.numel() // N
    x2d = x2.reshape(M, N)

    has_weight = weight is not None

    can_use_triton = (
        _cuda_ok(x2d)
        and (weight is None or _cuda_ok(weight))
        and N <= _RMSNORM_BLOCK * 8   # avoid silly tile sizes
    )

    if can_use_triton:
        x_c = x2d.contiguous()
        w_c = weight.contiguous() if has_weight else x_c  # unused when HAS_WEIGHT=False
        out = torch.empty_like(x_c)

        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        _rmsnorm_kernel[grid](
            x_c, w_c, out,
            N, eps,
            HAS_WEIGHT=has_weight,
            BLOCK_N=BLOCK_N,
        )
        return out.reshape(orig_shape)

    # PyTorch fallback.
    variance = x2d.pow(2).mean(-1, keepdim=True)
    y = x2d * torch.rsqrt(variance + eps)
    if has_weight:
        y = y * weight
    return y.reshape(orig_shape)


# ═══════════════════════════════════════════════════════════════════
#  SpectraFormer: Fused Complex Spectral Gate & RMS Variance Match
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _fused_spectral_gate_kernel(
        X_re_ptr, X_im_ptr, W_re_ptr, W_im_ptr, Tau_ptr,
        Y_re_ptr, Y_im_ptr,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements

        x_re = tl.load(X_re_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_im = tl.load(X_im_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w_re = tl.load(W_re_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w_im = tl.load(W_im_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tau = tl.load(Tau_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Complex multiplication: (x_re + i x_im) * (w_re + i w_im)
        y_hat_re = x_re * w_re - x_im * w_im
        y_hat_im = x_re * w_im + x_im * w_re

        # Magnitude and soft thresholding shrinkage
        mag = tl.sqrt(y_hat_re * y_hat_re + y_hat_im * y_hat_im + 1e-8)
        sig_tau = 1.0 / (1.0 + tl.exp(-tau))
        diff = mag - sig_tau
        shrink = tl.maximum(diff, 0.0) / mag

        out_re = shrink * y_hat_re
        out_im = shrink * y_hat_im

        tl.store(Y_re_ptr + offs, out_re.to(Y_re_ptr.dtype.element_ty), mask=mask)
        tl.store(Y_im_ptr + offs, out_im.to(Y_im_ptr.dtype.element_ty), mask=mask)


def fused_spectral_gate_triton(
    x_complex: torch.Tensor,
    w_complex: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Fused complex spectral gate + soft-threshold shrinkage in SRAM."""
    if x_complex.is_cuda and _cuda_ok(x_complex.real) and HAS_TRITON:
        x_re = x_complex.real.contiguous()
        x_im = x_complex.imag.contiguous()
        w_re = w_complex.real.contiguous()
        w_im = w_complex.imag.contiguous()
        tau_c = tau.expand_as(x_re).contiguous()

        y_re = torch.empty_like(x_re)
        y_im = torch.empty_like(x_im)
        total_elements = x_re.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

        _fused_spectral_gate_kernel[grid](
            x_re, x_im, w_re, w_im, tau_c,
            y_re, y_im,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return torch.complex(y_re, y_im)

    # PyTorch Fallback
    y_hat = x_complex * w_complex
    mag = torch.abs(y_hat).clamp_min(1e-8)
    sig_tau = torch.sigmoid(tau)
    scale = (torch.relu(mag - sig_tau) / mag)
    return y_hat * scale


def fused_rms_variance_match_triton(
    y: torch.Tensor,
    v_ref: torch.Tensor,
    clamp_min: float = 0.25,
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """Normalizes output y to match reference v_ref's RMS energy in a single pass."""
    v_rms = torch.sqrt(torch.mean(v_ref.float() ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
    y_rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
    gain = (v_rms / y_rms).clamp(clamp_min, clamp_max).to(y.dtype)
    return y * gain


# ═══════════════════════════════════════════════════════════════════
#  Fast Walsh-Hadamard Transform (FWHT) & Orthogonal Rotation
# ═══════════════════════════════════════════════════════════════════

def _pytorch_fwht_butterfly(x: torch.Tensor) -> torch.Tensor:
    """Vectorized in-place butterfly stages for Walsh-Hadamard Transform."""
    orig_shape = x.shape
    dim = orig_shape[-1]
    
    # Must be power of 2
    if dim & (dim - 1) != 0:
        next_pow2 = 1 << (dim - 1).bit_length()
        x_pad = torch.nn.functional.pad(x, (0, next_pow2 - dim))
        res = _pytorch_fwht_butterfly(x_pad)
        return res[..., :dim]
        
    x_flat = x.reshape(-1, dim).float().clone()
    h = 1
    while h < dim:
        for i in range(0, dim, h * 2):
            u = x_flat[:, i : i + h].clone()
            v = x_flat[:, i + h : i + 2 * h].clone()
            x_flat[:, i : i + h] = u + v
            x_flat[:, i + h : i + 2 * h] = u - v
        h *= 2
        
    x_flat /= (dim ** 0.5)
    return x_flat.view(orig_shape).to(x.dtype)


def fast_hadamard_transform(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Computes normalized block-diagonal Fast Walsh-Hadamard Transform:
      Y = X @ H where H^T H = I and H = H^T (involutory and orthogonal).
    Smears activation and weight channel outliers uniformly without parameters.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    
    # Ensure block_size is power of 2 and <= d
    b = min(block_size, 1 << (d.bit_length() - 1))
    if b < 2:
        return x
        
    # Chunk along last dimension
    rem = d % b
    if rem == 0:
        x_blocks = x.view(-1, d // b, b)
        y_blocks = _pytorch_fwht_butterfly(x_blocks)
        return y_blocks.view(orig_shape)
    else:
        # Pad last dimension to multiple of b
        pad_len = b - rem
        x_pad = torch.nn.functional.pad(x, (0, pad_len))
        y_pad = fast_hadamard_transform(x_pad, block_size=b)
        return y_pad[..., :d].contiguous()


# ═══════════════════════════════════════════════════════════════════
#  ALiBi Bias Fusion: Fused Relative Position Bias Kernel
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _fused_alibi_bias_kernel(
        Bias_ptr,
        Slopes_ptr,
        N_q,
        N_k,
        past_len,
        stride_bh,
        stride_q,
        stride_k,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        head_idx = tl.program_id(2)

        slope = tl.load(Slopes_ptr + head_idx)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < N_q
        mask_n = offs_n < N_k

        q_pos = offs_m[:, None] + past_len
        k_pos = offs_n[None, :]

        diff = q_pos - k_pos
        if IS_CAUSAL:
            dist = tl.where(diff >= 0, diff, 0.0)
        else:
            dist = tl.where(diff >= 0, diff, -diff)

        bias = -slope * dist

        bias_offs = (
            head_idx * stride_bh
            + offs_m[:, None] * stride_q
            + offs_n[None, :] * stride_k
        )
        mask = mask_m[:, None] & mask_n[None, :]
        tl.store(Bias_ptr + bias_offs, bias, mask=mask)


def fused_alibi_bias_pytorch(
    slopes: torch.Tensor,
    N_q: int,
    N_k: int,
    past_len: int = 0,
    is_causal: bool = True,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for ALiBi relative bias."""
    device = slopes.device
    num_heads = slopes.numel()
    q_pos = torch.arange(past_len, past_len + N_q, device=device).unsqueeze(1)
    k_pos = torch.arange(0, N_k, device=device).unsqueeze(0)
    if is_causal:
        dist = (q_pos - k_pos).clamp(min=0).float()
    else:
        dist = (q_pos - k_pos).abs().float()
    slopes_reshaped = slopes.view(1, num_heads, 1, 1).to(device)
    return -slopes_reshaped * dist.unsqueeze(0).unsqueeze(0)


def fused_alibi_bias_triton(
    slopes: torch.Tensor,
    N_q: int,
    N_k: int,
    past_len: int = 0,
    is_causal: bool = True,
) -> torch.Tensor:
    """Fused ALiBi relative distance bias matrix computation in SRAM."""
    if not (HAS_TRITON and slopes.is_cuda):
        return fused_alibi_bias_pytorch(slopes, N_q, N_k, past_len=past_len, is_causal=is_causal)

    device = slopes.device
    num_heads = slopes.numel()
    out = torch.empty((1, num_heads, N_q, N_k), device=device, dtype=slopes.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        triton.cdiv(N_q, BLOCK_M),
        triton.cdiv(N_k, BLOCK_N),
        num_heads,
    )

    try:
        _fused_alibi_bias_kernel[grid](
            out,
            slopes,
            N_q,
            N_k,
            past_len,
            out.stride(1),
            out.stride(2),
            out.stride(3),
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return out
    except Exception:
        return fused_alibi_bias_pytorch(slopes, N_q, N_k, past_len=past_len, is_causal=is_causal)


# ═══════════════════════════════════════════════════════════════════
#  DynamicConv1D Fusion: Fused Causal Depthwise Conv Kernel
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _fused_dynconv1d_fwd_kernel(
        v_ptr,
        w_ptr,
        out_ptr,
        B,
        H,
        N,
        d,
        K,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_wb,
        stride_wh,
        stride_wn,
        stride_wk,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < d

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k in range(0, K):
            w_offset = b * stride_wb + h * stride_wh + pid_n * stride_wn + k * stride_wk
            weight = tl.load(w_ptr + w_offset)

            v_offset = b * stride_vb + h * stride_vh + (pid_n + k) * stride_vt + offs_d * stride_vd
            v_val = tl.load(v_ptr + v_offset, mask=mask_d, other=0.0)

            acc += weight * v_val.to(tl.float32)

        out_offset = b * stride_ob + h * stride_oh + pid_n * stride_on + offs_d * stride_od
        tl.store(out_ptr + out_offset, acc.to(out_ptr.dtype.element_ty), mask=mask_d)


def fused_dynamic_conv1d_pytorch(
    v_full: torch.Tensor,
    dyn_w: torch.Tensor,
    chunk: int | None = 64,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for chunked dynamic causal conv."""
    B, H, T, d = v_full.shape
    N = dyn_w.shape[2]
    K = dyn_w.shape[3]
    out = torch.empty((B, H, N, d), dtype=v_full.dtype, device=v_full.device)
    v_win_all = v_full.unfold(dimension=2, size=K, step=1)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        v_chunk = v_win_all[:, :, start:end, :, :]
        w_chunk = dyn_w[:, :, start:end, :]
        out[:, :, start:end, :] = torch.einsum("bhcdk,bhck->bhcd", v_chunk, w_chunk)
    return out


def fused_dynamic_conv1d_triton(
    v_full: torch.Tensor,
    dyn_w: torch.Tensor,
) -> torch.Tensor:
    """Fused Dynamic Causal Depthwise Conv1D kernel executed entirely in GPU SRAM."""
    if not (HAS_TRITON and v_full.is_cuda):
        return fused_dynamic_conv1d_pytorch(v_full, dyn_w)

    B, H, T, d = v_full.shape
    N = dyn_w.shape[2]
    K = dyn_w.shape[3]
    device = v_full.device

    out = torch.empty((B, H, N, d), dtype=v_full.dtype, device=device)
    BLOCK_D = triton.next_power_of_2(d)
    grid = (N, B * H)

    try:
        _fused_dynconv1d_fwd_kernel[grid](
            v_full, dyn_w, out,
            B, H, N, d, K,
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            dyn_w.stride(0), dyn_w.stride(1), dyn_w.stride(2), dyn_w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_D=BLOCK_D,
        )
        return out
    except Exception:
        return fused_dynamic_conv1d_pytorch(v_full, dyn_w)


# ═══════════════════════════════════════════════════════════════════
#  Adaptive Span Penalty Fusion: Fused Span Mask Kernel
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _fused_adaptive_span_kernel(
        penalty_ptr,
        spans_ptr,
        q_pos_ptr,
        k_pos_ptr,
        ramp_size,
        B,
        H,
        N_q,
        N_k,
        stride_pb,
        stride_ph,
        stride_pq,
        stride_pk,
        stride_qb,
        stride_qq,
        stride_kb,
        stride_kk,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bh = tl.program_id(2)

        b = pid_bh // H
        h = pid_bh % H

        span = tl.load(spans_ptr + h)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < N_q
        mask_n = offs_n < N_k

        q_pos = tl.load(q_pos_ptr + b * stride_qb + offs_m * stride_qq, mask=mask_m, other=0.0)
        k_pos = tl.load(k_pos_ptr + b * stride_kb + offs_n * stride_kk, mask=mask_n, other=0.0)

        diff = q_pos[:, None] - k_pos[None, :]

        if IS_CAUSAL:
            dist = tl.where(diff >= 0.0, diff, 0.0)
        else:
            dist = tl.where(diff >= 0.0, diff, -diff)

        ramp = (ramp_size + span - dist) / ramp_size
        ramp = tl.maximum(0.0, tl.minimum(1.0, ramp))
        penalty = tl.log(tl.maximum(1e-9, ramp))

        offs_penalty = (
            b * stride_pb
            + h * stride_ph
            + offs_m[:, None] * stride_pq
            + offs_n[None, :] * stride_pk
        )
        mask = mask_m[:, None] & mask_n[None, :]
        tl.store(penalty_ptr + offs_penalty, penalty, mask=mask)


def fused_adaptive_span_penalty_pytorch(
    spans: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    ramp_size: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for adaptive span penalty."""
    B, N_q = q_pos.shape
    _, N_k = k_pos.shape
    H = spans.shape[0]

    if is_causal:
        dist = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).clamp(min=0.0)
    else:
        dist = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).abs()

    span_h = spans.view(H, 1, 1).unsqueeze(1)
    dist_e = dist.unsqueeze(0)
    ramp = ((ramp_size + span_h - dist_e) / ramp_size).clamp(0.0, 1.0)
    span_mask = torch.log(ramp.clamp(min=1e-9))
    return span_mask.permute(1, 0, 2, 3).contiguous()


def fused_adaptive_span_penalty_triton(
    spans: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    ramp_size: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """Fused Adaptive Span Penalty computation in GPU SRAM."""
    if not (HAS_TRITON and spans.is_cuda and q_pos.is_cuda and k_pos.is_cuda):
        return fused_adaptive_span_penalty_pytorch(spans, q_pos, k_pos, ramp_size, is_causal=is_causal)

    B, N_q = q_pos.shape
    _, N_k = k_pos.shape
    H = spans.shape[0]
    device = spans.device

    penalty = torch.empty((B, H, N_q, N_k), device=device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        triton.cdiv(N_q, BLOCK_M),
        triton.cdiv(N_k, BLOCK_N),
        B * H,
    )

    try:
        _fused_adaptive_span_kernel[grid](
            penalty,
            spans.float(),
            q_pos.float(),
            k_pos.float(),
            float(ramp_size),
            B, H, N_q, N_k,
            penalty.stride(0), penalty.stride(1), penalty.stride(2), penalty.stride(3),
            q_pos.stride(0), q_pos.stride(1),
            k_pos.stride(0), k_pos.stride(1),
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return penalty
    except Exception:
        return fused_adaptive_span_penalty_pytorch(spans, q_pos, k_pos, ramp_size, is_causal=is_causal)


# ═══════════════════════════════════════════════════════════════════
#  Backward-compatibility aliases
# ═══════════════════════════════════════════════════════════════════

fused_fast_gelu = fast_gelu

__all__ = [
    "fused_swiglu",
    "fast_gelu",
    "fused_rmsnorm",
    "fused_fast_gelu",
    "fused_spectral_gate_triton",
    "fused_rms_variance_match_triton",
    "fast_hadamard_transform",
    "fused_alibi_bias_triton",
    "fused_alibi_bias_pytorch",
    "fused_dynamic_conv1d_triton",
    "fused_dynamic_conv1d_pytorch",
    "fused_adaptive_span_penalty_triton",
    "fused_adaptive_span_penalty_pytorch",
]