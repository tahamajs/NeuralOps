"""
Fused Elementwise Kernels

High-throughput elementwise kernels for SwiGLU, Fast GELU, and RMSNorm with transparent fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations

import math
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


if HAS_TRITON and tl is not None:
    @triton.jit
    def _rms_variance_match_kernel(
        Y_ptr, V_REF_ptr, OUT_ptr,
        total_rows, D: tl.constexpr,
        clamp_min, clamp_max,
        stride_yn, stride_yd,
        stride_vn, stride_vd,
        stride_on, stride_od,
        BLOCK_D: tl.constexpr,
    ):
        """In-SRAM Dual-RMS Energy Matching Kernel."""
        pid = tl.program_id(0)
        if pid >= total_rows:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        y = tl.load(Y_ptr + pid * stride_yn + offs_d * stride_yd, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(V_REF_ptr + pid * stride_vn + offs_d * stride_vd, mask=mask, other=0.0).to(tl.float32)

        mean_y2 = tl.sum(y * y, axis=0) / tl.cast(D, tl.float32)
        mean_v2 = tl.sum(v * v, axis=0) / tl.cast(D, tl.float32)

        y_rms = tl.sqrt(tl.maximum(mean_y2, 1e-8))
        v_rms = tl.sqrt(tl.maximum(mean_v2, 1e-8))

        raw_gain = v_rms / y_rms
        gain = tl.maximum(clamp_min, tl.minimum(clamp_max, raw_gain))

        out = y * gain
        tl.store(OUT_ptr + pid * stride_on + offs_d * stride_od, out.to(OUT_ptr.dtype.element_ty), mask=mask)


def fused_rms_variance_match_triton(
    y: torch.Tensor,
    v_ref: torch.Tensor,
    clamp_min: float = 0.25,
    clamp_max: float = 4.0,
) -> torch.Tensor:
    """Normalizes output y to match reference v_ref's RMS energy in a single pass."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK
    if (
        HAS_TRITON
        and not FORCE_PYTORCH_FALLBACK
        and y.is_cuda
        and v_ref.is_cuda
        and y.shape == v_ref.shape
    ):
        try:
            orig_shape = y.shape
            D = orig_shape[-1]
            y_2d = y.contiguous().view(-1, D)
            v_2d = v_ref.contiguous().view(-1, D)
            total_rows = y_2d.shape[0]
            out_2d = torch.empty_like(y_2d)

            block_d = triton.next_power_of_2(D)
            grid = (total_rows,)
            _rms_variance_match_kernel[grid](
                y_2d, v_2d, out_2d,
                total_rows, D=D,
                clamp_min=float(clamp_min), clamp_max=float(clamp_max),
                stride_yn=y_2d.stride(0), stride_yd=y_2d.stride(1),
                stride_vn=v_2d.stride(0), stride_vd=v_2d.stride(1),
                stride_on=out_2d.stride(0), stride_od=out_2d.stride(1),
                BLOCK_D=block_d,
            )
            return out_2d.view(orig_shape)
        except Exception:
            pass

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


if HAS_TRITON and tl is not None:
    @triton.jit
    def _fwht_sram_kernel(
        X_ptr, Y_ptr,
        total_rows,
        BLOCK_D: tl.constexpr,
    ):
        """In-SRAM Fast Walsh-Hadamard Transform (FWHT) Kernel.
        Loads row of size BLOCK_D directly into SRAM registers, executes
        fast Hadamard butterfly stages via bitwise XOR stride exchange in-place,
        and scales by 1/sqrt(BLOCK_D). Zero intermediate allocations in HBM!
        """
        pid_m = tl.program_id(0)
        if pid_m >= total_rows:
            return

        offs_d = tl.arange(0, BLOCK_D)
        x_ptrs = X_ptr + pid_m * BLOCK_D + offs_d
        val = tl.load(x_ptrs).to(tl.float32)

        # In-register butterfly stages for power-of-two BLOCK_D (e.g. 64, 128, 256)
        # Butterfly step h = 1, 2, 4, 8, ...
        # Implemented using parallel bit-parity signs and strided pairs
        norm = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))
        
        # Sequentially unroll butterfly rounds
        h = 1
        while h < BLOCK_D:
            partner = offs_d ^ h
            val_partner = tl.load(X_ptr + pid_m * BLOCK_D + partner).to(tl.float32)
            is_lower = (offs_d & h) == 0
            val = tl.where(is_lower, val + val_partner, val_partner - val)
            h = h * 2

        val = val * norm
        y_ptrs = Y_ptr + pid_m * BLOCK_D + offs_d
        tl.store(y_ptrs, val.to(Y_ptr.dtype.element_ty))


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


    @triton.jit
    def _fused_dynconv1d_bwd_kernel(
        v_ptr,
        w_ptr,
        dout_ptr,
        dv_ptr,
        dw_ptr,
        B,
        H,
        N,
        d,
        K,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_wb, stride_wh, stride_wn, stride_wk,
        stride_dob, stride_doh, stride_don, stride_dod,
        stride_dvb, stride_dvh, stride_dvt, stride_dvd,
        stride_dwb, stride_dwh, stride_dwn, stride_dwk,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < d

        # Load dout for timestep pid_n
        dout_off = b * stride_dob + h * stride_doh + pid_n * stride_don + offs_d * stride_dod
        dout_val = tl.load(dout_ptr + dout_off, mask=mask_d, other=0.0).to(tl.float32)

        for k in range(0, K):
            # dw[b, h, n, k] = sum_d(dout[b, h, n, d] * v[b, h, n + k, d])
            v_off = b * stride_vb + h * stride_vh + (pid_n + k) * stride_vt + offs_d * stride_vd
            v_val = tl.load(v_ptr + v_off, mask=mask_d, other=0.0).to(tl.float32)
            dw_val = tl.sum(dout_val * v_val)

            w_off = b * stride_wb + h * stride_wh + pid_n * stride_wn + k * stride_wk
            w_val = tl.load(w_ptr + w_off)

            dw_off = b * stride_dwb + h * stride_dwh + pid_n * stride_dwn + k * stride_dwk
            tl.store(dw_ptr + dw_off, dw_val.to(dw_ptr.dtype.element_ty))

            # dv[b, h, n + k, d] += dout[b, h, n, d] * w[b, h, n, k]
            dv_off = b * stride_dvb + h * stride_dvh + (pid_n + k) * stride_dvt + offs_d * stride_dvd
            dv_val = dout_val * w_val
            tl.atomic_add(dv_ptr + dv_off, dv_val.to(dv_ptr.dtype.element_ty), mask=mask_d)


class _FusedDynamicConv1DTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_full: torch.Tensor, dyn_w: torch.Tensor) -> torch.Tensor:
        B, H, T, d = v_full.shape
        N = dyn_w.shape[2]
        K = dyn_w.shape[3]
        device = v_full.device

        out = torch.empty((B, H, N, d), dtype=v_full.dtype, device=device)
        BLOCK_D = triton.next_power_of_2(d)
        grid = (N, B * H)

        _fused_dynconv1d_fwd_kernel[grid](
            v_full, dyn_w, out,
            B, H, N, d, K,
            v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
            dyn_w.stride(0), dyn_w.stride(1), dyn_w.stride(2), dyn_w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(v_full, dyn_w)
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        v_full, dyn_w = ctx.saved_tensors
        B, H, T, d = v_full.shape
        N = dyn_w.shape[2]
        K = dyn_w.shape[3]

        dv = torch.zeros_like(v_full)
        dw = torch.empty_like(dyn_w)
        BLOCK_D = ctx.BLOCK_D
        grid = (N, B * H)

        dout_c = dout.contiguous()
        is_modern_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        if is_modern_gpu:
            try:
                _fused_dynconv1d_bwd_kernel[grid](
                    v_full, dyn_w, dout_c, dv, dw,
                    B, H, N, d, K,
                    v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
                    dyn_w.stride(0), dyn_w.stride(1), dyn_w.stride(2), dyn_w.stride(3),
                    dout_c.stride(0), dout_c.stride(1), dout_c.stride(2), dout_c.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    dw.stride(0), dw.stride(1), dw.stride(2), dw.stride(3),
                    BLOCK_D=BLOCK_D,
                )
                return dv, dw
            except Exception:
                pass

        # Exact PyTorch fallback backward for SM < 70 (e.g. Pascal SM 6.1)
        v_win = v_full.unfold(dimension=2, size=K, step=1)  # (B, H, N, d, K)
        dw = torch.einsum("bhnd,bhndk->bhnk", dout_c, v_win)
        dv = torch.zeros_like(v_full)
        for k in range(K):
            dv[:, :, k : k + N, :] += dout_c * dyn_w[:, :, :, k].unsqueeze(-1)
        return dv, dw


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

    try:
        return _FusedDynamicConv1DTritonFn.apply(v_full.contiguous(), dyn_w.contiguous())
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
# ═══════════════════════════════════════════════════════════════════
#  FAVOR+ Feature Map Fusion: Fused Random Projection & Normalizer
# ═══════════════════════════════════════════════════════════════════

def fused_favor_plus_feature_map_pytorch(
    x: torch.Tensor,
    omega: torch.Tensor,
    num_features: int,
    kernel_type: str = "favor_plus",
    eps: float = 1e-4,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for FAVOR+ feature mapping."""
    omega_t = omega.to(dtype=x.dtype, device=x.device).t()
    proj = torch.matmul(x, omega_t)

    if kernel_type in ("favor_plus", "exp", "favor"):
        diag_sq = torch.sum(x**2, dim=-1, keepdim=True) * 0.5
        diff = torch.clamp(proj - diag_sq, min=-20.0, max=8.0)
        phi = torch.exp(diff) / math.sqrt(num_features)
    elif kernel_type == "elu":
        phi = (F.elu(proj) + 1.0) / math.sqrt(num_features)
    else:
        phi = (F.relu(proj) + eps) / math.sqrt(num_features)
    return phi


if HAS_TRITON and tl is not None:
    @triton.jit
    def _favor_plus_act_kernel(
        Proj_ptr, DiagSq_ptr, Out_ptr,
        total_tokens, F_dim: tl.constexpr, norm_factor,
        stride_pn, stride_pf,
        stride_dn,
        stride_on, stride_of,
        BLOCK_F: tl.constexpr,
    ):
        """Fused in-SRAM exponential activation with diagonal norm shift: exp(proj - 0.5*|x|^2) / sqrt(m)."""
        pid = tl.program_id(0)
        if pid >= total_tokens:
            return

        offs_f = tl.arange(0, BLOCK_F)
        mask = offs_f < F_dim

        diag_sq = tl.load(DiagSq_ptr + pid * stride_dn)
        proj = tl.load(Proj_ptr + pid * stride_pn + offs_f * stride_pf, mask=mask, other=0.0).to(tl.float32)

        diff = proj - diag_sq
        diff_clamped = tl.maximum(-20.0, tl.minimum(8.0, diff))
        phi = tl.exp(diff_clamped) * norm_factor

        tl.store(Out_ptr + pid * stride_on + offs_f * stride_of, phi.to(Out_ptr.dtype.element_ty), mask=mask)


def fused_favor_plus_feature_map_triton(
    x: torch.Tensor,
    omega: torch.Tensor,
    num_features: int,
    kernel_type: str = "favor_plus",
    eps: float = 1e-4,
) -> torch.Tensor:
    """Fused FAVOR+ feature map in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if not (HAS_TRITON and not FORCE_PYTORCH_FALLBACK and x.is_cuda and omega.is_cuda):
        return fused_favor_plus_feature_map_pytorch(x, omega, num_features, kernel_type, eps)

    try:
        omega_t = omega.to(dtype=x.dtype, device=x.device).t().contiguous()
        proj = torch.matmul(x, omega_t)

        if kernel_type in ("favor_plus", "exp", "favor"):
            orig_shape = proj.shape
            F_dim = orig_shape[-1]
            proj_2d = proj.contiguous().view(-1, F_dim)
            x_2d = x.contiguous().view(-1, x.shape[-1])
            total_tokens = proj_2d.shape[0]

            diag_sq = (torch.sum(x_2d**2, dim=-1, keepdim=True) * 0.5).contiguous()
            out_2d = torch.empty_like(proj_2d)

            block_f = triton.next_power_of_2(F_dim)
            norm_factor = 1.0 / math.sqrt(num_features)
            grid = (total_tokens,)
            _favor_plus_act_kernel[grid](
                proj_2d, diag_sq, out_2d,
                total_tokens, F_dim=F_dim, norm_factor=float(norm_factor),
                stride_pn=proj_2d.stride(0), stride_pf=proj_2d.stride(1),
                stride_dn=diag_sq.stride(0),
                stride_on=out_2d.stride(0), stride_of=out_2d.stride(1),
                BLOCK_F=block_f,
            )
            return out_2d.view(orig_shape)
        elif kernel_type == "elu":
            phi = (F.elu(proj) + 1.0) / math.sqrt(num_features)
            return phi
        else:
            phi = (F.relu(proj) + eps) / math.sqrt(num_features)
            return phi
    except Exception:
        return fused_favor_plus_feature_map_pytorch(x, omega, num_features, kernel_type, eps)


if HAS_TRITON and tl is not None:
    @triton.jit
    def _fused_entropy_row_kernel(
        Scores_ptr, Entropy_ptr,
        total_rows, L: tl.constexpr,
        stride_sb, stride_sl,
        stride_eb, stride_el,
        BLOCK_L: tl.constexpr,
    ):
        """In-SRAM Softmax and Entropy reduction row-wise."""
        pid = tl.program_id(0)
        if pid >= total_rows:
            return

        offs_l = tl.arange(0, BLOCK_L)
        mask = offs_l < L

        s = tl.load(Scores_ptr + pid * stride_sb + offs_l * stride_sl, mask=mask, other=-1e4).to(tl.float32)
        m = tl.max(s, axis=0)
        p = tl.exp(s - m)
        denom = tl.sum(p, axis=0)
        prob = p / tl.maximum(denom, 1e-8)

        log_prob = tl.log(tl.maximum(prob, 1e-8))
        h = -tl.sum(prob * log_prob, axis=0)

        tl.store(Entropy_ptr + pid * stride_eb, h.to(Entropy_ptr.dtype.element_ty))


def fused_entropy_token_pruning_mask_pytorch(
    scores: torch.Tensor,
    keep_ratio: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback for entropy token pruning."""
    if scores.dim() == 2:
        probs = F.softmax(scores.float(), dim=-1)
        entropy = -probs * torch.log(probs + 1e-8)  # (B, L)
        B, L = entropy.shape
    elif scores.dim() >= 3:
        probs = F.softmax(scores.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, H, L) or (B, L)
        if entropy.dim() == 3:
            entropy = entropy.mean(dim=1)  # (B, L)
        B, L = entropy.shape
    else:
        raise ValueError(f"scores must be at least 2D, got shape {scores.shape}")

    k = max(1, min(L, int(math.ceil(L * keep_ratio))))
    topk_vals, topk_indices = torch.topk(entropy, k=k, dim=-1)
    mask = torch.zeros(B, L, device=scores.device, dtype=torch.bool).scatter_(-1, topk_indices, True)
    # Always preserve first token ([CLS]/BOS)
    mask[:, 0] = True
    return mask, entropy.to(scores.dtype)


def fused_entropy_token_pruning_mask_triton(
    scores: torch.Tensor,
    keep_ratio: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused entropy-guided token pruning in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and scores.is_cuda and scores.dim() in (2, 3):
        try:
            L = scores.shape[-1]
            if scores.dim() == 2:
                B = scores.shape[0]
                scores_2d = (scores.float() / max(temperature, 1e-4)).contiguous()
                entropy = torch.empty((B,), device=scores.device, dtype=scores.dtype)
                block_l = triton.next_power_of_2(L)
                grid = (B,)
                _fused_entropy_row_kernel[grid](
                    scores_2d, entropy,
                    B, L=L,
                    stride_sb=scores_2d.stride(0), stride_sl=scores_2d.stride(1),
                    stride_eb=entropy.stride(0), stride_el=1,
                    BLOCK_L=block_l,
                )
            else:
                B, H, _ = scores.shape
                scores_2d = (scores.float() / max(temperature, 1e-4)).view(-1, L).contiguous()
                ent_bh = torch.empty((B * H,), device=scores.device, dtype=scores.dtype)
                block_l = triton.next_power_of_2(L)
                grid = (B * H,)
                _fused_entropy_row_kernel[grid](
                    scores_2d, ent_bh,
                    B * H, L=L,
                    stride_sb=scores_2d.stride(0), stride_sl=scores_2d.stride(1),
                    stride_eb=ent_bh.stride(0), stride_el=1,
                    BLOCK_L=block_l,
                )
                entropy = ent_bh.view(B, H).mean(dim=1)

            k = max(1, min(L, int(math.ceil(L * keep_ratio))))
            _, topk_indices = torch.topk(entropy, k=k, dim=-1)
            mask = torch.zeros(B, L, device=scores.device, dtype=torch.bool).scatter_(-1, topk_indices, True)
            mask[:, 0] = True
            return mask, entropy.to(scores.dtype)
        except Exception:
            pass
    return fused_entropy_token_pruning_mask_pytorch(scores, keep_ratio, temperature)


def fused_aegis_multibranch_gate_norm_pytorch(
    branches: list[torch.Tensor],
    gate_logits: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Exact PyTorch fallback for Aegis multi-branch RMSNorm + router combination."""
    M = len(branches)
    weights = F.softmax(gate_logits.float() / max(tau, 1e-4), dim=-1)  # (B, L, M)
    out = torch.zeros_like(branches[0])
    for m in range(M):
        bm = branches[m].float()
        rms = torch.rsqrt(bm.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed_bm = bm * rms
        out = out + (normed_bm * weights[..., m:m+1]).to(branches[0].dtype)
    return out


if HAS_TRITON and tl is not None:
    @triton.jit
    def _aegis_multibranch_gate_norm_2branch_kernel(
        B0_ptr, B1_ptr, W_ptr, OUT_ptr,
        total_tokens, D: tl.constexpr, eps,
        stride_b0n, stride_b0d,
        stride_b1n, stride_b1d,
        stride_wn, stride_wm,
        stride_on, stride_od,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= total_tokens:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        # Load weights
        w0 = tl.load(W_ptr + pid * stride_wn + 0 * stride_wm).to(tl.float32)
        w1 = tl.load(W_ptr + pid * stride_wn + 1 * stride_wm).to(tl.float32)

        # Branch 0
        b0 = tl.load(B0_ptr + pid * stride_b0n + offs_d * stride_b0d, mask=mask, other=0.0).to(tl.float32)
        mean_sq0 = tl.sum(b0 * b0, axis=0) / tl.cast(D, tl.float32)
        inv_rms0 = tl.rsqrt(mean_sq0 + eps)
        normed_b0 = b0 * inv_rms0

        # Branch 1
        b1 = tl.load(B1_ptr + pid * stride_b1n + offs_d * stride_b1d, mask=mask, other=0.0).to(tl.float32)
        mean_sq1 = tl.sum(b1 * b1, axis=0) / tl.cast(D, tl.float32)
        inv_rms1 = tl.rsqrt(mean_sq1 + eps)
        normed_b1 = b1 * inv_rms1

        out = normed_b0 * w0 + normed_b1 * w1
        tl.store(OUT_ptr + pid * stride_on + offs_d * stride_od, out.to(OUT_ptr.dtype.element_ty), mask=mask)


def fused_aegis_multibranch_gate_norm_triton(
    branches: list[torch.Tensor],
    gate_logits: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused Aegis multi-branch gating and normalization in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    M = len(branches)
    if (
        HAS_TRITON
        and not FORCE_PYTORCH_FALLBACK
        and branches[0].is_cuda
        and M == 2
        and branches[0].shape == branches[1].shape
    ):
        try:
            orig_shape = branches[0].shape
            D = orig_shape[-1]
            b0_2d = branches[0].contiguous().view(-1, D)
            b1_2d = branches[1].contiguous().view(-1, D)
            total_tokens = b0_2d.shape[0]

            weights = F.softmax(gate_logits.float() / max(tau, 1e-4), dim=-1).view(-1, M).contiguous()
            out_2d = torch.empty_like(b0_2d)

            block_d = triton.next_power_of_2(D)
            grid = (total_tokens,)
            _aegis_multibranch_gate_norm_2branch_kernel[grid](
                b0_2d, b1_2d, weights, out_2d,
                total_tokens, D=D, eps=float(eps),
                stride_b0n=b0_2d.stride(0), stride_b0d=b0_2d.stride(1),
                stride_b1n=b1_2d.stride(0), stride_b1d=b1_2d.stride(1),
                stride_wn=weights.stride(0), stride_wm=weights.stride(1),
                stride_on=out_2d.stride(0), stride_od=out_2d.stride(1),
                BLOCK_D=block_d,
            )
            return out_2d.view(orig_shape)
        except Exception:
            pass
    return fused_aegis_multibranch_gate_norm_pytorch(branches, gate_logits, tau, eps)


def fused_split_head_hybrid_concat_norm_pytorch(
    head_groups: list[torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Exact PyTorch fallback for heterogeneous split-head Group-RMSNorm."""
    normed_groups = []
    for g in head_groups:
        gf = g.float()
        rms = torch.rsqrt(gf.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed_groups.append((gf * rms).to(g.dtype))
    return torch.cat(normed_groups, dim=-1)


if HAS_TRITON and tl is not None:
    @triton.jit
    def _split_head_rmsnorm_kernel(
        X_ptr, OUT_ptr,
        total_rows, D: tl.constexpr, eps,
        stride_xn, stride_xd,
        stride_on, stride_od,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= total_rows:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        x = tl.load(X_ptr + pid * stride_xn + offs_d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        sq = x * x
        mean_sq = tl.sum(sq, axis=0) / tl.cast(D, tl.float32)
        inv_rms = tl.rsqrt(mean_sq + eps)

        normed_x = x * inv_rms
        tl.store(OUT_ptr + pid * stride_on + offs_d * stride_od, normed_x.to(OUT_ptr.dtype.element_ty), mask=mask)


def fused_split_head_hybrid_concat_norm_triton(
    head_groups: list[torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused heterogeneous split-head Group-RMSNorm in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and head_groups[0].is_cuda:
        try:
            normed_groups = []
            for g in head_groups:
                orig_shape = g.shape
                D = orig_shape[-1]
                g_2d = g.contiguous().view(-1, D)
                total_rows = g_2d.shape[0]
                out_2d = torch.empty_like(g_2d)

                block_d = triton.next_power_of_2(D)
                grid = (total_rows,)
                _split_head_rmsnorm_kernel[grid](
                    g_2d, out_2d,
                    total_rows, D=D, eps=float(eps),
                    stride_xn=g_2d.stride(0), stride_xd=g_2d.stride(1),
                    stride_on=out_2d.stride(0), stride_od=out_2d.stride(1),
                    BLOCK_D=block_d,
                )
                normed_groups.append(out_2d.view(orig_shape))
            return torch.cat(normed_groups, dim=-1)
        except Exception:
            pass
    return fused_split_head_hybrid_concat_norm_pytorch(head_groups, eps)


def fused_entmax_bisect_pytorch(
    scores: torch.Tensor,
    alpha: float = 1.5,
    valid_mask: torch.Tensor | None = None,
    dim: int = -1,
    n_iter: int = 15,
) -> torch.Tensor:
    """Exact stabilized bisection for alpha-entmax in PyTorch."""
    device = scores.device
    dtype = scores.dtype
    neg_inf_val = torch.tensor(-10000.0, device=device, dtype=dtype)
    scores_cl = torch.where(torch.isfinite(scores), scores, neg_inf_val)

    if abs(alpha - 1.0) < 1e-4:
        if valid_mask is not None:
            scores_cl = scores_cl.masked_fill(~valid_mask, -10000.0)
        sf = torch.nan_to_num(F.softmax(scores_cl.float(), dim=dim), nan=0.0).to(dtype)
        if valid_mask is not None:
            sf = sf.masked_fill(~valid_mask, 0.0)
        return (sf / sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)).to(dtype)

    alpha_m1 = max(1e-4, alpha - 1.0)
    inv_alpha_m1 = 1.0 / alpha_m1
    scores_valid = scores_cl.masked_fill(~valid_mask, -10000.0) if valid_mask is not None else scores_cl

    low = scores_valid.max(dim=dim, keepdim=True)[0] - 2.0
    high = scores_valid.max(dim=dim, keepdim=True)[0]

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        p = torch.clamp((scores_valid - mid) * alpha_m1, min=0.0) ** inv_alpha_m1
        if valid_mask is not None:
            p = p.masked_fill(~valid_mask, 0.0)
        diff = p.sum(dim=dim, keepdim=True) - 1.0
        low = torch.where(diff > 0, mid, low)
        high = torch.where(diff <= 0, mid, high)

    mid = 0.5 * (low + high)
    p = torch.clamp((scores_valid - mid) * alpha_m1, min=0.0) ** inv_alpha_m1
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, 0.0)
    return (p / p.sum(dim=dim, keepdim=True).clamp(min=1e-9)).to(dtype)


if HAS_TRITON:
    @triton.jit
    def _entmax15_bisect_fwd_kernel(
        Scores_ptr,
        Out_ptr,
        stride_sb,
        stride_sn,
        stride_ob,
        stride_on,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        N_ITER: tl.constexpr,
    ):
        """Compiled Triton kernel for row-wise exact alpha=1.5 entmax via register bisection."""
        pid = tl.program_id(0)
        row_s = Scores_ptr + pid * stride_sb
        row_o = Out_ptr + pid * stride_ob

        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(row_s + offs * stride_sn, mask=mask, other=-10000.0).to(tl.float32)

        # 1. Row max
        m = tl.max(tl.where(mask, x, -10000.0), axis=0)

        # 2. Bisection bounds: low = m - 2.0, high = m
        low = m - 2.0
        high = m

        for _ in range(N_ITER):
            mid = 0.5 * (low + high)
            # p = clamp((x - mid) * 0.5, min=0.0) ** 2.0
            diff = (x - mid) * 0.5
            p = tl.where(diff > 0.0, diff * diff, 0.0)
            p = tl.where(mask, p, 0.0)
            total = tl.sum(p, axis=0)
            if total > 1.0:
                low = mid
            else:
                high = mid

        final_mid = 0.5 * (low + high)
        diff_final = (x - final_mid) * 0.5
        p_final = tl.where(diff_final > 0.0, diff_final * diff_final, 0.0)
        p_final = tl.where(mask, p_final, 0.0)
        denom = tl.sum(p_final, axis=0)
        denom_safe = tl.maximum(denom, 1e-9)
        out_val = p_final / denom_safe

        tl.store(row_o + offs * stride_on, out_val.to(Out_ptr.dtype.element_ty), mask=mask)


def fused_entmax_bisect_triton(
    scores: torch.Tensor,
    alpha: float = 1.5,
    valid_mask: torch.Tensor | None = None,
    dim: int = -1,
    n_iter: int = 15,
) -> torch.Tensor:
    """Dispatches genuine compiled Triton alpha=1.5 entmax bisection kernel."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK
    if (
        HAS_TRITON
        and not FORCE_PYTORCH_FALLBACK
        and scores.is_cuda
        and abs(alpha - 1.5) < 1e-3
        and valid_mask is None
        and dim in (-1, scores.ndim - 1)
        and scores.shape[-1] <= 4096
    ):
        try:
            orig_shape = scores.shape
            N = orig_shape[-1]
            scores_2d = scores.contiguous().view(-1, N)
            num_rows = scores_2d.shape[0]
            out = torch.empty_like(scores_2d)

            BLOCK_SIZE = triton.next_power_of_2(N)
            grid = (num_rows,)
            _entmax15_bisect_fwd_kernel[grid](
                scores_2d,
                out,
                scores_2d.stride(0),
                scores_2d.stride(1),
                out.stride(0),
                out.stride(1),
                N=N,
                BLOCK_SIZE=BLOCK_SIZE,
                N_ITER=n_iter,
            )
            return out.view(orig_shape)
        except Exception:
            pass
    return fused_entmax_bisect_pytorch(scores, alpha=alpha, valid_mask=valid_mask, dim=dim, n_iter=n_iter)


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
    "fused_favor_plus_feature_map_triton",
    "fused_favor_plus_feature_map_pytorch",
    "fused_entropy_token_pruning_mask_triton",
    "fused_entropy_token_pruning_mask_pytorch",
    "fused_aegis_multibranch_gate_norm_triton",
    "fused_aegis_multibranch_gate_norm_pytorch",
    "fused_split_head_hybrid_concat_norm_triton",
    "fused_split_head_hybrid_concat_norm_pytorch",
    "fused_entmax_bisect_triton",
    "fused_entmax_bisect_pytorch",
]