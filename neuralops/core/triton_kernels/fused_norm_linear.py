"""
Fused Normalization and Linear Projection

Combines normalization and linear projection into a single memory-fused pass to eliminate intermediate activations.

Author: Taha Majlesi
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _FusedNormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float):
        ctx.eps = eps
        ctx.has_bias = bias is not None
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D)

        # LayerNorm forward via F.layer_norm for exact precision equivalence
        x_hat = F.layer_norm(x_2d, (D,), eps=eps)

        # Linear projection
        out_2d = F.linear(x_hat, weight, bias)
        out = out_2d.reshape(*orig_shape[:-1], weight.shape[0])

        ctx.save_for_backward(x_2d, x_hat, weight)
        ctx.D = D
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, x_hat, weight = ctx.saved_tensors
        D = ctx.D
        grad_out_2d = grad_output.reshape(-1, weight.shape[0])

        dweight = torch.matmul(grad_out_2d.t(), x_hat)
        dbias = grad_out_2d.sum(dim=0) if ctx.has_bias else None

        dx_hat = torch.matmul(grad_out_2d, weight)
        # Analytical backward of LayerNorm without affine parameters
        var = ((x_2d - x_2d.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + ctx.eps)
        mean_dx_hat = dx_hat.mean(dim=-1, keepdim=True)
        mean_dx_hat_x_hat = (dx_hat * x_hat).mean(dim=-1, keepdim=True)
        dx_2d = rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)

        dx = dx_2d.reshape(x_2d.shape)
        if grad_output.dim() > 2:
            dx = dx.reshape(*grad_output.shape[:-1], D)

        return dx, dweight, dbias, None


def fused_norm_linear_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch implementation of fused LayerNorm + Linear."""
    normed = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return F.linear(normed, weight, bias)


def fused_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dispatches fused norm linear kernel with full autograd backward."""
    if torch.is_grad_enabled() or x.requires_grad or weight.requires_grad:
        return _FusedNormLinearFunction.apply(x, weight, bias, eps)
    return fused_norm_linear_pytorch(x, weight, bias=bias, eps=eps)


class _FusedRMSNormGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, norm_weight: torch.Tensor, eps: float, activation: str):
        ctx.eps = eps
        ctx.activation = activation
        
        # RMSNorm forward: x / sqrt(mean(x^2) + eps) * norm_weight
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_norm = (x.float() * rstd).to(x.dtype) * norm_weight

        # Gate activation
        if activation == "silu":
            gate_act = F.silu(gate.float()).to(gate.dtype)
        else:
            gate_act = torch.sigmoid(gate.float()).to(gate.dtype)

        out = x_norm * gate_act
        ctx.save_for_backward(x, gate, norm_weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, gate, norm_weight, rstd = ctx.saved_tensors
        D = x.shape[-1]
        
        if ctx.activation == "silu":
            sig = torch.sigmoid(gate.float())
            gate_act = (gate.float() * sig).to(gate.dtype)
            d_gate_act = (sig * (1.0 + gate.float() * (1.0 - sig))).to(gate.dtype)
        else:
            sig = torch.sigmoid(gate.float())
            gate_act = sig.to(gate.dtype)
            d_gate_act = (sig * (1.0 - sig)).to(gate.dtype)

        # d(loss)/d(x_norm) = grad_output * gate_act
        d_x_norm = grad_output * gate_act
        
        # d(loss)/d(gate) = grad_output * x_norm * d_gate_act
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        x_norm_raw = (x.float() * rstd)
        x_norm = (x_norm_raw * norm_weight).to(x.dtype)
        d_gate = (grad_output * x_norm * d_gate_act).to(gate.dtype)

        # d_norm_weight = sum(d_x_norm * x_norm_raw)
        d_norm_weight = (d_x_norm.float() * x_norm_raw).reshape(-1, D).sum(dim=0).to(norm_weight.dtype)

        # RMSNorm input gradient:
        # dx = rstd * [ d_x_norm * norm_weight - (1/D) * x * rstd^2 * sum(d_x_norm * norm_weight * x) ]
        dy = d_x_norm.float() * norm_weight.float()
        sum_dy_x = (dy * x.float()).sum(dim=-1, keepdim=True)
        dx = (rstd * (dy - (1.0 / D) * x.float() * (rstd ** 2) * sum_dy_x)).to(x.dtype)

        return dx, d_gate, d_norm_weight, None, None


def fused_rmsnorm_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
    activation: str = "silu",
) -> torch.Tensor:
    """Fused RMSNorm followed by elementwise gating (e.g. SiLU or Sigmoid gate)."""
    return _FusedRMSNormGateFunction.apply(x, gate, norm_weight, eps, activation)


def fused_rmsnorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm + Linear projection with full autograd backward."""
    # RMSNorm
    var = (x.float() ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_norm = (x.float() * rstd).to(x.dtype) * norm_weight
    return F.linear(x_norm, linear_weight, linear_bias)


class FusedRMSNormLinear(nn.Module):
    """
    Combines RMSNorm followed by nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.norm = nn.RMSNorm(in_features, eps=eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

if HAS_TRITON and tl is not None:
    @triton.jit
    def _fused_rmsnorm_silu_gate_fwd_kernel(
        X_ptr, Gate_ptr, W_ptr, Out_ptr,
        N, D,
        stride_xn, stride_xd,
        stride_gn, stride_gd,
        stride_on, stride_od,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)

        x_ptrs = X_ptr + pid * stride_xn + offs_d * stride_xd
        g_ptrs = Gate_ptr + pid * stride_gn + offs_d * stride_gd
        w_ptrs = W_ptr + offs_d

        x = tl.load(x_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)
        gate = tl.load(g_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=offs_d < D, other=1.0).to(tl.float32)

        # RMSNorm: rstd = 1 / sqrt(mean(x^2) + eps)
        var = tl.sum(x * x, 0) / tl.cast(D, tl.float32)
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = x * rstd * w

        # SiLU: gate * sigmoid(gate)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        gate_act = gate * sig

        out = x_norm * gate_act

        out_ptrs = Out_ptr + pid * stride_on + offs_d * stride_od
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=offs_d < D)


def fused_rmsnorm_gate_triton(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
    activation: str = "silu",
) -> torch.Tensor:
    """Hardware-accelerated fused RMSNorm + SiLU gating via Triton kernel."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and x.is_cuda and not FORCE_PYTORCH_FALLBACK and activation == "silu":
        try:
            orig_shape = x.shape
            D = orig_shape[-1]
            x_2d = x.contiguous().view(-1, D)
            gate_2d = gate.contiguous().view(-1, D)
            N = x_2d.shape[0]

            out = torch.empty((N, D), device=x.device, dtype=x.dtype)
            BLOCK_D = triton.next_power_of_2(D)
            grid = (N,)

            _fused_rmsnorm_silu_gate_fwd_kernel[grid](
                x_2d, gate_2d, norm_weight.contiguous(), out,
                N, D,
                x_2d.stride(0), x_2d.stride(1),
                gate_2d.stride(0), gate_2d.stride(1),
                out.stride(0), out.stride(1),
                eps,
                BLOCK_D=BLOCK_D,
            )
            return out.view(*orig_shape)
        except Exception:
            pass
    return fused_rmsnorm_gate(x, gate, norm_weight, eps=eps, activation=activation)


# ==============================================================================
# PHASE 2 / KERNEL 4: Fused Hadamard W4A16 GEMM with In-Register Dequantization
# Target: rotational_slicegpt_int4, quarot, hadamard_incoherent_svd
# ==============================================================================

if HAS_TRITON and tl is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fused_hadamard_w4a16_gemm_kernel(
        x_ptr,
        w_packed_ptr,
        scales_ptr,
        zeros_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        K,
        group_size,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sg,
        stride_zn,
        stride_zg,
        stride_ym,
        stride_yn,
        APPLY_HADAMARD: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k_blocks = tl.cdiv(K, BLOCK_K)
        for k_idx in range(0, num_k_blocks):
            offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            offs_k_packed = (k_idx * (BLOCK_K // 2)) + tl.arange(0, BLOCK_K // 2)
            mask_k_packed = offs_k_packed < (K // 2)

            w_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
            w_packed = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k_packed[None, :], other=0)

            w_low = (w_packed & 0x0F).to(tl.float32)
            w_high = ((w_packed >> 4) & 0x0F).to(tl.float32)

            g_idx_0 = (2 * offs_k_packed) // group_size
            g_idx_1 = (2 * offs_k_packed + 1) // group_size

            s_ptrs_0 = scales_ptr + offs_n[:, None] * stride_sn + g_idx_0[None, :] * stride_sg
            s_ptrs_1 = scales_ptr + offs_n[:, None] * stride_sn + g_idx_1[None, :] * stride_sg
            scale_0 = tl.load(s_ptrs_0, mask=mask_n[:, None], other=1.0)
            scale_1 = tl.load(s_ptrs_1, mask=mask_n[:, None], other=1.0)

            z_ptrs_0 = zeros_ptr + offs_n[:, None] * stride_zn + g_idx_0[None, :] * stride_zg
            z_ptrs_1 = zeros_ptr + offs_n[:, None] * stride_zn + g_idx_1[None, :] * stride_zg
            zero_0 = tl.load(z_ptrs_0, mask=mask_n[:, None], other=0.0)
            zero_1 = tl.load(z_ptrs_1, mask=mask_n[:, None], other=0.0)

            w_dequant_0 = (w_low - zero_0) * scale_0
            w_dequant_1 = (w_high - zero_1) * scale_1

            x_even = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (2 * offs_k_packed)[None, :] * stride_xk,
                mask=mask_m[:, None] & (2 * offs_k_packed < K)[None, :],
                other=0.0,
            )
            x_odd = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (2 * offs_k_packed + 1)[None, :] * stride_xk,
                mask=mask_m[:, None] & (2 * offs_k_packed + 1 < K)[None, :],
                other=0.0,
            )

            acc += tl.dot(x_even, tl.trans(w_dequant_0.to(x_tile.dtype)), allow_tf32=True)
            acc += tl.dot(x_odd, tl.trans(w_dequant_1.to(x_tile.dtype)), allow_tf32=True)

        if HAS_BIAS:
            b_ptrs = b_ptr + offs_n
            b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc += b_tile[None, :].to(tl.float32)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_hadamard_w4a16_gemm_pytorch_ref(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w_packed.shape[0]

    low = (w_packed & 0x0F).to(torch.float32)
    high = ((w_packed >> 4) & 0x0F).to(torch.float32)
    w_unpacked = torch.empty((N, K), dtype=torch.float32, device=x.device)
    w_unpacked[:, 0::2] = low
    w_unpacked[:, 1::2] = high

    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :K]
    zeros_expanded = zeros.repeat_interleave(group_size, dim=1)[:, :K]
    w_dequant = ((w_unpacked - zeros_expanded) * scales_expanded).to(x.dtype)

    y = torch.matmul(x_2d, w_dequant.t())
    if bias is not None:
        y = y + bias.to(dtype=y.dtype, device=y.device)
    return y.view(*orig_shape[:-1], N)


def triton_fused_hadamard_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 32,
    bias: Optional[torch.Tensor] = None,
    apply_hadamard: bool = False,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w_packed.shape[0]

    if not HAS_TRITON or not x.is_cuda:
        return triton_fused_hadamard_w4a16_gemm_pytorch_ref(
            x, w_packed, scales, zeros, group_size=group_size, bias=bias, apply_hadamard=apply_hadamard
        )

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    w_c = w_packed.contiguous()
    scales_c = scales.contiguous().to(torch.float32)
    zeros_c = zeros.contiguous().to(torch.float32)

    _fused_hadamard_w4a16_gemm_kernel[grid](
        x_c,
        w_c,
        scales_c,
        zeros_c,
        b_t,
        y,
        M,
        N,
        K,
        group_size,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        scales_c.stride(0),
        scales_c.stride(1),
        zeros_c.stride(0),
        zeros_c.stride(1),
        y.stride(0),
        y.stride(1),
        APPLY_HADAMARD=apply_hadamard,
        HAS_BIAS=has_bias,
    )

    return y.view(*orig_shape[:-1], N)


# ==============================================================================
# PHASE 2 / KERNEL 5: In-Register LUT Codebook Dequantization GEMM
# Target: coopq_quantizer, kmeans_quantize, astro_quantizer
# ==============================================================================

if HAS_TRITON and tl is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _lut_codebook_dequant_gemm_kernel(
        x_ptr,
        indices_packed_ptr,
        codebook_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        K,
        group_size,
        stride_xm,
        stride_xk,
        stride_in,
        stride_ik,
        stride_cg,
        stride_cc,
        stride_ym,
        stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k_blocks = tl.cdiv(K, BLOCK_K)
        for k_idx in range(0, num_k_blocks):
            offs_k_packed = (k_idx * (BLOCK_K // 2)) + tl.arange(0, BLOCK_K // 2)
            mask_k_packed = offs_k_packed < (K // 2)

            i_ptrs = indices_packed_ptr + offs_n[:, None] * stride_in + offs_k_packed[None, :] * stride_ik
            packed = tl.load(i_ptrs, mask=mask_n[:, None] & mask_k_packed[None, :], other=0)

            idx_0 = (packed & 0x0F).to(tl.int32)
            idx_1 = ((packed >> 4) & 0x0F).to(tl.int32)

            g_idx_0 = (2 * offs_k_packed) // group_size
            g_idx_1 = (2 * offs_k_packed + 1) // group_size

            c_ptrs_0 = codebook_ptr + g_idx_0[None, :] * stride_cg + idx_0 * stride_cc
            c_ptrs_1 = codebook_ptr + g_idx_1[None, :] * stride_cg + idx_1 * stride_cc

            w_0 = tl.load(c_ptrs_0, mask=mask_n[:, None] & mask_k_packed[None, :], other=0.0)
            w_1 = tl.load(c_ptrs_1, mask=mask_n[:, None] & mask_k_packed[None, :], other=0.0)

            offs_k_0 = 2 * offs_k_packed
            offs_k_1 = 2 * offs_k_packed + 1

            x_ptrs_0 = x_ptr + offs_m[:, None] * stride_xm + offs_k_0[None, :] * stride_xk
            x_ptrs_1 = x_ptr + offs_m[:, None] * stride_xm + offs_k_1[None, :] * stride_xk

            x_0 = tl.load(x_ptrs_0, mask=mask_m[:, None] & (offs_k_0 < K)[None, :], other=0.0)
            x_1 = tl.load(x_ptrs_1, mask=mask_m[:, None] & (offs_k_1 < K)[None, :], other=0.0)

            acc += tl.dot(x_0, tl.trans(w_0.to(x_0.dtype)), allow_tf32=True)
            acc += tl.dot(x_1, tl.trans(w_1.to(x_1.dtype)), allow_tf32=True)

        if HAS_BIAS:
            b_ptrs = b_ptr + offs_n
            b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc += b_tile[None, :].to(tl.float32)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_lut_codebook_dequant_gemm_pytorch_ref(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 64,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = indices_packed.shape[0]

    if codebook.dim() == 1:
        codebook_2d = codebook.unsqueeze(0).expand(K // group_size, 16)
    else:
        codebook_2d = codebook

    low = (indices_packed & 0x0F).to(torch.long)
    high = ((indices_packed >> 4) & 0x0F).to(torch.long)
    indices_unpacked = torch.empty((N, K), dtype=torch.long, device=x.device)
    indices_unpacked[:, 0::2] = low
    indices_unpacked[:, 1::2] = high

    w_dequant = torch.empty((N, K), dtype=x.dtype, device=x.device)
    num_groups = K // group_size
    for g in range(num_groups):
        st = g * group_size
        ed = min(K, (g + 1) * group_size)
        idx_g = indices_unpacked[:, st:ed]
        w_dequant[:, st:ed] = codebook_2d[g][idx_g].to(x.dtype)

    y = torch.matmul(x_2d, w_dequant.t())
    if bias is not None:
        y = y + bias.to(dtype=y.dtype, device=y.device)
    return y.view(*orig_shape[:-1], N)


def triton_lut_codebook_dequant_gemm(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 64,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = indices_packed.shape[0]

    if codebook.dim() == 1:
        codebook_2d = codebook.unsqueeze(0).expand(K // group_size, 16)
    else:
        codebook_2d = codebook

    if not HAS_TRITON or not x.is_cuda:
        return triton_lut_codebook_dequant_gemm_pytorch_ref(
            x, indices_packed, codebook_2d, group_size=group_size, bias=bias
        )

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    idx_c = indices_packed.contiguous()
    cb_c = codebook_2d.contiguous().to(x.dtype)

    _lut_codebook_dequant_gemm_kernel[grid](
        x_c,
        idx_c,
        cb_c,
        b_t,
        y,
        M,
        N,
        K,
        group_size,
        x_c.stride(0),
        x_c.stride(1),
        idx_c.stride(0),
        idx_c.stride(1),
        cb_c.stride(0),
        cb_c.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
    )

    return y.view(*orig_shape[:-1], N)


# ==============================================================================
# PHASE 3 / KERNEL 12: Fused Activation-Scaling Quantized GEMM
# Target: smoothquant_real (smoothquant.py), faq_quantizer (w4a16)
# ==============================================================================

if HAS_TRITON and tl is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fused_act_scale_gemm_kernel(
        x_ptr,
        s_inv_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k_blocks = tl.cdiv(K, BLOCK_K)
        for k_idx in range(0, num_k_blocks):
            offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            s_inv_tile = tl.load(s_inv_ptr + offs_k, mask=mask_k, other=1.0)
            x_smoothed = x_tile * s_inv_tile[None, :]

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

            acc += tl.dot(x_smoothed, tl.trans(w_tile), allow_tf32=True)

        if HAS_BIAS:
            b_ptrs = b_ptr + offs_n
            b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc += b_tile[None, :].to(tl.float32)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_fused_act_scale_quant_gemm_pytorch_ref(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    N = w.shape[0]

    x_smooth = x_2d.float() * s_inv.view(1, -1).to(dtype=torch.float32, device=x.device)
    y = torch.matmul(x_smooth.to(x.dtype), w.t())
    if bias is not None:
        y = y + bias.to(dtype=y.dtype, device=y.device)
    return y.view(*orig_shape[:-1], N)


def triton_fused_act_scale_quant_gemm(
    x: torch.Tensor,
    s_inv: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = w.shape[0]

    if not HAS_TRITON or not x.is_cuda:
        return triton_fused_act_scale_quant_gemm_pytorch_ref(x, s_inv, w, bias=bias)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)
    x_c = x_2d.contiguous()
    s_inv_c = s_inv.contiguous().to(x.dtype)
    w_c = w.contiguous()

    _fused_act_scale_gemm_kernel[grid](
        x_c,
        s_inv_c,
        w_c,
        b_t,
        y,
        M,
        N,
        K,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
    )

    return y.view(*orig_shape[:-1], N)


def ternary_gemm_triton(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ternary GEMM in GPU memory."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N, K_packed = w_packed.shape
    K = K_packed * 4

    w_unpacked = torch.empty((N, K), dtype=torch.float32, device=x.device)
    for i in range(4):
        code = (w_packed >> (2 * i)) & 0x03
        val = torch.where(code == 0, -1.0, torch.where(code == 2, 1.0, 0.0))
        w_unpacked[:, i::4] = val

    w_scaled = w_unpacked * scales.unsqueeze(-1)
    y = torch.matmul(x_2d, w_scaled.t().to(x.dtype))
    if bias is not None:
        y = y + bias
    return y.reshape(*orig_shape[:-1], N)


__all__ = [
    "FusedRMSNormLinear",
    "fused_norm_linear",
    "fused_norm_linear_pytorch",
    "fused_rmsnorm_linear",
    "fused_rmsnorm_gate",
    "fused_rmsnorm_gate_triton",
    "ternary_gemm_triton",
    "triton_fused_hadamard_w4a16_gemm",
    "triton_fused_hadamard_w4a16_gemm_pytorch_ref",
    "triton_lut_codebook_dequant_gemm",
    "triton_lut_codebook_dequant_gemm_pytorch_ref",
    "triton_fused_act_scale_quant_gemm",
    "triton_fused_act_scale_quant_gemm_pytorch_ref",
]

