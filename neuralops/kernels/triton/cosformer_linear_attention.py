"""
Triton Cosformer Linear Attention Kernel (Phase 3 / Kernel 8).

Target Modules: cosformer, chela_slay, performer, decay_linear_attention
Solves:
    Applies non-negative feature map phi(x) = ELU(x) + 1 and cos/sin positional modulation:
        q_cos = phi(q) * cos(pi * i / (2 * L)), q_sin = phi(q) * sin(pi * i / (2 * L))
        k_cos = phi(k) * cos(pi * j / (2 * L)), k_sin = phi(k) * sin(pi * j / (2 * L))
    without materializing the 2x expanded channel dimensions in HBM.

Fuses feature map, positional encoding, and linear attention accumulation in registers.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _elu_plus_one(x):
    """In-register ELU(x) + 1."""
    return tl.where(x > 0.0, x + 1.0, tl.exp(x))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    ],
    key=["L", "D"],
)
@triton.jit
def _cosformer_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    H,
    L,
    D: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < L

    # Load Q block and compute modulated q_cos, q_sin on the fly
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    raw_q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    phi_q = _elu_plus_one(raw_q)

    # Positional angles theta = pi * i / (2 * L)
    theta_q = (3.141592653589793 / (2.0 * L)) * offs_m[:, None]
    cos_q = tl.cos(theta_q)
    sin_q = tl.sin(theta_q)

    q_cos = phi_q * cos_q
    q_sin = phi_q * sin_q

    # Numerator accumulator: shape (BLOCK_M, D)
    num_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    # Denominator accumulator: shape (BLOCK_M,)
    den_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Scan over KV sequence blocks
    max_n_blocks = tl.cdiv(L, BLOCK_N)
    for n_idx in range(0, max_n_blocks):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L

        k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd

        raw_k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        phi_k = _elu_plus_one(raw_k)
        theta_k = (3.141592653589793 / (2.0 * L)) * offs_n[:, None]
        k_cos = phi_k * tl.cos(theta_k)
        k_sin = phi_k * tl.sin(theta_k)

        # Dot product in modulated space: S = q_cos @ k_cos^T + q_sin @ k_sin^T
        s_tile = tl.dot(q_cos, tl.trans(k_cos), allow_tf32=True) + tl.dot(q_sin, tl.trans(k_sin), allow_tf32=True)

        if CAUSAL:
            # Mask out j > i
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_tile = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], s_tile, 0.0)
        else:
            s_tile = tl.where(mask_m[:, None] & mask_n[None, :], s_tile, 0.0)

        # Accumulate numerator = S @ V
        num_acc += tl.dot(s_tile, v_tile, allow_tf32=True)
        # Accumulate denominator = row sums of S
        den_acc += tl.sum(s_tile, axis=1)

    # Safe normalization: max(den, 1e-5)
    safe_den = tl.maximum(den_acc[:, None], 1e-5)
    out_tile = num_acc / safe_den

    o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_tile.to(q_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_cosformer_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    High-performance Triton Cosformer linear attention:
        Y_i = sum_j (phi(q_i) phi(k_j) * cos(theta_i - theta_j) * v_j) / sum_j (phi(q_i) phi(k_j) * cos(theta_i - theta_j))

    Args:
        q: (B, H, L, D)
        k: (B, H, L, D)
        v: (B, H, L, D)
        is_causal: If True, applies causal mask j <= i.

    Returns:
        Output tensor of shape (B, H, L, D).
    """
    B, H, L, D = q.shape

    if not q.is_cuda or D > 128:
        # PyTorch reference
        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0

        idx = torch.arange(L, device=q.device, dtype=torch.float32).view(1, 1, L, 1)
        theta = (math.pi / (2.0 * L)) * idx

        q_cos = phi_q * torch.cos(theta)
        q_sin = phi_q * torch.sin(theta)
        k_cos = phi_k * torch.cos(theta)
        k_sin = phi_k * torch.sin(theta)

        # Similarity matrix S = q_cos @ k_cos^T + q_sin @ k_sin^T
        scores = torch.matmul(q_cos, k_cos.transpose(-1, -2)) + torch.matmul(q_sin, k_sin.transpose(-1, -2))
        if is_causal:
            mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, 0.0)

        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        out = torch.matmul(weights, v.float()).to(q.dtype)
        return out

    out = torch.empty_like(v)

    grid = lambda META: (
        B,
        H,
        triton.cdiv(L, META["BLOCK_M"]),
    )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    _cosformer_fwd_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        B,
        H,
        L,
        D=D,
        stride_qb=q_c.stride(0),
        stride_qh=q_c.stride(1),
        stride_ql=q_c.stride(2),
        stride_qd=q_c.stride(3),
        stride_kb=k_c.stride(0),
        stride_kh=k_c.stride(1),
        stride_kl=k_c.stride(2),
        stride_kd=k_c.stride(3),
        stride_vb=v_c.stride(0),
        stride_vh=v_c.stride(1),
        stride_vl=v_c.stride(2),
        stride_vd=v_c.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
        CAUSAL=is_causal,
    )

    return out
