"""
Spectral Entropy SVD Dynamic Gating

Applies dynamic singular value truncation and entropy-based rank gating for adaptive low-rank projections.

Author: Taha Majlesi
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "R_MAX"],
)
@triton.jit
def _spectral_entropy_svd_kernel(
    x_ptr,
    v_ptr,
    u_ptr,
    s_weights_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    R_MAX,
    effective_rank,
    stride_xm,
    stride_xk,
    stride_vk,
    stride_vr,
    stride_ur,
    stride_un,
    stride_ym,
    stride_yn,
    HAS_WEIGHTS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    mask_m = offs_m < M
    mask_n = offs_n < N
    # Dynamic mask: only keep ranks < effective_rank and < R_MAX
    mask_r = (offs_r < effective_rank) & (offs_r < R_MAX)

    # 1. Compute Z_tile = X[offs_m, :] @ V_r^T [:, offs_r]
    z_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k_idx in range(0, num_k_blocks):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        v_ptrs = v_ptr + offs_k[:, None] * stride_vk + offs_r[None, :] * stride_vr

        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_k[:, None] & mask_r[None, :], other=0.0)

        z_acc += tl.dot(x_tile, v_tile, allow_tf32=True)

    # 2. In-Register Spectral Weighting
    if HAS_WEIGHTS:
        w_tile = tl.load(s_weights_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)
        z_acc = z_acc * w_tile[None, :]

    z_tile = z_acc.to(x_ptr.dtype.element_ty)

    # 3. Up-projection: Y_tile = Z_tile @ U_r^T [offs_r, offs_n]
    u_ptrs = u_ptr + offs_r[:, None] * stride_ur + offs_n[None, :] * stride_un
    u_tile = tl.load(u_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0)

    y_acc = tl.dot(z_tile, u_tile, allow_tf32=True)

    if HAS_BIAS:
        b_ptrs = b_ptr + offs_n
        b_tile = tl.load(b_ptrs, mask=mask_n, other=0.0)
        y_acc += b_tile[None, :].to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y_acc.to(x_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_spectral_entropy_svd_gate(
    x: torch.Tensor,
    V_r: torch.Tensor,
    U_r: torch.Tensor,
    effective_rank: int,
    s_weights: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    High-performance Triton Dynamic-Rank Entropy-Gated SVD GEMM:
        Y = (X @ V_r[:r_eff]^T * s_weights[:r_eff]) @ U_r[:r_eff]^T + bias

    Args:
        x: (M, K) or (B, Seq, K)
        V_r: (R_MAX, K) or (K, R_MAX)
        U_r: (N, R_MAX) or (R_MAX, N)
        effective_rank: Active rank r_eff <= R_MAX.
        s_weights: Optional singular importance weights of shape (R_MAX,).
        bias: Optional bias of shape (N,).

    Returns:
        Tensor of shape (*x.shape[:-1], N).
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, K = x_2d.shape

    if V_r.shape[0] == K:
        V_mat = V_r
    else:
        V_mat = V_r.t().contiguous()

    R_MAX = V_mat.shape[1]
    eff_r = min(int(effective_rank), R_MAX)

    if U_r.shape[0] == R_MAX:
        U_mat = U_r
    else:
        U_mat = U_r.t().contiguous()

    N = U_mat.shape[1]

    if not x.is_cuda or R_MAX > 256:
        # PyTorch reference
        V_sub = V_mat[:, :eff_r]
        U_sub = U_mat[:eff_r, :]
        z = torch.matmul(x_2d, V_sub)
        if s_weights is not None:
            z = z * s_weights[:eff_r].view(1, -1).to(dtype=z.dtype, device=z.device)
        y = torch.matmul(z, U_sub)
        if bias is not None:
            y = y + bias.to(dtype=y.dtype, device=y.device)
        return y.view(*orig_shape[:-1], N)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_R = max(16, triton.next_power_of_2(R_MAX))

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_weights = s_weights is not None
    w_t = s_weights.contiguous() if has_weights else torch.empty(0, device=x.device, dtype=x.dtype)
    has_bias = bias is not None
    b_t = bias.contiguous() if has_bias else torch.empty(0, device=x.device, dtype=x.dtype)

    x_c = x_2d.contiguous()
    V_c = V_mat.contiguous()
    U_c = U_mat.contiguous()

    _spectral_entropy_svd_kernel[grid](
        x_c,
        V_c,
        U_c,
        w_t,
        b_t,
        y,
        M,
        N,
        K,
        R_MAX,
        eff_r,
        x_c.stride(0),
        x_c.stride(1),
        V_c.stride(0),
        V_c.stride(1),
        U_c.stride(0),
        U_c.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_WEIGHTS=has_weights,
        HAS_BIAS=has_bias,
        BLOCK_R=BLOCK_R,
    )

    return y.view(*orig_shape[:-1], N)


# Aliases
spectral_entropy_svd_gate = triton_spectral_entropy_svd_gate

__all__ = [
    "triton_spectral_entropy_svd_gate",
    "spectral_entropy_svd_gate",
]
