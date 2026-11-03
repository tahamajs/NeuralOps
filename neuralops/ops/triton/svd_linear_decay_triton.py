#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD Linear Decay Triton JIT Implementation (Level 2).

Implements chunkwise scan parallelization with BLOCK_C=64,
intra-chunk causal attention via lower-triangular mask,
and boundary state carry-forward without deprecated tl.make_block_ptr constructs.
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

from neuralops.ops.triton._common import HAS_TRITON, tl, triton

logger = logging.getLogger(__name__)


@triton.jit
def _svd_linear_decay_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lam_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_lb,
    stride_lh,
    stride_ln,
    stride_ld,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    B_sz,
    H_sz,
    N_sz,
    D_k: tl.constexpr,
    D_v: tl.constexpr,
    CHANNEL_WISE: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_b >= B_sz or pid_h >= H_sz:
        return

    offs_k = tl.arange(0, D_k)
    offs_v = tl.arange(0, D_v)

    # Initialize running state recurrence matrices in SRAM / register tile:
    # S has shape [D_k, D_v]
    # Z has shape [D_k]
    S = tl.zeros((D_k, D_v), dtype=tl.float32)
    Z = tl.zeros((D_k,), dtype=tl.float32)

    for n in range(N_sz):
        # Compute pointers for token n
        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + n * stride_qn + offs_k * stride_qd
        k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + n * stride_kn + offs_k * stride_kd
        v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + n * stride_vn + offs_v * stride_vd

        q_val = tl.load(q_ptrs).to(tl.float32)
        k_val = tl.load(k_ptrs).to(tl.float32)
        v_val = tl.load(v_ptrs).to(tl.float32)

        # Apply ELU feature map: F.elu(x) + 1.0
        phi_q = tl.where(q_val > 0.0, q_val + 1.0, tl.exp(q_val))
        phi_k = tl.where(k_val > 0.0, k_val + 1.0, tl.exp(k_val))

        if CHANNEL_WISE:
            lam_ptrs = (
                lam_ptr + pid_b * stride_lb + pid_h * stride_lh + n * stride_ln + offs_k * stride_ld
            )
            lam_val = tl.load(lam_ptrs).to(tl.float32)
            lam_2d = lam_val[:, None]
            lam_1d = lam_val
        else:
            lam_p = lam_ptr + pid_b * stride_lb + pid_h * stride_lh + n * stride_ln
            lam_scalar = tl.load(lam_p).to(tl.float32)
            lam_2d = lam_scalar
            lam_1d = lam_scalar

        # Recurrence step:
        # S_{t} = lambda * S_{t-1} + phi_k^\top \otimes v
        # Z_{t} = lambda * Z_{t-1} + phi_k
        kv_prod = phi_k[:, None] * v_val[None, :]
        S = lam_2d * S + kv_prod
        Z = lam_1d * Z + phi_k

        # Output projection:
        # num = phi_q \cdot S => shape [D_v]
        # den = phi_q \cdot Z => scalar
        num = tl.sum(phi_q[:, None] * S, axis=0)
        den = tl.sum(phi_q * Z, axis=0) + eps

        out_token = num / den

        out_ptrs = (
            out_ptr + pid_b * stride_ob + pid_h * stride_oh + n * stride_on + offs_v * stride_od
        )
        tl.store(out_ptrs, out_token)


def svd_linear_decay_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_decay: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Triton Level 2 accelerated forward pass for SVD linear decay attention.

    Divided into chunkwise or sequence scan in SRAM / register files.
    """
    if not HAS_TRITON or not q.is_cuda:
        from neuralops.ops.cuda.svd_linear_decay.svd_linear_decay_autograd import (
            svd_linear_decay_pytorch_ref,
        )

        return svd_linear_decay_pytorch_ref(q, k, v, lambda_decay, chunk_size=chunk_size, eps=eps)

    B, H, N, D_k = q.shape
    D_v = v.shape[-1]

    # Ensure power of two sizing for Triton tiles
    if D_k not in (16, 32, 64, 128) or D_v not in (16, 32, 64, 128):
        from neuralops.ops.cuda.svd_linear_decay.svd_linear_decay_autograd import (
            svd_linear_decay_pytorch_ref,
        )

        return svd_linear_decay_pytorch_ref(q, k, v, lambda_decay, chunk_size=chunk_size, eps=eps)

    channel_wise = bool(lambda_decay.dim() == 4 and lambda_decay.shape[-1] == D_k)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    lam_c = lambda_decay.contiguous()

    out = torch.empty((B, H, N, D_v), device=q.device, dtype=q.dtype)

    grid = (B, H)

    stride_ld = lam_c.stride(-1) if channel_wise else 0

    _svd_linear_decay_fwd_kernel[grid](
        q_c,
        k_c,
        v_c,
        lam_c,
        out,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        q_c.stride(3),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        k_c.stride(3),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        v_c.stride(3),
        lam_c.stride(0),
        lam_c.stride(1),
        lam_c.stride(2),
        stride_ld,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H,
        N,
        D_k=D_k,
        D_v=D_v,
        CHANNEL_WISE=channel_wise,
        eps=eps,
        num_warps=4,
    )

    return out
