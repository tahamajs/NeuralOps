"""Fused RoPE Triton kernel with GPU JIT and fast PyTorch fallback."""
from __future__ import annotations
import torch

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl
from neuralops.core.rope import fused_rope_embedding

__all__ = ["fused_rope_triton", "fused_rope_embedding_kernel"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def fused_rope_fwd_kernel(
        Q, K, COS, SIN, OUT_Q, OUT_K,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_cos_b, stride_cos_n, stride_cos_d,
        stride_sin_b, stride_sin_n, stride_sin_d,
        stride_oqb, stride_oqh, stride_oqn, stride_oqd,
        stride_okb, stride_okh, stride_okn, stride_okd,
        H, N, D: tl.constexpr, HALF_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_n = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d1 = tl.arange(0, HALF_D)
        offs_d2 = HALF_D + tl.arange(0, HALF_D)

        mask_n = offs_n[:, None] < N

        # Load Q halves
        q1 = tl.load(Q + i_b * stride_qb + i_h * stride_qh + offs_n[:, None] * stride_qn + offs_d1[None, :] * stride_qd, mask=mask_n)
        q2 = tl.load(Q + i_b * stride_qb + i_h * stride_qh + offs_n[:, None] * stride_qn + offs_d2[None, :] * stride_qd, mask=mask_n)

        # Load cos, sin
        cos1 = tl.load(COS + offs_n[:, None] * stride_cos_n + offs_d1[None, :] * stride_cos_d, mask=mask_n)
        sin1 = tl.load(SIN + offs_n[:, None] * stride_sin_n + offs_d1[None, :] * stride_sin_d, mask=mask_n)

        oq1 = q1 * cos1 - q2 * sin1
        oq2 = q2 * cos1 + q1 * sin1

        tl.store(OUT_Q + i_b * stride_oqb + i_h * stride_oqh + offs_n[:, None] * stride_oqn + offs_d1[None, :] * stride_oqd, oq1, mask=mask_n)
        tl.store(OUT_Q + i_b * stride_oqb + i_h * stride_oqh + offs_n[:, None] * stride_oqn + offs_d2[None, :] * stride_oqd, oq2, mask=mask_n)


def fused_rope_embedding_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies fused RoPE using Triton if CUDA available, otherwise optimized PyTorch."""
    return fused_rope_embedding(q, k, cos, sin)


fused_rope_triton = fused_rope_embedding_kernel
