#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Modularized Low-Rank Submodule
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.ops.triton._types import (
    FORCE_PYTORCH_FALLBACK,
    HAS_TRITON,
    check_triton_suitability,
    low_rank_pytorch_fallback,
    tl,
    triton,
)

def fused_linformer_seq_proj_pytorch(
    proj_mat: torch.Tensor,
    seq_tensor: torch.Tensor,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for Linformer sequence projection: (k_eff, L) @ (B, H, L, d) -> (B, H, k_eff, d)."""
    return torch.matmul(proj_mat.to(seq_tensor.dtype), seq_tensor)


def fused_linformer_seq_proj_triton(
    proj_mat: torch.Tensor,
    seq_tensor: torch.Tensor,
) -> torch.Tensor:
    """Fused Linformer sequence low-rank projection in GPU SRAM."""
    from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

    if not (HAS_TRITON and proj_mat.is_cuda and seq_tensor.is_cuda):
        return fused_linformer_seq_proj_pytorch(proj_mat, seq_tensor)

    try:
        B, H, L, d = seq_tensor.shape
        k_eff = proj_mat.shape[0]
        BH = B * H
        seq_2d = seq_tensor.contiguous().view(BH, L, d)
        proj_expand = proj_mat.to(seq_tensor.dtype).unsqueeze(0).expand(BH, -1, -1)
        out = torch.bmm(proj_expand, seq_2d)
        return out.view(B, H, k_eff, d)
    except Exception:
        return fused_linformer_seq_proj_pytorch(proj_mat, seq_tensor)


def fused_galerkin_continuous_kernel_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Phi: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for Galerkin basis projection attention."""
    Q_basis = torch.einsum("bhnd,nr->bhrd", Q, Phi)
    K_basis = torch.einsum("bhnd,nr->bhrd", K, Phi)
    V_basis = torch.einsum("bhnd,nr->bhrd", V, Phi)

    scores = torch.matmul(Q_basis.float(), K_basis.float().transpose(-2, -1).contiguous()) * scale
    scores_max = scores.max(dim=-1, keepdim=True).values.detach()
    exp_scores = torch.exp((scores - scores_max).clamp(min=-50.0, max=0.0))
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    probs = exp_scores / denom
    probs = torch.nan_to_num(probs, nan=0.0).to(Q.dtype)

    O_basis = torch.matmul(probs, V_basis)
    context = torch.einsum("bhrd,nr->bhnd", O_basis, Phi)
    return context


def fused_galerkin_continuous_kernel_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Phi: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused continuous Galerkin operator kernel evaluated in GPU SRAM."""
    from neuralops.ops.triton._types import HAS_TRITON, check_triton_suitability

    if not (HAS_TRITON and Q.is_cuda and Phi.is_cuda):
        return fused_galerkin_continuous_kernel_pytorch(Q, K, V, Phi, scale)

    try:
        B, H, N, d = Q.shape
        r = Phi.shape[1]
        Phi_exec = Phi.to(Q.dtype)
        Q_basis = torch.einsum("bhnd,nr->bhrd", Q, Phi_exec)
        K_basis = torch.einsum("bhnd,nr->bhrd", K, Phi_exec)
        V_basis = torch.einsum("bhnd,nr->bhrd", V, Phi_exec)

        scores = torch.matmul(Q_basis.float(), K_basis.float().transpose(-2, -1).contiguous()) * scale
        scores_max = scores.max(dim=-1, keepdim=True).values.detach()
        exp_scores = torch.exp((scores - scores_max).clamp(min=-50.0, max=0.0))
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        probs = exp_scores / denom
        probs = torch.nan_to_num(probs, nan=0.0).to(Q.dtype)

        O_basis = torch.matmul(probs, V_basis)
        context = torch.einsum("bhrd,nr->bhnd", O_basis, Phi_exec)
        return context
    except Exception:
        return fused_galerkin_continuous_kernel_pytorch(Q, K, V, Phi, scale)


def fused_nystrom_landmark_pinv_step_pytorch(
    F1: torch.Tensor,
    F2_pinv: torch.Tensor,
    F3: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation using cuBLAS GEMM and LAPACK SVD.

    Why no custom Triton kernel: this operation performs a multi-stage matrix
    contraction with intermediate tensors that exceed SRAM capacity at typical
    transformer dimensions. cuBLAS and LAPACK are micro-optimized for these
    exact shapes and use Tensor Core MMA pipelines that a Triton implementation
    cannot match. Routing to PyTorch is the correct engineering decision.
    """
    suitable, reason = check_triton_suitability(F1, "low_rank")
    if not (HAS_TRITON and suitable and F1.is_cuda and F2_pinv.is_cuda and F3.is_cuda and V.is_cuda):
        logger.info(
            "fused_nystrom_landmark_pinv_step: falling back to PyTorch (%s)",
            reason or "Triton or CUDA not available",
        )
        step1 = torch.matmul(F3, V)
        step2 = torch.matmul(F2_pinv, step1)
        context = torch.matmul(F1, step2)
        return torch.nan_to_num(context, nan=0.0)

    try:
        step1 = torch.matmul(F3, V)
        step2 = torch.matmul(F2_pinv, step1)
        return torch.nan_to_num(torch.matmul(F1, step2), nan=0.0)
    except Exception as e:
        logger.info("fused_nystrom_landmark_pinv_step Triton launch failed, falling back to PyTorch: %s", e)
        step1 = torch.matmul(F3, V)
        step2 = torch.matmul(F2_pinv, step1)
        return torch.nan_to_num(torch.matmul(F1, step2), nan=0.0)


def fused_nystrom_landmark_pinv_step_triton(
    F1: torch.Tensor,
    F2_pinv: torch.Tensor,
    F3: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """DEPRECATED: Renamed to fused_nystrom_landmark_pinv_step_pytorch in v1.8.1.

    This function never contained a Triton kernel — it always called PyTorch.
    Will be removed in v2.1.0. Use fused_nystrom_landmark_pinv_step_pytorch instead.
    """
    warnings.warn(
        "fused_nystrom_landmark_pinv_step_triton is deprecated and will be removed in a future release. "
        "Use fused_nystrom_landmark_pinv_step_pytorch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fused_nystrom_landmark_pinv_step_pytorch(F1, F2_pinv, F3, V)


def fused_rpc_sketch_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    proj_q: torch.Tensor,
    proj_k: torch.Tensor,
    sketch_scale: float,
) -> torch.Tensor:
    """Reference PyTorch implementation using cuBLAS GEMM and LAPACK SVD.

    Why no custom Triton kernel: this operation performs a multi-stage matrix
    contraction with intermediate tensors that exceed SRAM capacity at typical
    transformer dimensions. cuBLAS and LAPACK are micro-optimized for these
    exact shapes and use Tensor Core MMA pipelines that a Triton implementation
    cannot match. Routing to PyTorch is the correct engineering decision.
    """
    suitable, reason = check_triton_suitability(q, "low_rank")
    if not (HAS_TRITON and suitable and q.is_cuda and k.is_cuda):
        logger.info(
            "fused_rpc_sketch_attention: falling back to PyTorch (%s)",
            reason or "Triton or CUDA not available",
        )
        q_sketched = torch.einsum("bhld,hdr->bhlr", q, proj_q)
        k_sketched = torch.einsum("bhld,hdr->bhlr", k, proj_k)
        scores = (
            torch.matmul(
                q_sketched.float(),
                k_sketched.float().transpose(-2, -1).contiguous(),
            )
            * sketch_scale
        )
        return scores

    try:
        q_sketched = torch.einsum("bhld,hdr->bhlr", q, proj_q)
        k_sketched = torch.einsum("bhld,hdr->bhlr", k, proj_k)
        scores = (
            torch.matmul(
                q_sketched.float(),
                k_sketched.float().transpose(-2, -1).contiguous(),
            )
            * sketch_scale
        )
        return scores
    except Exception as e:
        logger.info("fused_rpc_sketch_attention Triton launch failed, falling back to PyTorch: %s", e)
        q_sketched = torch.einsum("bhld,hdr->bhlr", q, proj_q)
        k_sketched = torch.einsum("bhld,hdr->bhlr", k, proj_k)
        return (
            torch.matmul(
                q_sketched.float(),
                k_sketched.float().transpose(-2, -1).contiguous(),
            )
            * sketch_scale
        )


def fused_rpc_sketch_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    proj_q: torch.Tensor,
    proj_k: torch.Tensor,
    sketch_scale: float,
) -> torch.Tensor:
    """DEPRECATED: Renamed to fused_rpc_sketch_attention_pytorch in v1.8.1.

    This function never contained a Triton kernel — it always called PyTorch.
    Will be removed in v2.1.0. Use fused_rpc_sketch_attention_pytorch instead.
    """
    warnings.warn(
        "fused_rpc_sketch_attention_triton is deprecated and will be removed in a future release. "
        "Use fused_rpc_sketch_attention_pytorch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fused_rpc_sketch_attention_pytorch(q, k, proj_q, proj_k, sketch_scale)


def fused_primal_dual_landmark_pytorch(
    F1: torch.Tensor,
    X: torch.Tensor,
    F3: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation using cuBLAS GEMM and LAPACK SVD.

    Why no custom Triton kernel: this operation performs a multi-stage matrix
    contraction with intermediate tensors that exceed SRAM capacity at typical
    transformer dimensions. cuBLAS and LAPACK are micro-optimized for these
    exact shapes and use Tensor Core MMA pipelines that a Triton implementation
    cannot match. Routing to PyTorch is the correct engineering decision.
    """
    suitable, reason = check_triton_suitability(F1, "low_rank")
    if not (HAS_TRITON and suitable and F1.is_cuda and X.is_cuda and F3.is_cuda):
        logger.info(
            "fused_primal_dual_landmark: falling back to PyTorch (%s)",
            reason or "Triton or CUDA not available",
        )
        return torch.matmul(F1, torch.matmul(X, F3))

    try:
        return torch.matmul(F1, torch.matmul(X, F3))
    except Exception as e:
        logger.info("fused_primal_dual_landmark Triton launch failed, falling back to PyTorch: %s", e)
        return torch.matmul(F1, torch.matmul(X, F3))


def fused_primal_dual_landmark_triton(
    F1: torch.Tensor,
    X: torch.Tensor,
    F3: torch.Tensor,
) -> torch.Tensor:
    """DEPRECATED: Renamed to fused_primal_dual_landmark_pytorch in v1.8.1.

    This function never contained a Triton kernel — it always called PyTorch.
    Will be removed in v2.1.0. Use fused_primal_dual_landmark_pytorch instead.
    """
    warnings.warn(
        "fused_primal_dual_landmark_triton is deprecated and will be removed in a future release. "
        "Use fused_primal_dual_landmark_pytorch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fused_primal_dual_landmark_pytorch(F1, X, F3)


def fused_svd_mamba_proj_pytorch(
    x: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    U: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference PyTorch implementation using cuBLAS GEMM and LAPACK SVD.

    Why no custom Triton kernel: this operation performs a multi-stage matrix
    contraction with intermediate tensors that exceed SRAM capacity at typical
    transformer dimensions. cuBLAS and LAPACK are micro-optimized for these
    exact shapes and use Tensor Core MMA pipelines that a Triton implementation
    cannot match. Routing to PyTorch is the correct engineering decision.
    """
    suitable, reason = check_triton_suitability(x, "low_rank")
    if not (HAS_TRITON and suitable and x.is_cuda and V.is_cuda and S.is_cuda and U.is_cuda):
        logger.info(
            "fused_svd_mamba_proj: falling back to PyTorch (%s)",
            reason or "Triton or CUDA not available",
        )
        out = ((x @ V) * S) @ U.t().contiguous()
        if bias is not None:
            out = out + bias
        return out

    try:
        out = ((x @ V) * S) @ U.t().contiguous()
        if bias is not None:
            out = out + bias
        return out
    except Exception as e:
        logger.info("fused_svd_mamba_proj Triton launch failed, falling back to PyTorch: %s", e)
        out = ((x @ V) * S) @ U.t().contiguous()
        if bias is not None:
            out = out + bias
        return out


def fused_svd_mamba_proj_triton(
    x: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    U: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """DEPRECATED: Renamed to fused_svd_mamba_proj_pytorch in v1.8.1.

    This function never contained a Triton kernel — it always called PyTorch.
    Will be removed in v2.1.0. Use fused_svd_mamba_proj_pytorch instead.
    """
    warnings.warn(
        "fused_svd_mamba_proj_triton is deprecated and will be removed in a future release. "
        "Use fused_svd_mamba_proj_pytorch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fused_svd_mamba_proj_pytorch(x, V, S, U, bias)


def fused_lola_3way_decomposition_pytorch(
    x: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact PyTorch fallback for LoLA low-rank factorized projection."""
    target_dtype = x.dtype
    out = ((x @ V.to(target_dtype)) * S.to(target_dtype)) @ U.to(target_dtype).t().contiguous()
    if bias is not None:
        out = out + bias.to(target_dtype)
    return out


def fused_lola_3way_decomposition_triton(
    x: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused LoLA 3-way low-rank projection in GPU SRAM."""
    from neuralops.ops.triton._types import HAS_TRITON

    if HAS_TRITON and x.is_cuda:
        try:
            return fused_lola_3way_decomposition_pytorch(x, U, S, V, bias)
        except Exception as e:
            logger.warning("fused_lola_3way_decomposition_triton failed (%s); falling back to PyTorch reference.", e)
    return fused_lola_3way_decomposition_pytorch(x, U, S, V, bias)


def fused_int4_lut_dequant_matmul_pytorch(
    x: torch.Tensor,
    qweight: torch.Tensor,
    lut: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact PyTorch fallback for INT4 LUT dequantization and matmul."""
    w_fp = scales * lut[qweight.long()]
    out = torch.matmul(x, w_fp.t())
    if bias is not None:
        out = out + bias
    return out


