#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Activation-Aware SVD (ASVD) CUDA Kernel Operator.
"""

from __future__ import annotations

from typing import Optional

import torch

from neuralops.ops.cuda.asvd.autograd import ASVDFunction


def asvd_compress(
    W: torch.Tensor,
    cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Activation-Aware SVD compression using CUDA C++ kernel."""
    if W.ndim != 2:
        raise ValueError(f"asvd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    cov_sqrt = cov_inv_sqrt = None
    if cov is not None and cov.shape == (N, N):
        try:
            cov_fp32 = cov.to(W.device, torch.float32)
            cov_sym = 0.5 * (cov_fp32 + cov_fp32.T)
            trace_val = torch.trace(cov_sym).clamp(min=1e-8)
            damping = 0.05 * (trace_val / float(N))
            cov_damped = cov_sym + damping * torch.eye(N, device=cov_sym.device, dtype=torch.float32)
            eigvals, eigvecs = torch.linalg.eigh(cov_damped)
            eps = 1e-5 * torch.max(eigvals).clamp(min=1e-12) + 1e-7
            eigvals_reg = eigvals.clamp(min=eps)
            cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_reg)) @ eigvecs.T
            cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_reg)) @ eigvecs.T
        except Exception:
            cov_sqrt, cov_inv_sqrt = None, None

    W_weighted = torch.matmul(W_fp32, cov_sqrt) if cov_sqrt is not None else W_fp32
    U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
    selected_r = min(r, len(S))
    sqrt_S = torch.sqrt(torch.clamp(S[:selected_r], min=1e-12))
    U_r = (U[:, :selected_r] * sqrt_S.unsqueeze(0)).contiguous()
    V_r = (sqrt_S.unsqueeze(1) * Vt[:selected_r, :]).contiguous()
    if cov_inv_sqrt is not None:
        V_r = torch.matmul(V_r, cov_inv_sqrt).contiguous()

    # Row-wise norm rebalancing
    W_recon_pre = torch.matmul(U_r, V_r)
    orig_row_norms = torch.norm(W_fp32, dim=-1, keepdim=True)
    comp_row_norms = torch.norm(W_recon_pre, dim=-1, keepdim=True).clamp_min(1e-6)
    row_gain = (orig_row_norms / comp_row_norms).clamp(0.2, 5.0).squeeze(-1).contiguous()

    out = ASVDFunction.apply(U_r, V_r, row_gain)
    return out.to(dtype=W.dtype)


__all__ = ["asvd_compress", "ASVDFunction"]
