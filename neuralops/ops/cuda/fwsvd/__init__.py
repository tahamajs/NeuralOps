#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fisher-Weighted SVD (FWSVD) CUDA Kernel Operator.
"""

from __future__ import annotations

from typing import Optional

import torch

from neuralops.ops.cuda.fwsvd.autograd import FWSVDFunction


def fwsvd_compress(
    W: torch.Tensor,
    fisher_diag: Optional[torch.Tensor] = None,
    fisher_matrix: Optional[torch.Tensor] = None,
    rank: int = 32,
    damping: float = 1e-4,
    max_error: float = 1.0,
) -> torch.Tensor:
    """Fisher-Weighted SVD compression using CUDA C++ kernel."""
    if W.ndim != 2:
        raise ValueError(f"fwsvd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    F_mat = fisher_matrix
    if F_mat is None and fisher_diag is not None and fisher_diag.ndim == 2:
        F_mat = fisher_diag

    if F_mat is not None and F_mat.shape == (N, N):
        F_dev = F_mat.to(device=W.device, dtype=torch.float32)
        F_sym = 0.5 * (F_dev + F_dev.t())
        tr_F = torch.trace(F_sym).item()
        damp_val = max(1e-7, float(damping * tr_F / max(1, N)))
        F_damped = F_sym + damp_val * torch.eye(N, device=F_dev.device, dtype=torch.float32)

        try:
            eigvals, eigvecs = torch.linalg.eigh(F_damped)
            eigvals_clamped = torch.clamp(eigvals, min=damp_val)
            sqrt_eig = torch.sqrt(eigvals_clamped)
            inv_sqrt_eig = 1.0 / sqrt_eig
            F_sqrt = eigvecs @ torch.diag(sqrt_eig) @ eigvecs.t()
            F_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eig) @ eigvecs.t()

            W_weighted = torch.matmul(W_fp32, F_sqrt)
            U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
            V_r = torch.matmul(sqrt_S.unsqueeze(1) * Vt[:r, :], F_inv_sqrt).contiguous()
        except Exception:
            U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()
    elif fisher_diag is not None and fisher_diag.shape == (N,):
        f_clamped = fisher_diag.to(device=W.device, dtype=torch.float32).clamp(min=1e-8)
        sqrt_f = torch.sqrt(f_clamped)
        inv_sqrt_f = 1.0 / sqrt_f
        W_weighted = W_fp32 * sqrt_f.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
        V_r = ((sqrt_S.unsqueeze(1) * Vt[:r, :]) * inv_sqrt_f.unsqueeze(0)).contiguous()
    else:
        U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()

    out = FWSVDFunction.apply(U_r, V_r)
    return out.to(dtype=W.dtype)


__all__ = ["fwsvd_compress", "FWSVDFunction"]
