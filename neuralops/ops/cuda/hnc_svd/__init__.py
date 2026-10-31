#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HNC-SVD CUDA Kernel Operator.
"""

from __future__ import annotations

from typing import Optional

import torch
from neuralops.ops.cuda.hnc_svd.autograd import HNCSVDFunction


def hnc_svd_compress(
    W: torch.Tensor,
    cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """HNC-SVD compression using CUDA C++ kernel."""
    if W.ndim != 2:
        raise ValueError(f"hnc_svd_compress requires a 2D weight matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1
    W_fp32 = W.float()

    if cov is not None and cov.shape == (N, N):
        Hx = cov.to(device=W.device, dtype=torch.float32)
        try:
            reg = 1e-3 * (torch.trace(Hx) / float(N)).item() + 1e-6
            Sx = Hx + reg * torch.eye(N, device=W.device, dtype=torch.float32)
            L = torch.linalg.cholesky(Sx)
            W_whitened = torch.matmul(W_fp32, L)
            U, S, Vh = torch.linalg.svd(W_whitened, full_matrices=False)
            U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
            V_unwhitened = torch.linalg.solve_triangular(
                L.t(), (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).t(), upper=True
            ).t()
            A_r = U_r.contiguous()
            B_r = V_unwhitened.contiguous()
        except Exception:
            U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
            A_r = (U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).contiguous()
            B_r = (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).contiguous()
    else:
        U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
        A_r = (U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).contiguous()
        B_r = (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).contiguous()

    out = HNCSVDFunction.apply(A_r, B_r)
    return out.to(dtype=W.dtype)


__all__ = ["hnc_svd_compress", "HNCSVDFunction"]
