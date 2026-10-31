#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Direct SVD CUDA Kernel Operator.
"""

from __future__ import annotations

import torch
from neuralops.ops.cuda.direct_svd.autograd import DirectSVDFunction


def direct_svd_compress(
    W: torch.Tensor,
    rank: int = 32,
    max_error: float = 0.35,
) -> torch.Tensor:
    """Direct SVD compression using CUDA C++ kernel."""
    if W.ndim != 2:
        raise ValueError(f"direct_svd_compress requires a 2D matrix, got shape {W.shape}")

    M, N = W.shape
    r = max(1, min(rank, min(M, N) - 1)) if min(M, N) > 1 else 1

    W_fp32 = W.float()
    norm_w = torch.norm(W_fp32, p="fro").item()

    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
    sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
    U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).contiguous()
    V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).contiguous()

    W_recon_pre = torch.matmul(U_r, V_r)
    recon_norm = torch.norm(W_recon_pre, p="fro").item()
    scale = 1.0
    if recon_norm > 1e-12 and norm_w > 1e-12:
        scale = norm_w / recon_norm

    out = DirectSVDFunction.apply(U_r, V_r, scale)
    return out.to(dtype=W.dtype)


__all__ = ["direct_svd_compress", "DirectSVDFunction"]
