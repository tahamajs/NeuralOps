"""
Centralized SVD & Factorization Utilities for Attention and Compression Modules.
Provides numerical stability (FP32 execution for half-precision inputs),
truncated SVD factor initialization, and energy-based rank selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def stable_svd(
    weight: torch.Tensor,
    full_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes SVD with guaranteed FP32 precision to prevent CUDA half-precision solver crashes (e.g. geqrf_cuda).
    Returns (U, S, Vh) matching the original device and floating dtype.
    """
    orig_dtype = weight.dtype
    orig_device = weight.device
    W_fp32 = weight.detach().to(device=orig_device, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(W_fp32, full_matrices=full_matrices)
    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)


def init_factorized_linear(
    src_weight: torch.Tensor,
    lin1: nn.Linear,
    lin2: nn.Linear,
    rank: int,
    src_bias: torch.Tensor | None = None,
) -> None:
    """
    Initializes a two-stage linear projection (x @ lin1.T @ lin2.T) from a dense source weight
    using truncated SVD:
        lin1.weight = Vh[:r, :]
        lin2.weight = U[:, :r] @ diag(S[:r])
    Ensures zeroed parameters for excess capacity and copies bias to lin2 if present.
    """
    with torch.no_grad():
        U, S, Vh = stable_svd(src_weight, full_matrices=False)
        r = min(rank, lin1.weight.shape[0], lin2.weight.shape[1], U.size(1), Vh.size(0))

        lin1.weight.data.zero_()
        lin2.weight.data.zero_()

        # lin1 takes input and projects to rank: shape (rank, in_features)
        min_in = min(lin1.weight.shape[1], Vh.shape[1])
        lin1.weight.data[:r, :min_in].copy_(Vh[:r, :min_in].to(lin1.weight.dtype))

        # lin2 takes rank and projects to out_features: shape (out_features, rank)
        min_out = min(lin2.weight.shape[0], U.shape[0])
        U_scaled = (U[:min_out, :r] * S[:r].unsqueeze(0)).to(lin2.weight.dtype)
        lin2.weight.data[:min_out, :r].copy_(U_scaled)

        if src_bias is not None and getattr(lin2, "bias", None) is not None:
            min_b = min(lin2.bias.shape[0], src_bias.shape[0])
            lin2.bias.data[:min_b].copy_(src_bias[:min_b].to(lin2.bias.dtype))


def select_rank_by_energy(
    singular_values: torch.Tensor,
    energy_threshold: float = 0.90,
    min_rank: int = 4,
    max_rank: int | None = None,
) -> int:
    """
    Selects truncation rank r such that sum(S[:r]^2) / sum(S^2) >= energy_threshold.
    """
    if singular_values.numel() == 0:
        return min_rank
    energy = torch.cumsum(singular_values ** 2, dim=0)
    total_energy = energy[-1]
    if total_energy <= 0:
        return min_rank
    fraction = energy / total_energy
    mask = fraction >= energy_threshold
    if not torch.any(mask):
        r = int(singular_values.numel())
    else:
        r = int(torch.nonzero(mask)[0].item()) + 1
    r = max(min_rank, r)
    if max_rank is not None:
        r = min(max_rank, r)
    return r
