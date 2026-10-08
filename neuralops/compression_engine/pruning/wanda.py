"""Wanda (Pruning by Weights and activations)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["wanda_prune_layer"]


def wanda_prune_layer(
    layer: nn.Linear,
    act_norm: torch.Tensor,
    sparsity: float = 0.5,
) -> torch.Tensor:
    """Prunes weights using metric S_ij = |W_ij| * ||X_j||_2.

    Args:
        layer: nn.Linear module to prune.
        act_norm: 1D tensor of L2 norms of column activations (shape: [in_features]).
        sparsity: Fraction of weights to prune to zero (0.0 to 1.0).

    Returns:
        Binary mask tensor of kept weights.
    """
    W = layer.weight.data
    # Importance metric S = |W| * ||X||_2
    score = torch.abs(W) * act_norm.unsqueeze(0)

    # Sort per row and zero out smallest elements
    k = int(sparsity * W.shape[1])
    if k > 0:
        thresh = torch.kthvalue(score, k, dim=1, keepdim=True).values
        mask = score > thresh
        layer.weight.data *= mask.to(layer.weight.dtype)
    else:
        mask = torch.ones_like(W, dtype=torch.bool)

    return mask
