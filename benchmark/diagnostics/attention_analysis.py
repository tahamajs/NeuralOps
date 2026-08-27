"""Attention matrix entropy and sparsity diagnostics."""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import torch


def compute_attention_entropy(
    attention_weights: Optional[Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]]
) -> float:
    """
    Compute average Shannon entropy H(P) = - sum(p * log(p + eps)) across query positions and attention heads.

    Args:
        attention_weights: Tensor of shape [B, heads, L, L], [B, L, L], or a list/tuple of layer-wise tensors.

    Returns:
        Average entropy in nats as a float (or 0.0 if unavailable).
    """
    if attention_weights is None:
        return 0.0

    if isinstance(attention_weights, (list, tuple)):
        valid_tensors = [t for t in attention_weights if isinstance(t, torch.Tensor)]
        if not valid_tensors:
            return 0.0
        entropies = [compute_attention_entropy(t) for t in valid_tensors]
        return float(sum(entropies) / len(entropies))

    if not isinstance(attention_weights, torch.Tensor):
        return 0.0

    if attention_weights.numel() == 0:
        return 0.0

    eps = 1e-12
    # Ensure non-negative probabilities
    p = torch.clamp(attention_weights.float(), min=0.0)
    # Re-normalize along last dimension
    denom = p.sum(dim=-1, keepdim=True) + eps
    p = p / denom

    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    return float(entropy.mean().item())


def compute_attention_sparsity(
    attention_weights: Optional[Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]],
    threshold: float = 1e-4,
) -> float:
    """
    Compute the percentage of attention weight elements below the specified sparsity threshold.

    Args:
        attention_weights: Tensor of shape [B, heads, L, L], [B, L, L], or a list/tuple of layer-wise tensors.
        threshold: Sparsity cutoff value.

    Returns:
        Sparsity ratio expressed as a percentage (0.0 to 100.0).
    """
    if attention_weights is None:
        return 0.0

    if isinstance(attention_weights, (list, tuple)):
        valid_tensors = [t for t in attention_weights if isinstance(t, torch.Tensor)]
        if not valid_tensors:
            return 0.0
        sparsities = [compute_attention_sparsity(t, threshold=threshold) for t in valid_tensors]
        return float(sum(sparsities) / len(sparsities))

    if not isinstance(attention_weights, torch.Tensor):
        return 0.0

    if attention_weights.numel() == 0:
        return 0.0

    sparse_elements = (attention_weights.abs() < threshold).float()
    return float(sparse_elements.mean().item() * 100.0)
