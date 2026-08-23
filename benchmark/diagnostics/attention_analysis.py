"""Attention matrix entropy and sparsity diagnostics."""

from __future__ import annotations
from typing import Optional
import torch


def compute_attention_entropy(attention_weights: Optional[torch.Tensor]) -> float:
    """
    Compute average Shannon entropy H(P) = - sum(p * log(p + eps)) across query positions and attention heads.

    Args:
        attention_weights: Tensor of shape [B, heads, L, L] or [B, heads, L, K].

    Returns:
        Average entropy in nats.
    """
    if attention_weights is None or not isinstance(attention_weights, torch.Tensor):
        return 0.0

    eps = 1e-12
    # Ensure non-negative probabilities
    p = torch.clamp(attention_weights, min=0.0)
    # Re-normalize if needed along last dimension
    denom = p.sum(dim=-1, keepdim=True) + eps
    p = p / denom

    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    return float(entropy.mean().item())


def compute_attention_sparsity(attention_weights: Optional[torch.Tensor], threshold: float = 1e-4) -> float:
    """
    Compute the percentage of attention weight elements below the specified sparsity threshold.

    Args:
        attention_weights: Attention matrix tensor.
        threshold: Sparsity cutoff value.

    Returns:
        Sparsity ratio expressed as a percentage (0.0 to 100.0).
    """
    if attention_weights is None or not isinstance(attention_weights, torch.Tensor):
        return 0.0

    sparse_elements = (attention_weights.abs() < threshold).float()
    return float(sparse_elements.mean().item() * 100.0)
