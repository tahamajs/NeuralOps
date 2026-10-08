"""Production Wanda (Pruning by Weights and Activations)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["wanda_prune_layer", "wanda_prune_model"]


def wanda_prune_layer(
    module: nn.Linear,
    sparsity: float = 0.5,
    calib_data: torch.Tensor | list[torch.Tensor] | None = None,
    act_norm: torch.Tensor | None = None,
) -> tuple[nn.Linear, torch.Tensor]:
    """Prunes linear layer weights based on the Wanda importance score S_ij = |W_ij| * ||X_j||_2.

    Args:
        module: The target nn.Linear module.
        sparsity: Desired sparsity level in range [0.0, 1.0).
        calib_data: Calibration activations tensor or list of batch tensors.
        act_norm: Precomputed L2 norm vector across input features.

    Returns:
        Tuple of (pruned module, boolean mask of kept weights).
    """
    if sparsity <= 0.0:
        mask = torch.ones_like(module.weight.data, dtype=torch.bool)
        return module, mask

    W = module.weight.data

    if act_norm is None:
        if calib_data is None:
            # Fallback to pure magnitude pruning
            act_norm = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
        else:
            if isinstance(calib_data, list):
                stacked = torch.cat([x.reshape(-1, x.shape[-1]) for x in calib_data], dim=0)
            else:
                stacked = calib_data.reshape(-1, calib_data.shape[-1])
            # L2 norm across samples for each input column
            act_norm = torch.norm(stacked.to(torch.float32), p=2, dim=0).to(W.device, dtype=W.dtype)

    # Compute Wanda importance metric: S = |W| * ||X||_2
    metric = torch.abs(W) * act_norm.unsqueeze(0)

    # Per-row (per-output channel) structured/unstructured pruning threshold
    k = max(int(metric.shape[1] * sparsity), 1)
    threshold = torch.kthvalue(metric, k, dim=1, keepdim=True).values
    mask = metric > threshold

    module.weight.data *= mask.to(module.weight.dtype)
    return module, mask


def wanda_prune_model(
    model: nn.Module,
    calib_data: list[torch.Tensor],
    sparsity: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Applies Wanda pruning to all linear layers of model."""
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _, mask = wanda_prune_layer(module, sparsity=sparsity, calib_data=calib_data)
            masks[name] = mask
    return masks
