"""SmoothQuant (migrating activation outliers to weights)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["smoothquant_layer"]


def smoothquant_layer(
    layer: nn.Linear,
    act_scales: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Migrates difficulty from activation outliers to weights using per-channel scale s = act^alpha / w^(1-alpha).

    Returns:
        Per-channel scale factor s applied to input activations (layer scales inversely).
    """
    W = torch.abs(layer.weight.data.clone().to(torch.float32))
    w_scales = torch.amax(W, dim=0).clamp(min=1e-5)
    a_scales = act_scales.clamp(min=1e-5).to(torch.float32)

    # s = a^alpha / w^(1-alpha)
    scales = (a_scales.pow(alpha) / w_scales.pow(1.0 - alpha)).clamp(min=1e-5)

    # W' = W * diag(s)
    layer.weight.data = (layer.weight.data * scales.unsqueeze(0)).to(layer.weight.dtype)
    return scales
