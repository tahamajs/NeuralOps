"""Production SmoothQuant: outlier migration from activations to weights."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["smoothquant_layer", "smoothquant_model"]


def smoothquant_layer(
    module: nn.Linear,
    alpha: float = 0.5,
    calib_data: torch.Tensor | list[torch.Tensor] | None = None,
    act_scales: torch.Tensor | None = None,
) -> tuple[nn.Linear, torch.Tensor]:
    """Applies SmoothQuant to a linear layer.

    Migrates activation outliers into weights:
      s = (a_max^alpha) / (w_max^(1-alpha))
      W' = W * diag(s)
      X' = X * diag(s)^-1

    Returns:
        Tuple of (modified module, scale factor s).
    """
    if calib_data is None and act_scales is None:
        s_id = torch.ones(module.weight.shape[1], device=module.weight.device)
        return module, s_id

    if act_scales is None:
        if isinstance(calib_data, list):
            stacked = torch.cat([x.reshape(-1, x.shape[-1]) for x in calib_data], dim=0)
        else:
            stacked = calib_data.reshape(-1, calib_data.shape[-1])
        a_max = stacked.abs().amax(dim=0).clamp(min=1e-5)
    else:
        a_max = act_scales.clamp(min=1e-5)

    w_max = module.weight.abs().amax(dim=0).clamp(min=1e-5)

    # Compute smoothing scale factor
    s = (a_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)

    # Apply to weights in-place
    module.weight.data = (module.weight.data * s.unsqueeze(0)).to(module.weight.dtype)
    return module, s


def smoothquant_model(
    model: nn.Module,
    calib_data: list[torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Applies SmoothQuant to all nn.Linear layers in model and returns scale dict."""
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _, s = smoothquant_layer(module, alpha=alpha, calib_data=calib_data)
            scales[name] = s
    return scales
