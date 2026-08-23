"""Layer-wise gradient norm profiling and numerical health audit."""

from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn


def compute_gradient_norm_stats(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Execute a forward and backward pass on synthetic loss to measure layer-wise gradient magnitudes.

    Args:
        model: Transformer model with trainable parameters.
        inputs: Input dictionary.

    Returns:
        Dictionary containing gradient statistics, NaN detection, and per-layer norms.
    """
    model.train()
    model.zero_grad()

    device = next(model.parameters()).device
    dev_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    has_nan = False
    layer_norms: Dict[str, float] = {}

    try:
        outputs = model(**dev_inputs)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif hasattr(outputs, "logits"):
            loss = outputs.logits.sum()
        elif isinstance(outputs, tuple):
            loss = outputs[0].sum()
        else:
            loss = outputs.sum()

        loss.backward()

        all_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    has_nan = True
                norm = float(grad.norm(2).item())
                layer_norms[name] = norm
                all_norms.append(norm)

        mean_norm = float(sum(all_norms) / max(len(all_norms), 1))
        max_norm = float(max(all_norms)) if all_norms else 0.0

        return {
            "mean_norm": mean_norm,
            "max_norm": max_norm,
            "has_nan": has_nan,
            "vanishing": mean_norm < 1e-7,
            "exploding": max_norm > 100.0,
            "layer_norms": layer_norms,
        }

    except Exception as e:
        return {
            "mean_norm": 0.0,
            "max_norm": 0.0,
            "has_nan": True,
            "error": str(e),
            "layer_norms": {},
        }
    finally:
        model.zero_grad()
