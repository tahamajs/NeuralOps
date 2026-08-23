"""Mixture-of-Experts (MoE) router load balancing and expert utilization diagnostics."""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn


def compute_moe_load_balance(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Inspect router gate logits and evaluate expert utilization balance across tokens.

    Args:
        model: Transformer model containing potential MoE sub-modules.
        inputs: Input token dictionary.

    Returns:
        Dictionary detailing routing coefficient of variation (CV) and expert counts.
    """
    expert_activations: List[int] = []

    # Hook into potential router / gate modules
    handles = []

    def make_hook():
        def hook(module, input_tensors, output_tensors):
            # Capture routing logits or indices
            if isinstance(output_tensors, (tuple, list)):
                routing_tensor = output_tensors[0]
            elif isinstance(output_tensors, dict):
                routing_tensor = output_tensors.get("router_logits", output_tensors.get("indices"))
            else:
                routing_tensor = output_tensors

            if isinstance(routing_tensor, torch.Tensor):
                if routing_tensor.dim() >= 2:
                    top_experts = torch.argmax(routing_tensor, dim=-1).view(-1)
                    counts = torch.bincount(top_experts).cpu().tolist()
                    expert_activations.extend(counts)
        return hook

    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ("router", "gate", "expert_choice", "moe")):
            handles.append(module.register_forward_hook(make_hook()))

    model.eval()
    with torch.no_grad():
        try:
            device = next(model.parameters()).device
            dev_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            model(**dev_inputs)
        except Exception:
            pass
        finally:
            for h in handles:
                h.remove()

    if not expert_activations:
        return {"has_moe": False, "cv": 0.0, "imbalanced": False, "expert_counts": []}

    counts_arr = np.array(expert_activations, dtype=np.float32)
    mean_val = float(np.mean(counts_arr))
    std_val = float(np.std(counts_arr))
    cv = std_val / (mean_val + 1e-8)

    return {
        "has_moe": True,
        "cv": float(cv),
        "imbalanced": bool(cv > 1.0),
        "expert_counts": expert_activations,
    }
