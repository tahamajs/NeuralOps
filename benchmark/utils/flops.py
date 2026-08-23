"""Theoretical floating-point operation (FLOP) estimator for transformer models."""

from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def compute_flops(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Optional[float]:
    """
    Estimate total forward GFLOPs for a model given input tensors.

    Args:
        model: Transformer model instance.
        inputs: Input dictionary containing 'input_ids' and 'attention_mask'.

    Returns:
        Total GFLOPs as a float, or None if profiling tools are unavailable.
    """
    if FlopCountAnalysis is not None and "input_ids" in inputs:
        try:
            model_device = next(model.parameters()).device
            device_inputs = (inputs["input_ids"].to(model_device),)
            flops = FlopCountAnalysis(model, device_inputs)
            flops.unsupported_ops_warnings(False)
            return float(flops.total()) / 1e9
        except Exception:
            pass

    # Analytical heuristic fallback: ~ 2 * N_params * Seq_len per sample
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        b, l = inputs["input_ids"].shape[:2]
        approx_flops = 2.0 * num_params * l * b
        return float(approx_flops) / 1e9
    except Exception:
        return None
