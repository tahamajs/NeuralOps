"""Lightweight public injection entry point used by sweep runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def apply_module_to_model(model: nn.Module, module_name: str, module_type: str,
                          rank: int = 16, device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, float]:
    original = sum(p.numel() for p in model.parameters())
    if module_name in ATTENTION_REGISTRY:
        module_type = "attention"
    elif module_name in COMPRESSOR_REGISTRY:
        module_type = "compressor"
    if module_type == "attention":
        inject_attention(model, module_name, rank=rank, copy_weights=True)
    elif module_type == "compressor":
        compressor = COMPRESSOR_REGISTRY.resolve(module_name)
        instance = None
        for args, kwargs in (((model,), {"rank": rank}), ((model,), {}), ((), {"rank": rank}), ((), {})):
            try:
                instance = compressor(*args, **kwargs)
                break
            except (TypeError, ValueError):
                continue
        if instance is None:
            raise RuntimeError(f"Could not instantiate compressor {module_name}")
        if isinstance(instance, nn.Module) and not any(callable(getattr(instance, n, None)) for n in ("compress", "apply", "prune")):
            model = instance
        else:
            result = None
            for method_name in ("compress", "apply", "prune"):
                method = getattr(instance, method_name, None)
                if callable(method):
                    for call in (lambda: method(model), lambda: method(), lambda: method(calibration_data=None)):
                        try:
                            result = call()
                            break
                        except (TypeError, ValueError, RuntimeError):
                            continue
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, nn.Module):
                        model = result
                        break
    else:
        raise RuntimeError(f"Unknown module type: {module_type}")
    model.to(device)
    current = sum(p.numel() for p in model.parameters())
    return model, original / max(1, current)
