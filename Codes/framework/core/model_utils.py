"""
Generic helpers: calibration, weight‑copying, safe module replacement,
activation collection.
"""

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


def run_calibration_forward(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device | None = None,
    max_length: int = 128,
) -> nn.Module:
    """Run a single forward pass on calibration texts; return model (in eval)."""
    model.eval()
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    return model


# ------------------------------------------------------------------ #
# Weight‑copying for attention modules
# ------------------------------------------------------------------ #
ATTENTION_WEIGHT_MAP = {
    "query": ["q_lin", "q_proj", "query"],
    "key": ["k_lin", "k_proj", "key"],
    "value": ["v_lin", "v_proj", "value"],
    "out_proj": ["out_lin", "o_proj", "output", "dense"],
}


def copy_attention_weights(
    src_module: nn.Module,
    dst_module: nn.Module,
    source_map: dict[str, list[str]] | None = None,
) -> None:
    """
    Copy pretrained attention weights from src to dst.
    Tries multiple possible attribute names for each projection.
    """
    if source_map is None:
        source_map = ATTENTION_WEIGHT_MAP
    for target_attr, src_attrs in source_map.items():
        src = None
        for attr in src_attrs:
            if hasattr(src_module, attr):
                src = getattr(src_module, attr)
                break
        if src is None:
            raise ValueError(
                f"Cannot find source for '{target_attr}' in "
                f"{src_module.__class__.__name__}. Tried: {src_attrs}"
            )
        dst = getattr(dst_module, target_attr)
        dst.weight.data.copy_(src.weight.data)
        if src.bias is not None and dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)
        elif src.bias is None and dst.bias is not None:
            nn.init.zeros_(dst.bias)


def replace_module(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    """Replace a sub-module identified by its dotted name, supporting ModuleList and ModuleDict."""
    parts = [p for p in full_name.split(".") if p]
    parent = model
    for part in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, list)):
            parent = parent[int(part)]
        elif isinstance(parent, (nn.ModuleDict, dict)):
            parent = parent[part]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if isinstance(parent, (nn.ModuleList, list)):
        parent[int(last)] = new_module
    elif isinstance(parent, (nn.ModuleDict, dict)):
        parent[last] = new_module
    else:
        setattr(parent, last, new_module)


def get_parent_and_child_name(model: nn.Module, full_name: str) -> Any:
    """Return (parent_module, child_attribute_name), supporting ModuleList and ModuleDict."""
    parts = [p for p in full_name.split(".") if p]
    parent = model
    for part in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, list)):
            parent = parent[int(part)]
        elif isinstance(parent, (nn.ModuleDict, dict)):
            parent = parent[part]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@contextmanager
def activation_hooks(
    model: nn.Module,
    target_module_type: type = nn.Linear,
    collect_input: bool = True,
    collect_output: bool = False,
) -> Any:
    """Context manager that collects activations for all modules of a given type."""
    activations: dict[str, Any] = {}
    hooks = []

    def make_hook(name) -> Any:
        """Make hook."""

        def hook(module, input, output) -> None:
            """Hook."""
            data = {}
            if collect_input:
                data["input"] = input[0].detach()
            if collect_output:
                data["output"] = (
                    output.detach()
                    if isinstance(output, torch.Tensor)
                    else output[0].detach()
                )
            activations[name] = data

        return hook

    for name, module in model.named_modules():
        if isinstance(module, target_module_type):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        yield activations
    finally:
        for hook in hooks:
            hook.remove()
