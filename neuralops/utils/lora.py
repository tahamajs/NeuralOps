import torch
import torch.nn as nn
"""Backward-compatibility shim. Moved to neuralops.core.utils.lora."""
import warnings
warnings.warn(
    "neuralops.utils.lora is deprecated. Use neuralops.core.utils.lora.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.core.utils.lora import *  # noqa: F401,F403
from neuralops.core.utils.lora import LoRALinear  # noqa: F401


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_proj", "c_fc",
    "q_lin", "k_lin", "v_lin", "out_lin",
]

def find_all_linear_submodule_names(model: nn.Module) -> set[str]:
    """Dynamically extract all Linear/Conv1D layer leaf names from model, excluding task heads."""
    names = set()
    for name, module in model.named_modules():
        is_linear_like = isinstance(
            module, (nn.Linear, nn.Conv1d)
        ) or module.__class__.__name__ in ("Conv1D", "Linear", "LoRALinear")
        if is_linear_like:
            if not any(head in name.lower() for head in ("score", "classifier", "lm_head")):
                leaf_name = name.split(".")[-1]
                names.add(leaf_name)
    return names

def get_lora_trainable_parameters(model: nn.Module):
    """Return all parameters that require gradients (LoRA adapters + task heads)."""
    return [p for p in model.parameters() if p.requires_grad]

def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules = None,
):
    """Freeze base model parameters and inject LoRALinear into matching target modules."""
    if target_modules is None:
        targets = set(DEFAULT_TARGET_MODULES)
    elif isinstance(target_modules, str):
        targets = {t.strip() for t in target_modules.split(",") if t.strip()}
    else:
        targets = set(target_modules)

    for p in model.parameters():
        p.requires_grad = False

    adapted_count = 0

    def _replace_linear_recursive(module: nn.Module, prefix: str = "") -> None:
        nonlocal adapted_count
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            is_target = any(target in name for target in targets) or any(
                target == name for target in targets
            )
            is_linear_like = isinstance(child, nn.Linear) or child.__class__.__name__ == "Conv1D"

            if is_linear_like and is_target:
                if any(head in name.lower() for head in ("score", "classifier", "lm_head")):
                    continue
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, name, wrapped)
                adapted_count += 1
            else:
                _replace_linear_recursive(child, full_name)

    _replace_linear_recursive(model)
    return model, adapted_count
