"""Native, zero-dependency LoRA (Low-Rank Adaptation) module for PyTorch models.

Provides standalone low-rank adapters for linear layers without requiring external
PEFT packages. Supports Qwen, BERT, DistilBERT, RoBERTa, GPT-2, and T5 architectures.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_


class LoRALinear(nn.Module):
    """Linear layer wrapped with Low-Rank Adaptation (LoRA) adapters."""

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, self.r) if self.r > 0 else 1.0
        self.merged = False

        # Base linear layer (frozen)
        self.base_layer = linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA trainable low-rank matrices
        if self.r > 0:
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def reset_parameters(self) -> None:
        """Initialize lora_A with Kaiming uniform and lora_B with zeros."""
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Expose base layer weight for compatibility with hooks and inspectors."""
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Expose base layer bias for compatibility with hooks and inspectors."""
        return self.base_layer.bias

    def merge(self) -> None:
        """In-place merge LoRA delta weight into the base linear layer."""
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.data.add_(delta_w.to(self.base_layer.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        """Subtract LoRA delta weight to revert the base linear layer to unmerged state."""
        if self.r > 0 and self.merged and self.lora_A is not None and self.lora_B is not None:
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.data.sub_(delta_w.to(self.base_layer.weight.dtype))
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base projection with low-rank adapter delta."""
        # Ensure base layer matches input dtype
        if self.base_layer.weight.dtype != x.dtype and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.base_layer.to(dtype=x.dtype)
        if self.r > 0 and self.lora_A is not None and self.lora_A.weight.dtype != x.dtype:
            self.lora_A.to(dtype=x.dtype)
            self.lora_B.to(dtype=x.dtype)

        base_out = self.base_layer(x)
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return base_out + lora_out
        return base_out

    def __repr__(self) -> str:
        return (
            f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f})"
        )


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "query",
    "key",
    "value",
    "dense",
)


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[Union[Iterable[str], str]] = None,
) -> Tuple[nn.Module, int]:
    """Freeze base model parameters and inject LoRALinear into matching target modules.

    Args:
        model: PyTorch module (e.g. Qwen2, DistilBERT, RoBERTa).
        r: LoRA rank dimension.
        alpha: LoRA scaling factor (scaling = alpha / r).
        dropout: Dropout probability applied to LoRA adapter input.
        target_modules: Iterable or comma-separated string of target submodule names.

    Returns:
        Tuple of (model, number_of_adapted_linear_layers).
    """
    if target_modules is None:
        targets = set(DEFAULT_TARGET_MODULES)
    elif isinstance(target_modules, str):
        targets = {t.strip() for t in target_modules.split(",") if t.strip()}
    else:
        targets = set(target_modules)

    # 1. Freeze all existing parameters in the model
    for p in model.parameters():
        p.requires_grad = False

    adapted_count = 0

    def _replace_linear_recursive(module: nn.Module, prefix: str = "") -> None:
        nonlocal adapted_count
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if this child's name matches any target module
            is_target = any(target in name for target in targets) or any(target == name for target in targets)

            if isinstance(child, nn.Linear) and is_target:
                # Do not adapt classification heads
                if any(head in name.lower() for head in ("score", "classifier", "lm_head")):
                    continue
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, name, wrapped)
                adapted_count += 1
            else:
                _replace_linear_recursive(child, full_name)

    _replace_linear_recursive(model)
    return model, adapted_count


def get_lora_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return all parameters that require gradients (LoRA adapters + task heads)."""
    return [p for p in model.parameters() if p.requires_grad]


def save_lora_weights(model: nn.Module, save_path: str, metadata: Optional[dict] = None) -> None:
    """Save only trainable parameters (LoRA weights + task head) to disk."""
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    payload = {
        "state_dict": trainable_state,
        "metadata": metadata or {},
    }
    torch.save(payload, save_path)


def load_lora_weights(model: nn.Module, load_path: str, strict: bool = False) -> dict:
    """Load trainable parameters from saved checkpoint into model."""
    payload = torch.load(load_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
    }
