"""Native, zero-dependency LoRA (Low-Rank Adaptation) module for PyTorch models.

Provides standalone low-rank adapters for linear layers without requiring external
PEFT packages. Supports Qwen, BERT, DistilBERT, RoBERTa, GPT-2, and T5 architectures.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, zeros_


class LoRALinear(nn.Module):
    """Linear layer wrapped with Low-Rank Adaptation (LoRA) adapters."""

    def __init__(
        self,
        linear: nn.Module,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hasattr(linear, "in_features"):
            self.in_features = linear.in_features
            self.out_features = linear.out_features
        elif hasattr(linear, "weight"):
            # HuggingFace Conv1D (used in GPT-2) has weight shape [in_features, out_features]
            self.in_features = linear.weight.shape[0]
            self.out_features = linear.weight.shape[1]
        else:
            raise ValueError(f"Unsupported linear module: {type(linear)}")
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(1, self.r) if self.r > 0 else 1.0
        self.merged = False

        # Base linear layer (frozen)
        self.base_layer = linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        device = linear.weight.device
        # Trainable LoRA parameters are kept in FP32 for numerical stability
        # and compatibility with torch.amp.GradScaler in mixed precision.
        dtype = torch.float32

        # LoRA trainable low-rank matrices
        if self.r > 0:
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False, device=device, dtype=dtype)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False, device=device, dtype=dtype)
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
    def bias(self) -> torch.Tensor | None:
        """Expose base layer bias for compatibility with hooks and inspectors."""
        return self.base_layer.bias

    def merge(self) -> None:
        """In-place merge LoRA delta weight into the base linear layer."""
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            is_conv1d = "conv1d" in self.base_layer.__class__.__name__.lower() or (
                not isinstance(self.base_layer, nn.Linear) and self.base_layer.weight.shape == (self.in_features, self.out_features)
            )
            if is_conv1d:
                delta_w = delta_w.t()
            self.base_layer.weight.data.add_(delta_w.to(self.base_layer.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        """Subtract LoRA delta weight to revert the base linear layer to unmerged state."""
        if self.r > 0 and self.merged and self.lora_A is not None and self.lora_B is not None:
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            is_conv1d = "conv1d" in self.base_layer.__class__.__name__.lower() or (
                not isinstance(self.base_layer, nn.Linear) and self.base_layer.weight.shape == (self.in_features, self.out_features)
            )
            if is_conv1d:
                delta_w = delta_w.t()
            self.base_layer.weight.data.sub_(delta_w.to(self.base_layer.weight.dtype))
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base projection with low-rank adapter delta."""
        base_out = self.base_layer(x)
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            lora_in = self.lora_dropout(x).to(dtype=self.lora_A.weight.dtype)
            lora_out = self.lora_B(self.lora_A(lora_in)) * self.scaling
            return base_out + lora_out.to(dtype=base_out.dtype)
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
    "q",
    "k",
    "v",
    "o",
    "wi",
    "wo",
    "wi_0",
    "wi_1",
    "c_attn",
    "c_proj",
    "c_fc",
    "in_proj",
    "out_proj",
    "down",
    "up",
    "fc1",
    "fc2",
    "Wqkv",
    "qkv_proj",
    "qkv",
)


def find_all_linear_submodule_names(model: nn.Module) -> set[str]:
    """Dynamically extract all Linear/Conv1D layer leaf names from model, excluding task heads."""
    names = set()
    for name, module in model.named_modules():
        is_linear_like = isinstance(module, (nn.Linear, nn.Conv1d)) or module.__class__.__name__ in ("Conv1D", "Linear", "LoRALinear")
        if is_linear_like:
            if not any(head in name.lower() for head in ("score", "classifier", "lm_head")):
                leaf_name = name.split(".")[-1]
                names.add(leaf_name)
    return names


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Iterable[str] | str | None = None,
) -> tuple[nn.Module, int]:
    """Freeze base model parameters and inject LoRALinear into matching target modules.

    Args:
        model: PyTorch module (e.g. Qwen2, DistilBERT, RoBERTa, GPT-2).
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
            is_linear_like = isinstance(child, nn.Linear) or child.__class__.__name__ == "Conv1D"

            if is_linear_like and is_target:
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


def get_lora_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all parameters that require gradients (LoRA adapters + task heads)."""
    return [p for p in model.parameters() if p.requires_grad]


def save_lora_weights(model: nn.Module, save_path: str, metadata: dict | None = None) -> None:
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
    with open(save_path, "wb") as f:
        torch.save(payload, f)


def load_lora_weights(model: nn.Module, load_path: str, strict: bool = False) -> dict:
    """Load trainable parameters from saved checkpoint into model."""
    with open(load_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
    }
