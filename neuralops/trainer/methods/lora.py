#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LoRA Adaptation Suite

Implements Low-Rank Adaptation (LoRA), DoRA, and DisLoRA parameter-efficient fine-tuning methods.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_

from .registry import register_method

logger = logging.getLogger(__name__)


from neuralops.utils.lora import LoRALinear as _BaseLoRALinear
from .dora import DoRALinear, apply_dora


class LoRALinear(_BaseLoRALinear):
    """Universal LoRA linear layer with bitwise-reversible merge/unmerge and strict contract conformance."""

    def __init__(
        self,
        base_or_in_features: nn.Module | int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(base_or_in_features, nn.Module):
            # Map positional arguments: LoRALinear(module, r, alpha, dropout)
            if len(args) >= 1 and "r" not in kwargs and "lora_rank" not in kwargs:
                kwargs["r"] = args[0]
            if len(args) >= 2 and "alpha" not in kwargs and "lora_alpha" not in kwargs:
                kwargs["alpha"] = args[1]
            if len(args) >= 3 and "dropout" not in kwargs and "lora_dropout" not in kwargs:
                kwargs["dropout"] = args[2]
            args = ()
        super().__init__(base_or_in_features, *args, **kwargs)
        self._unmerged_weight: torch.Tensor | None = None

        # Canonical LoRA zero-initializes lora_B so delta is exactly zero at start
        if self.lora_rank > 0 and self.lora_B is not None:
            with torch.no_grad():
                nn.init.zeros_(self.lora_B)

    def merge(self) -> None:
        """Permanently fold adapter delta (lora_B @ lora_A) * scaling into base weight."""
        if self.merged or self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return
        target = self.base_layer.weight if self.is_wrapper else self.weight
        self._unmerged_weight = target.data.clone()
        super().merge()

    def unmerge(self) -> None:
        """Reverse merge exactly, recovering original unmerged weights bitwise."""
        if not self.merged or self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return
        if self._unmerged_weight is not None:
            target = self.base_layer.weight if self.is_wrapper else self.weight
            target.data.copy_(self._unmerged_weight)
            self._unmerged_weight = None
            self.merged = False
        else:
            super().unmerge()


class DisLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        beta: float = 0.5,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        self.r = r
        self.beta = beta
        self.capacity_factor = capacity_factor
        if r > 0:
            self.lora_A = nn.Linear(linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, linear.out_features, bias=False)
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout)
            self.register_buffer("_eye", torch.eye(r))
        else:
            self.lora_A = self.lora_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()

    def forward(self, x) -> Any:
        """Forward pass — compute module output from inputs."""
        base = self.linear(x)
        if self.r > 0:
            try:
                from neuralops.core.triton_kernels.low_rank import fused_down_up_gemm_bias_triton

                drop_x = self.dropout(x)
                lora = (
                    fused_down_up_gemm_bias_triton(drop_x, self.lora_A.weight.t(), self.lora_B.weight.t())
                    * self.scaling
                )
            except (ImportError, RuntimeError, Exception) as exc:
                # allowed-swallow: triton kernel fallback to PyTorch
                logger.debug("Fused Triton GEMM unavailable, using PyTorch fallback: %s", exc)
                lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return base + lora
        return base

    def ortho_loss(self) -> Any:
        """Ortho loss."""
        if self.lora_A is None:
            return 0.0
        A = self.lora_A.weight
        B = self.lora_B.weight
        reg = (A @ A.T - self._eye).pow(2).sum() + (B.T @ B - self._eye).pow(2).sum()
        return self.beta * reg


def _apply_adapters(model, r, alpha, dropout, method_type="lora", **kwargs) -> None:
    """apply adapters."""
    target_substrings = [
        "query",
        "key",
        "value",
        "dense",
        "intermediate.dense",
        "output.dense",
        "q_lin",
        "k_lin",
        "v_lin",
        "out_lin",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
    ]
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(ts in name for ts in target_substrings):
            parent = model.get_submodule(".".join(name.split(".")[:-1]) or "")
            child = name.split(".")[-1]
            try:
                if method_type == "dora":
                    new_module = DoRALinear(module, r, alpha, dropout)
                elif method_type == "dislora":
                    new_module = DisLoRALinear(
                        module,
                        r,
                        alpha,
                        dropout,
                        beta=kwargs.get("dislora_beta", 0.5),
                        capacity_factor=kwargs.get("dislora_capacity_factor", 1.0),
                    )
                else:
                    new_module = LoRALinear(module, r, alpha, dropout)
                setattr(parent, child, new_module)
            except Exception as e:
                logger.warning(f"Failed to replace {name}: {e}")


@register_method("lora")
def apply_lora(model, config) -> None:
    """Apply lora."""
    r = getattr(config, "lora_r", getattr(config, "r", 16))
    alpha = getattr(config, "lora_alpha", getattr(config, "alpha", 32.0))
    dropout = getattr(config, "lora_dropout", getattr(config, "dropout", 0.0))
    _apply_adapters(model, r, alpha, dropout, "lora")


__all__ = [
    "LoRALinear",
    "DoRALinear",
    "DisLoRALinear",
    "apply_lora",
    "apply_dora",
    "apply_dislora",
]


@register_method("dislora")
def apply_dislora(model, config) -> None:
    """Apply dislora."""
    r = getattr(config, "lora_r", getattr(config, "r", 16))
    alpha = getattr(config, "lora_alpha", getattr(config, "alpha", 32.0))
    dropout = getattr(config, "lora_dropout", getattr(config, "dropout", 0.0))
    beta = getattr(config, "dislora_beta", 0.5)
    capacity_factor = getattr(config, "dislora_capacity_factor", 1.0)
    _apply_adapters(
        model,
        r,
        alpha,
        dropout,
        "dislora",
        dislora_beta=beta,
        dislora_capacity_factor=capacity_factor,
    )
