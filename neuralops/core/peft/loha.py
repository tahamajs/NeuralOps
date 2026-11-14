#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Low-Rank Hadamard Product (LoHa) Module.

Implements LoHa matrix adaptation, decomposing weight deltas into the Hadamard (elementwise)
product of two low-rank matrix pairs:
    ΔW = ((B_1 @ A_1) ⊙ (B_2 @ A_2)) * (alpha / r)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.peft.lora import LoRAParameter


class LoHaLinear(nn.Module):
    r"""
    Low-Rank Hadamard Product (LoHa) Linear Layer.

    Decomposes the adapted delta into the elementwise Hadamard product of two independent
    low-rank factorizations:
        ΔW = ((B_1 @ A_1) ⊙ (B_2 @ A_2)) * (alpha / r)

    Args:
        base_or_in_features: Underlying nn.Linear/nn.Module or input feature dimension.
        out_features: Output feature dimension (if base_or_in_features is int).
        rank: Target rank r for both low-rank factor pairs.
        alpha: LoHa scaling factor.
        dropout: Input dropout probability.
        bias: Whether to include bias if initializing standalone.
        freeze_base: Whether to freeze base layer parameters.
    """

    def __init__(
        self,
        base_or_in_features: nn.Module | int,
        out_features: int | None = None,
        rank: int = 16,
        r: int | None = None,
        alpha: float = 16.0,
        lora_alpha: float | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        effective_r = r if r is not None else rank
        self.rank = int(effective_r)
        self.r = self.rank

        effective_alpha = lora_alpha if lora_alpha is not None else alpha
        self.alpha = float(effective_alpha)
        self.lora_alpha = self.alpha
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False
        self._orig_base_weight: torch.Tensor | None = None

        factory_kwargs = {"device": device, "dtype": dtype}
        is_wrapper = isinstance(base_or_in_features, nn.Module)
        if freeze_base is None:
            self.freeze_base = is_wrapper
        else:
            self.freeze_base = bool(freeze_base)

        if is_wrapper:
            self.base_layer = base_or_in_features
            in_f = getattr(self.base_layer, "in_features", None)
            out_f = getattr(self.base_layer, "out_features", None)
            if in_f is None and hasattr(self.base_layer, "weight"):
                out_f, in_f = self.base_layer.weight.shape
            self.in_features = int(in_f) if in_f is not None else 0
            self.out_features = int(out_f) if out_f is not None else 0
            self.is_wrapper = True
        else:
            self.in_features = int(base_or_in_features)
            self.out_features = int(out_features if out_features is not None else base_or_in_features)
            self.base_layer = nn.Linear(self.in_features, self.out_features, bias=bias, **factory_kwargs)
            self.is_wrapper = False

        if self.rank > 0 and self.in_features > 0 and self.out_features > 0:
            self.lora_A1 = LoRAParameter(torch.empty((self.rank, self.in_features), **factory_kwargs))
            self.lora_B1 = LoRAParameter(torch.empty((self.out_features, self.rank), **factory_kwargs))
            self.lora_A2 = LoRAParameter(torch.empty((self.rank, self.in_features), **factory_kwargs))
            self.lora_B2 = LoRAParameter(torch.empty((self.out_features, self.rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A1", None)
            self.register_parameter("lora_B1", None)
            self.register_parameter("lora_A2", None)
            self.register_parameter("lora_B2", None)

        self.reset_parameters()
        if self.freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        """Initialize adapter matrices: Kaiming for A1, A2, B1, and zero for B2."""
        if not self.is_wrapper and hasattr(self.base_layer, "reset_parameters"):
            self.base_layer.reset_parameters()
        if self.rank > 0 and self.lora_A1 is not None:
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)

    def freeze_base_weights(self) -> None:
        """Freeze underlying base parameters."""
        for p in self.base_layer.parameters():
            p.requires_grad = False
        for param in (self.lora_A1, self.lora_B1, self.lora_A2, self.lora_B2):
            if param is not None:
                param.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze underlying base parameters."""
        for p in self.base_layer.parameters():
            p.requires_grad = True

    def compute_delta_weight(self) -> torch.Tensor:
        """Compute the full low-rank Hadamard delta weight matrix."""
        if self.rank <= 0 or self.lora_A1 is None or self.lora_B1 is None:
            return torch.zeros((self.out_features, self.in_features), device=self.base_layer.weight.device)
        w1 = self.lora_B1 @ self.lora_A1
        w2 = self.lora_B2 @ self.lora_A2
        return (w1 * w2) * self.scaling

    def merge(self) -> None:
        """Merge LoHa delta into base layer weight in-place."""
        if self.merged or self.rank <= 0 or self.lora_A1 is None:
            return
        with torch.no_grad():
            delta = self.compute_delta_weight()
            self._orig_base_weight = self.base_layer.weight.data.clone()
            self.base_layer.weight.data.add_(delta.to(self.base_layer.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        """Revert base layer weight back to unmerged state."""
        if not self.merged or self.rank <= 0 or self.lora_A1 is None:
            return
        with torch.no_grad():
            if self._orig_base_weight is not None:
                self.base_layer.weight.data.copy_(self._orig_base_weight)
            else:
                delta = self.compute_delta_weight()
                self.base_layer.weight.data.sub_(delta.to(self.base_layer.weight.dtype))
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base projection with LoHa adapter delta."""
        base_out = self.base_layer(x)
        if not self.merged and self.rank > 0 and self.lora_A1 is not None and self.lora_B2 is not None:
            drop_x = self.dropout(x)
            delta = self.compute_delta_weight()
            return base_out + F.linear(drop_x, delta)
        return base_out


__all__ = ["LoHaLinear"]
