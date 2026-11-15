#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Weight-Decomposed Low-Rank Adaptation (DoRA)

Decomposes full weight updates into directional and magnitude components:
    W = m * (W_0 + Delta W) / ||W_0 + Delta W||_F
where Delta W = (alpha / r) * B @ A.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_

from .registry import register_method

logger = logging.getLogger(__name__)


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation layer for linear projections."""

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.r) if self.r > 0 else 1.0
        self.merged = False
        self._orig_weight: torch.Tensor | None = None
        self._saved_lora_A: nn.Linear | None = None
        self._saved_lora_B: nn.Linear | None = None
        self._saved_r: int = self.r
        self._saved_m: torch.Tensor | None = None

        if self.r > 0:
            # Magnitude parameter m: initialize to row-wise Frobenius norm of base weight
            with torch.no_grad():
                init_m = linear.weight.norm(p=2, dim=1, keepdim=True).squeeze(1)
            self.m = nn.Parameter(init_m.clone())
            self.lora_A = nn.Linear(linear.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, linear.out_features, bias=False)
            kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            zeros_(self.lora_B.weight)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.m = nn.Parameter(torch.ones(linear.out_features))
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def get_effective_weight(self) -> torch.Tensor:
        """Compute the effective combined weight matrix W_eff = m * (W_0 + Delta W) / ||W_0 + Delta W||."""
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return self.linear.weight
        delta = self.lora_B.weight @ self.lora_A.weight
        effective = self.linear.weight + (self.scaling * delta)
        direction = effective / effective.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return self.m.unsqueeze(1) * direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying weight-decomposed low-rank adaptation."""
        if self.merged or self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return self.linear(x)

        effective_w = self.get_effective_weight().to(dtype=x.dtype, device=x.device)
        bias = self.linear.bias.to(dtype=x.dtype, device=x.device) if self.linear.bias is not None else None
        return F.linear(x, effective_w, bias)

    def merge_weights(self) -> None:
        """Permanently fold DoRA directional and magnitude weights into base linear weight."""
        if self.merged or self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return
        with torch.no_grad():
            self._orig_weight = self.linear.weight.data.clone()
            self._saved_m = self.m.data.clone()
            self._saved_lora_A = self.lora_A
            self._saved_lora_B = self.lora_B
            self._saved_r = self.r

            merged = self.get_effective_weight()
            self.linear.weight.copy_(merged)
            self.r = 0
            self.lora_A = None
            self.lora_B = None
            self.merged = True

    def unmerge(self) -> None:
        """Reverse weight merge, recovering base weights bitwise and restoring adapters."""
        if not self.merged or self._orig_weight is None:
            return
        with torch.no_grad():
            self.linear.weight.copy_(self._orig_weight)
            if self._saved_m is not None:
                self.m.data.copy_(self._saved_m)
            self.lora_A = self._saved_lora_A
            self.lora_B = self._saved_lora_B
            self.r = self._saved_r
            self._orig_weight = None
            self._saved_m = None
            self.merged = False

    merge = merge_weights
    unmerge_weights = unmerge


def merge_dora_into_linear(dora_mod: DoRALinear) -> nn.Linear:
    """Fold DoRA adapter parameters into the underlying linear layer and return it."""
    dora_mod.merge_weights()
    return dora_mod.linear


@register_method("dora")
def apply_dora(model: nn.Module, config: Any) -> None:
    """Recursively inject DoRA layers into model attention and feedforward projections."""
    r = getattr(config, "lora_r", getattr(config, "r", 8))
    alpha = getattr(config, "lora_alpha", getattr(config, "alpha", 16.0))
    dropout = getattr(config, "lora_dropout", getattr(config, "dropout", 0.0))

    target_substrings = [
        "query",
        "key",
        "value",
        "dense",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "c_attn",
        "c_proj",
        "q_lin",
        "k_lin",
        "v_lin",
        "out_lin",
    ]
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(ts in name for ts in target_substrings):
            parent_name = ".".join(name.split(".")[:-1]) or ""
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            try:
                new_mod = DoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, new_mod)
            except Exception as e:
                logger.warning("Failed to inject DoRA into %s: %s", name, e)


__all__ = [
    "DoRALinear",
    "apply_dora",
    "merge_dora_into_linear",
]
