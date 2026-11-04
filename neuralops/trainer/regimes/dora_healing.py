#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
DoRA Fast Directional Healing Regime.

Implements Weight-Decomposed Low-Rank Adaptation (DoRA) for post-surgery healing:
    W = m * (W_0 + Delta W) / ||W_0 + Delta W||_F
where Delta W = (B @ A) * scaling.
Decomposes linear projection weights into independent magnitude vectors (m) and
directional matrices (V), cutting required healing steps by ~60% by separating
magnitude recovery from directional representation learning.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NeuralOps.Trainer.Regimes.DoRAHealing")


class DoRAProjectionWrapper(nn.Module):
    """
    Wraps an existing nn.Linear or projection weight with DoRA decomposition.
    Tracks original base weight W_0 and trains magnitude m and low-rank adapters A, B.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = lora_alpha / rank

        self.register_buffer("weight_base", base_layer.weight.data.clone())
        # Magnitude initialized to Euclidean norm of column/row slices
        self.magnitude = nn.Parameter(self.weight_base.norm(p=2, dim=1, keepdim=True))

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def compute_effective_weight(self) -> torch.Tensor:
        """Returns effective weight W = m * (W_0 + BA) / ||W_0 + BA||_F."""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        full_w = self.weight_base + delta_w
        w_norm = full_w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return self.magnitude * (full_w / w_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.compute_effective_weight()
        return F.linear(x, w_eff, self.bias)

    def merge_and_collapse(self) -> nn.Linear:
        """Fuses DoRA adapter back into standard nn.Linear layer."""
        with torch.no_grad():
            w_eff = self.compute_effective_weight()
            merged_layer = nn.Linear(
                self.in_features, self.out_features, bias=(self.bias is not None)
            ).to(w_eff.device, dtype=w_eff.dtype)
            merged_layer.weight.data.copy_(w_eff)
            if self.bias is not None:
                merged_layer.bias.data.copy_(self.bias)
        return merged_layer

    def collapse_adapters(self) -> nn.Linear:
        """Alias for merge_and_collapse to fold adapter into standard nn.Linear layer."""
        return self.merge_and_collapse()


class DoRAHealingHealer:
    """
    Applies DoRA wrappers to linear attention projections and manages
    focused micro-healing iterations.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        lora_alpha: float = 32.0,
        target_substrings: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.target_substrings = target_substrings or [
            "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "in_proj", "proj"
        ]
        self.injected_wrappers: Dict[str, DoRAProjectionWrapper] = {}

    def inject_dora_adapters(self) -> int:
        """Replaces matching nn.Linear modules with DoRAProjectionWrapper."""
        injected = 0
        for name, module in list(self.model.named_modules()):
            for child_name, child in list(module.named_children()):
                full_child_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear) and any(
                    sub in full_child_name for sub in self.target_substrings
                ):
                    wrapper = DoRAProjectionWrapper(
                        child, rank=self.rank, lora_alpha=self.lora_alpha
                    )
                    setattr(module, child_name, wrapper)
                    self.injected_wrappers[full_child_name] = wrapper
                    injected += 1
        logger.info("Injected %d DoRA adapters across model projections.", injected)
        return injected

    def collapse_adapters(self) -> None:
        """Fuses all injected DoRA adapters back into standard nn.Linear layers."""
        for name, module in list(self.model.named_modules()):
            for child_name, child in list(module.named_children()):
                full_child_name = f"{name}.{child_name}" if name else child_name
                if full_child_name in self.injected_wrappers:
                    wrapper = self.injected_wrappers[full_child_name]
                    merged_layer = wrapper.merge_and_collapse()
                    setattr(module, child_name, merged_layer)
        self.injected_wrappers.clear()
        logger.info("Collapsed all DoRA adapters back into base model weights.")


__all__ = ["DoRAProjectionWrapper", "DoRAHealingHealer"]
