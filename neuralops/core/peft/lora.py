#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical LoRA Linear Implementation.

Provides the universal low-rank adaptation (LoRA) linear module supporting both
standalone usage and wrapping an existing nn.Linear or nn.Module layer.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralops.core.triton_kernels._base import HAS_TRITON, triton_dispatch
except ImportError:
    HAS_TRITON = False
    triton_dispatch = None


class LoRALinear(nn.Module):
    r"""
    Universal LoRA linear layer:
    Supports both wrapper mode: LoRALinear(base_layer, r=16, ...)
    and standalone mode: LoRALinear(in_features, out_features, bias=True, lora_rank=16, ...)
    """

    def __init__(
        self,
        base_or_in_features: nn.Module | int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        lora_rank: int = 16,
        r: int | None = None,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        dropout: float | None = None,
        freeze_base: bool = False,
        merge_weights: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        in_features: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if base_or_in_features is None:
            if in_features is not None:
                base_or_in_features = in_features
            else:
                base_or_in_features = 0

        effective_r = kwargs.get("rank", r if r is not None else lora_rank)
        effective_alpha = kwargs.get("alpha", lora_alpha)
        effective_dropout = dropout if dropout is not None else lora_dropout

        self.lora_rank = int(effective_r)
        self.r = self.lora_rank
        self.rank = self.lora_rank
        self.lora_alpha = float(effective_alpha)
        self.alpha = self.lora_alpha
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.scaling = self.lora_scaling
        self.lora_dropout = (
            nn.Dropout(p=effective_dropout) if effective_dropout > 0.0 else nn.Identity()
        )
        self.merged = False
        self.merge_weights = merge_weights
        self.freeze_base = freeze_base

        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(base_or_in_features, nn.Module):
            self.base_layer = base_or_in_features
            in_feat = getattr(self.base_layer, "in_features", None)
            out_feat = getattr(self.base_layer, "out_features", None)
            if in_feat is None and hasattr(self.base_layer, "weight"):
                if hasattr(self.base_layer, "nf"):
                    in_feat, out_feat = (
                        self.base_layer.weight.shape[0],
                        self.base_layer.weight.shape[1],
                    )
                else:
                    out_feat, in_feat = (
                        self.base_layer.weight.shape[0],
                        self.base_layer.weight.shape[1],
                    )
            self.in_features = in_feat
            self.out_features = out_feat
            self.is_wrapper = True
        else:
            self.is_wrapper = False
            self.in_features = int(base_or_in_features)
            self.out_features = int(
                out_features if out_features is not None else base_or_in_features
            )
            self.weight = nn.Parameter(
                torch.empty((self.out_features, self.in_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)

        if self.lora_rank > 0 and self.in_features and self.out_features:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_rank, self.in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((self.out_features, self.lora_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        if not self.is_wrapper and hasattr(self, "weight"):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        if self.is_wrapper:
            for p in self.base_layer.parameters():
                p.requires_grad = False
        else:
            if hasattr(self, "weight") and self.weight is not None:
                self.weight.requires_grad = False
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        if self.is_wrapper:
            for p in self.base_layer.parameters():
                p.requires_grad = True
        else:
            if hasattr(self, "weight") and self.weight is not None:
                self.weight.requires_grad = True
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.requires_grad = True

    def merge(self) -> None:
        if self.merged or self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.lora_scaling
            if self.is_wrapper:
                if hasattr(self.base_layer, "nf"):
                    delta = delta.t()
                self.base_layer.weight.data.add_(delta.to(self.base_layer.weight.dtype))
            else:
                self.weight.data.add_(delta.to(self.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        if not self.merged or self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.lora_scaling
            if self.is_wrapper:
                if hasattr(self.base_layer, "nf"):
                    delta = delta.t()
                self.base_layer.weight.data.sub_(delta.to(self.base_layer.weight.dtype))
            else:
                self.weight.data.sub_(delta.to(self.weight.dtype))
            self.merged = False

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> LoRALinear:
        """Wrap an existing nn.Linear layer with a LoRA adapter."""
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"from_linear expects nn.Linear, got {type(layer).__name__}")

        mod = cls(
            base_or_in_features=layer.in_features,
            out_features=layer.out_features,
            bias=(layer.bias is not None),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
            dtype=layer.weight.dtype,
            device=layer.weight.device,
        )
        with torch.no_grad():
            mod.weight.data.copy_(layer.weight.data)
            if layer.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(layer.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_wrapper:
            res = self.base_layer(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if (
            not self.merged
            and self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            drop_x = self.lora_dropout(x)
            if HAS_TRITON and triton_dispatch is not None and drop_x.is_cuda:
                try:
                    lora_out = (
                        triton_dispatch("low_rank", drop_x, self.lora_B, self.lora_A, bias=None)
                        * self.lora_scaling
                    )
                except Exception:
                    lora_out = (
                        F.linear(F.linear(drop_x, self.lora_A), self.lora_B) * self.lora_scaling
                    )
            else:
                lora_out = F.linear(F.linear(drop_x, self.lora_A), self.lora_B) * self.lora_scaling
            res = res + lora_out
        return res


__all__ = ["LoRALinear"]
