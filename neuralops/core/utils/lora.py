"""LoRA and Factorized Linear shared layers."""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn

__all__ = ["LoRALinear", "FactorizedLinear"]


class LoRALinear(nn.Module):
    """Flexible, standard LoRA linear layer supporting either (in_features, out_features)
    or wrapping an existing nn.Linear base layer.

    Follows PEFT conventions:
    - lora_A: (lora_rank, in_features) with kaiming uniform initialization.
    - lora_B: (out_features, lora_rank) with zero initialization (or small normal init).
    """

    def __init__(
        self,
        in_features: int | nn.Linear,
        out_features: int | None = None,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        fk = {"device": device, "dtype": dtype}

        if isinstance(in_features, nn.Linear):
            base_linear = in_features
            self.in_features = base_linear.in_features
            self.out_features = base_linear.out_features
            self.weight = base_linear.weight
            self.bias = base_linear.bias
            lora_rank = kwargs.get("rank", lora_rank)
            lora_alpha = kwargs.get("alpha", lora_alpha)
        else:
            self.in_features = int(in_features)
            self.out_features = int(out_features) if out_features is not None else int(in_features)
            self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **fk))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features, **fk))
            else:
                self.register_parameter("bias", None)

        self.lora_rank = int(kwargs.get("rank", lora_rank))
        self.lora_alpha = float(kwargs.get("alpha", lora_alpha))
        self.rank = self.lora_rank
        self.alpha = self.lora_alpha
        self.scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.lora_scaling = self.scaling
        self.freeze_base = freeze_base
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, self.in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, self.lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

        if self.freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        if hasattr(self, "weight") and self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if hasattr(self, "bias") and self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_rank > 0:
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                nn.init.zeros_(self.lora_B)
                self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        if hasattr(self, "weight") and self.weight is not None:
            self.weight.requires_grad = False
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        if hasattr(self, "weight") and self.weight is not None:
            self.weight.requires_grad = True
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.lora_scaling
            return out + lora_out
        return out


class FactorizedLinear(nn.Module):
    """Low-rank factorized linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x))
